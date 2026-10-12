#include "rt64_water_renderer.h"
#include "rt64_render_target.h"
#include "hle/rt64_workload.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include "shaders/WaterVS.hlsl.spirv.h"
#include "shaders/WaterPS.hlsl.spirv.h"
#include "shaders/WaterClassicPS.hlsl.spirv.h"
#include "shaders/WaterClassicMSPS.hlsl.spirv.h"
#include "shaders/WaterNormalsCS.hlsl.spirv.h"
#include "shaders/WaterCaptureCS.hlsl.spirv.h"
#include "shaders/WaterCaptureMSCS.hlsl.spirv.h"
#include "shaders/WaterOriginalCS.hlsl.spirv.h"
#include "shaders/WaterOriginalMSCS.hlsl.spirv.h"
#include "shaders/WaterDepthReduceCS.hlsl.spirv.h"
#include "shaders/WaterFieldCS.hlsl.spirv.h"
#include "shaders/WaterStatsCS.hlsl.spirv.h"
#include "shaders/WaterSurfacePS.hlsl.spirv.h"
#include "shaders/WaterSprayVS.hlsl.spirv.h"
#include "shaders/WaterSprayPS.hlsl.spirv.h"
#include "shaders/WaterSprayMSPS.hlsl.spirv.h"
#ifdef _WIN32
#include "shaders/WaterVS.hlsl.dxil.h"
#include "shaders/WaterPS.hlsl.dxil.h"
#include "shaders/WaterClassicPS.hlsl.dxil.h"
#include "shaders/WaterClassicMSPS.hlsl.dxil.h"
#include "shaders/WaterNormalsCS.hlsl.dxil.h"
#include "shaders/WaterCaptureCS.hlsl.dxil.h"
#include "shaders/WaterCaptureMSCS.hlsl.dxil.h"
#include "shaders/WaterOriginalCS.hlsl.dxil.h"
#include "shaders/WaterOriginalMSCS.hlsl.dxil.h"
#include "shaders/WaterDepthReduceCS.hlsl.dxil.h"
#include "shaders/WaterFieldCS.hlsl.dxil.h"
#include "shaders/WaterStatsCS.hlsl.dxil.h"
#include "shaders/WaterSurfacePS.hlsl.dxil.h"
#include "shaders/WaterSprayVS.hlsl.dxil.h"
#include "shaders/WaterSprayPS.hlsl.dxil.h"
#include "shaders/WaterSprayMSPS.hlsl.dxil.h"
#elif defined(__APPLE__)
#include "shaders/WaterVS.hlsl.metal.h"
#include "shaders/WaterPS.hlsl.metal.h"
#include "shaders/WaterClassicPS.hlsl.metal.h"
#include "shaders/WaterClassicMSPS.hlsl.metal.h"
#include "shaders/WaterNormalsCS.hlsl.metal.h"
#include "shaders/WaterCaptureCS.hlsl.metal.h"
#include "shaders/WaterCaptureMSCS.hlsl.metal.h"
#include "shaders/WaterOriginalCS.hlsl.metal.h"
#include "shaders/WaterOriginalMSCS.hlsl.metal.h"
#include "shaders/WaterDepthReduceCS.hlsl.metal.h"
#include "shaders/WaterFieldCS.hlsl.metal.h"
#include "shaders/WaterStatsCS.hlsl.metal.h"
#include "shaders/WaterSurfacePS.hlsl.metal.h"
#include "shaders/WaterSprayVS.hlsl.metal.h"
#include "shaders/WaterSprayPS.hlsl.metal.h"
#include "shaders/WaterSprayMSPS.hlsl.metal.h"
#endif

#ifdef _WIN32
#   define CREATE_SHADER_INPUTS(DXIL_BLOB, SPIRV_BLOB, MSL_BLOB, ENTRY_NAME, SHADER_FORMAT)\
        (SHADER_FORMAT == RenderShaderFormat::DXIL) ? DXIL_BLOB : (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? SPIRV_BLOB : nullptr,\
        (SHADER_FORMAT == RenderShaderFormat::DXIL) ? sizeof(DXIL_BLOB) : (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? sizeof(SPIRV_BLOB) : 0,\
        ENTRY_NAME,\
        SHADER_FORMAT
#elif defined(__APPLE__)
#   define CREATE_SHADER_INPUTS(DXIL_BLOB, SPIRV_BLOB, MSL_BLOB, ENTRY_NAME, SHADER_FORMAT)\
        (SHADER_FORMAT == RenderShaderFormat::METAL) ? MSL_BLOB : (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? SPIRV_BLOB : nullptr,\
        (SHADER_FORMAT == RenderShaderFormat::METAL) ? sizeof(MSL_BLOB) : (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? sizeof(SPIRV_BLOB) : 0,\
        ENTRY_NAME,\
        SHADER_FORMAT
#else
#   define CREATE_SHADER_INPUTS(DXIL_BLOB, SPIRV_BLOB, MSL_BLOB, ENTRY_NAME, SHADER_FORMAT)\
        (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? SPIRV_BLOB : nullptr,\
        (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? sizeof(SPIRV_BLOB) : 0,\
        ENTRY_NAME,\
        SHADER_FORMAT
#endif

namespace RT64 {
namespace {
template <typename T> std::unique_ptr<T> checkedWaterResource(std::unique_ptr<T> resource) {
    if (!resource) throw std::runtime_error("water shader or pipeline creation failed");
    return resource;
}
}
WaterRenderer::WaterRenderer(RenderDevice *device, RenderShaderFormat format, bool hdr) : usesHDR(hdr) {
    if (std::getenv("WR64_TEST_WATER_SHADER_FAILURE")) throw std::runtime_error("injected water shader failure");
    if (std::getenv("WR64_WATER_PROFILE") && device->getCapabilities().queryPools) profilePool = device->createQueryPool(128);
    descriptorSet = std::make_unique<WaterDescriptorSet>(device);
    RenderPipelineLayoutBuilder builder;
    builder.begin(false, true); // Water vertices use the input assembler on D3D12.
    builder.addPushConstant(0, 0, sizeof(interop::WaterPushConstants), RenderShaderStageFlag::VERTEX | RenderShaderStageFlag::PIXEL);
    builder.addDescriptorSet(*descriptorSet);
    builder.end();
    layout = checkedWaterResource(builder.create(device));
    vertexShader = checkedWaterResource(device->createShader(CREATE_SHADER_INPUTS(WaterVSBlobDXIL, WaterVSBlobSPIRV, WaterVSBlobMSL, "VSMain", format)));
    pixelShader = checkedWaterResource(device->createShader(CREATE_SHADER_INPUTS(WaterPSBlobDXIL, WaterPSBlobSPIRV, WaterPSBlobMSL, "PSMain", format)));
    classicShader=checkedWaterResource(device->createShader(CREATE_SHADER_INPUTS(WaterClassicPSBlobDXIL,WaterClassicPSBlobSPIRV,WaterClassicPSBlobMSL,"PSMain",format)));
    classicMSShader=checkedWaterResource(device->createShader(CREATE_SHADER_INPUTS(WaterClassicMSPSBlobDXIL,WaterClassicMSPSBlobSPIRV,WaterClassicMSPSBlobMSL,"PSMain",format)));
    surfaceShader= checkedWaterResource(device->createShader(CREATE_SHADER_INPUTS(WaterSurfacePSBlobDXIL,WaterSurfacePSBlobSPIRV,WaterSurfacePSBlobMSL,"PSMain",format)));
    sprayVertexShader= checkedWaterResource(device->createShader(CREATE_SHADER_INPUTS(WaterSprayVSBlobDXIL,WaterSprayVSBlobSPIRV,WaterSprayVSBlobMSL,"VSMain",format)));
    sprayPixelShader= checkedWaterResource(device->createShader(CREATE_SHADER_INPUTS(WaterSprayPSBlobDXIL,WaterSprayPSBlobSPIRV,WaterSprayPSBlobMSL,"PSMain",format)));
    sprayMSPixelShader= checkedWaterResource(device->createShader(CREATE_SHADER_INPUTS(WaterSprayMSPSBlobDXIL,WaterSprayMSPSBlobSPIRV,WaterSprayMSPSBlobMSL,"PSMain",format)));
    WaterSpraySet spraySet;
    builder.begin();
    builder.addPushConstant(0,0,sizeof(interop::WaterPushConstants),RenderShaderStageFlag::VERTEX|RenderShaderStageFlag::PIXEL);
    builder.addDescriptorSet(spraySet);
    builder.end();
    sprayLayout= checkedWaterResource(builder.create(device));
    normalSet = std::make_unique<WaterNormalSet>(device);
    builder.begin();
    builder.addPushConstant(0,0,sizeof(uint32_t),RenderShaderStageFlag::COMPUTE);
    builder.addDescriptorSet(*normalSet);
    builder.end();
    normalShader.pipelineLayout = checkedWaterResource(builder.create(device));
    auto cs = checkedWaterResource(device->createShader(CREATE_SHADER_INPUTS(WaterNormalsCSBlobDXIL, WaterNormalsCSBlobSPIRV, WaterNormalsCSBlobMSL, "CSMain", format)));
    normalShader.pipeline = checkedWaterResource(device->createComputePipeline(RenderComputePipelineDesc(normalShader.pipelineLayout.get(),cs.get(),64,1,1)));
    WaterCaptureSet captureSet;
    builder.begin();
    builder.addPushConstant(0,0,16,RenderShaderStageFlag::COMPUTE);
    builder.addDescriptorSet(captureSet);
    builder.end();
    captureShader.pipelineLayout = checkedWaterResource(builder.create(device));
    captureMSShader.pipelineLayout = checkedWaterResource(builder.create(device));
    auto capture = checkedWaterResource(device->createShader(CREATE_SHADER_INPUTS(WaterCaptureCSBlobDXIL, WaterCaptureCSBlobSPIRV, WaterCaptureCSBlobMSL, "CSMain", format)));
    auto captureMS = checkedWaterResource(device->createShader(CREATE_SHADER_INPUTS(WaterCaptureMSCSBlobDXIL, WaterCaptureMSCSBlobSPIRV, WaterCaptureMSCSBlobMSL, "CSMain", format)));
    captureShader.pipeline = checkedWaterResource(device->createComputePipeline(RenderComputePipelineDesc(captureShader.pipelineLayout.get(),capture.get(),8,8,1)));
    captureMSShader.pipeline = checkedWaterResource(device->createComputePipeline(RenderComputePipelineDesc(captureMSShader.pipelineLayout.get(),captureMS.get(),8,8,1)));
    WaterOriginalSet originalSet;
    builder.begin();
    builder.addPushConstant(0,0,16,RenderShaderStageFlag::COMPUTE);
    builder.addDescriptorSet(originalSet);
    builder.end();
    originalShader.pipelineLayout = checkedWaterResource(builder.create(device));
    originalMSShader.pipelineLayout = checkedWaterResource(builder.create(device));
    auto original = checkedWaterResource(device->createShader(CREATE_SHADER_INPUTS(WaterOriginalCSBlobDXIL, WaterOriginalCSBlobSPIRV, WaterOriginalCSBlobMSL, "CSMain", format)));
    auto originalMS = checkedWaterResource(device->createShader(CREATE_SHADER_INPUTS(WaterOriginalMSCSBlobDXIL, WaterOriginalMSCSBlobSPIRV, WaterOriginalMSCSBlobMSL, "CSMain", format)));
    originalShader.pipeline = checkedWaterResource(device->createComputePipeline(RenderComputePipelineDesc(originalShader.pipelineLayout.get(),original.get(),8,8,1)));
    originalMSShader.pipeline = checkedWaterResource(device->createComputePipeline(RenderComputePipelineDesc(originalMSShader.pipelineLayout.get(),originalMS.get(),8,8,1)));
    WaterDepthReduceSet reduceSet;
    builder.begin();
    builder.addPushConstant(0,0,16,RenderShaderStageFlag::COMPUTE);
    builder.addDescriptorSet(reduceSet);
    builder.end();
    depthReduceShader.pipelineLayout = checkedWaterResource(builder.create(device));
    auto reduce = checkedWaterResource(device->createShader(CREATE_SHADER_INPUTS(WaterDepthReduceCSBlobDXIL, WaterDepthReduceCSBlobSPIRV, WaterDepthReduceCSBlobMSL, "CSMain", format)));
    depthReduceShader.pipeline = checkedWaterResource(device->createComputePipeline(RenderComputePipelineDesc(depthReduceShader.pipelineLayout.get(),reduce.get(),8,8,1)));
    WaterFieldSet fieldSet;
    builder.begin();
    builder.addPushConstant(0,0,sizeof(interop::WaterFieldPushConstants),RenderShaderStageFlag::COMPUTE);
    builder.addDescriptorSet(fieldSet);
    builder.end();
    fieldShader.pipelineLayout= checkedWaterResource(builder.create(device));
    auto field= checkedWaterResource(device->createShader(CREATE_SHADER_INPUTS(WaterFieldCSBlobDXIL,WaterFieldCSBlobSPIRV,WaterFieldCSBlobMSL,"CSMain",format)));
    fieldShader.pipeline= checkedWaterResource(device->createComputePipeline(RenderComputePipelineDesc(fieldShader.pipelineLayout.get(),field.get(),8,8,1)));
    if (std::getenv("WR64_WATER_STATS")) {
        statsSet=std::make_unique<WaterStatsSet>(device);
        builder.begin(); builder.addDescriptorSet(*statsSet); builder.end();
        statsShader.pipelineLayout= checkedWaterResource(builder.create(device));
        auto stats= checkedWaterResource(device->createShader(CREATE_SHADER_INPUTS(WaterStatsCSBlobDXIL,WaterStatsCSBlobSPIRV,WaterStatsCSBlobMSL,"CSMain",format)));
        statsShader.pipeline= checkedWaterResource(device->createComputePipeline(RenderComputePipelineDesc(statsShader.pipelineLayout.get(),stats.get(),64,1,1)));
        statsBuffer=device->createBuffer(RenderBufferDesc::DefaultBuffer(128*16,RenderBufferFlag::STORAGE|RenderBufferFlag::UNORDERED_ACCESS));
        statsReadback=device->createBuffer(RenderBufferDesc::ReadbackBuffer(128*16));
        statsSet->setBuffer(statsSet->output,statsBuffer.get(),RenderBufferStructuredView(16));
    }
}

void WaterRenderer::bindBuffer() {
    for (auto &set:fieldSets) {
        if (set) set->setBuffer(set->draws,drawBuffer.get(),RenderBufferStructuredView(sizeof(interop::WaterDrawParams)));
    }
    descriptorSet->setBuffer(descriptorSet->draws, drawBuffer.get(), RenderBufferStructuredView(sizeof(interop::WaterDrawParams)));
    for (const auto &[key,index] : sceneKeys) {
        auto &spray=scenes[index].spraySet;
        spray->setBuffer(spray->draws,drawBuffer.get(),RenderBufferStructuredView(sizeof(interop::WaterDrawParams)));
        if (particleBuffer.get()) spray->setBuffer(spray->particles,particleBuffer.get(),RenderBufferStructuredView(sizeof(interop::WaterSprayParticle)));
        auto &set = scenes[index].materialSet;
        set->setBuffer(set->draws,drawBuffer.get(),RenderBufferStructuredView(sizeof(interop::WaterDrawParams)));
        set->setTexture(set->currentField,fieldTextures[fieldCurrent].get(),RenderTextureLayout::SHADER_READ,fieldViews[fieldCurrent].get());
        set->setTexture(set->previousField,fieldTextures[1-fieldCurrent].get(),RenderTextureLayout::SHADER_READ,fieldViews[1-fieldCurrent].get());
    }
}

void WaterRenderer::reset() {
    draws.clear(); normalGroups.clear(); sceneKeys.clear(); drawScenes.clear();
    wireIndices.clear(); wireStarts.clear();
    profileKinds.clear();
    for (auto &scene : scenes) { scene.captured=false; scene.surfaceCleared=false; }
}

uint32_t WaterRenderer::prepareScene(RenderWorker *worker, RenderTarget *color, RenderTarget *depth, uint64_t key, bool classic, bool selfDepth) {
    const auto sceneKey=std::make_pair(key,selfDepth);
    auto found = sceneKeys.find(sceneKey);
    if (found != sceneKeys.end()) return found->second;
    const uint32_t index = uint32_t(sceneKeys.size());
    sceneKeys.emplace(sceneKey,index);
    if (index == scenes.size()) scenes.emplace_back();
    auto &scene = scenes[index];
    scene.selfDepth = selfDepth;
    if (!scene.color || scene.width != color->width || scene.height != color->height) {
        scene.width = color->width; scene.height = color->height;
        scene.originalSet.reset(); scene.originalColorView.reset(); scene.originalColor.reset();
        scene.color = worker->device->createTexture(RenderTextureDesc::Texture2D(scene.width,scene.height,1,RenderFormat::R16G16B16A16_FLOAT,RenderTextureFlag::UNORDERED_ACCESS));
        scene.mipCount = std::min(6u,1u+uint32_t(std::floor(std::log2(float(std::min(scene.width,scene.height))))));
        scene.depth = worker->device->createTexture(RenderTextureDesc::Texture2D(scene.width,scene.height,scene.mipCount,RenderFormat::R32_FLOAT,RenderTextureFlag::UNORDERED_ACCESS));
        scene.colorView = scene.color->createTextureView(RenderTextureViewDesc::Texture2D(RenderFormat::R16G16B16A16_FLOAT));
        scene.depthView = scene.depth->createTextureView(RenderTextureViewDesc::Texture2D(RenderFormat::R32_FLOAT));
        scene.depthMipViews.clear(); scene.reduceSets.clear();
        for (uint32_t mip=0;mip<scene.mipCount;mip++) {
            auto view = RenderTextureViewDesc::Texture2D(RenderFormat::R32_FLOAT);
            view.mipSlice = mip; view.mipLevels = 1;
            scene.depthMipViews.push_back(scene.depth->createTextureView(view));
            if (mip > 0) {
                auto set = std::make_unique<WaterDepthReduceSet>(worker->device);
                set->setTexture(set->input,scene.depth.get(),RenderTextureLayout::GENERAL,scene.depthMipViews[mip-1].get());
                set->setTexture(set->output,scene.depth.get(),RenderTextureLayout::GENERAL,scene.depthMipViews[mip].get());
                scene.reduceSets.push_back(std::move(set));
            }
        }
        scene.color->setName("Water immutable scene color");
        scene.depth->setName("Water immutable scene depth");
        scene.captureSet = std::make_unique<WaterCaptureSet>(worker->device);
        scene.materialSet = std::make_unique<WaterDescriptorSet>(worker->device);
    }
    if (classic && (!scene.originalColor || scene.originalSamples!=color->multisampling.sampleCount)) {
        scene.originalSet.reset(); scene.originalColorView.reset(); scene.originalColor.reset();
        scene.originalSamples=color->multisampling.sampleCount;
        const uint32_t columns=scene.originalSamples>1 ? 2 : 1;
        const uint32_t rows=(scene.originalSamples+1)/2;
        scene.originalColor = worker->device->createTexture(RenderTextureDesc::Texture2D(scene.width*columns,scene.height*rows,1,RenderFormat::R16G16B16A16_FLOAT,RenderTextureFlag::UNORDERED_ACCESS));
        scene.originalColorView = scene.originalColor->createTextureView(RenderTextureViewDesc::Texture2D(RenderFormat::R16G16B16A16_FLOAT));
        scene.originalColor->setName("Water original palette and transparency");
        scene.originalSet = std::make_unique<WaterOriginalSet>(worker->device);
    }
    if (!scene.surface || scene.surfaceWidth!=scene.width || scene.surfaceHeight!=scene.height || scene.surfaceSamples!=color->multisampling.sampleCount) {
        scene.surfaceFramebuffer.reset();
        scene.surfaceView.reset();
        scene.surface=worker->device->createTexture(RenderTextureDesc::ColorTarget(scene.width,scene.height,RenderFormat::R32_FLOAT,color->multisampling));
        scene.surfaceView=scene.surface->createTextureView(RenderTextureViewDesc::Texture2D(RenderFormat::R32_FLOAT));
        scene.surfaceSamples=color->multisampling.sampleCount;
        scene.surfaceWidth=scene.width; scene.surfaceHeight=scene.height;
        scene.surface->setName("Water view surface distance and mask");
        scene.spraySet=std::make_unique<WaterSpraySet>(worker->device);
    }
    if (scene.selfDepth && (!scene.waterDepth || scene.waterDepthWidth!=depth->width || scene.waterDepthHeight!=depth->height ||
        scene.waterDepthSamples!=depth->multisampling.sampleCount)) {
        scene.surfaceFramebuffer.reset(); scene.waterColorFramebuffer.reset();
        scene.waterDepthWidth=depth->width; scene.waterDepthHeight=depth->height;
        scene.waterDepthSamples=depth->multisampling.sampleCount;
        scene.waterDepth=worker->device->createTexture(RenderTextureDesc::DepthTarget(depth->width,depth->height,RenderFormat::D32_FLOAT,depth->multisampling));
        scene.waterDepth->setName("Water isolated self-occlusion depth");
    }
    // The source target itself may be recreated without changing dimensions.
    // Modern water owns its depth writes; later original translucent effects
    // must still see the game's unchanged depth attachment.
    const RenderTexture *surfaceAttachment=scene.surface.get();
    scene.surfaceFramebuffer=worker->device->createFramebuffer(RenderFramebufferDesc(&surfaceAttachment,1,
        scene.selfDepth ? scene.waterDepth.get() : depth->texture.get(),!scene.selfDepth));
    if (scene.selfDepth) {
        const RenderTexture *colorAttachment=color->texture.get();
        scene.waterColorFramebuffer=worker->device->createFramebuffer(RenderFramebufferDesc(&colorAttachment,1,scene.waterDepth.get(),true));
    }
    else scene.waterColorFramebuffer.reset();
    scene.spraySet->setTexture(scene.spraySet->sceneDepth,scene.depth.get(),RenderTextureLayout::SHADER_READ,scene.depthMipViews[0].get());
    scene.spraySet->setTexture(scene.spraySet->waterDepth,scene.surface.get(),RenderTextureLayout::SHADER_READ,scene.surfaceView.get());
    scene.sourceColor = color; scene.sourceDepth = depth;
    auto &cs = scene.captureSet;
    cs->setTexture(cs->color,color->texture.get(),RenderTextureLayout::SHADER_READ,color->textureView.get());
    cs->setTexture(cs->depth,depth->texture.get(),RenderTextureLayout::SHADER_READ,depth->textureView.get());
    cs->setTexture(cs->outputColor,scene.color.get(),RenderTextureLayout::GENERAL,scene.colorView.get());
    cs->setTexture(cs->outputDepth,scene.depth.get(),RenderTextureLayout::GENERAL,scene.depthMipViews[0].get());
    if (classic) {
        auto &os=scene.originalSet;
        os->setTexture(os->color,color->texture.get(),RenderTextureLayout::SHADER_READ,color->textureView.get());
        os->setTexture(os->output,scene.originalColor.get(),RenderTextureLayout::GENERAL,scene.originalColorView.get());
    }
    auto &ms = scene.materialSet;
    ms->setTexture(ms->originalColor,classic ? scene.originalColor.get() : scene.color.get(),RenderTextureLayout::SHADER_READ,
        classic ? scene.originalColorView.get() : scene.colorView.get());
    ms->setTexture(ms->sceneColor,scene.color.get(),RenderTextureLayout::SHADER_READ,scene.colorView.get());
    ms->setTexture(ms->sceneDepth,scene.depth.get(),RenderTextureLayout::SHADER_READ,scene.depthView.get());
    return index;
}

bool WaterRenderer::captureScene(RenderWorker *worker, uint32_t drawIndex, bool depthReadOnly) {
    auto &scene = scenes[drawScenes[drawIndex]];
    if (scene.captured) return false;
    scene.captured = true;
    const uint32_t profileToken = beginProfile(worker,1);
    worker->commandList->setFramebuffer(nullptr);
    if (scene.selfDepth) {
        // Preserve every MSAA sample of the original scene depth, then keep
        // this copy across all water calls in the view. No copy-back occurs.
        worker->commandList->barriers(RenderBarrierStage::COPY, {
            RenderTextureBarrier(scene.sourceDepth->texture.get(),RenderTextureLayout::COPY_SOURCE),
            RenderTextureBarrier(scene.waterDepth.get(),RenderTextureLayout::COPY_DEST)
        });
        worker->commandList->copyTexture(scene.waterDepth.get(),scene.sourceDepth->texture.get());
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS,
            RenderTextureBarrier(scene.waterDepth.get(),RenderTextureLayout::DEPTH_WRITE));
    }
    worker->commandList->barriers(RenderBarrierStage::COMPUTE, {
        RenderTextureBarrier(scene.sourceColor->texture.get(),RenderTextureLayout::SHADER_READ),
        RenderTextureBarrier(scene.sourceDepth->texture.get(),RenderTextureLayout::SHADER_READ),
        RenderTextureBarrier(scene.color.get(),RenderTextureLayout::GENERAL),
        RenderTextureBarrier(scene.depth.get(),RenderTextureLayout::GENERAL)
    });
    const uint32_t samples = scene.sourceColor->multisampling.sampleCount;
    const auto &shader = samples > 1 ? captureMSShader : captureShader;
    const uint32_t params[] = {scene.width,scene.height,samples,0};
    worker->commandList->setComputePipelineLayout(shader.pipelineLayout.get());
    worker->commandList->setPipeline(shader.pipeline.get());
    worker->commandList->setComputeDescriptorSet(scene.captureSet->get(),0);
    worker->commandList->setComputePushConstants(0,params);
    worker->commandList->dispatch((scene.width+7)/8,(scene.height+7)/8,1);
    if (draws[drawIndex].material.identity.x >= 2) {
        uint32_t sourceWidth = scene.width, sourceHeight = scene.height;
        for (uint32_t mip=1;mip<scene.mipCount;mip++) {
            worker->commandList->barriers(RenderBarrierStage::COMPUTE,RenderTextureBarrier(scene.depth.get(),RenderTextureLayout::GENERAL));
            const uint32_t w = std::max(1u,sourceWidth/2), h = std::max(1u,sourceHeight/2);
            const uint32_t reduceParams[] = {w,h,sourceWidth,sourceHeight};
            worker->commandList->setComputePipelineLayout(depthReduceShader.pipelineLayout.get());
            worker->commandList->setPipeline(depthReduceShader.pipeline.get());
            worker->commandList->setComputeDescriptorSet(scene.reduceSets[mip-1]->get(),0);
            worker->commandList->setComputePushConstants(0,reduceParams);
            worker->commandList->dispatch((w+7)/8,(h+7)/8,1);
            sourceWidth = w; sourceHeight = h;
        }
    }
    worker->commandList->barriers(RenderBarrierStage::GRAPHICS, {
        RenderTextureBarrier(scene.color.get(),RenderTextureLayout::SHADER_READ),
        RenderTextureBarrier(scene.depth.get(),RenderTextureLayout::SHADER_READ),
        RenderTextureBarrier(scene.sourceColor->texture.get(),RenderTextureLayout::COLOR_WRITE),
        RenderTextureBarrier(scene.sourceDepth->texture.get(),depthReadOnly ? RenderTextureLayout::DEPTH_READ : RenderTextureLayout::DEPTH_WRITE)
    });
    endProfile(worker,profileToken);
    return true;
}

void WaterRenderer::captureOriginal(RenderWorker *worker, uint32_t drawIndex) {
    auto &scene=scenes[drawScenes[drawIndex]];
    worker->commandList->setFramebuffer(nullptr);
    worker->commandList->barriers(RenderBarrierStage::COMPUTE,{
        RenderTextureBarrier(scene.sourceColor->texture.get(),RenderTextureLayout::SHADER_READ),
        RenderTextureBarrier(scene.originalColor.get(),RenderTextureLayout::GENERAL)
    });
    const uint32_t samples=scene.sourceColor->multisampling.sampleCount;
    const auto &shader=samples>1 ? originalMSShader : originalShader;
    const uint32_t params[]={scene.width,scene.height,samples,0};
    worker->commandList->setComputePipelineLayout(shader.pipelineLayout.get());
    worker->commandList->setPipeline(shader.pipeline.get());
    worker->commandList->setComputeDescriptorSet(scene.originalSet->get(),0);
    worker->commandList->setComputePushConstants(0,params);
    worker->commandList->dispatch((scene.width+7)/8,(scene.height+7)/8,1);
    worker->commandList->barriers(RenderBarrierStage::GRAPHICS,{
        RenderTextureBarrier(scene.originalColor.get(),RenderTextureLayout::SHADER_READ),
        RenderTextureBarrier(scene.sourceColor->texture.get(),RenderTextureLayout::COLOR_WRITE)
    });
}

RenderDescriptorSet *WaterRenderer::materialSet(uint32_t drawIndex) {
    return scenes[drawScenes[drawIndex]].materialSet->get();
}

void WaterRenderer::beginSurface(RenderWorker *worker,uint32_t drawIndex) {
    auto &scene=scenes[drawScenes[drawIndex]];
    worker->commandList->setFramebuffer(nullptr);
    worker->commandList->barriers(RenderBarrierStage::GRAPHICS,{
        RenderTextureBarrier(scene.surface.get(),RenderTextureLayout::COLOR_WRITE),
        RenderTextureBarrier(scene.selfDepth ? scene.waterDepth.get() : scene.sourceDepth->texture.get(),
            scene.selfDepth ? RenderTextureLayout::DEPTH_WRITE : RenderTextureLayout::DEPTH_READ)
    });
    worker->commandList->setFramebuffer(scene.surfaceFramebuffer.get());
    if (!scene.surfaceCleared) {
        worker->commandList->clearColor(0,RenderColor(0,0,0,0));
        scene.surfaceCleared=true;
    }
}

void WaterRenderer::beginColor(RenderWorker *worker,uint32_t drawIndex) {
    auto &scene=scenes[drawScenes[drawIndex]];
    assert(scene.selfDepth && scene.waterColorFramebuffer);
    worker->commandList->setFramebuffer(nullptr);
    worker->commandList->barriers(RenderBarrierStage::GRAPHICS,
        RenderTextureBarrier(scene.waterDepth.get(),RenderTextureLayout::DEPTH_READ));
    worker->commandList->setFramebuffer(scene.waterColorFramebuffer.get());
}

void WaterRenderer::stepSpray(const interop::WaterMaterial &m,float fromTime,float toTime,bool reset) {
    if (reset || m.identity.x<2) {
        particles.clear(); sprayDistance.fill(0); sprayHeading.fill({0,0,0,0});
        spraySeed=0x57A064u^uint32_t(m.identity.w);
        return;
    }
    particles.erase(std::remove_if(particles.begin(),particles.end(),[&](const auto &p) {
        return toTime>p.origin.w+p.velocity.w;
    }),particles.end());
    const float interval=std::max(m.animation.y-m.animation.x,0.00001f);
    const float a=std::clamp((fromTime-m.animation.x)/interval,0.0f,1.0f);
    const float b=std::clamp((toTime-m.animation.x)/interval,0.0f,1.0f);
    const uint32_t budget=m.identity.x>=3 ? 4096 : 2048;
    auto random=[&]() { spraySeed=1664525u*spraySeed+1013904223u; return float(spraySeed>>8)*(1.0f/16777216); };
    for (uint32_t i=0;i<4;i++) {
        const auto &old=m.craftPrevious[i], &current=m.craftCurrent[i];
        if (current.w<=0) { sprayDistance[i]=0; continue; }
        const float dx=current.x-old.x,dz=current.z-old.z;
        const float distance=std::sqrt(dx*dx+dz*dz), speed=distance/interval;
        const float hx=dx/std::max(distance,0.001f), hz=dz/std::max(distance,0.001f);
        const float turn=std::min(1.0f,std::abs(hx*sprayHeading[i].z-hz*sprayHeading[i].x)*4);
        sprayHeading[i]={hx,0,hz,0};
        sprayDistance[i]+=distance*(b-a)*std::clamp(speed/250.0f,0.0f,1.0f)*std::min(current.w*3,1.0f);
        uint32_t count=std::min(64u,uint32_t(sprayDistance[i]/10));
        sprayDistance[i]-=float(count)*10;
        const bool landing=old.w==0 && current.w>0 && fromTime<m.animation.y && toTime>=m.animation.y;
        if (landing) count+=uint32_t(std::clamp((old.y-current.y)/interval,0.0f,250.0f)*0.15f);
        for (uint32_t j=0;j<count && particles.size()<budget;j++) {
            const float f=a+(b-a)*(float(j)+0.5f)/float(count);
            const float side=random()<0.5f ? -1.0f : 1.0f;
            const float lateral=side*(35+random()*45+turn*90);
            interop::WaterSprayParticle p{};
            p.origin={old.x+dx*f-hx*12+hz*side*8,old.y+(current.y-old.y)*f+4,old.z+dz*f-hz*12-hx*side*8,fromTime+(toTime-fromTime)*(float(j)+0.5f)/float(count)};
            p.velocity={hx*speed*0.10f+hz*lateral,45+random()*65+(landing ? 50 : 0),hz*speed*0.10f-hx*lateral,0.35f+random()*0.65f};
            p.appearance={1.2f+random()*2.8f,0.20f+random()*0.20f,random(),0};
            particles.push_back(p);
        }
    }
}

void WaterRenderer::drawSpray(RenderWorker *worker,uint32_t drawIndex) {
    const auto &draw=draws[drawIndex];
    if (particles.empty() || draw.material.identity.x<2 || draw.material.identity.y!=0 || draw.material.effects.x<0.5f) return;
    const auto &scene=scenes[drawScenes[drawIndex]];
    const uint32_t key=scene.surfaceSamples;
    auto found=sprayPipelines.find(key);
    if (found==sprayPipelines.end()) {
        RenderGraphicsPipelineDesc desc;
        desc.pipelineLayout=sprayLayout.get();
        desc.vertexShader=sprayVertexShader.get();
        desc.pixelShader=key>1 ? sprayMSPixelShader.get() : sprayPixelShader.get();
        desc.renderTargetCount=1;
        desc.renderTargetFormat[0]=RenderTarget::colorBufferFormat(usesHDR);
        desc.renderTargetBlend[0]=RenderBlendDesc::AlphaBlend();
        desc.depthEnabled=true; desc.depthWriteEnabled=false;
        desc.depthFunction=RenderComparisonFunction::LESS;
        desc.depthTargetFormat=RenderFormat::D32_FLOAT;
        desc.cullMode=RenderCullMode::NONE;
        desc.multisampling=scene.sourceColor->multisampling;
        found=sprayPipelines.emplace(key,worker->device->createGraphicsPipeline(desc)).first;
    }
    if (!found->second) return;
    worker->commandList->setGraphicsPipelineLayout(sprayLayout.get());
    worker->commandList->setGraphicsDescriptorSet(scene.spraySet->get(),0);
    worker->commandList->setPipeline(found->second.get());
    interop::WaterPushConstants pc{drawIndex};
    worker->commandList->setGraphicsPushConstants(0,&pc);
    worker->commandList->drawInstanced(6,uint32_t(particles.size()),0,0);
}

void WaterRenderer::prepareField(RenderWorker *worker) {
    fieldSteps.clear();
    statsStep=UINT32_MAX;
    if (draws.empty()) return;
    const auto sourceDraw=draws.front();
    const auto m=sourceDraw.material;
    const uint32_t resolution=m.identity.x>=2 ? 512 : 256;
    constexpr float extent=4096, dt=1.0f/60;
    if (fieldResolution!=resolution) {
        fieldResolution=resolution;
        fieldGeneration=UINT32_MAX;
        for (uint32_t i=0;i<2;i++) {
            fieldTextures[i]=worker->device->createTexture(RenderTextureDesc::Texture2D(resolution*4,resolution,1,RenderFormat::R16G16B16A16_FLOAT,RenderTextureFlag::UNORDERED_ACCESS));
            fieldViews[i]=fieldTextures[i]->createTextureView(RenderTextureViewDesc::Texture2D(RenderFormat::R16G16B16A16_FLOAT));
            fieldSets[i]=std::make_unique<WaterFieldSet>(worker->device);
            fieldTextures[i]->setName("Water persistent ripple and foam field "+std::to_string(i));
        }
        for (uint32_t i=0;i<2;i++) {
            fieldSets[i]->setTexture(fieldSets[i]->previous,fieldTextures[1-i].get(),RenderTextureLayout::SHADER_READ,fieldViews[1-i].get());
            fieldSets[i]->setTexture(fieldSets[i]->next,fieldTextures[i].get(),RenderTextureLayout::GENERAL,fieldViews[i].get());
        }
    }
    const float desired=draws.front().camera.w;
    const bool reset=fieldGeneration!=uint32_t(m.identity.w) || desired<fieldTime-0.1f || desired>fieldTime+0.5f;
    if (reset) fieldInputs.clear();
    if (fieldInputs.empty() || m.animation.y>fieldInputs.back().animation.y) fieldInputs.push_back(m);
    while (fieldInputs.size()>32) fieldInputs.pop_front();
    const float cell=extent/resolution;
    auto step=[&](float nextTime,bool clear) {
        interop::WaterFieldPushConstants pc{};
        // A presentation frame can cross the preceding game update. Use the
        // immutable input that owns this fixed step, not whichever update is
        // newest at 30/60/120 Hz. Otherwise clamping loses part of each trail.
        const interop::WaterMaterial *input=&m;
        for (const auto &candidate:fieldInputs) {
            if (nextTime>candidate.animation.x+0.000001f && nextTime<=candidate.animation.y+0.00001f) { input=&candidate; break; }
        }
        auto stepDraw=sourceDraw;
        stepDraw.material=*input;
        pc.drawIndex=uint32_t(draws.size());
        draws.push_back(stepDraw);
        pc.resolution=resolution; pc.reset=clear;
        pc.simulation={clear ? 0 : dt,fieldTime,nextTime,extent};
        const auto &motion=*input;
        float f=std::clamp((nextTime-motion.animation.x)/std::max(motion.animation.y-motion.animation.x,0.00001f),0.0f,1.0f);
        for (uint32_t i=0;i<4;i++) {
            const float x=motion.craftPrevious[i].x+(motion.craftCurrent[i].x-motion.craftPrevious[i].x)*f;
            const float z=motion.craftPrevious[i].z+(motion.craftCurrent[i].z-motion.craftPrevious[i].z)*f;
            const float ox=std::floor((x-extent*0.5f)/cell)*cell, oz=std::floor((z-extent*0.5f)/cell)*cell;
            pc.origins[i]={ox,oz,clear ? ox : fieldOrigins[i].x,clear ? oz : fieldOrigins[i].y};
            fieldOrigins[i]=pc.origins[i];
        }
        stepSpray(motion,fieldTime,nextTime,clear);
        if (statsSet && !clear && fieldTick%60==0) {
            statsStep=uint32_t(fieldSteps.size());
            statsTime=nextTime; statsMaterial=motion; statsParticles=uint32_t(particles.size());
        }
        fieldCurrent=1-fieldCurrent;
        fieldSteps.emplace_back(pc,fieldCurrent);
        fieldTime=nextTime;
    };
    if (reset) {
        // Reset at the immutable game interval boundary, then integrate to
        // this presentation time. Starting at floor(desired * 60) skipped a
        // different number of initial steps at 30/60/120 Hz after race cuts.
        // The game clock is an integer 60 Hz count exported as float; round
        // its boundary back to that grid rather than flooring float error.
        fieldTick=int64_t(std::llround(double(m.animation.x)*60));
        fieldTime=float(double(fieldTick)/60);
        step(fieldTime,true); step(fieldTime,true);
        fieldGeneration=uint32_t(m.identity.w);
        if (statsSet) std::fprintf(stderr,"[water-field] reset generation=%u interval=(%.6f,%.6f) presentation=%.6f boundary=%lld\n",
            fieldGeneration,m.animation.x,m.animation.y,desired,static_cast<long long>(fieldTick));
    }
    while (fieldTime+0.00001f<desired) {
        ++fieldTick;
        step(float(double(fieldTick)/60),false);
    }
    for (auto &draw:draws) {
        for (uint32_t i=0;i<4;i++) draw.fieldOrigins[i]=fieldOrigins[i];
        draw.flags.z=std::clamp(1-(fieldTime-desired)/dt,0.0f,1.0f);
        draw.flags.w=float(resolution);
    }
}

void WaterRenderer::recordField(RenderWorker *worker) {
    if (fieldSteps.empty()) return;
    const uint32_t token=beginProfile(worker,3);
    uint32_t stepIndex=0;
    for (const auto &[pc,index]:fieldSteps) {
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, {
            RenderTextureBarrier(fieldTextures[1-index].get(),RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(fieldTextures[index].get(),RenderTextureLayout::GENERAL)
        });
        worker->commandList->setComputePipelineLayout(fieldShader.pipelineLayout.get());
        worker->commandList->setPipeline(fieldShader.pipeline.get());
        worker->commandList->setComputeDescriptorSet(fieldSets[index]->get(),0);
        worker->commandList->setComputePushConstants(0,&pc);
        worker->commandList->dispatch((fieldResolution*4+7)/8,(fieldResolution+7)/8,1);
        if (stepIndex==statsStep) {
            statsSet->setTexture(statsSet->field,fieldTextures[index].get(),RenderTextureLayout::SHADER_READ,fieldViews[index].get());
            worker->commandList->barriers(RenderBarrierStage::COMPUTE,RenderTextureBarrier(fieldTextures[index].get(),RenderTextureLayout::SHADER_READ));
            worker->commandList->barriers(RenderBarrierStage::COMPUTE,RenderBufferBarrier(statsBuffer.get(),RenderBufferAccess::WRITE));
            worker->commandList->setComputePipelineLayout(statsShader.pipelineLayout.get());
            worker->commandList->setPipeline(statsShader.pipeline.get());
            worker->commandList->setComputeDescriptorSet(statsSet->get(),0);
            worker->commandList->dispatch(1,1,1);
            worker->commandList->barriers(RenderBarrierStage::COPY,RenderBufferBarrier(statsBuffer.get(),RenderBufferAccess::READ));
            worker->commandList->copyBufferRegion(RenderBufferReference(statsReadback.get()),RenderBufferReference(statsBuffer.get()),128*16);
            statsPending=true;
        }
        ++stepIndex;
    }
    worker->commandList->barriers(RenderBarrierStage::GRAPHICS, {
        RenderTextureBarrier(fieldTextures[0].get(),RenderTextureLayout::SHADER_READ),
        RenderTextureBarrier(fieldTextures[1].get(),RenderTextureLayout::SHADER_READ)
    });
    endProfile(worker,token);

}

uint32_t WaterRenderer::beginProfile(RenderWorker *worker, uint32_t kind) {
    if (!profilePool || profileKinds.size() >= 64) return UINT32_MAX;
    const uint32_t token = uint32_t(profileKinds.size())*2;
    profileKinds.push_back(kind);
    worker->commandList->resetQueryPool(profilePool.get(),token,2);
    worker->commandList->writeTimestamp(profilePool.get(),token);
    return token;
}

void WaterRenderer::endProfile(RenderWorker *worker, uint32_t token) {
    if (token != UINT32_MAX) worker->commandList->writeTimestamp(profilePool.get(),token+1);
}

void WaterRenderer::finishProfile(double frameMilliseconds) {
    if (statsPending && !draws.empty()) {
        // WorkloadQueue calls this only after the graphics completion fence.
        const auto *data=static_cast<const interop::uint4*>(statsReadback->map());
        uint64_t count=0,sum=0; uint32_t peak=0;
        for (uint32_t i=0;i<64;i++) { count+=data[i].x;sum+=data[i].y;peak=std::max(peak,data[i].z); }
        uint64_t shoreCount=0,shoreSum=0; uint32_t shorePeak=0;
        for (uint32_t i=64;i<128;i++) { shoreCount+=data[i].x;shoreSum+=data[i].y;shorePeak=std::max(shorePeak,data[i].z); }
        statsReadback->unmap(); statsPending=false;
        static FILE *f=[]() {
            FILE *file=std::fopen(std::getenv("WR64_WATER_STATS"),"w");
            if (file) std::fprintf(file,"time,generation,course,quality,resolution,foam_pixels,foam_sum,foam_peak,particles,craft_speed,wet,shore_pixels,shore_sum,shore_peak\n");
            return file;
        }();
        if (f) {
            const auto &m=statsMaterial;
            const float dx=m.craftCurrent[0].x-m.craftPrevious[0].x,dz=m.craftCurrent[0].z-m.craftPrevious[0].z;
            std::fprintf(f,"%.4f,%.0f,%.0f,%.0f,%u,%llu,%.3f,%.4f,%u,%.3f,%.4f,%llu,%.3f,%.4f\n",statsTime,m.identity.w,m.identity.z,m.identity.x,fieldResolution,
                (unsigned long long)count,double(sum)/65535,double(peak)/65535,statsParticles,std::sqrt(dx*dx+dz*dz)/std::max(m.animation.y-m.animation.x,0.00001f),m.craftCurrent[0].w,
                (unsigned long long)shoreCount,double(shoreSum)/65535,double(shorePeak)/65535);
            std::fflush(f);
        }
    }
    if (!profilePool || profileKinds.empty() || draws.empty()) return;
    profilePool->queryResults();
    const auto *times = profilePool->getResults();
    double milliseconds[4] = {};
    for (uint32_t i=0;i<profileKinds.size();i++) {
        if (times[i*2+1] >= times[i*2]) milliseconds[profileKinds[i]] += double(times[i*2+1]-times[i*2])/1000000.0;
    }
    static FILE *csv = []() {
        const char *name = std::getenv("WR64_WATER_PROFILE");
        FILE *file = name ? std::fopen(name,"w") : nullptr;
        if (file) std::fprintf(file,"course,quality,debug,views,width,height,water_draws,normal_vertices,normal_ms,capture_ms,surface_ms,field_ms,frame_gpu_ms\n");
        return file;
    }();
    if (csv) {
        const auto &d = draws.front();
        std::fprintf(csv,"%.0f,%.0f,%.0f,%zu,%.0f,%.0f,%zu,%zu,%.5f,%.5f,%.5f,%.5f,%.5f\n",d.material.identity.z,d.material.identity.x,d.material.identity.y,sceneKeys.size(),d.screen.z,d.screen.w,
            drawScenes.size(),normalVertices.size(),milliseconds[0],milliseconds[1],milliseconds[2],milliseconds[3],frameMilliseconds);
        std::fflush(csv);
    }
}

uint64_t WaterRenderer::allocationBytes() const {
    // Texture payload and upload/default buffers; excludes driver alignment,
    // pipelines and the original renderer's own attachments.
    uint64_t bytes=uint64_t(fieldResolution)*fieldResolution*4*8*2+normalBufferSize;
    bytes+=2*(drawBuffer.allocatedSize+normalVertexBuffer.allocatedSize+normalFaceBuffer.allocatedSize+wireBuffer.allocatedSize+particleBuffer.allocatedSize);
    for (const auto &scene:scenes) {
        bytes+=uint64_t(scene.width)*scene.height*8*(1+(scene.originalColor ? scene.originalSamples : 0));
        bytes+=uint64_t(scene.surfaceWidth)*scene.surfaceHeight*4*std::max(scene.surfaceSamples,1u);
        if (scene.waterDepth) bytes+=uint64_t(scene.waterDepthWidth)*scene.waterDepthHeight*4*scene.waterDepthSamples;
        for (uint32_t mip=0;mip<scene.mipCount;mip++) bytes+=uint64_t(std::max(scene.width>>mip,1u))*std::max(scene.height>>mip,1u)*4;
    }
    return bytes;
}

RenderPipeline *WaterRenderer::pipeline(RenderDevice *device, const RenderMultisampling &samples, bool zCompare, bool zWrite, bool wireframe, bool surface, bool classic, bool selfDepth) {
    uint32_t key = uint32_t(samples.sampleCount) | (uint32_t(zCompare) << 16) | (uint32_t(zWrite) << 17) | (uint32_t(wireframe) << 18) | (uint32_t(surface) << 19) | (uint32_t(classic) << 20) | (uint32_t(selfDepth) << 21);
    auto it = pipelines.find(key);
    if (it != pipelines.end()) return it->second.get();
    const RenderInputSlot slots[] = {{0, 16}, {1, 8}, {2, 16}, {3, 16}, {4,16}};
    const RenderInputElement elements[] = {
        {"POSITION", 0, 0, RenderFormat::R32G32B32A32_FLOAT, 0, 0},
        {"TEXCOORD", 0, 1, RenderFormat::R32G32_FLOAT, 1, 0},
        {"COLOR", 0, 2, RenderFormat::R32G32B32A32_FLOAT, 2, 0},
        {"POSITION", 1, 3, RenderFormat::R32G32B32A32_FLOAT, 3, 0},
        {"NORMAL", 0, 4, RenderFormat::R32G32B32A32_FLOAT, 4, 0}
    };
    RenderGraphicsPipelineDesc desc;
    desc.primitiveTopology=wireframe ? RenderPrimitiveTopology::LINE_LIST : RenderPrimitiveTopology::TRIANGLE_LIST;
    desc.pipelineLayout = layout.get();
    desc.vertexShader = vertexShader.get();
    desc.pixelShader = surface ? surfaceShader.get() : classic ?
        (samples.sampleCount>1 ? classicMSShader.get() : classicShader.get()) : pixelShader.get();
    desc.inputSlots = slots;
    desc.inputSlotsCount = 5;
    desc.inputElements = elements;
    desc.inputElementsCount = 5;
    desc.renderTargetCount = 1;
    desc.renderTargetFormat[0] = surface ? RenderFormat::R32_FLOAT : RenderTarget::colorBufferFormat(usesHDR);
    if (!surface) desc.renderTargetBlend[0] = RenderBlendDesc::AlphaBlend();
    desc.depthEnabled = zCompare || zWrite;
    desc.depthWriteEnabled = zWrite;
    const bool equalDepth = classic || (selfDepth && !surface);
    desc.depthFunction = zCompare ? (equalDepth ? RenderComparisonFunction::LESS_EQUAL : RenderComparisonFunction::LESS) : RenderComparisonFunction::ALWAYS;
    desc.depthTargetFormat = RenderFormat::D32_FLOAT;
    desc.cullMode = RenderCullMode::NONE;
    desc.multisampling = samples;
    auto result = device->createGraphicsPipeline(desc);
    auto *ptr = result.get();
    pipelines.emplace(key, std::move(result));
    return ptr;
}

void WaterRenderer::collectNormals(const DrawData &data, uint32_t start, uint32_t count, uint64_t group) {
    wireStarts.push_back(uint32_t(wireIndices.size()));
    for (uint32_t i=0; i+2<count; i+=3) {
        interop::uint4 f{data.faceIndices[start+i],data.faceIndices[start+i+1],data.faceIndices[start+i+2],0};
        if (draws.back().material.identity.y==12) wireIndices.insert(wireIndices.end(),{f.x,f.y,f.y,f.z,f.z,f.x});
        for (uint32_t j=0;j<3;j++) {
            const uint32_t v = f[j];
            auto key = std::make_tuple(group,int(data.posFloats[v*3]),int(data.posFloats[v*3+1]),int(data.posFloats[v*3+2]));
            auto &g = normalGroups[key];
            g.vertices.insert(v);
            g.faces.push_back(f);
        }
    }
}

void WaterRenderer::prepareNormals(RenderWorker *worker, const OutputBuffers &outputs, std::vector<BufferUploader::Upload> &uploads) {
    normalVertices.clear(); normalFaces.clear();
    for (const auto &[key,g] : normalGroups) {
        uint32_t first = uint32_t(normalFaces.size());
        normalFaces.insert(normalFaces.end(),g.faces.begin(),g.faces.end());
        for (uint32_t v : g.vertices) normalVertices.push_back({v,first,uint32_t(g.faces.size()),0});
    }
    const uint64_t required = outputs.rasterWorldPosBuffer.allocatedSize;
    if (!normalBuffer || required > normalBufferSize) {
        normalBufferSize = required;
        normalBuffer = worker->device->createBuffer(RenderBufferDesc::DefaultBuffer(required,
            RenderBufferFlag::VERTEX | RenderBufferFlag::STORAGE | RenderBufferFlag::UNORDERED_ACCESS));
    }
    normalView = RenderVertexBufferView(RenderBufferReference(normalBuffer.get()),normalBufferSize);
    uploads.push_back({normalVertices.data(),{0,normalVertices.size()},sizeof(interop::uint4),RenderBufferFlag::STORAGE,{},&normalVertexBuffer});
    uploads.push_back({normalFaces.data(),{0,normalFaces.size()},sizeof(interop::uint4),RenderBufferFlag::STORAGE,{},&normalFaceBuffer});
    if (!particles.empty()) uploads.push_back({particles.data(),{0,particles.size()},sizeof(interop::WaterSprayParticle),RenderBufferFlag::STORAGE,{},&particleBuffer});
    if (!wireIndices.empty()) uploads.push_back({wireIndices.data(),{0,wireIndices.size()},sizeof(uint32_t),RenderBufferFlag::INDEX,{},&wireBuffer});
}

void WaterRenderer::bindNormals(const OutputBuffers &outputs) {
    if (!wireIndices.empty()) wireView=RenderIndexBufferView(RenderBufferReference(wireBuffer.get()),wireIndices.size()*sizeof(uint32_t),RenderFormat::R32_UINT);
    normalSet->setBuffer(normalSet->positions,outputs.rasterWorldPosBuffer.buffer.get(),RenderBufferStructuredView(16));
    normalSet->setBuffer(normalSet->vertices,normalVertexBuffer.get(),RenderBufferStructuredView(16));
    normalSet->setBuffer(normalSet->faces,normalFaceBuffer.get(),RenderBufferStructuredView(16));
    normalSet->setBuffer(normalSet->normals,normalBuffer.get(),RenderBufferStructuredView(16));
}

void WaterRenderer::recordNormals(RenderWorker *worker, const OutputBuffers &outputs) {
    if (normalVertices.empty()) return;
    const uint32_t profileToken = beginProfile(worker,0);
    const uint32_t count = uint32_t(normalVertices.size());
    worker->commandList->barriers(RenderBarrierStage::COMPUTE,RenderBufferBarrier(outputs.rasterWorldPosBuffer.buffer.get(),RenderBufferAccess::READ));
    worker->commandList->barriers(RenderBarrierStage::COMPUTE,RenderBufferBarrier(normalBuffer.get(),RenderBufferAccess::WRITE));
    worker->commandList->setComputePipelineLayout(normalShader.pipelineLayout.get());
    worker->commandList->setPipeline(normalShader.pipeline.get());
    worker->commandList->setComputeDescriptorSet(normalSet->get(),0);
    worker->commandList->setComputePushConstants(0,&count);
    worker->commandList->dispatch((count+63)/64,1,1);
    worker->commandList->barriers(RenderBarrierStage::GRAPHICS,RenderBufferBarrier(normalBuffer.get(),RenderBufferAccess::READ));
    endProfile(worker,profileToken);
}
}
