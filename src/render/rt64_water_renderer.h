#pragma once
#include "rt64_buffer_uploader.h"
#include "rt64_shader_library.h"
#include "shared/rt64_water_params.h"
#include <array>
#include <map>
#include <deque>
#include <set>
#include <tuple>

namespace RT64 {
struct DrawData;
struct OutputBuffers;
struct RenderTarget;
struct WaterStatsSet : RenderDescriptorSetBase {
    uint32_t field,output;
    explicit WaterStatsSet(RenderDevice *device=nullptr) {
        builder.begin(); field=builder.addTexture(1); output=builder.addReadWriteStructuredBuffer(2); builder.end();
        if (device) create(device);
    }
};
struct WaterSpraySet : RenderDescriptorSetBase {
    uint32_t draws, particles, sceneDepth, waterDepth;
    explicit WaterSpraySet(RenderDevice *device = nullptr) {
        builder.begin();
        draws=builder.addStructuredBuffer(1);
        particles=builder.addStructuredBuffer(2);
        sceneDepth=builder.addTexture(3);
        waterDepth=builder.addTexture(4);
        builder.end();
        if (device) create(device);
    }
};
struct WaterFieldSet : RenderDescriptorSetBase {
    uint32_t draws, previous, next;
    explicit WaterFieldSet(RenderDevice *device = nullptr) {
        builder.begin();
        draws=builder.addStructuredBuffer(1);
        previous=builder.addTexture(2);
        next=builder.addReadWriteTexture(3);
        builder.end();
        if (device) create(device);
    }
};
struct WaterDepthReduceSet : RenderDescriptorSetBase {
    uint32_t input, output;
    explicit WaterDepthReduceSet(RenderDevice *device = nullptr) {
        builder.begin();
        input = builder.addTexture(1);
        output = builder.addReadWriteTexture(2);
        builder.end();
        if (device) create(device);
    }
};
struct WaterCaptureSet : RenderDescriptorSetBase {
    uint32_t color, depth, outputColor, outputDepth;
    explicit WaterCaptureSet(RenderDevice *device = nullptr) {
        builder.begin();
        color = builder.addTexture(1);
        depth = builder.addTexture(2);
        outputColor = builder.addReadWriteTexture(3);
        outputDepth = builder.addReadWriteTexture(4);
        builder.end();
        if (device) create(device);
    }
};
struct WaterOriginalSet : RenderDescriptorSetBase {
    uint32_t color, output;
    explicit WaterOriginalSet(RenderDevice *device=nullptr) {
        builder.begin(); color=builder.addTexture(1); output=builder.addReadWriteTexture(2); builder.end();
        if (device) create(device);
    }
};
struct WaterNormalSet : RenderDescriptorSetBase {
    uint32_t positions, vertices, faces, normals;
    explicit WaterNormalSet(RenderDevice *device = nullptr) {
        builder.begin();
        positions = builder.addStructuredBuffer(1);
        vertices = builder.addStructuredBuffer(2);
        faces = builder.addStructuredBuffer(3);
        normals = builder.addReadWriteStructuredBuffer(4);
        builder.end();
        if (device) create(device);
    }
};
struct WaterDescriptorSet : RenderDescriptorSetBase {
    uint32_t draws, sceneColor, sceneDepth, currentField, previousField, originalColor;
    explicit WaterDescriptorSet(RenderDevice *device = nullptr) {
        builder.begin();
        draws = builder.addStructuredBuffer(1);
        sceneColor = builder.addTexture(2);
        sceneDepth = builder.addTexture(3);
        currentField = builder.addTexture(4);
        previousField = builder.addTexture(5);
        originalColor = builder.addTexture(6);
        builder.end();
        if (device) create(device);
    }
};

// Owned by the framebuffer renderer, so descriptor and upload lifetimes follow its workload fences.
struct WaterRenderer {
    std::vector<interop::WaterDrawParams> draws;
    BufferPair drawBuffer;
    std::unique_ptr<WaterDescriptorSet> descriptorSet;
    std::unique_ptr<RenderPipelineLayout> layout;
    std::unique_ptr<RenderShader> vertexShader;
    std::unique_ptr<RenderShader> pixelShader, classicShader, classicMSShader;
    std::unique_ptr<RenderShader> surfaceShader, sprayVertexShader, sprayPixelShader, sprayMSPixelShader;
    std::unique_ptr<RenderPipelineLayout> sprayLayout;
    std::map<uint32_t,std::unique_ptr<RenderPipeline>> sprayPipelines;
    std::vector<interop::WaterSprayParticle> particles;
    BufferPair particleBuffer;
    std::array<float,4> sprayDistance{};
    std::array<interop::float4,4> sprayHeading{};
    uint32_t spraySeed=1;
    std::map<uint32_t, std::unique_ptr<RenderPipeline>> pipelines;
    RenderVertexBufferView worldView;
    RenderVertexBufferView normalView;
    struct NormalGroup {
        std::set<uint32_t> vertices;
        std::vector<interop::uint4> faces;
    };
    std::map<std::tuple<uint64_t,int,int,int>, NormalGroup> normalGroups;
    std::vector<interop::uint4> normalVertices, normalFaces;
    std::vector<uint32_t> wireIndices, wireStarts;
    BufferPair wireBuffer;
    RenderIndexBufferView wireView;
    BufferPair normalVertexBuffer, normalFaceBuffer;
    std::unique_ptr<RenderBuffer> normalBuffer;
    uint64_t normalBufferSize = 0;
    ShaderRecord normalShader;
    std::unique_ptr<WaterNormalSet> normalSet;
    struct SceneInputs {
        std::unique_ptr<RenderTexture> color, depth, originalColor;
        std::unique_ptr<RenderTextureView> originalColorView;
        std::unique_ptr<WaterOriginalSet> originalSet;
        std::unique_ptr<RenderTexture> surface;
        std::unique_ptr<RenderTexture> waterDepth;
        std::unique_ptr<RenderTextureView> surfaceView;
        std::unique_ptr<RenderFramebuffer> surfaceFramebuffer;
        std::unique_ptr<RenderFramebuffer> waterColorFramebuffer;
        std::unique_ptr<WaterSpraySet> spraySet;
        uint32_t surfaceSamples=0, originalSamples=0;
        uint32_t surfaceWidth=0,surfaceHeight=0;
        uint32_t waterDepthWidth=0,waterDepthHeight=0,waterDepthSamples=0;
        bool selfDepth=false;
        bool surfaceCleared=false;
        std::unique_ptr<RenderTextureView> colorView, depthView;
        std::vector<std::unique_ptr<RenderTextureView>> depthMipViews;
        std::vector<std::unique_ptr<WaterDepthReduceSet>> reduceSets;
        std::unique_ptr<WaterCaptureSet> captureSet;
        std::unique_ptr<WaterDescriptorSet> materialSet;
        RenderTarget *sourceColor = nullptr, *sourceDepth = nullptr;
        uint32_t width = 0, height = 0, mipCount = 0;
        bool captured = false;
    };
    std::vector<SceneInputs> scenes;
    std::map<std::pair<uint64_t,bool>,uint32_t> sceneKeys;
    std::vector<uint32_t> drawScenes;
    ShaderRecord captureShader, captureMSShader, originalShader, originalMSShader;
    ShaderRecord depthReduceShader;
    ShaderRecord fieldShader;
    std::unique_ptr<RenderTexture> fieldTextures[2];
    std::unique_ptr<RenderTextureView> fieldViews[2];
    std::unique_ptr<WaterFieldSet> fieldSets[2];
    std::vector<std::pair<interop::WaterFieldPushConstants,uint32_t>> fieldSteps;
    std::array<interop::float4,4> fieldOrigins{};
    std::deque<interop::WaterMaterial> fieldInputs;
    float fieldTime = 0;
    int64_t fieldTick=0;
    uint32_t fieldGeneration = UINT32_MAX, fieldResolution = 0, fieldCurrent = 0;
    std::unique_ptr<RenderQueryPool> profilePool;
    ShaderRecord statsShader;
    std::unique_ptr<WaterStatsSet> statsSet;
    std::unique_ptr<RenderBuffer> statsBuffer,statsReadback;
    bool statsPending=false;
    float statsTime=-1;
    uint32_t statsStep=UINT32_MAX, statsParticles=0;
    interop::WaterMaterial statsMaterial{};
    std::vector<uint32_t> profileKinds;
    bool usesHDR;

    WaterRenderer(RenderDevice *device, RenderShaderFormat format, bool usesHDR);
    void bindBuffer();
    void collectNormals(const DrawData &data, uint32_t start, uint32_t count, uint64_t group);
    void prepareNormals(RenderWorker *worker, const OutputBuffers &outputs, std::vector<BufferUploader::Upload> &uploads);
    void bindNormals(const OutputBuffers &outputs);
    void recordNormals(RenderWorker *worker, const OutputBuffers &outputs);
    void reset();
    uint32_t prepareScene(RenderWorker *worker, RenderTarget *color, RenderTarget *depth, uint64_t key, bool classic, bool selfDepth);
    bool captureScene(RenderWorker *worker, uint32_t drawIndex, bool depthReadOnly);
    void captureOriginal(RenderWorker *worker, uint32_t drawIndex);
    RenderDescriptorSet *materialSet(uint32_t drawIndex);
    uint32_t beginProfile(RenderWorker *worker, uint32_t kind);
    void endProfile(RenderWorker *worker, uint32_t token);
    void finishProfile(double frameMilliseconds);
    uint64_t allocationBytes() const;
    void prepareField(RenderWorker *worker);
    void recordField(RenderWorker *worker);
    void stepSpray(const interop::WaterMaterial &material, float previousTime, float nextTime, bool reset);
    void beginSurface(RenderWorker *worker,uint32_t drawIndex);
    void beginColor(RenderWorker *worker,uint32_t drawIndex);
    void drawSpray(RenderWorker *worker,uint32_t drawIndex);
    RenderPipeline *pipeline(RenderDevice *device, const RenderMultisampling &samples, bool zCompare, bool zWrite, bool wireframe, bool surface=false, bool classic=false, bool selfDepth=false);
};
}
