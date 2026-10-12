//
// RT64
//

#pragma once

#include <stdint.h>

#include "render/rt64_shader_common.h"
#include "shared/rt64_rsp_vertex_test_z.h"

namespace RT64 {
    struct InstanceDrawCall {
        enum class Type {
            Unknown,
            Raytracing,
            IndexedTriangles,
            RawTriangles,
            RegularRect,
            FillRect,
            VertexTestZ
        };

        Type type;
        uint32_t waterIndex = UINT32_MAX;
        const RenderPipeline *originalWaterPipeline = nullptr;

        union {
            struct {
                uint32_t hitGroupIndex;
                uint32_t queryMask;
                bool cullDisable;
            } raytracing;
            
            struct {
                ShaderDescription shaderDesc;
                const RenderPipeline *pipeline;
                interop::float2 screenScale;
                interop::float2 screenOffset;
                RenderRect scissor;
                uint32_t indexStart;
                uint32_t faceCount;
                bool vertexTestZ : 1;
                bool postBlendDitherNoise : 1;
                bool postBlendDitherNoiseNegative : 1;
            } triangles;

            struct {
                RenderRect rect;

                union {
                    RenderColor color;
                    float depth;
                };
            } clearRect;

            interop::RSPVertexTestZCB vertexTestZ;
        };

        InstanceDrawCall() {
            type = Type::Unknown;
        }

        ~InstanceDrawCall() { }
    };
};