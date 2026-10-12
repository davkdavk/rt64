//
// RT64
//

#pragma once

#include "rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    struct RDPParams {
        float4 primColor;
        float2 primLOD;
        float2 primDepth;
        float4 envColor;
        float4 fogColor;
        float4 blendColor;
        float4 keyCenter;
        float4 keyScale;
#ifdef HLSL_CPU
        int convertK[6];
#else
        int4 convertK[2];
#endif
    };
#ifdef HLSL_CPU
};
#endif
