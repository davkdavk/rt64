//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    struct RasterParams {
        float2 screenScale;
        float2 screenOffset;
        uint renderIndex;
        uint padding0;
        uint padding1;
        uint padding2;
    };
#ifdef HLSL_CPU
};
#endif
