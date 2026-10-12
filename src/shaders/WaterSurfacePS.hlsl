#include "shared/rt64_water_params.h"
[[vk::push_constant]] ConstantBuffer<WaterPushConstants> gConstants : register(b0);
StructuredBuffer<WaterDrawParams> waterDraws : register(t1);
// Zero is no water; positive values are eye distance. Keep the original opaque
// depth attachment read-only, including when the N64 water doesn't write Z.
float PSMain(float4 position : SV_POSITION, float3 world : TEXCOORD0) : SV_TARGET0 {
    if (position.z < 0 || position.z > 1022.0/1024.0) discard;
    return max(length(world-waterDraws[gConstants.drawIndex].camera.xyz),0.001);
}
