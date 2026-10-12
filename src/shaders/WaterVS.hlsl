#include "shared/rt64_water_params.h"
[[vk::push_constant]] ConstantBuffer<WaterPushConstants> gConstants : register(b0);
StructuredBuffer<WaterDrawParams> waterDraws : register(t1);

void VSMain(float4 position : POSITION0, float2 uv : TEXCOORD0, float4 color : COLOR0,
            float4 world : POSITION1, float4 normal : NORMAL0, out float4 clip : SV_POSITION,
            out float3 worldPosition : TEXCOORD0, out float4 sourceColor : COLOR0,
            out float3 surfaceNormal : TEXCOORD1) {
    WaterDrawParams p = waterDraws[gConstants.drawIndex];
    clip = position;
    clip.xy = (clip.xy - p.screen.xy * 0.5) / (p.screen.xy * float2(0.5, -0.5));
    clip.xyz *= clip.w;
    clip.xy = clip.xy * p.screenTransform.xy + p.screenTransform.zw * clip.w;
    if (p.flags.x > 0.5) clip.z = p.flags.y * clip.w;
    worldPosition = world.xyz;
    sourceColor = color;
    surfaceNormal = normal.xyz;
}
