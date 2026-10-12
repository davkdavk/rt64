#include "shared/rt64_water_params.h"
[[vk::push_constant]] ConstantBuffer<WaterPushConstants> gConstants : register(b0);
StructuredBuffer<WaterDrawParams> waterDraws : register(t1);
Texture2D<float> sceneDepth : register(t3);
#ifdef SPRAY_MSAA
Texture2DMS<float> waterDepth : register(t4);
#else
Texture2D<float> waterDepth : register(t4);
#endif
float4 PSMain(float4 position : SV_POSITION, float2 uv : TEXCOORD0,
    float3 world : TEXCOORD1, float2 fade : TEXCOORD2) : SV_TARGET0 {
    WaterDrawParams p=waterDraws[gConstants.drawIndex];
    float radius=dot(uv,uv);
    float alpha=(1-smoothstep(0.10,1,radius))*fade.x;
    int2 pixel=int2(clamp(position.xy,p.viewBounds.xy,p.viewBounds.zw-1));
    float water=0;
#ifdef SPRAY_MSAA
    uint width,height,samples; waterDepth.GetDimensions(width,height,samples);
    for (uint i=0;i<samples;i++) {
        float d=waterDepth.Load(pixel,i);
        if (d>0) water=water>0 ? min(water,d) : d;
    }
#else
    water=waterDepth.Load(int3(pixel,0));
#endif
    if (water>0) alpha*=saturate((water-fade.y)/12);
    float depth=sceneDepth.Load(int3(pixel,0));
    if (depth<0.9999) {
        float2 ndc=position.xy/float2(p.viewportScale.w,p.viewportTranslate.w)*float2(2,-2)+float2(-1,1);
        ndc=(ndc-p.screenTransform.zw)/p.screenTransform.xy;
        float2 screen=(ndc*float2(0.5,-0.5)+0.5)*p.screen.xy;
        float3 projected=float3((screen-p.viewportTranslate.xy)/p.viewportScale.xy*float2(1,-1),(depth-p.viewportTranslate.z)/p.viewportScale.z);
        float4 background=mul(p.inverseViewProjection,float4(projected,1));
        alpha*=saturate((length(background.xyz/background.w-p.camera.xyz)-fade.y)/12);
    }
    float3 light=(p.material.skyColor.rgb*0.3+p.material.sunColor.rgb*0.7)*p.material.sunColor.a;
    // Match the course haze without introducing a second exposure system.
    float fog=p.fogColor.a*saturate((fade.y-1200)/5000);
    light=lerp(light,p.fogColor.rgb,fog);
    light=pow(saturate(light/(1+light*0.35)),1.0/2.2);
    return float4(light,alpha);
}
