#include "shared/rt64_water_params.h"
[[vk::push_constant]] ConstantBuffer<WaterPushConstants> gConstants : register(b0);
StructuredBuffer<WaterDrawParams> waterDraws : register(t1);
StructuredBuffer<WaterSprayParticle> particles : register(t2);

void VSMain(uint vertex : SV_VertexID, uint instance : SV_InstanceID,
    out float4 position : SV_POSITION, out float2 uv : TEXCOORD0,
    out float3 world : TEXCOORD1, out float2 fade : TEXCOORD2) {
    // Return a finite, degenerate offscreen primitive for invisible particles.
    // Alpha zero alone cannot make invalid projected positions safe.
    position=float4(2,2,2,1); uv=0; world=0; fade=0;
    WaterDrawParams p=waterDraws[gConstants.drawIndex];
    WaterSprayParticle particle=particles[instance];
    float age=p.camera.w-particle.origin.w;
    if (age<0 || age>=particle.velocity.w || particle.velocity.w<=0) return;
    float drag=1.5, response=(1-exp(-drag*age))/drag;
    world=particle.origin.xyz+particle.velocity.xyz*response;
    world.y-=140*(age-response)/drag;
    float4 clip=mul(p.viewProjection,float4(world,1));
    float4 top=mul(p.viewProjection,float4(world+float3(0,particle.appearance.x,0),1));
    // Both points must be in front of the eye before either perspective divide.
    // A top edge crossing the eye plane used to create an enormous white quad.
    if (!all(isfinite(clip)) || !all(isfinite(top)) || clip.w<=0.01 || top.w<=0.01) return;
    float3 ndc=clip.xyz/clip.w;
    float3 screen=ndc*float3(1,-1,1)*p.viewportScale.xyz+p.viewportTranslate.xyz;
    float2 xy=(screen.xy/p.screen.xy*float2(2,-2)+float2(-1,1))*p.screenTransform.xy+p.screenTransform.zw;
    float projectedRadius=abs(top.y/top.w-ndc.y)*abs(p.viewportScale.y/p.screen.y)*2;
    if (!all(isfinite(ndc)) || !all(isfinite(screen)) || !all(isfinite(xy)) || !isfinite(projectedRadius)) return;
    const float2 corners[6]={float2(-1,-1),float2(1,-1),float2(-1,1),float2(-1,1),float2(1,-1),float2(1,1)};
    uv=corners[vertex];
    position=float4(xy*clip.w,screen.z*clip.w,clip.w);
    float radius=clamp(projectedRadius,0.0005,0.035);
    position.xy+=uv*radius*float2(p.screen.y/p.screen.x,1)*clip.w;
    float life=saturate(age/particle.velocity.w);
    float distance=length(world-p.camera.xyz);
    float nearFade=smoothstep(particle.appearance.x*4,particle.appearance.x*12,distance);
    float sizeFade=1-smoothstep(0.012,0.035,projectedRadius);
    fade=float2(particle.appearance.y*smoothstep(0,0.07,age)*(1-smoothstep(0.55,1,life))*nearFade*sizeFade,distance);
}
