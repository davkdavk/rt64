#include "shared/rt64_water_params.h"
[[vk::push_constant]] ConstantBuffer<WaterPushConstants> gConstants : register(b0);
StructuredBuffer<WaterDrawParams> waterDraws : register(t1);
Texture2D<float4> sceneColor : register(t2);
Texture2D<float> sceneDepth : register(t3);
Texture2D<float4> currentField : register(t4);
Texture2D<float4> previousField : register(t5);
Texture2D<float4> originalColor : register(t6);

static const float PI = 3.14159265359;

float foamHash(float2 cell) {
    float3 p=frac(float3(cell.x,cell.y,cell.x)*0.1031);
    p+=dot(p,p.yzx+33.33);
    return frac((p.x+p.y)*p.z);
}
float foamNoise(float2 at) {
    float2 cell=floor(at), f=frac(at);
    f=f*f*f*(f*(f*6-15)+10);
    return lerp(lerp(foamHash(cell),foamHash(cell+float2(1,0)),f.x),
        lerp(foamHash(cell+float2(0,1)),foamHash(cell+1),f.x),f.y);
}
float foamBreakup(float2 world,float time,float2 wind) {
    // Advect with the wake field. The old product of two axis-aligned sine
    // waves cut a repeating checkerboard of square holes through the foam.
    // Rotated, warped octaves give the wake irregular connected patches.
    float2 at=(world-wind*(8*time))*0.043;
    at=float2(at.x*0.9239-at.y*0.3827,at.x*0.3827+at.y*0.9239);
    at+=float2(sin(dot(at,float2(0.71,0.53))+time*0.12),
        sin(dot(at,float2(-0.47,0.83))-time*0.09))*0.65;
    float2 fineAt=float2(at.x*0.8-at.y*0.6,at.x*0.6+at.y*0.8)*2.17+float2(7.13,19.91);
    float broadFilter=1-smoothstep(0.35,1.2,max(fwidth(at.x),fwidth(at.y)));
    float fineFilter=1-smoothstep(0.35,1.2,max(fwidth(fineAt.x),fwidth(fineAt.y)));
    float detail=lerp(0.5,foamNoise(fineAt),fineFilter);
    float breakup=smoothstep(0.29,0.65,foamNoise(at)*0.65+detail*0.35);
    // Preserve average coverage when the pattern becomes subpixel, avoiding
    // sparkling along the distant part of a wake and at grazing angles.
    return lerp(0.60,breakup,broadFilter);
}

float3 fieldLoad(int2 pixel,uint tile,uint resolution,bool previous) {
    if (any(pixel<0) || any(pixel>=int(resolution))) return 0;
    int3 at=int3(pixel+int2(tile*resolution,0),0);
    return previous ? previousField.Load(at).xzw : currentField.Load(at).xzw;
}
float3 fieldSample(float2 at,uint tile,uint resolution,bool previous) {
    int2 pixel=int2(floor(at)); float2 f=frac(at);
    return lerp(lerp(fieldLoad(pixel,tile,resolution,previous),fieldLoad(pixel+int2(1,0),tile,resolution,previous),f.x),
        lerp(fieldLoad(pixel+int2(0,1),tile,resolution,previous),fieldLoad(pixel+1,tile,resolution,previous),f.x),f.y);
}
float3 interaction(float2 world,WaterDrawParams p) {
    uint tile=4; float closest=1e20;
    [unroll] for (uint i=0;i<4;i++) {
        float2 local=world-p.fieldOrigins[i].xy;
        float distance=dot(local-2048,local-2048);
        if (p.material.craftCurrent[i].w>=0 && all(local>=0) && all(local<4096) && distance<closest) { tile=i;closest=distance; }
    }
    if (tile==4) return 0;
    float resolution=p.flags.w;
    float2 now=(world-p.fieldOrigins[tile].xy)*(resolution/4096)-0.5;
    float2 old=(world-p.fieldOrigins[tile].zw)*(resolution/4096)-0.5;
    return lerp(fieldSample(old,tile,uint(resolution),true),fieldSample(now,tile,uint(resolution),false),p.flags.z);
}

float2 boundPixel(float2 pixel, WaterDrawParams p) {
    return clamp(pixel,p.viewBounds.xy+1,p.viewBounds.zw-2);
}

float3 backgroundColor(float2 pixel, WaterDrawParams p) {
    float2 at = boundPixel(pixel,p)-0.5;
    int2 i = int2(floor(at));
    float2 f = frac(at);
    return lerp(lerp(sceneColor.Load(int3(i,0)).rgb,sceneColor.Load(int3(i+int2(1,0),0)).rgb,f.x),
        lerp(sceneColor.Load(int3(i+int2(0,1),0)).rgb,sceneColor.Load(int3(i+1,0)).rgb,f.x),f.y);
}

float4 submergedColor(float2 pixel, float surfaceDepth, WaterDrawParams p) {
    float2 at=boundPixel(pixel,p)-0.5;
    int2 base=int2(floor(at));
    float2 f=frac(at);
    float3 color=0;
    float coverage=0;
    // A valid center depth does not make its bilinear neighbors submerged.
    // Reject foreground and clear-sky taps individually so moving refraction
    // cannot pull their bright colors through a shoreline or a wave edge.
    [unroll] for (uint y=0;y<2;y++) {
        [unroll] for (uint x=0;x<2;x++) {
            int2 tap=base+int2(x,y);
            float depth=sceneDepth.Load(int3(tap,0));
            float weight=(x ? f.x : 1-f.x)*(y ? f.y : 1-f.y);
            weight*=depth>surfaceDepth+0.000001 && depth<0.9999 ? 1 : 0;
            color+=sceneColor.Load(int3(tap,0)).rgb*weight;
            coverage+=weight;
        }
    }
    return float4(color/max(coverage,0.000001),coverage);
}

float3 worldFromPixel(float2 pixel, float depth, WaterDrawParams p) {
    // Undo the actual raster screen transform, then the original N64 viewport.
    // Its Z scale/translate differ from a conventional 0..1 camera projection.
    float2 ndc = pixel / float2(p.viewportScale.w,p.viewportTranslate.w) * float2(2,-2) + float2(-1,1);
    ndc = (ndc-p.screenTransform.zw)/p.screenTransform.xy;
    float2 screen = (ndc * float2(0.5,-0.5)+0.5)*p.screen.xy;
    float3 projected = float3((screen-p.viewportTranslate.xy)/p.viewportScale.xy*float2(1,-1),
        (depth-p.viewportTranslate.z)/p.viewportScale.z);
    float4 world = mul(p.inverseViewProjection,float4(projected,1));
    return world.xyz / world.w;
}

float opticalDepth(float2 pixel, float surfaceDepth, float3 world, WaterDrawParams p) {
    float depth = sceneDepth.Load(int3(int2(boundPixel(pixel,p)),0));
    if (depth >= 0.9999 || depth <= surfaceDepth+0.000001) return -1;
    float3 background = worldFromPixel(pixel,depth,p);
    float distance = length(background-p.camera.xyz)-length(world-p.camera.xyz);
    return distance > 0 && distance < 100000 ? min(distance,1600) : -1;
}

float4 pixelFromWorld(float3 world, WaterDrawParams p) {
    float4 clip = mul(p.viewProjection,float4(world,1));
    float3 ndc = clip.xyz / clip.w;
    float3 screen = ndc*float3(1,-1,1)*p.viewportScale.xyz+p.viewportTranslate.xyz;
    float2 xy = (screen.xy/p.screen.xy*float2(2,-2)+float2(-1,1))*p.screenTransform.xy+p.screenTransform.zw;
    return float4((xy*float2(0.5,-0.5)+0.5)*float2(p.viewportScale.w,p.viewportTranslate.w),screen.z,clip.w);
}

float4 screenReflection(float3 world, float3 N, float3 V, float roughness, WaterDrawParams p) {
    if (p.material.identity.x < 2) return 0;
    uint depthWidth,depthHeight,depthLevels; sceneDepth.GetDimensions(0,depthWidth,depthHeight,depthLevels);
    float3 ray = reflect(-V,N);
    float start = 5, distance = 16;
    float3 origin = world + N*3;
    float2 previousPixel = pixelFromWorld(origin,p).xy;
    [loop] for (uint step=0;step<36;step++) {
        float3 rayPoint = origin+ray*distance;
        float4 pixel = pixelFromWorld(rayPoint,p);
        if (pixel.w <= 0 || any(pixel.xy < p.viewBounds.xy+2) || any(pixel.xy >= p.viewBounds.zw-2)) break;
        float footprint = max(length(pixel.xy-previousPixel),1);
        uint mip = min(uint(clamp(floor(log2(footprint)),0,5)),depthLevels-1);
        uint mipWidth,mipHeight,levels; sceneDepth.GetDimensions(mip,mipWidth,mipHeight,levels);
        uint2 mipPixel=min(uint2(pixel.xy)/(1u<<mip),uint2(mipWidth-1,mipHeight-1));
        float minimum = sceneDepth.Load(int3(mipPixel,mip));
        if (pixel.z >= minimum && minimum < 0.9999) {
            float depth = sceneDepth.Load(int3(int2(pixel.xy),0));
            float3 background = worldFromPixel(pixel.xy,depth,p);
            float gap = length(rayPoint-p.camera.xyz)-length(background-p.camera.xyz);
            if (depth < 0.9999 && gap >= 0) {
                float lo=start, hi=distance;
                [unroll] for (uint refine=0;refine<5;refine++) {
                    float mid=(lo+hi)*0.5;
                    float4 at=pixelFromWorld(origin+ray*mid,p);
                    float z=sceneDepth.Load(int3(int2(boundPixel(at.xy,p)),0));
                    if (at.z >= z) hi=mid; else lo=mid;
                }
                float3 hit=origin+ray*hi;
                float4 at=pixelFromWorld(hit,p);
                float z=sceneDepth.Load(int3(int2(boundPixel(at.xy,p)),0));
                float3 sampled=worldFromPixel(at.xy,z,p);
                float separation=length(hit-sampled);
                float edge=min(min(at.x-p.viewBounds.x,p.viewBounds.z-at.x),min(at.y-p.viewBounds.y,p.viewBounds.w-at.y));
                float confidence=saturate(1-separation/45)*saturate(edge/40)*saturate(1-hi/3000);
                confidence *= z < 0.9999 && sampled.y > p.material.surface.x-20 ? 1 : 0;
                float radius=roughness*5;
                float3 color=(backgroundColor(at.xy+float2(radius,0),p)+backgroundColor(at.xy-float2(radius,0),p)
                    +backgroundColor(at.xy+float2(0,radius),p)+backgroundColor(at.xy-float2(0,radius),p))*0.25;
                return float4(color,confidence);
            }
        }
        previousPixel=pixel.xy;
        start=distance;
        distance=distance*1.16+12;
    }
    return 0;
}

float2 rippleCornerGradient(float2 cell) {
    float2 gradient=float2(foamHash(cell),foamHash(cell+float2(73.19,19.73)))*2-1;
    return gradient*rsqrt(max(dot(gradient,gradient),0.0001));
}

// Analytic slope of compact, randomly oriented simplex kernels. Unlike a
// handful of global cosine waves, this has no continuous crossed stripe grid.
float2 rippleNoiseGradient(float2 at) {
    const float skew=0.3660254038, unskew=0.2113248654;
    float2 cell=floor(at+(at.x+at.y)*skew);
    float2 first=at-cell+(cell.x+cell.y)*unskew;
    float2 corner=first.x>first.y ? float2(1,0) : float2(0,1);
    float2 offsets[3]={first,first-corner+unskew,first-1+2*unskew};
    float2 cells[3]={cell,cell+corner,cell+1};
    float2 slope=0;
    [unroll] for (uint i=0;i<3;i++) {
        float2 offset=offsets[i], direction=rippleCornerGradient(cells[i]);
        float support=max(0.5-dot(offset,offset),0);
        slope+=support*support*support*(support*direction-8*dot(direction,offset)*offset);
    }
    return slope*70;
}

float2 rippleGradient(float2 x, float t, float strength, float2 wind) {
    float windSpeed=length(wind);
    float2 axis=windSpeed>0.001 ? wind/windSpeed : float2(1,0);
    float2 crosswind=float2(-axis.y,axis.x);
    x=float2(dot(x,axis),dot(x,crosswind));
    t*=0.45+0.55*min(windSpeed,2.0);
    const float frequencies[3]={0.012,0.030,0.075};
    const float speeds[3]={24,17,11};
    const float weights[3]={1,0.62,0.35};
    const float2 rotations[3]={float2(0.95534,0.29552),float2(0.87758,-0.47943),float2(0.76484,0.64422)};
    const float2 offsets[3]={float2(13.17,7.93),float2(-31.41,47.23),float2(83.71,-19.37)};
    float2 gradient=0;
    [unroll] for (uint i=0;i<3;i++) {
        // Stretch crests across the wind and advect each scale continuously.
        float2 local=(x-float2(t*speeds[i],0))*float2(1,0.60);
        float2 rotation=rotations[i];
        float2 at=float2(local.x*rotation.x-local.y*rotation.y,local.x*rotation.y+local.y*rotation.x)*frequencies[i]+offsets[i];
        float2 footprintX=ddx(at),footprintY=ddy(at);
        float footprint=sqrt(dot(footprintX,footprintX)+dot(footprintY,footprintY));
        float filter=1-smoothstep(0.18,0.65,footprint);
        float2 slope=rippleNoiseGradient(at);
        // Inverse rotation and stretch form the domain's gradient Jacobian.
        // Height amplitude is inverse-frequency so fine bands do not dominate.
        slope=float2(slope.x*rotation.x+slope.y*rotation.y,-slope.x*rotation.y+slope.y*rotation.x)*float2(1,0.60);
        gradient+=slope*weights[i]*filter;
    }
    // Match the prior unfiltered RMS slope (~0.1205 at Normal/detail 1).
    return (axis*gradient.x+crosswind*gradient.y)*(0.04623*strength);
}

float3 sky(float3 r, WaterMaterial m) {
    float h = saturate(r.y);
    // Haze inherits the course's illumination. A fixed blue-white horizon
    // erases Sunset Bay's warm light even when its sky preset is orange.
    float3 haze = lerp(m.skyColor.rgb, m.sunColor.rgb * 0.72, 0.65);
    float3 horizon = lerp(m.skyColor.rgb, haze, m.skyColor.a);
    float3 result = lerp(horizon, m.skyColor.rgb, sqrt(h));
    // Very broad cloud variations rather than a second animated texture layer.
    float cloud = smoothstep(0.55, 0.85, sin(r.x * 8 + r.z * 3) * 0.28 + sin(r.z * 13 - r.x * 5) * 0.16 + 0.5);
    return lerp(result, haze, cloud * (1-h) * 0.24);
}

float4 PSMain(float4 position : SV_POSITION, float3 worldPosition : TEXCOORD0,
              float4 sourceColor : COLOR0, float3 surfaceNormal : TEXCOORD1
#ifdef CLASSIC_MSAA
              , uint sampleIndex : SV_SampleIndex
#endif
              ) : SV_TARGET0 {
    WaterDrawParams p = waterDraws[gConstants.drawIndex];
    WaterMaterial m = p.material;
    if (position.z < 0 || position.z > 1022.0/1024.0) discard;
    float3 geometric = normalize(surfaceNormal);
    if (geometric.y < 0) geometric = -geometric;
    float2 ripples = rippleGradient(worldPosition.xz, p.camera.w, m.animation.z,m.surface.yz);
    float3 wake=interaction(worldPosition.xz,p);
    float2 dx=ddx(worldPosition.xz),dy=ddy(worldPosition.xz);
    float determinant=dx.x*dy.y-dx.y*dy.x;
    float2 wakeGradient=float2(ddx(wake.x)*dy.y-ddy(wake.x)*dx.y,ddy(wake.x)*dx.x-ddx(wake.x)*dy.x)/max(abs(determinant),0.001)*sign(determinant);
    ripples += clamp(wakeGradient*8,-0.10,0.10);
    float3 N = normalize(geometric + float3(-ripples.x, 0, -ripples.y));
    float3 V = normalize(p.camera.xyz - worldPosition);
    float3 L = normalize(m.sunDirection.xyz);
    float3 H = normalize(V + L);
    float nv = max(dot(N, V), 0.015);
    float nl = saturate(dot(N,L));
    float nh = saturate(dot(N,H));
    float vh = saturate(dot(V,H));
    float fresnel = 0.0204 + 0.9796 * pow(1-nv, 5);
    float roughness = clamp(m.shallowColor.a + length(float2(ddx(N.x), ddy(N.z))) * 0.2, 0.065, 0.65);
    float a2 = pow(roughness, 4);
    float d = a2 / max(PI * pow(nh*nh*(a2-1)+1, 2), 0.00001);
    float k = pow(roughness + 1, 2) * 0.125;
    float g = nv / (nv*(1-k)+k) * nl / max(nl*(1-k)+k, 0.0001);
    float f = 0.0204 + 0.9796 * pow(1-vh, 5);
    float specular = min(d*g*f / max(4*nv, 0.001), 12);
    float waveLight = saturate((worldPosition.y - m.surface.x) * 0.005 + 0.35 + nl * 0.25);
    float3 body = lerp(m.deepColor.rgb, m.shallowColor.rgb, waveLight * 0.65);
    float thickness = opticalDepth(position.xy,position.z,worldPosition,p);
    float2 refractedPixel = boundPixel(position.xy + ripples * 13 * saturate(thickness/80),p);
    float refractedThickness = opticalDepth(refractedPixel,position.z,worldPosition,p);
    // Reject foreground silhouettes at the proposed sample before distorting.
    if (refractedThickness < 0) { refractedPixel = position.xy; refractedThickness = thickness; }
    // Derivatives must be evaluated for the whole pixel quad, before the
    // depth-validity branch. A branch crossing a shoreline otherwise leaves
    // caustic receiver normals/filter widths undefined on some backends.
    float bottomDepth=sceneDepth.Load(int3(int2(boundPixel(refractedPixel,p)),0));
    float3 bottom=worldFromPixel(refractedPixel,bottomDepth,p);
    float3 bottomNormal=cross(ddx(bottom),ddy(bottom));
    bottomNormal*=rsqrt(max(dot(bottomNormal,bottomNormal),0.000001));
    if (bottomNormal.y<0) bottomNormal=-bottomNormal;
    float2 causticUV=bottom.xz*0.035;
    float causticFilter=1-smoothstep(0.25,1.5,length(fwidth(causticUV)));
    if (thickness >= 0 && refractedThickness >= 0) {
        float3 transmission = exp(-m.deepColor.a * float3(1.8,0.55,0.30) * refractedThickness);
        // Course geometry often ends a short distance below the shoreline.
        // Limit optical visibility smoothly before those uncapped triangle
        // edges become visible. Higher-absorption courses reach deep color
        // sooner; no seabed is invented where the game submitted none.
        float visibilityRange=max(m.optics.x,1);
        transmission*=1-smoothstep(visibilityRange*0.30,visibilityRange,refractedThickness);
        float4 receiverColor=submergedColor(refractedPixel,position.z,p);
        float3 bottomColor=receiverColor.rgb;
        // Uncovered taps transition to the water body, not an unrelated
        // background color. Keep fractional coverage for a soft boundary.
        transmission*=receiverColor.a;
        // Stable world-space light focusing, confined to an actual shallow
        // submerged receiver. Dark source lighting and steep walls reject it.
        float phaseA=causticUV.x+sin(causticUV.y*1.3+p.camera.w*0.65);
        float phaseB=causticUV.y-sin(causticUV.x*1.1-p.camera.w*0.5);
        float focusing=pow(saturate(1-abs(sin(phaseA)+sin(phaseB)*0.55)),12);
        float receiver=smoothstep(0.2,0.65,bottomNormal.y)*saturate(dot(bottomNormal,L));
        receiver*=smoothstep(0.015,0.08,dot(bottomColor,float3(0.2126,0.7152,0.0722)));
        receiver*=bottom.y<worldPosition.y-2 && refractedThickness<visibilityRange ? 1 : 0;
        bottomColor+=m.sunColor.rgb*focusing*causticFilter*receiver*m.optics.z;
        body = bottomColor * transmission + body * (1-transmission);
    }
    float3 color = body * (0.55 + nl * 0.45) * m.sunColor.a * (1-fresnel)
        + m.sunColor.rgb * m.sunDirection.a * specular;
    float4 reflection=screenReflection(worldPosition,N,V,roughness,p);
    float3 reflected=lerp(sky(reflect(-V,N),m),reflection.rgb,reflection.a);
    float3 authoredReflection=0;
    float authoredConfidence=0;
    if (m.optics.y>0) {
        // Some original course scenes contain authored mirrored scenery.
        // It is already in the immutable pre-water color. Preserve that
        // course-specific input with rough filtering and foreground rejection.
        float2 at=boundPixel(position.xy+ripples*10,p);
        if (sceneDepth.Load(int3(int2(at),0))<=position.z+0.000001) at=position.xy;
        float radius=roughness*8;
        authoredReflection=(backgroundColor(at+float2(radius,0),p)+backgroundColor(at-float2(radius,0),p)
            +backgroundColor(at+float2(0,radius),p)+backgroundColor(at-float2(0,radius),p))*0.25;
        authoredConfidence=m.optics.y*(1-smoothstep(15,70,abs(worldPosition.y-m.surface.x)));
    }
    color += reflected*fresnel;
    float crest=smoothstep(0.12,0.32,1-geometric.y)*smoothstep(25,85,worldPosition.y-m.surface.x)*0.15;
    float contact=thickness>=0 ? smoothstep(0,8,thickness)*(1-smoothstep(10,45,thickness))*0.10 : 0;
    float breakup=foamBreakup(worldPosition.xz,p.camera.w,m.surface.yz);
    float shore=wake.z*(1-smoothstep(30,100,abs(worldPosition.y-m.surface.x)));
    float foam=saturate((wake.y*1.8+shore*1.3+crest+contact)*m.animation.w*breakup);
    color=lerp(color,float3(0.73,0.79,0.80)*(0.65+0.35*nl)*m.sunColor.a,foam);
    if (p.fogColor.a > 0.5) color = lerp(color, p.fogColor.rgb, saturate(sourceColor.a));
    // The original mirrored scene has already passed through course fog.
    // Compose it here to avoid applying that fog a second time and erasing
    // Drake Lake's distant shoreline reflection.
    // The original reflection is LDR artwork rather than scene radiance.
    // Retain a restrained base contribution in this explicitly authored
    // course mode, then strengthen it toward grazing angles.
    color=lerp(color,authoredReflection,authoredConfidence*(0.60+0.40*fresnel)*(1-foam));
    float alpha = 1;
    // Compress only the new lighting before conversion into the legacy display-referred target.
#ifdef CLASSIC_STYLE
    {
        // The original raster pass already contains the course palette,
        // broad wave highlights, transparency and fog. Preserve that artwork
        // without reapplying absorption, sky tint or tone compression.
        uint2 originalPixel=uint2(position.xy);
#ifdef CLASSIC_MSAA
        uint originalWidth,originalHeight;
        sceneColor.GetDimensions(originalWidth,originalHeight);
        originalPixel+=uint2((sampleIndex&1)*originalWidth,(sampleIndex>>1)*originalHeight);
#endif
        float3 base=originalColor.Load(int3(originalPixel,0)).rgb;
        float3 classic=base;
        float reflectionWeight=reflection.a*fresnel*0.32*(1-foam);
        classic=lerp(classic,reflection.rgb,reflectionWeight);
        classic+=m.sunColor.rgb*m.sunDirection.a*specular*0.16;
        classic=lerp(classic,float3(0.73,0.79,0.80)*(0.65+0.35*nl)*m.sunColor.a,foam);
        // The captured base is already fogged; only attenuate additions.
        float visibility=p.fogColor.a>0.5 ? 1-saturate(sourceColor.a) : 1;
        color=pow(saturate(lerp(base,classic,visibility)),1.0/2.2);
    }
#else
    {
        color = pow(saturate(color / (1 + color * 0.35)), 1.0/2.2);
    }
#endif
    uint debug = (uint)m.identity.y;
    if (debug == 1) return float4(1,0,0.8,1);
    if (debug == 2) return float4(N*0.5+0.5,1);
    if (debug == 3) return float4(frac(worldPosition * 0.005),1);
    if (debug == 4) return float4(sourceColor.rgb,1);
    if (debug == 5) return float4(thickness < 0 ? float3(0,0,0) : float3(saturate(thickness/500),saturate(thickness/100),0),1);
    if (debug == 6) return float4(pow(saturate(backgroundColor(position.xy,p)),1.0/2.2),1);
    if (debug == 7) return float4(thickness < 0 ? float3(0.8,0.1,0.1) : float3(0.1,0.8,0.2),1);
    if (debug == 8) return float4(reflection.aaa,1);
    if (debug == 9) return float4(saturate(abs(worldFromPixel(position.xy,position.z,p)-worldPosition)*0.1),1);
    if (debug == 10) {
        float3 marker=float3(0,0,0);
        [unroll] for (uint i=0;i<4;i++) {
            float f=saturate((p.camera.w-m.animation.x)/max(m.animation.y-m.animation.x,0.00001));
            float2 craft=lerp(m.craftPrevious[i].xz,m.craftCurrent[i].xz,f);
            if (m.craftCurrent[i].w>=0 && length(worldPosition.xz-craft)<45) marker=m.craftCurrent[i].w>0 ? float3(0,1,0) : float3(1,0,0);
        }
        return float4(marker,1);
    }
    if (debug == 11) return float4(wake.y,shore,crest+contact,1);
    if (debug == 12) return float4(0.25,1,0.65,1);
    if (debug == 13) {
        uint w,h; currentField.GetDimensions(w,h);
        int2 samplePixel=int2(saturate((position.xy-p.viewBounds.xy)/(p.viewBounds.zw-p.viewBounds.xy))*float2(w-1,h-1));
        float4 value=currentField.Load(int3(samplePixel,0));
        return float4(value.zzz,1);
    }
    if (debug == 14) {
        float2 local=(worldPosition.xz-p.fieldOrigins[0].xy)/4096;
        return float4(saturate(local),all(local>=0) && all(local<=1) ? 0 : 1,1);
    }
    return float4(color, alpha);
}
