#pragma once
#include "rt64_hlsl.h"
#define WATER_SHORE_SEGMENTS_PER_TILE 20

#ifdef HLSL_CPU
namespace interop {
#endif
// All fields are full vectors to keep the display-list, C++, SPIR-V and MSL ABI identical.
struct WaterMaterial {
    float4 deepColor;       // linear RGB, absorption scale per game unit
    float4 shallowColor;    // linear RGB, roughness
    float4 sunDirection;    // direction toward sun, intensity
    float4 sunColor;        // linear RGB, ambient intensity
    float4 skyColor;        // linear RGB, horizon haze
    float4 animation;       // previous/current simulation seconds, detail strength, foam strength
    float4 identity;        // quality (0 disables), debug view, course, generation
    float4 surface;         // mean height, wind X/Z, view index
    float4 optics;          // visibility range, authored reflection weight, shallow caustics, classic style
    float4 craftPrevious[4]; // world XYZ, wet contact fraction (-1 means inactive)
    float4 craftCurrent[4];
    float4 shoreCounts;     // bounded static shoreline segments in each craft's world tile
    float4 shoreSegments[4 * WATER_SHORE_SEGMENTS_PER_TILE]; // immutable mean-water intersections, XZ endpoints
    float4 effects;         // payload v2: added spray enabled, reserved YZW
};

struct WaterDrawParams {
    WaterMaterial material;
    float4 screen;          // original framebuffer width/height, target width/height
    float4 screenTransform;// scale XY, offset XY
    float4 camera;          // world camera XYZ, interpolated animation time
    float4 fogColor;        // linear RGB, enabled
    float4 flags;           // z source, prim depth, field interpolation, field resolution
    float4 viewportScale;   // original RSP viewport XYZ, render viewport width
    float4 viewportTranslate; // original RSP viewport XYZ, render viewport height
    float4 viewBounds;      // pixel scissor, including the player viewport
    float4 fieldOrigins[4]; // current X/Z, previous X/Z, in world units
    float4x4 viewProjection;
    float4x4 inverseViewProjection;
};

struct WaterPushConstants {
    uint drawIndex;
};
struct WaterSprayParticle {
    float4 origin;          // world XYZ, birth time
    float4 velocity;        // initial XYZ velocity, lifetime
    float4 appearance;      // radius, opacity, seed, reserved
};
struct WaterFieldPushConstants {
    float4 origins[4];      // destination X/Z, source X/Z
    uint drawIndex;
    uint resolution;
    uint reset;
    uint unused;
    float4 simulation;      // step seconds, previous time, current time, tile extent
};
#ifdef HLSL_CPU
static_assert(sizeof(WaterMaterial) == 1584, "Water display-list ABI");
static_assert(sizeof(WaterDrawParams) == 1904, "Water shader ABI");
static_assert(sizeof(WaterDrawParams) <= 2048, "D3D structured-buffer stride limit");
static_assert(sizeof(WaterFieldPushConstants) == 96, "Water field shader ABI");
}
#endif
