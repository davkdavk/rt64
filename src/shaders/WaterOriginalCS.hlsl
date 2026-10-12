// Copy the game's actual water texture, alpha blend and course fog before
// adding modern surface effects. This is a separate image, never feedback.
struct CaptureParams { uint width; uint height; uint samples; uint unused; };
[[vk::push_constant]] ConstantBuffer<CaptureParams> params : register(b0);
#ifdef CAPTURE_MSAA
Texture2DMS<float4> sourceColor : register(t1);
#else
Texture2D<float4> sourceColor : register(t1);
#endif
RWTexture2D<float4> originalColor : register(u2);
[numthreads(8,8,1)]
void CSMain(uint2 pixel : SV_DispatchThreadID) {
    if (pixel.x >= params.width || pixel.y >= params.height) return;
#ifdef CAPTURE_MSAA
    // Preserve individual coverage samples. Resolving here and writing the
    // result back through the water mesh would apply edge coverage twice.
    // A two-column atlas fits every supported sample count within 2D limits.
    for (uint i=0; i<params.samples; ++i) {
        float4 color=sourceColor.Load(pixel,i);
        uint2 at=pixel+uint2((i&1)*params.width,(i>>1)*params.height);
        originalColor[at]=float4(pow(max(color.rgb,0),2.2),1);
    }
#else
    float4 color = sourceColor.Load(uint3(pixel,0));
    originalColor[pixel] = float4(pow(max(color.rgb,0),2.2),1);
#endif
}
