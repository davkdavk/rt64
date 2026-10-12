// Separate immutable inputs: no attachment feedback and no water recursion.
struct CaptureParams { uint width; uint height; uint samples; uint unused; };
[[vk::push_constant]] ConstantBuffer<CaptureParams> params : register(b0);
#ifdef CAPTURE_MSAA
Texture2DMS<float4> sceneColor : register(t1);
Texture2DMS<float> sceneDepth : register(t2);
#else
Texture2D<float4> sceneColor : register(t1);
Texture2D<float> sceneDepth : register(t2);
#endif
RWTexture2D<float4> capturedColor : register(u3);
RWTexture2D<float> capturedDepth : register(u4);

[numthreads(8,8,1)]
void CSMain(uint2 pixel : SV_DispatchThreadID) {
    if (pixel.x >= params.width || pixel.y >= params.height) return;
#ifdef CAPTURE_MSAA
    float4 color = 0;
    float depth = 1;
    for (uint i=0; i<params.samples; ++i) {
        color += sceneColor.Load(pixel,i);
        depth = min(depth,sceneDepth.Load(pixel,i));
    }
    color /= params.samples;
#else
    float4 color = sceneColor.Load(uint3(pixel,0));
    float depth = sceneDepth.Load(uint3(pixel,0));
#endif
    // Inputs are the existing display-referred N64 scene. Decode once here.
    capturedColor[pixel] = float4(pow(max(color.rgb,0),2.2),1);
    capturedDepth[pixel] = depth;
}
