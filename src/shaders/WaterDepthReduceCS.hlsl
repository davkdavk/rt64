struct ReduceParams { uint width; uint height; uint sourceWidth; uint sourceHeight; };
[[vk::push_constant]] ConstantBuffer<ReduceParams> params : register(b0);
Texture2D<float> inputDepth : register(t1);
RWTexture2D<float> outputDepth : register(u2);
[numthreads(8,8,1)]
void CSMain(uint2 pixel : SV_DispatchThreadID) {
    if (pixel.x >= params.width || pixel.y >= params.height) return;
    float depth = 1;
    // Standard mip sizes round down. Include the extra edge row/column on
    // odd-sized targets, otherwise thin shore geometry disappears from Hi-Z.
    uint2 first=pixel*2;
    uint2 end=min(first+2,uint2(params.sourceWidth,params.sourceHeight));
    if (pixel.x+1==params.width) end.x=params.sourceWidth;
    if (pixel.y+1==params.height) end.y=params.sourceHeight;
    for (uint y=first.y;y<end.y;y++) for (uint x=first.x;x<end.x;x++) {
        uint2 source = uint2(x,y);
        depth = min(depth,inputDepth.Load(uint3(source,0)));
    }
    outputDepth[pixel] = depth;
}
