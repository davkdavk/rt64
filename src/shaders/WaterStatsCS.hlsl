// Optional test instrumentation; never enabled for normal play.
Texture2D<float4> field : register(t1);
RWStructuredBuffer<uint4> statistics : register(u2);
[numthreads(64,1,1)]
void CSMain(uint lane : SV_DispatchThreadID) {
    uint width,height; field.GetDimensions(width,height);
    uint count=0,sum=0,peak=0,shoreCount=0,shoreSum=0,shorePeak=0;
    for (uint i=lane;i<width*height;i+=64) {
        float4 sample=field.Load(int3(i%width,i/width,0));
        float foam=sample.z;
        uint value=uint(saturate(foam)*65535);
        count+=foam>0.01;
        sum+=value;
        peak=max(peak,value);
        uint shoreValue=uint(saturate(sample.w)*65535);
        shoreCount+=sample.w>0.01;
        shoreSum+=shoreValue;
        shorePeak=max(shorePeak,shoreValue);
    }
    statistics[lane]=uint4(count,sum,peak,0);
    statistics[lane+64]=uint4(shoreCount,shoreSum,shorePeak,0);
}
