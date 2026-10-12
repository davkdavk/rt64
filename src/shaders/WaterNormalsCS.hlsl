struct NormalPushConstants { uint vertexCount; };
[[vk::push_constant]] ConstantBuffer<NormalPushConstants> params : register(b0);
StructuredBuffer<float4> positions : register(t1);
StructuredBuffer<uint4> vertices : register(t2);
StructuredBuffer<uint4> faces : register(t3);
RWStructuredBuffer<float4> normals : register(u4);

[numthreads(64,1,1)]
void CSMain(uint i : SV_DispatchThreadID) {
    if (i >= params.vertexCount) return;
    uint4 vertex = vertices[i];
    float3 sum = 0;
    for (uint j = 0; j < vertex.z; ++j) {
        uint3 f = faces[vertex.y+j].xyz;
        float3 a = positions[f.x].xyz;
        float3 b = positions[f.y].xyz;
        float3 c = positions[f.z].xyz;
        float3 n = cross(b-a,c-a);
        sum += n.y < 0 ? -n : n;
    }
    normals[vertex.x] = float4(dot(sum,sum)>0.00001 ? normalize(sum) : float3(0,1,0), 0);
}
