#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct CONSTS
{
    float3 light_pos;
    float3 view_pos;
};

struct main0_out
{
    float4 frag_color [[color(0)]];
};

struct main0_in
{
    float4 color [[user(locn0)]];
    float2 uv [[user(locn1)]];
    float3 normal [[user(locn2)]];
    float3 frag_pos [[user(locn3)]];
};

fragment main0_out main0(main0_in in [[stage_in]], constant CONSTS& _44 [[buffer(0)]], texture2d<float> tex_sampler [[texture(0)]], sampler tex_samplerSmplr [[sampler(0)]])
{
    main0_out out = {};
    float4 base_color = tex_sampler.sample(tex_samplerSmplr, in.uv) * in.color;
    float3 lightColor = float3(3.0);
    float ambientStrength = 0.100000001490116119384765625;
    float3 ambient = lightColor * ambientStrength;
    float3 norm = fast::normalize(in.normal);
    float3 lightDir = fast::normalize(_44.light_pos - in.frag_pos);
    float diff = fast::max(dot(norm, lightDir), 0.0);
    float3 diffuse = lightColor * diff;
    float specularStrength = 0.5;
    float3 viewDir = fast::normalize(_44.view_pos - in.frag_pos);
    float3 reflectDir = reflect(-lightDir, norm);
    float spec = powr(fast::max(dot(viewDir, reflectDir), 0.0), 32.0);
    float3 specular = lightColor * (specularStrength * spec);
    float3 result = ((ambient + diffuse) + specular) * base_color.xyz;
    out.frag_color = float4(result, 1.0);
    return out;
}

