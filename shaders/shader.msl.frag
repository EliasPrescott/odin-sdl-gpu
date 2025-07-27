#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct main0_out
{
    float4 color [[color(0)]];
};

fragment main0_out main0()
{
    main0_out out = {};
    out.color = float4(0.5, 0.0, 0.20000000298023223876953125, 1.0);
    return out;
}

