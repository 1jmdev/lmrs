#include <cuda_bf16.h>

extern "C" __global__ void rotary_inplace_bf16(
    __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ cos,
    const __nv_bfloat16 *__restrict__ sin,
    const int tokens,
    const int dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = dim / 2;
    const int total = tokens * half;
    if (idx >= total) return;
    const int token = idx / half;
    const int d = idx % half;
    const float c = __bfloat162float(cos[token * half + d]);
    const float s = __bfloat162float(sin[token * half + d]);
    const int base = token * dim + d;
    const float a = __bfloat162float(input[base]);
    const float b = __bfloat162float(input[base + half]);
    input[base] = __float2bfloat16(a * c - b * s);
    input[base + half] = __float2bfloat16(b * c + a * s);
}
