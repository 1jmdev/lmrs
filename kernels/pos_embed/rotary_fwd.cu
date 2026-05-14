#include <cuda_bf16.h>

extern "C" __global__ void rotary_fwd_bf16(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ cos,
    const __nv_bfloat16 *__restrict__ sin,
    __nv_bfloat16 *__restrict__ output,
    const int tokens,
    const int seq_len,
    const int dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * dim;
    if (idx >= total) return;
    const int half = dim / 2;
    const int token = idx / dim;
    const int seq = token % seq_len;
    const int d = idx % dim;
    const int pair = d % half;
    const float c = __bfloat162float(cos[seq * half + pair]);
    const float s = __bfloat162float(sin[seq * half + pair]);
    const float a = __bfloat162float(input[token * dim + pair]);
    const float b = __bfloat162float(input[token * dim + pair + half]);
    output[idx] = d < half ? __float2bfloat16(a * c - b * s) : __float2bfloat16(b * c + a * s);
}
