#include <cuda_bf16.h>

extern "C" __global__ void paged_attn_bwd_value_bf16(
    const __nv_bfloat16 *__restrict__ grad_out,
    const __nv_bfloat16 *__restrict__ probs,
    __nv_bfloat16 *__restrict__ grad_v,
    const int rows,
    const int cols
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;
    const float go = __bfloat162float(grad_out[idx]);
    const float p = __bfloat162float(probs[idx]);
    grad_v[idx] = __float2bfloat16(go * p);
}
