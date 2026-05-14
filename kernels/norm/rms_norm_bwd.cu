#include <cuda_bf16.h>

extern "C" __global__ void rms_norm_bwd_bf16(
    const __nv_bfloat16 *__restrict__ grad_out,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ grad_x,
    const int elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) return;
    grad_x[idx] = __float2bfloat16(__bfloat162float(grad_out[idx]) * __bfloat162float(weight[idx]));
}
