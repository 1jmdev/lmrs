#include <cuda_bf16.h>

extern "C" __global__ void flash_mha_bwd_copy_bf16(
    const __nv_bfloat16 *__restrict__ grad_out,
    __nv_bfloat16 *__restrict__ grad_q,
    const int elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) grad_q[idx] = grad_out[idx];
}
