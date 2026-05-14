#include <cuda_bf16.h>
#include <stdint.h>

extern "C" __global__ void generic_causal_mask_bf16(
    __nv_bfloat16 *__restrict__ dst,
    const int seq_len,
    const int total_len,
    const int start_pos
) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len || col >= total_len) return;
    dst[row * total_len + col] = col > start_pos + row
        ? __float2bfloat16(-INFINITY)
        : __float2bfloat16(0.0f);
}
