#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>

extern "C" __global__ void apply_causal_mask_bf16(
    const __nv_bfloat16 *__restrict__ scores,
    __nv_bfloat16 *__restrict__ out,
    const int total_elements,
    const int query_len,
    const int key_len,
    const int start_pos
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const int col = idx % key_len;
    const int row = (idx / key_len) % query_len;
    out[idx] = col > start_pos + row ? __float2bfloat16(-CUDART_INF_F) : scores[idx];
}
