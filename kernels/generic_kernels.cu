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

extern "C" __global__ void generic_argmax_bf16(
    const __nv_bfloat16 *__restrict__ logits,
    int32_t *__restrict__ out,
    const int vocab_size
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    float best = -INFINITY;
    int best_idx = 0;
    for (int i = tid; i < vocab_size; i += block_size) {
        const float v = __bfloat162float(logits[i]);
        if (v > best) {
            best = v;
            best_idx = i;
        }
    }

    __shared__ float vals[256];
    __shared__ int idxs[256];
    vals[tid] = best;
    idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && vals[tid + stride] > vals[tid]) {
            vals[tid] = vals[tid + stride];
            idxs[tid] = idxs[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[0] = idxs[0];
}
