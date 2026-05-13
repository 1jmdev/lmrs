#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

static constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

extern "C" __global__ void qwen_fused_silu_mul_bf16(
    const __nv_bfloat16 *__restrict__ gate_up,
    __nv_bfloat16 *__restrict__ dst,
    const int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const __nv_bfloat16 *gate = gate_up + row * 2 * intermediate_size;
    const __nv_bfloat16 *up = gate + intermediate_size;
    __nv_bfloat16 *out = dst + row * intermediate_size;
    for (int i = tid; i < intermediate_size; i += block_size) {
        const float g = __bfloat162float(gate[i]);
        const float u = __bfloat162float(up[i]);
        out[i] = __float2bfloat16(fast_silu(g) * u);
    }
}

extern "C" __global__ void qwen_causal_mask_bf16(
    const __nv_bfloat16 *__restrict__ src,
    __nv_bfloat16 *__restrict__ dst,
    const int seq_len,
    const int total_len,
    const int start_pos
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq_len * total_len;
    if (idx >= total) return;
    const int row = idx / total_len;
    const int col = idx - row * total_len;
    dst[idx] = col > start_pos + row ? __float2bfloat16(-INFINITY) : __float2bfloat16(0.0f);
}

extern "C" __global__ void qwen_argmax_bf16_phase1(
    const __nv_bfloat16 *__restrict__ logits,
    float *__restrict__ block_vals,
    int32_t *__restrict__ block_idxs,
    const int vocab_size
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int chunk = (vocab_size + gridDim.x - 1) / gridDim.x;
    const int start = bid * chunk;
    const int end = min(start + chunk, vocab_size);

    float best = -INFINITY;
    int best_idx = start;
    for (int i = start + tid; i < end; i += block_size) {
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

    if (tid == 0) {
        block_vals[bid] = vals[0];
        block_idxs[bid] = idxs[0];
    }
}

extern "C" __global__ void qwen_argmax_phase2(
    const float *__restrict__ block_vals,
    const int32_t *__restrict__ block_idxs,
    int32_t *__restrict__ out,
    const int num_blocks
) {
    const int tid = threadIdx.x;
    float best = tid < num_blocks ? block_vals[tid] : -INFINITY;
    int best_idx = tid < num_blocks ? block_idxs[tid] : 0;

    __shared__ float vals[256];
    __shared__ int idxs[256];
    vals[tid] = best;
    idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && vals[tid + stride] > vals[tid]) {
            vals[tid] = vals[tid + stride];
            idxs[tid] = idxs[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[0] = idxs[0];
}
