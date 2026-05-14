#include <cuda_bf16.h>
#include <stdint.h>
#include <math_constants.h>

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

extern "C" __global__ void paged_attention_bf16(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k_cache,
    const __nv_bfloat16 *__restrict__ v_cache,
    const int32_t *__restrict__ block_tables,
    const int32_t *__restrict__ context_lens,
    __nv_bfloat16 *__restrict__ out,
    const int batch,
    const int num_heads,
    const int num_kv_heads,
    const int query_len,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq,
    const float scale,
    const int causal
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int qi = blockIdx.z;
    const int tid = threadIdx.x;
    if (b >= batch || h >= num_heads || qi >= query_len) return;

    const int context_len = context_lens[b];
    const int kv_head = h % num_kv_heads;
    int key_limit = context_len;
    if (causal) {
        const int start = context_len - query_len;
        key_limit = start + qi + 1;
        key_limit = key_limit < context_len ? key_limit : context_len;
    }
    if (key_limit <= 0) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            const int out_idx = ((b * num_heads + h) * query_len + qi) * head_dim + d;
            out[out_idx] = __float2bfloat16(0.0f);
        }
        return;
    }

    __shared__ float reduce[256];
    float local_max = -CUDART_INF_F;
    for (int key = tid; key < key_limit; key += blockDim.x) {
        const int block_idx = block_tables[b * max_blocks_per_seq + key / block_size];
        const int block_offset = key - (key / block_size) * block_size;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const int q_idx = ((b * num_heads + h) * query_len + qi) * head_dim + d;
            const int k_idx = (((block_idx * block_size + block_offset) * num_kv_heads + kv_head) * head_dim) + d;
            score += __bfloat162float(q[q_idx]) * __bfloat162float(k_cache[k_idx]);
        }
        local_max = fmaxf(local_max, score * scale);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
        __syncthreads();
    }
    const float max_score = reduce[0];

    float local_sum = 0.0f;
    for (int key = tid; key < key_limit; key += blockDim.x) {
        const int block_idx = block_tables[b * max_blocks_per_seq + key / block_size];
        const int block_offset = key - (key / block_size) * block_size;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const int q_idx = ((b * num_heads + h) * query_len + qi) * head_dim + d;
            const int k_idx = (((block_idx * block_size + block_offset) * num_kv_heads + kv_head) * head_dim) + d;
            score += __bfloat162float(q[q_idx]) * __bfloat162float(k_cache[k_idx]);
        }
        local_sum += expf(score * scale - max_score);
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    const float inv_denom = 1.0f / reduce[0];

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int key = 0; key < key_limit; ++key) {
            const int block_idx = block_tables[b * max_blocks_per_seq + key / block_size];
            const int block_offset = key - (key / block_size) * block_size;
            float score = 0.0f;
            for (int kd = 0; kd < head_dim; ++kd) {
                const int q_idx = ((b * num_heads + h) * query_len + qi) * head_dim + kd;
                const int k_idx = (((block_idx * block_size + block_offset) * num_kv_heads + kv_head) * head_dim) + kd;
                score += __bfloat162float(q[q_idx]) * __bfloat162float(k_cache[k_idx]);
            }
            const float weight = expf(score * scale - max_score) * inv_denom;
            const int v_idx = (((block_idx * block_size + block_offset) * num_kv_heads + kv_head) * head_dim) + d;
            acc += weight * __bfloat162float(v_cache[v_idx]);
        }
        const int out_idx = ((b * num_heads + h) * query_len + qi) * head_dim + d;
        out[out_idx] = __float2bfloat16(acc);
    }
}
