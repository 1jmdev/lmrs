#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>

extern "C" __global__ void repeat_kv_bf16(
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ out,
    const int total_elements,
    const int kv_heads,
    const int n_rep,
    const int seq_len,
    const int head_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const int d = idx % head_dim;
    const int s = (idx / head_dim) % seq_len;
    const int out_head = (idx / (head_dim * seq_len)) % (kv_heads * n_rep);
    const int b = idx / (head_dim * seq_len * kv_heads * n_rep);
    const int kv_head = out_head / n_rep;
    const int src_idx = ((b * kv_heads + kv_head) * seq_len + s) * head_dim + d;
    out[idx] = x[src_idx];
}

extern "C" __global__ void sdpa_bf16(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    __nv_bfloat16 *__restrict__ out,
    const int batch,
    const int heads,
    const int query_len,
    const int key_len,
    const int head_dim,
    const float scale,
    const int causal,
    const int start_pos
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int qi = blockIdx.z;
    const int tid = threadIdx.x;
    if (b >= batch || h >= heads || qi >= query_len) return;

    int key_limit = key_len;
    if (causal) {
        const int visible = start_pos + qi + 1;
        key_limit = visible < key_len ? visible : key_len;
    }

    if (key_limit <= 0) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            const int out_idx = ((b * heads + h) * query_len + qi) * head_dim + d;
            out[out_idx] = __float2bfloat16(0.0f);
        }
        return;
    }

    __shared__ float reduce[256];
    float local_max = -CUDART_INF_F;
    for (int key = tid; key < key_limit; key += blockDim.x) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const int q_idx = ((b * heads + h) * query_len + qi) * head_dim + d;
            const int k_idx = ((b * heads + h) * key_len + key) * head_dim + d;
            score += __bfloat162float(q[q_idx]) * __bfloat162float(k[k_idx]);
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
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const int q_idx = ((b * heads + h) * query_len + qi) * head_dim + d;
            const int k_idx = ((b * heads + h) * key_len + key) * head_dim + d;
            score += __bfloat162float(q[q_idx]) * __bfloat162float(k[k_idx]);
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
            float score = 0.0f;
            for (int kd = 0; kd < head_dim; ++kd) {
                const int q_idx = ((b * heads + h) * query_len + qi) * head_dim + kd;
                const int k_idx = ((b * heads + h) * key_len + key) * head_dim + kd;
                score += __bfloat162float(q[q_idx]) * __bfloat162float(k[k_idx]);
            }
            const float weight = expf(score * scale - max_score) * inv_denom;
            const int v_idx = ((b * heads + h) * key_len + key) * head_dim + d;
            acc += weight * __bfloat162float(v[v_idx]);
        }
        const int out_idx = ((b * heads + h) * query_len + qi) * head_dim + d;
        out[out_idx] = __float2bfloat16(acc);
    }
}
