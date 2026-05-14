#include <cuda_bf16.h>
#include <stdint.h>

extern "C" __global__ void add_bias_bf16(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ bias,
    __nv_bfloat16 *__restrict__ out,
    const int total_elements,
    const int cols
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    const int col = idx % cols;
    out[idx] = __float2bfloat16(__bfloat162float(x[idx]) + __bfloat162float(bias[col]));
}

extern "C" __global__ void linear_bf16(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ bias,
    __nv_bfloat16 *__restrict__ out,
    const int rows,
    const int in_features,
    const int out_features,
    const int has_bias
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= out_features) return;
    float acc = has_bias ? __bfloat162float(bias[col]) : 0.0f;
    for (int i = 0; i < in_features; ++i) {
        acc += __bfloat162float(x[row * in_features + i]) * __bfloat162float(weight[col * in_features + i]);
    }
    out[row * out_features + col] = __float2bfloat16(acc);
}

extern "C" __global__ void embedding_lookup_i32_bf16(
    const int32_t *__restrict__ input_ids,
    const __nv_bfloat16 *__restrict__ embeddings,
    __nv_bfloat16 *__restrict__ out,
    const int tokens,
    const int hidden_size,
    const int vocab_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * hidden_size;
    if (idx >= total) return;
    const int token = idx / hidden_size;
    const int hidden = idx % hidden_size;
    const int id = input_ids[token];
    out[idx] = (id >= 0 && id < vocab_size) ? embeddings[id * hidden_size + hidden] : __float2bfloat16(0.0f);
}

extern "C" __global__ void zero_f32(
    float *__restrict__ out,
    const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    out[idx] = 0.0f;
}

extern "C" __global__ void concat_dim2_bf16(
    const __nv_bfloat16 *__restrict__ left,
    const __nv_bfloat16 *__restrict__ right,
    __nv_bfloat16 *__restrict__ out,
    const int total_elements,
    const int batch,
    const int heads,
    const int left_seq,
    const int right_seq,
    const int head_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    const int d = idx % head_dim;
    const int s = (idx / head_dim) % (left_seq + right_seq);
    const int h = (idx / (head_dim * (left_seq + right_seq))) % heads;
    const int b = idx / (head_dim * (left_seq + right_seq) * heads);
    if (b >= batch) return;
    if (s < left_seq) {
        out[idx] = left[((b * heads + h) * left_seq + s) * head_dim + d];
    } else {
        out[idx] = right[((b * heads + h) * right_seq + (s - left_seq)) * head_dim + d];
    }
}
