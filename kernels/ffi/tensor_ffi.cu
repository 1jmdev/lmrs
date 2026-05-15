#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../ops/activation_kernels.cuh"

namespace {

constexpr int kThreads = 256;

int grid_1d(int64_t n) {
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

template <typename T>
__device__ T from_float(float value) {
  return cuda_kernels::from_float<T>(value);
}

template <typename T>
__global__ void fused_silu_split_kernel(
    T* out,
    const T* gate_up,
    int64_t rows,
    int inner) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = rows * inner;
  if (idx >= total) {
    return;
  }

  const int64_t row = idx / inner;
  const int col = static_cast<int>(idx % inner);
  const T* row_base = gate_up + row * inner * 2;

  const float gate = cuda_kernels::to_float(row_base[col]);
  const float up = cuda_kernels::to_float(row_base[inner + col]);
  const float silu = gate / (1.0f + expf(-gate));
  out[idx] = from_float<T>(silu * up);
}

template <typename T>
__global__ void affine_kernel(
    T* out,
    const T* input,
    int64_t numel,
    float scale,
    float offset) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }

  const float value = cuda_kernels::to_float(input[idx]);
  out[idx] = from_float<T>(value * scale + offset);
}

template <typename T>
__global__ void greater_equal_kernel(
    T* out,
    const T* left,
    const T* right,
    int64_t numel) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }

  const float lhs = cuda_kernels::to_float(left[idx]);
  const float rhs = cuda_kernels::to_float(right[idx]);
  out[idx] = from_float<T>(lhs >= rhs ? 1.0f : 0.0f);
}

template <typename T>
__global__ void where_kernel(
    T* out,
    const T* cond,
    const T* true_values,
    const T* false_values,
    int64_t numel) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }

  out[idx] = cuda_kernels::to_float(cond[idx]) > 0.0f
                 ? true_values[idx]
                 : false_values[idx];
}

template <typename T>
__global__ void add_bias_kernel(
    T* out,
    const T* input,
    const T* bias,
    int64_t numel,
    int last_dim) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }

  const float x = cuda_kernels::to_float(input[idx]);
  const float b = cuda_kernels::to_float(bias[idx % last_dim]);
  out[idx] = from_float<T>(x + b);
}

template <typename T>
__global__ void narrow_dim1_kernel(
    T* out,
    const T* input,
    int dim0,
    int dim1,
    int dim2,
    int start,
    int len) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(dim0) * len * dim2;
  if (idx >= total) {
    return;
  }

  const int col = static_cast<int>(idx % dim2);
  const int narrowed = static_cast<int>((idx / dim2) % len);
  const int batch = static_cast<int>(idx / (len * dim2));
  const int source_dim1 = start + narrowed;

  const int64_t source_idx =
      (static_cast<int64_t>(batch) * dim1 + source_dim1) * dim2 + col;
  out[idx] = input[source_idx];
}

template <typename T>
__global__ void transpose_1_2_kernel(
    T* out,
    const T* input,
    int dim0,
    int dim1,
    int dim2,
    int dim3) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(dim0) * dim1 * dim2 * dim3;
  if (idx >= total) {
    return;
  }

  const int col = static_cast<int>(idx % dim3);
  const int out_dim1 = static_cast<int>((idx / dim3) % dim2);
  const int out_dim2 = static_cast<int>((idx / (dim3 * dim2)) % dim1);
  const int batch = static_cast<int>(idx / (dim3 * dim2 * dim1));

  const int64_t source_idx =
      ((static_cast<int64_t>(batch) * dim1 + out_dim2) * dim2 + out_dim1) *
          dim3 +
      col;
  out[idx] = input[source_idx];
}

template <typename T>
__global__ void concat_dim2_kernel(
    T* out,
    const T* left,
    const T* right,
    int dim0,
    int dim1,
    int left_dim2,
    int right_dim2,
    int dim3) {
  const int out_dim2 = left_dim2 + right_dim2;
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(dim0) * dim1 * out_dim2 * dim3;
  if (idx >= total) {
    return;
  }

  const int col = static_cast<int>(idx % dim3);
  const int seq = static_cast<int>((idx / dim3) % out_dim2);
  const int head = static_cast<int>((idx / (dim3 * out_dim2)) % dim1);
  const int batch = static_cast<int>(idx / (dim3 * out_dim2 * dim1));

  if (seq < left_dim2) {
    const int64_t source_idx =
        ((static_cast<int64_t>(batch) * dim1 + head) * left_dim2 + seq) *
            dim3 +
        col;
    out[idx] = left[source_idx];
    return;
  }

  const int right_seq = seq - left_dim2;
  const int64_t source_idx =
      ((static_cast<int64_t>(batch) * dim1 + head) * right_dim2 + right_seq) *
          dim3 +
      col;
  out[idx] = right[source_idx];
}

template <typename T>
__global__ void repeat_kv_kernel(
    T* out,
    const T* input,
    int batch_size,
    int kv_heads,
    int seq_len,
    int head_dim,
    int repeats) {
  const int out_heads = kv_heads * repeats;
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total =
      static_cast<int64_t>(batch_size) * out_heads * seq_len * head_dim;
  if (idx >= total) {
    return;
  }

  const int col = static_cast<int>(idx % head_dim);
  const int seq = static_cast<int>((idx / head_dim) % seq_len);
  const int out_head = static_cast<int>((idx / (head_dim * seq_len)) % out_heads);
  const int batch = static_cast<int>(idx / (head_dim * seq_len * out_heads));
  const int source_head = out_head / repeats;

  const int64_t source_idx =
      ((static_cast<int64_t>(batch) * kv_heads + source_head) * seq_len + seq) *
          head_dim +
      col;
  out[idx] = input[source_idx];
}

template <typename T>
__global__ void linear_kernel(
    T* out,
    const T* input,
    const T* weight,
    const T* bias,
    int rows,
    int in_features,
    int out_features) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(rows) * out_features;
  if (idx >= total) {
    return;
  }

  const int out_col = static_cast<int>(idx % out_features);
  const int row = static_cast<int>(idx / out_features);
  float acc = bias == nullptr ? 0.0f : cuda_kernels::to_float(bias[out_col]);

  for (int in_col = 0; in_col < in_features; ++in_col) {
    const float x =
        cuda_kernels::to_float(input[static_cast<int64_t>(row) * in_features + in_col]);
    const float w = cuda_kernels::to_float(
        weight[static_cast<int64_t>(out_col) * in_features + in_col]);
    acc += x * w;
  }

  out[idx] = from_float<T>(acc);
}

template <typename T>
__global__ void embedding_kernel(
    T* out,
    const int* token_ids,
    const T* embeddings,
    int tokens,
    int hidden_size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(tokens) * hidden_size;
  if (idx >= total) {
    return;
  }

  const int hidden_col = static_cast<int>(idx % hidden_size);
  const int token_idx = static_cast<int>(idx / hidden_size);
  const int token_id = token_ids[token_idx];
  out[idx] = embeddings[static_cast<int64_t>(token_id) * hidden_size + hidden_col];
}

template <typename T>
__global__ void causal_mask_kernel(
    T* out,
    const T* scores,
    int64_t numel,
    int query_len,
    int key_len,
    int start_pos) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }

  const int key_idx = static_cast<int>(idx % key_len);
  const int query_idx = static_cast<int>((idx / key_len) % query_len);
  float value = cuda_kernels::to_float(scores[idx]);

  if (key_idx > start_pos + query_idx) {
    value = -INFINITY;
  }

  out[idx] = from_float<T>(value);
}

template <typename T>
__global__ void sdpa_kernel(
    T* out,
    const T* query,
    const T* key,
    const T* value,
    int batch_size,
    int heads,
    int query_len,
    int key_len,
    int head_dim,
    float scale,
    int causal,
    int start_pos) {
  const int row = blockIdx.x;
  const int col = threadIdx.x;
  if (col >= head_dim) {
    return;
  }

  const int query_idx = row % query_len;
  const int head = (row / query_len) % heads;
  const int batch = row / (query_len * heads);

  const T* query_row =
      query + ((static_cast<int64_t>(batch) * heads + head) * query_len + query_idx) *
                  head_dim;

  float max_score = -INFINITY;
  for (int key_idx = 0; key_idx < key_len; ++key_idx) {
    const T* key_row =
        key + ((static_cast<int64_t>(batch) * heads + head) * key_len + key_idx) *
                  head_dim;
    float score = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      score += cuda_kernels::to_float(query_row[i]) * cuda_kernels::to_float(key_row[i]);
    }
    score *= scale;
    if (causal && key_idx > start_pos + query_idx) {
      score = -INFINITY;
    }
    max_score = fmaxf(max_score, score);
  }

  float denom = 0.0f;
  float weighted_value = 0.0f;
  for (int key_idx = 0; key_idx < key_len; ++key_idx) {
    const T* key_row =
        key + ((static_cast<int64_t>(batch) * heads + head) * key_len + key_idx) *
                  head_dim;
    float score = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      score += cuda_kernels::to_float(query_row[i]) * cuda_kernels::to_float(key_row[i]);
    }
    score *= scale;
    if (causal && key_idx > start_pos + query_idx) {
      score = -INFINITY;
    }

    const float prob = expf(score - max_score);
    const int64_t value_idx =
        ((static_cast<int64_t>(batch) * heads + head) * key_len + key_idx) * head_dim +
        col;
    denom += prob;
    weighted_value += prob * cuda_kernels::to_float(value[value_idx]);
  }

  const int64_t out_idx =
      ((static_cast<int64_t>(batch) * heads + head) * query_len + query_idx) * head_dim +
      col;
  out[out_idx] = from_float<T>(weighted_value / denom);
}

template <typename T>
void launch_fused_silu_split(
    void* out,
    const void* gate_up,
    int64_t rows,
    int inner,
    cudaStream_t stream) {
  fused_silu_split_kernel<T><<<grid_1d(rows * inner), kThreads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(gate_up),
      rows,
      inner);
}

template <typename T>
void launch_affine(
    void* out,
    const void* input,
    int64_t numel,
    float scale,
    float offset,
    cudaStream_t stream) {
  affine_kernel<T><<<grid_1d(numel), kThreads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(input),
      numel,
      scale,
      offset);
}

template <typename T>
void launch_greater_equal(
    void* out,
    const void* left,
    const void* right,
    int64_t numel,
    cudaStream_t stream) {
  greater_equal_kernel<T><<<grid_1d(numel), kThreads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(left),
      static_cast<const T*>(right),
      numel);
}

template <typename T>
void launch_where(
    void* out,
    const void* cond,
    const void* true_values,
    const void* false_values,
    int64_t numel,
    cudaStream_t stream) {
  where_kernel<T><<<grid_1d(numel), kThreads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(cond),
      static_cast<const T*>(true_values),
      static_cast<const T*>(false_values),
      numel);
}

template <typename T>
void launch_add_bias(
    void* out,
    const void* input,
    const void* bias,
    int64_t numel,
    int last_dim,
    cudaStream_t stream) {
  add_bias_kernel<T><<<grid_1d(numel), kThreads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(input),
      static_cast<const T*>(bias),
      numel,
      last_dim);
}

template <typename T>
void launch_narrow_dim1(
    void* out,
    const void* input,
    int dim0,
    int dim1,
    int dim2,
    int start,
    int len,
    cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(dim0) * len * dim2;
  narrow_dim1_kernel<T><<<grid_1d(total), kThreads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(input),
      dim0,
      dim1,
      dim2,
      start,
      len);
}

template <typename T>
void launch_transpose_1_2(
    void* out,
    const void* input,
    int dim0,
    int dim1,
    int dim2,
    int dim3,
    cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(dim0) * dim1 * dim2 * dim3;
  transpose_1_2_kernel<T><<<grid_1d(total), kThreads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(input),
      dim0,
      dim1,
      dim2,
      dim3);
}

template <typename T>
void launch_concat_dim2(
    void* out,
    const void* left,
    const void* right,
    int dim0,
    int dim1,
    int left_dim2,
    int right_dim2,
    int dim3,
    cudaStream_t stream) {
  const int out_dim2 = left_dim2 + right_dim2;
  const int64_t total = static_cast<int64_t>(dim0) * dim1 * out_dim2 * dim3;
  concat_dim2_kernel<T><<<grid_1d(total), kThreads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(left),
      static_cast<const T*>(right),
      dim0,
      dim1,
      left_dim2,
      right_dim2,
      dim3);
}

template <typename T>
void launch_repeat_kv(
    void* out,
    const void* input,
    int batch_size,
    int kv_heads,
    int seq_len,
    int head_dim,
    int repeats,
    cudaStream_t stream) {
  const int64_t total =
      static_cast<int64_t>(batch_size) * kv_heads * repeats * seq_len * head_dim;
  repeat_kv_kernel<T><<<grid_1d(total), kThreads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(input),
      batch_size,
      kv_heads,
      seq_len,
      head_dim,
      repeats);
}

template <typename T>
void launch_linear(
    void* out,
    const void* input,
    const void* weight,
    const void* bias,
    int rows,
    int in_features,
    int out_features,
    cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(rows) * out_features;
  linear_kernel<T><<<grid_1d(total), kThreads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(input),
      static_cast<const T*>(weight),
      static_cast<const T*>(bias),
      rows,
      in_features,
      out_features);
}

template <typename T>
void launch_embedding(
    void* out,
    const int* token_ids,
    const void* embeddings,
    int tokens,
    int hidden_size,
    cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(tokens) * hidden_size;
  embedding_kernel<T><<<grid_1d(total), kThreads, 0, stream>>>(
      static_cast<T*>(out),
      token_ids,
      static_cast<const T*>(embeddings),
      tokens,
      hidden_size);
}

template <typename T>
void launch_causal_mask(
    void* out,
    const void* scores,
    int64_t numel,
    int query_len,
    int key_len,
    int start_pos,
    cudaStream_t stream) {
  causal_mask_kernel<T><<<grid_1d(numel), kThreads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(scores),
      numel,
      query_len,
      key_len,
      start_pos);
}

template <typename T>
void launch_sdpa(
    void* out,
    const void* query,
    const void* key,
    const void* value,
    int batch_size,
    int heads,
    int query_len,
    int key_len,
    int head_dim,
    float scale,
    int causal,
    int start_pos,
    cudaStream_t stream) {
  sdpa_kernel<T><<<batch_size * heads * query_len, head_dim, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(query),
      static_cast<const T*>(key),
      static_cast<const T*>(value),
      batch_size,
      heads,
      query_len,
      key_len,
      head_dim,
      scale,
      causal,
      start_pos);
}

}  // namespace

extern "C" void fused_silu_split_fwd(
    void* out,
    const void* gate_up,
    int64_t rows,
    int inner,
    int dtype,
    cudaStream_t stream) {
  switch (dtype) {
    case 0:
      launch_fused_silu_split<__nv_bfloat16>(out, gate_up, rows, inner, stream);
      break;
    case 1:
      launch_fused_silu_split<half>(out, gate_up, rows, inner, stream);
      break;
    case 2:
      launch_fused_silu_split<float>(out, gate_up, rows, inner, stream);
      break;
  }
}

extern "C" void affine_fwd(
    void* out,
    const void* input,
    int64_t numel,
    float scale,
    float offset,
    int dtype,
    cudaStream_t stream) {
  switch (dtype) {
    case 0:
      launch_affine<__nv_bfloat16>(out, input, numel, scale, offset, stream);
      break;
    case 1:
      launch_affine<half>(out, input, numel, scale, offset, stream);
      break;
    case 2:
      launch_affine<float>(out, input, numel, scale, offset, stream);
      break;
  }
}

extern "C" void ge_fwd(
    void* out,
    const void* left,
    const void* right,
    int64_t numel,
    int dtype,
    cudaStream_t stream) {
  switch (dtype) {
    case 0:
      launch_greater_equal<__nv_bfloat16>(out, left, right, numel, stream);
      break;
    case 1:
      launch_greater_equal<half>(out, left, right, numel, stream);
      break;
    case 2:
      launch_greater_equal<float>(out, left, right, numel, stream);
      break;
  }
}

extern "C" void where_fwd(
    void* out,
    const void* cond,
    const void* true_values,
    const void* false_values,
    int64_t numel,
    int dtype,
    cudaStream_t stream) {
  switch (dtype) {
    case 0:
      launch_where<__nv_bfloat16>(out, cond, true_values, false_values, numel, stream);
      break;
    case 1:
      launch_where<half>(out, cond, true_values, false_values, numel, stream);
      break;
    case 2:
      launch_where<float>(out, cond, true_values, false_values, numel, stream);
      break;
  }
}

extern "C" void add_bias_fwd(
    void* out,
    const void* input,
    const void* bias,
    int64_t numel,
    int last_dim,
    int dtype,
    cudaStream_t stream) {
  switch (dtype) {
    case 0:
      launch_add_bias<__nv_bfloat16>(out, input, bias, numel, last_dim, stream);
      break;
    case 1:
      launch_add_bias<half>(out, input, bias, numel, last_dim, stream);
      break;
    case 2:
      launch_add_bias<float>(out, input, bias, numel, last_dim, stream);
      break;
  }
}

extern "C" void narrow_dim1_fwd(
    void* out,
    const void* input,
    int dim0,
    int dim1,
    int dim2,
    int start,
    int len,
    int dtype,
    cudaStream_t stream) {
  switch (dtype) {
    case 0:
      launch_narrow_dim1<__nv_bfloat16>(out, input, dim0, dim1, dim2, start, len, stream);
      break;
    case 1:
      launch_narrow_dim1<half>(out, input, dim0, dim1, dim2, start, len, stream);
      break;
    case 2:
      launch_narrow_dim1<float>(out, input, dim0, dim1, dim2, start, len, stream);
      break;
  }
}

extern "C" void transpose12_fwd(
    void* out,
    const void* input,
    int dim0,
    int dim1,
    int dim2,
    int dim3,
    int dtype,
    cudaStream_t stream) {
  switch (dtype) {
    case 0:
      launch_transpose_1_2<__nv_bfloat16>(out, input, dim0, dim1, dim2, dim3, stream);
      break;
    case 1:
      launch_transpose_1_2<half>(out, input, dim0, dim1, dim2, dim3, stream);
      break;
    case 2:
      launch_transpose_1_2<float>(out, input, dim0, dim1, dim2, dim3, stream);
      break;
  }
}

extern "C" void concat_dim2_fwd(
    void* out,
    const void* left,
    const void* right,
    int dim0,
    int dim1,
    int left_dim2,
    int right_dim2,
    int dim3,
    int dtype,
    cudaStream_t stream) {
  switch (dtype) {
    case 0:
      launch_concat_dim2<__nv_bfloat16>(out, left, right, dim0, dim1, left_dim2, right_dim2, dim3, stream);
      break;
    case 1:
      launch_concat_dim2<half>(out, left, right, dim0, dim1, left_dim2, right_dim2, dim3, stream);
      break;
    case 2:
      launch_concat_dim2<float>(out, left, right, dim0, dim1, left_dim2, right_dim2, dim3, stream);
      break;
  }
}

extern "C" void repeat_kv_fwd(
    void* out,
    const void* input,
    int batch_size,
    int kv_heads,
    int seq_len,
    int head_dim,
    int repeats,
    int dtype,
    cudaStream_t stream) {
  switch (dtype) {
    case 0:
      launch_repeat_kv<__nv_bfloat16>(out, input, batch_size, kv_heads, seq_len, head_dim, repeats, stream);
      break;
    case 1:
      launch_repeat_kv<half>(out, input, batch_size, kv_heads, seq_len, head_dim, repeats, stream);
      break;
    case 2:
      launch_repeat_kv<float>(out, input, batch_size, kv_heads, seq_len, head_dim, repeats, stream);
      break;
  }
}

extern "C" void linear_fwd(
    void* out,
    const void* input,
    const void* weight,
    const void* bias,
    int rows,
    int in_features,
    int out_features,
    int dtype,
    cudaStream_t stream) {
  switch (dtype) {
    case 0:
      launch_linear<__nv_bfloat16>(out, input, weight, bias, rows, in_features, out_features, stream);
      break;
    case 1:
      launch_linear<half>(out, input, weight, bias, rows, in_features, out_features, stream);
      break;
    case 2:
      launch_linear<float>(out, input, weight, bias, rows, in_features, out_features, stream);
      break;
  }
}

extern "C" void embedding_fwd(
    void* out,
    const int* token_ids,
    const void* embeddings,
    int tokens,
    int hidden_size,
    int dtype,
    cudaStream_t stream) {
  switch (dtype) {
    case 0:
      launch_embedding<__nv_bfloat16>(out, token_ids, embeddings, tokens, hidden_size, stream);
      break;
    case 1:
      launch_embedding<half>(out, token_ids, embeddings, tokens, hidden_size, stream);
      break;
    case 2:
      launch_embedding<float>(out, token_ids, embeddings, tokens, hidden_size, stream);
      break;
  }
}

extern "C" void causal_mask_fwd(
    void* out,
    const void* scores,
    int64_t numel,
    int query_len,
    int key_len,
    int start_pos,
    int dtype,
    cudaStream_t stream) {
  switch (dtype) {
    case 0:
      launch_causal_mask<__nv_bfloat16>(out, scores, numel, query_len, key_len, start_pos, stream);
      break;
    case 1:
      launch_causal_mask<half>(out, scores, numel, query_len, key_len, start_pos, stream);
      break;
    case 2:
      launch_causal_mask<float>(out, scores, numel, query_len, key_len, start_pos, stream);
      break;
  }
}

extern "C" void sdpa_fwd(
    void* out,
    const void* query,
    const void* key,
    const void* value,
    int batch_size,
    int heads,
    int query_len,
    int key_len,
    int head_dim,
    float scale,
    int causal,
    int start_pos,
    int dtype,
    cudaStream_t stream) {
  switch (dtype) {
    case 0:
      launch_sdpa<__nv_bfloat16>(
          out,
          query,
          key,
          value,
          batch_size,
          heads,
          query_len,
          key_len,
          head_dim,
          scale,
          causal,
          start_pos,
          stream);
      break;
    case 1:
      launch_sdpa<half>(
          out,
          query,
          key,
          value,
          batch_size,
          heads,
          query_len,
          key_len,
          head_dim,
          scale,
          causal,
          start_pos,
          stream);
      break;
    case 2:
      launch_sdpa<float>(
          out,
          query,
          key,
          value,
          batch_size,
          heads,
          query_len,
          key_len,
          head_dim,
          scale,
          causal,
          start_pos,
          stream);
      break;
  }
}
