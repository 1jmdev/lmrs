#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#include "../ops/activation_kernels.cuh"

namespace cuda_kernels {

template <typename T>
__global__ void paged_attention_v1_kernel(
    T* __restrict__ out, const T* __restrict__ query,
    const T* __restrict__ key_cache, const T* __restrict__ value_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens, int32_t num_seqs,
    int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
    int32_t block_size, int32_t max_num_blocks_per_seq, float scale) {
  extern __shared__ float scores[];
  __shared__ float reduce[256];

  const int32_t seq_idx = blockIdx.x;
  const int32_t head_idx = blockIdx.y;
  if (seq_idx >= num_seqs || head_idx >= num_heads) {
    return;
  }

  const int32_t context_len = context_lens[seq_idx];
  const int32_t kv_head_idx = head_idx % num_kv_heads;
  const T* q = query + (static_cast<int64_t>(seq_idx) * num_heads + head_idx) * head_size;
  T* dst = out + (static_cast<int64_t>(seq_idx) * num_heads + head_idx) * head_size;

  if (context_len <= 0) {
    for (int32_t dim = threadIdx.x; dim < head_size; dim += blockDim.x) {
      dst[dim] = from_float<T>(0.0f);
    }
    return;
  }

  float max_score = -FLT_MAX;
  for (int32_t token_idx = threadIdx.x; token_idx < context_len; token_idx += blockDim.x) {
    const int32_t table_idx = seq_idx * max_num_blocks_per_seq + token_idx / block_size;
    const int32_t physical_block = block_tables[table_idx];
    const int32_t block_offset = token_idx % block_size;
    const int64_t cache_offset =
        (((static_cast<int64_t>(physical_block) * num_kv_heads + kv_head_idx) *
          block_size + block_offset) *
         head_size);

    float qk = 0.0f;
    for (int32_t dim = 0; dim < head_size; ++dim) {
      qk += to_float(q[dim]) * to_float(key_cache[cache_offset + dim]);
    }
    qk *= scale;
    scores[token_idx] = qk;
    max_score = fmaxf(max_score, qk);
  }

  reduce[threadIdx.x] = max_score;
  __syncthreads();
  for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  __shared__ float shared_max;
  if (threadIdx.x == 0) {
    shared_max = reduce[0];
  }
  __syncthreads();

  float exp_sum = 0.0f;
  for (int32_t token_idx = threadIdx.x; token_idx < context_len; token_idx += blockDim.x) {
    const float score = expf(scores[token_idx] - shared_max);
    scores[token_idx] = score;
    exp_sum += score;
  }
  reduce[threadIdx.x] = exp_sum;
  __syncthreads();
  for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    }
    __syncthreads();
  }
  __shared__ float shared_exp_sum;
  if (threadIdx.x == 0) {
    shared_exp_sum = reduce[0];
  }
  __syncthreads();

  for (int32_t dim = threadIdx.x; dim < head_size; dim += blockDim.x) {
    float acc = 0.0f;
    for (int32_t token_idx = 0; token_idx < context_len; ++token_idx) {
      const int32_t table_idx = seq_idx * max_num_blocks_per_seq + token_idx / block_size;
      const int32_t physical_block = block_tables[table_idx];
      const int32_t block_offset = token_idx % block_size;
      const int64_t cache_offset =
          (((static_cast<int64_t>(physical_block) * num_kv_heads + kv_head_idx) *
            block_size + block_offset) *
           head_size);
      acc += (scores[token_idx] / shared_exp_sum) * to_float(value_cache[cache_offset + dim]);
    }
    dst[dim] = from_float<T>(acc);
  }
}

template <typename T>
void launch_paged_attention_v1(T* out, const T* query, const T* key_cache,
                               const T* value_cache, const int32_t* block_tables,
                               const int32_t* context_lens, int32_t num_seqs,
                               int32_t num_heads, int32_t num_kv_heads,
                               int32_t head_size, int32_t block_size,
                               int32_t max_num_blocks_per_seq, float scale,
                               cudaStream_t stream) {
  if (num_seqs <= 0 || num_heads <= 0 || head_size <= 0 || block_size <= 0) {
    return;
  }
  const dim3 grid(num_seqs, num_heads);
  constexpr int threads = 256;
  const size_t shared = static_cast<size_t>(max_num_blocks_per_seq) * block_size * sizeof(float);
  paged_attention_v1_kernel<T><<<grid, threads, shared, stream>>>(
      out, query, key_cache, value_cache, block_tables, context_lens, num_seqs,
      num_heads, num_kv_heads, head_size, block_size, max_num_blocks_per_seq, scale);
}

}  // namespace cuda_kernels
