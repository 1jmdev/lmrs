#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../attention/paged_attention.cuh"

extern "C" void paged_attention_v1_fwd(
    void* out, const void* query, const void* key_cache, const void* value_cache,
    const int32_t* block_tables, const int32_t* context_lens, int32_t num_seqs,
    int32_t num_heads, int32_t num_kv_heads, int32_t head_size, int32_t block_size,
    int32_t max_num_blocks_per_seq, float scale, int dtype, cudaStream_t stream) {
  switch (dtype) {
    case 0:
      cuda_kernels::launch_paged_attention_v1<__nv_bfloat16>(
          static_cast<__nv_bfloat16*>(out), static_cast<const __nv_bfloat16*>(query),
          static_cast<const __nv_bfloat16*>(key_cache),
          static_cast<const __nv_bfloat16*>(value_cache), block_tables, context_lens,
          num_seqs, num_heads, num_kv_heads, head_size, block_size,
          max_num_blocks_per_seq, scale, stream);
      break;
    case 1:
      cuda_kernels::launch_paged_attention_v1<half>(
          static_cast<half*>(out), static_cast<const half*>(query),
          static_cast<const half*>(key_cache), static_cast<const half*>(value_cache),
          block_tables, context_lens, num_seqs, num_heads, num_kv_heads, head_size,
          block_size, max_num_blocks_per_seq, scale, stream);
      break;
    case 2:
      cuda_kernels::launch_paged_attention_v1<float>(
          static_cast<float*>(out), static_cast<const float*>(query),
          static_cast<const float*>(key_cache), static_cast<const float*>(value_cache),
          block_tables, context_lens, num_seqs, num_heads, num_kv_heads, head_size,
          block_size, max_num_blocks_per_seq, scale, stream);
      break;
  }
}
