#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../ops/activation_kernels.cuh"

extern "C" void silu_mul_fwd(void* out, const void* gate, const void* up,
                              int64_t numel, int dtype, cudaStream_t stream) {
  switch (dtype) {
    case 0:
      cuda_kernels::launch_silu_mul<__nv_bfloat16>(
          static_cast<__nv_bfloat16*>(out), static_cast<const __nv_bfloat16*>(gate),
          static_cast<const __nv_bfloat16*>(up), numel, stream);
      break;
    case 1:
      cuda_kernels::launch_silu_mul<half>(static_cast<half*>(out),
                                          static_cast<const half*>(gate),
                                          static_cast<const half*>(up), numel, stream);
      break;
    case 2:
      cuda_kernels::launch_silu_mul<float>(static_cast<float*>(out),
                                           static_cast<const float*>(gate),
                                           static_cast<const float*>(up), numel, stream);
      break;
  }
}
