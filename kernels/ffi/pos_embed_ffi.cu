#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../ops/activation_kernels.cuh"

template <typename T>
__global__ void rope_kernel(T* dst, const T* src, const T* cos, const T* sin,
                            uint32_t bh, uint32_t td, uint32_t d, uint32_t stride_b) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total = bh * td;
  if (idx >= total) return;
  const uint32_t row = idx / td;
  const uint32_t col = idx % td;
  const uint32_t half_d = d / 2;
  const uint32_t pair_col = col < half_d ? col + half_d : col - half_d;
  const float sign = col < half_d ? -1.0f : 1.0f;
  const int64_t base = static_cast<int64_t>(row) * stride_b;
  const float x = cuda_kernels::to_float(src[base + col]);
  const float y = cuda_kernels::to_float(src[base + pair_col]);
  const float c = cuda_kernels::to_float(cos[col % half_d]);
  const float s = cuda_kernels::to_float(sin[col % half_d]);
  dst[base + col] = cuda_kernels::from_float<T>(x * c + sign * y * s);
}

template <typename T>
static void launch_rope(void* dst, const void* src, const void* cos, const void* sin,
                        uint32_t bh, uint32_t td, uint32_t d, uint32_t stride_b,
                        cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = static_cast<int>((static_cast<uint64_t>(bh) * td + threads - 1) / threads);
  rope_kernel<T><<<blocks, threads, 0, stream>>>(static_cast<T*>(dst), static_cast<const T*>(src),
                                                static_cast<const T*>(cos), static_cast<const T*>(sin),
                                                bh, td, d, stride_b);
}

extern "C" void rope_fwd(void* dst, const void* src, const void* cos,
                         const void* sin, uint32_t bh, uint32_t td, uint32_t d,
                         uint32_t stride_b, int dtype, cudaStream_t stream) {
  switch (dtype) {
    case 0: launch_rope<__nv_bfloat16>(dst, src, cos, sin, bh, td, d, stride_b, stream); break;
    case 1: launch_rope<half>(dst, src, cos, sin, bh, td, d, stride_b, stream); break;
    case 2: launch_rope<float>(dst, src, cos, sin, bh, td, d, stride_b, stream); break;
  }
}
