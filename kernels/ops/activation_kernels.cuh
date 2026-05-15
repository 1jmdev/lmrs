#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace cuda_kernels {

template <typename T>
__device__ __forceinline__ float to_float(T v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<half>(half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
  return static_cast<T>(v);
}

template <>
__device__ __forceinline__ half from_float<half>(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <typename T>
__global__ void silu_mul_kernel(T* out, const T* gate, const T* up, int64_t numel) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float x = to_float(gate[i]);
    const float y = to_float(up[i]);
    out[i] = from_float<T>((x / (1.0f + expf(-x))) * y);
  }
}

template <typename T>
void launch_silu_mul(T* out, const T* gate, const T* up, int64_t numel,
                     cudaStream_t stream) {
  if (numel <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>((numel + threads - 1) / threads);
  silu_mul_kernel<T><<<blocks, threads, 0, stream>>>(out, gate, up, numel);
}

}  // namespace cuda_kernels
