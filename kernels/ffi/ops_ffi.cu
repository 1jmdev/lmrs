#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../ops/activation_kernels.cuh"

template <typename T>
__global__ void unary_kernel(T* out, const T* inp, int64_t numel, int op) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float x = cuda_kernels::to_float(inp ? inp[i] : out[i]);
    float y = x;
    if (op == 0) y = x / (1.0f + expf(-x));
    if (op == 1) y = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    if (op == 2) y = fmaxf(x, 0.0f);
    if (op == 3) y = expf(x);
    if (op == 4) y = logf(x);
    if (op == 5) y = 1.0f / (1.0f + expf(-x));
    out[i] = cuda_kernels::from_float<T>(y);
  }
}

template <typename T>
__global__ void binary_kernel(T* out, const T* lhs, const T* rhs, int64_t numel, int op) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float x = cuda_kernels::to_float(lhs[i]);
    const float y = cuda_kernels::to_float(rhs[i]);
    float z = x + y;
    if (op == 1) z = x * y;
    if (op == 2) z = x - y;
    if (op == 3) z = x / y;
    if (op == 4) z = fmaxf(x, y);
    if (op == 5) z = fminf(x, y);
    out[i] = cuda_kernels::from_float<T>(z);
  }
}

template <typename T>
__global__ void fill_kernel(T* out, T value, int64_t numel) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    out[i] = value;
  }
}

template <typename T>
static void launch_unary(void* out, const void* inp, int64_t numel, int op, cudaStream_t stream) {
  if (numel <= 0) return;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((numel + threads - 1) / threads);
  unary_kernel<T><<<blocks, threads, 0, stream>>>(static_cast<T*>(out), static_cast<const T*>(inp), numel, op);
}

template <typename T>
static void launch_binary(void* out, const void* lhs, const void* rhs, int64_t numel, int op, cudaStream_t stream) {
  if (numel <= 0) return;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((numel + threads - 1) / threads);
  binary_kernel<T><<<blocks, threads, 0, stream>>>(static_cast<T*>(out), static_cast<const T*>(lhs), static_cast<const T*>(rhs), numel, op);
}

extern "C" void unary_fwd(void* out, const void* inp, int64_t numel, int op, int dtype, cudaStream_t stream) {
  switch (dtype) {
    case 0: launch_unary<__nv_bfloat16>(out, inp, numel, op, stream); break;
    case 1: launch_unary<half>(out, inp, numel, op, stream); break;
    case 2: launch_unary<float>(out, inp, numel, op, stream); break;
  }
}

extern "C" void binary_fwd(void* out, const void* lhs, const void* rhs,
                           int64_t numel, int op, int dtype, cudaStream_t stream) {
  switch (dtype) {
    case 0: launch_binary<__nv_bfloat16>(out, lhs, rhs, numel, op, stream); break;
    case 1: launch_binary<half>(out, lhs, rhs, numel, op, stream); break;
    case 2: launch_binary<float>(out, lhs, rhs, numel, op, stream); break;
  }
}

extern "C" void fill_fwd(void* out, const void* value, int64_t numel, int dtype, cudaStream_t stream) {
  if (numel <= 0) return;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((numel + threads - 1) / threads);
  switch (dtype) {
    case 0: fill_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(static_cast<__nv_bfloat16*>(out), *static_cast<const __nv_bfloat16*>(value), numel); break;
    case 1: fill_kernel<half><<<blocks, threads, 0, stream>>>(static_cast<half*>(out), *static_cast<const half*>(value), numel); break;
    case 2: fill_kernel<float><<<blocks, threads, 0, stream>>>(static_cast<float*>(out), *static_cast<const float*>(value), numel); break;
  }
}
