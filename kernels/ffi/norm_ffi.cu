#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../ops/activation_kernels.cuh"

template <typename T>
__global__ void rmsnorm_kernel(T* dst, const T* src, const T* alpha,
                               int n_cols, float eps) {
  const int row = blockIdx.x;
  const int64_t base = static_cast<int64_t>(row) * n_cols;
  extern __shared__ float shared[];
  float sum = 0.0f;
  for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
    const float x = cuda_kernels::to_float(src[base + col]);
    sum += x * x;
  }
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv = rsqrtf(shared[0] / n_cols + eps);
  for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
    const float x = cuda_kernels::to_float(src[base + col]);
    const float a = cuda_kernels::to_float(alpha[col]);
    dst[base + col] = cuda_kernels::from_float<T>(x * inv * a);
  }
}

template <typename T>
__global__ void layernorm_kernel(T* dst, const T* src, const T* alpha,
                                 const T* beta, int n_cols, float eps) {
  const int row = blockIdx.x;
  const int64_t base = static_cast<int64_t>(row) * n_cols;
  extern __shared__ float shared[];
  float sum = 0.0f;
  for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
    sum += cuda_kernels::to_float(src[base + col]);
  }
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  const float mean = shared[0] / n_cols;
  float var_sum = 0.0f;
  for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
    const float d = cuda_kernels::to_float(src[base + col]) - mean;
    var_sum += d * d;
  }
  shared[threadIdx.x] = var_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv = rsqrtf(shared[0] / n_cols + eps);
  for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
    const float x = cuda_kernels::to_float(src[base + col]);
    const float a = cuda_kernels::to_float(alpha[col]);
    const float b = cuda_kernels::to_float(beta[col]);
    dst[base + col] = cuda_kernels::from_float<T>((x - mean) * inv * a + b);
  }
}

template <typename T>
static void launch_rmsnorm(void* dst, const void* src, const void* alpha,
                           int n_rows, int n_cols, int block_size, float eps,
                           cudaStream_t stream) {
  rmsnorm_kernel<T><<<n_rows, block_size, block_size * sizeof(float), stream>>>(
      static_cast<T*>(dst), static_cast<const T*>(src), static_cast<const T*>(alpha), n_cols, eps);
}

template <typename T>
static void launch_layernorm(void* dst, const void* src, const void* alpha,
                             const void* beta, int n_rows, int n_cols,
                             int block_size, float eps, cudaStream_t stream) {
  layernorm_kernel<T><<<n_rows, block_size, block_size * sizeof(float), stream>>>(
      static_cast<T*>(dst), static_cast<const T*>(src), static_cast<const T*>(alpha),
      static_cast<const T*>(beta), n_cols, eps);
}

extern "C" void rmsnorm_fwd(void* dst, const void* src, const void* alpha,
                            int n_rows, int n_cols, int block_size, float eps,
                            int dtype, cudaStream_t stream) {
  switch (dtype) {
    case 0: launch_rmsnorm<__nv_bfloat16>(dst, src, alpha, n_rows, n_cols, block_size, eps, stream); break;
    case 1: launch_rmsnorm<half>(dst, src, alpha, n_rows, n_cols, block_size, eps, stream); break;
    case 2: launch_rmsnorm<float>(dst, src, alpha, n_rows, n_cols, block_size, eps, stream); break;
  }
}

extern "C" void layernorm_fwd(void* dst, const void* src, const void* alpha,
                              const void* beta, int n_rows, int n_cols,
                              int block_size, float eps, int dtype,
                              cudaStream_t stream) {
  switch (dtype) {
    case 0: launch_layernorm<__nv_bfloat16>(dst, src, alpha, beta, n_rows, n_cols, block_size, eps, stream); break;
    case 1: launch_layernorm<half>(dst, src, alpha, beta, n_rows, n_cols, block_size, eps, stream); break;
    case 2: launch_layernorm<float>(dst, src, alpha, beta, n_rows, n_cols, block_size, eps, stream); break;
  }
}
