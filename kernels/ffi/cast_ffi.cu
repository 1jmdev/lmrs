#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../ops/activation_kernels.cuh"

template <typename S, typename D>
__global__ void cast_kernel(D* out, const S* inp, int64_t numel) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    out[i] = cuda_kernels::from_float<D>(cuda_kernels::to_float(inp[i]));
  }
}

template <typename S, typename D>
static void launch_cast(void* out, const void* inp, int64_t numel, cudaStream_t stream) {
  if (numel <= 0) return;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((numel + threads - 1) / threads);
  cast_kernel<S, D><<<blocks, threads, 0, stream>>>(static_cast<D*>(out), static_cast<const S*>(inp), numel);
}

extern "C" void cast_fwd(void* out, const void* inp, int64_t numel,
                         int src_dtype, int dst_dtype, cudaStream_t stream) {
  if (src_dtype == 0 && dst_dtype == 0) launch_cast<__nv_bfloat16, __nv_bfloat16>(out, inp, numel, stream);
  if (src_dtype == 0 && dst_dtype == 1) launch_cast<__nv_bfloat16, half>(out, inp, numel, stream);
  if (src_dtype == 0 && dst_dtype == 2) launch_cast<__nv_bfloat16, float>(out, inp, numel, stream);
  if (src_dtype == 1 && dst_dtype == 0) launch_cast<half, __nv_bfloat16>(out, inp, numel, stream);
  if (src_dtype == 1 && dst_dtype == 1) launch_cast<half, half>(out, inp, numel, stream);
  if (src_dtype == 1 && dst_dtype == 2) launch_cast<half, float>(out, inp, numel, stream);
  if (src_dtype == 2 && dst_dtype == 0) launch_cast<float, __nv_bfloat16>(out, inp, numel, stream);
  if (src_dtype == 2 && dst_dtype == 1) launch_cast<float, half>(out, inp, numel, stream);
  if (src_dtype == 2 && dst_dtype == 2) launch_cast<float, float>(out, inp, numel, stream);
}
