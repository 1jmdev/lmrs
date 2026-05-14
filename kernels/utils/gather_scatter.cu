#include <cuda_bf16.h>

extern "C" __global__ void gather_bf16(
    const __nv_bfloat16 *__restrict__ src,
    const int *__restrict__ indices,
    __nv_bfloat16 *__restrict__ dst,
    const int rows,
    const int cols
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;
    const int row = idx / cols;
    const int col = idx % cols;
    dst[idx] = src[indices[row] * cols + col];
}

extern "C" __global__ void scatter_bf16(
    const __nv_bfloat16 *__restrict__ src,
    const int *__restrict__ indices,
    __nv_bfloat16 *__restrict__ dst,
    const int rows,
    const int cols
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;
    const int row = idx / cols;
    const int col = idx % cols;
    dst[indices[row] * cols + col] = src[idx];
}
