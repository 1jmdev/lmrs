#include <cuda_bf16.h>

extern "C" __global__ void rms_norm_fwd_bf16(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out,
    const int rows,
    const int cols,
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float sum_sq = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float v = __bfloat162float(x[row * cols + col]);
        sum_sq += v * v;
    }
    __shared__ float shared[256];
    shared[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(shared[0] / cols + eps);
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float v = __bfloat162float(x[row * cols + col]) * inv * __bfloat162float(weight[col]);
        out[row * cols + col] = __float2bfloat16(v);
    }
}
