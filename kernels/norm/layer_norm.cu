#include <cuda_bf16.h>

extern "C" __global__ void layer_norm_fwd_bf16(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ bias,
    __nv_bfloat16 *__restrict__ out,
    const int rows,
    const int cols,
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) sum += __bfloat162float(x[row * cols + col]);
    __shared__ float shared[256];
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float mean = shared[0] / cols;
    float var = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float d = __bfloat162float(x[row * cols + col]) - mean;
        var += d * d;
    }
    shared[threadIdx.x] = var;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(shared[0] / cols + eps);
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float y = (__bfloat162float(x[row * cols + col]) - mean) * inv;
        out[row * cols + col] = __float2bfloat16(y * __bfloat162float(weight[col]) + __bfloat162float(bias[col]));
    }
}
