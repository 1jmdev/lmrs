#include <cuda_bf16.h>

extern "C" __global__ void softmax_rows_bf16(
    const __nv_bfloat16 *__restrict__ input,
    __nv_bfloat16 *__restrict__ output,
    const int rows,
    const int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float max_value = -INFINITY;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        max_value = fmaxf(max_value, __bfloat162float(input[row * cols + col]));
    }
    __shared__ float shared_max[256];
    shared_max[threadIdx.x] = max_value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        __syncthreads();
    }
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        sum += expf(__bfloat162float(input[row * cols + col]) - shared_max[0]);
    }
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        __syncthreads();
    }
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float p = expf(__bfloat162float(input[row * cols + col]) - shared_max[0]) / shared_sum[0];
        output[row * cols + col] = __float2bfloat16(p);
    }
}
