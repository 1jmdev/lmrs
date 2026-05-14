#include <cuda_bf16.h>

extern "C" __global__ void splitk_bf16_partial(
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b,
    float *__restrict__ partial,
    const int m,
    const int n,
    const int k,
    const int split_k
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int split = blockIdx.z;
    if (row >= m || col >= n || split >= split_k) return;
    const int chunk = (k + split_k - 1) / split_k;
    const int begin = split * chunk;
    const int end = min(k, begin + chunk);
    float acc = 0.0f;
    for (int i = begin; i < end; ++i) acc += __bfloat162float(a[row * k + i]) * __bfloat162float(b[i * n + col]);
    partial[(split * m + row) * n + col] = acc;
}

extern "C" __global__ void splitk_bf16_reduce(
    const float *__restrict__ partial,
    __nv_bfloat16 *__restrict__ c,
    const int m,
    const int n,
    const int split_k
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = m * n;
    if (idx >= total) return;
    float acc = 0.0f;
    for (int split = 0; split < split_k; ++split) acc += partial[split * total + idx];
    c[idx] = __float2bfloat16(acc);
}
