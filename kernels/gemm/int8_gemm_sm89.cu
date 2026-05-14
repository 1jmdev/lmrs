#include <cuda_bf16.h>
#include <stdint.h>

extern "C" __global__ void int8_gemm_sm89(
    const int8_t *__restrict__ a,
    const int8_t *__restrict__ b,
    __nv_bfloat16 *__restrict__ c,
    const int m,
    const int n,
    const int k,
    const float scale
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    int acc = 0;
    for (int i = 0; i < k; ++i) acc += static_cast<int>(a[row * k + i]) * static_cast<int>(b[i * n + col]);
    c[row * n + col] = __float2bfloat16(static_cast<float>(acc) * scale);
}
