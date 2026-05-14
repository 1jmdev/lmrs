#include <cuda_bf16.h>

extern "C" __global__ void bf16_gemm_sm100(
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b,
    __nv_bfloat16 *__restrict__ c,
    const int m,
    const int n,
    const int k
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        acc += __bfloat162float(a[row * k + i]) * __bfloat162float(b[i * n + col]);
    }
    c[row * n + col] = __float2bfloat16(acc);
}
