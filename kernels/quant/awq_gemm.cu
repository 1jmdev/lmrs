#include <cuda_bf16.h>
#include <stdint.h>

extern "C" __global__ void awq_gemm_bf16(
    const __nv_bfloat16 *__restrict__ a,
    const uint8_t *__restrict__ qweight,
    const __nv_bfloat16 *__restrict__ scales,
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
        const uint8_t packed = qweight[(i * n + col) / 2];
        const int q = ((col & 1) == 0 ? (packed & 0x0f) : (packed >> 4)) - 8;
        acc += __bfloat162float(a[row * k + i]) * static_cast<float>(q) * __bfloat162float(scales[col]);
    }
    c[row * n + col] = __float2bfloat16(acc);
}
