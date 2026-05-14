#include <cuda_bf16.h>
#include <stdint.h>
#include "../include/fp8_utils.cuh"

extern "C" __global__ void fp8_gemm_sm100(
    const uint8_t *__restrict__ a,
    const uint8_t *__restrict__ b,
    __nv_bfloat16 *__restrict__ c,
    const int m,
    const int n,
    const int k,
    const float a_scale,
    const float b_scale
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        acc += lmrs::e4m3_to_float(a[row * k + i], a_scale) * lmrs::e4m3_to_float(b[i * n + col], b_scale);
    }
    c[row * n + col] = __float2bfloat16(acc);
}
