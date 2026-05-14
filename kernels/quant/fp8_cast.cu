#include <cuda_bf16.h>
#include <stdint.h>
#include "../include/fp8_utils.cuh"

extern "C" __global__ void bf16_to_fp8_e4m3(
    const __nv_bfloat16 *__restrict__ input,
    uint8_t *__restrict__ output,
    const int elements,
    const float inv_scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) output[idx] = lmrs::float_to_e4m3(__bfloat162float(input[idx]), inv_scale);
}

extern "C" __global__ void fp8_e4m3_to_bf16(
    const uint8_t *__restrict__ input,
    __nv_bfloat16 *__restrict__ output,
    const int elements,
    const float scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) output[idx] = __float2bfloat16(lmrs::e4m3_to_float(input[idx], scale));
}
