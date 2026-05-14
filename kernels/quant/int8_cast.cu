#include <cuda_bf16.h>
#include <stdint.h>

extern "C" __global__ void bf16_to_int8(
    const __nv_bfloat16 *__restrict__ input,
    int8_t *__restrict__ output,
    const int elements,
    const float inv_scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) return;
    const int q = max(-128, min(127, static_cast<int>(roundf(__bfloat162float(input[idx]) * inv_scale))));
    output[idx] = static_cast<int8_t>(q);
}

extern "C" __global__ void int8_to_bf16(
    const int8_t *__restrict__ input,
    __nv_bfloat16 *__restrict__ output,
    const int elements,
    const float scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) output[idx] = __float2bfloat16(static_cast<float>(input[idx]) * scale);
}
