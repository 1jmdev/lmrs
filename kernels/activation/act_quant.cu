#include <cuda_bf16.h>
#include <stdint.h>

extern "C" __global__ void silu_quant_int8_bf16(
    const __nv_bfloat16 *__restrict__ input,
    int8_t *__restrict__ output,
    const int elements,
    const float inv_scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) return;
    const float x = __bfloat162float(input[idx]);
    const float y = x / (1.0f + expf(-x));
    const int q = max(-128, min(127, static_cast<int>(roundf(y * inv_scale))));
    output[idx] = static_cast<int8_t>(q);
}
