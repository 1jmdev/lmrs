#include <cuda_bf16.h>

extern "C" __global__ void fused_gelu_mul_bf16(
    const __nv_bfloat16 *__restrict__ gate_up,
    __nv_bfloat16 *__restrict__ dst,
    const int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const __nv_bfloat16 *gate = gate_up + row * 2 * intermediate_size;
    const __nv_bfloat16 *up = gate + intermediate_size;
    __nv_bfloat16 *out = dst + row * intermediate_size;
    for (int i = tid; i < intermediate_size; i += block_size) {
        const float g = __bfloat162float(gate[i]);
        const float u = __bfloat162float(up[i]);
        const float gelu = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
        out[i] = __float2bfloat16(gelu * u);
    }
}
