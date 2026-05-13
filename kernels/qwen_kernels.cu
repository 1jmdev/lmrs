#include <cuda_bf16.h>
__device__ __forceinline__ float fast_silu(float x) {
    return x / (1.0f + __expf(-x));
}

extern "C" __global__ void qwen_fused_silu_mul_bf16(
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
        out[i] = __float2bfloat16(fast_silu(g) * u);
    }
}
