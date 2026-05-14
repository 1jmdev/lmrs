#include <cuda_bf16.h>

// GELU tanh approximation from Hendrycks/Gimpel:
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).
// sqrt(2 / pi) scales the tanh input to approximate the Gaussian CDF.
constexpr float GELU_TANH_SQRT_2_OVER_PI = 0.7978845608028654f;

// Cubic correction coefficient fitted for the tanh approximation. It avoids the
// slower erf-based exact GELU while staying close to the original curve.
constexpr float GELU_TANH_CUBIC_COEFF = 0.044715f;

// Fuses GELU(gate) * up for gated MLP blocks. Input rows are laid out as
// [gate_0..gate_n, up_0..up_n], and each CUDA block processes one row.
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
        const float g3 = g * g * g;
        const float tanh_arg = GELU_TANH_SQRT_2_OVER_PI * (g + GELU_TANH_CUBIC_COEFF * g3);
        const float gelu = 0.5f * g * (1.0f + tanhf(tanh_arg));
        out[i] = __float2bfloat16(gelu * u);
    }
}
