#include <cuda_bf16.h>

// GELU tanh approximation from Hendrycks/Gimpel, also used by BERT/GPT-style
// MLPs: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).
// sqrt(2 / pi) scales the tanh argument to approximate the Gaussian CDF.
constexpr float GELU_TANH_SQRT_2_OVER_PI = 0.7978845608028654f;

// Cubic correction term fitted for the fast tanh approximation of GELU. It
// improves accuracy versus using sqrt(2 / pi) * x alone while avoiding erf().
constexpr float GELU_TANH_CUBIC_COEFF = 0.044715f;

// Applies the tanh-based GELU approximation in FP32. BF16 kernels upcast inputs
// to FP32 for the non-linear math, then downcast once for the output.
__device__ __forceinline__ float fast_gelu(float x) {
    const float x3 = x * x * x;
    const float tanh_arg = GELU_TANH_SQRT_2_OVER_PI * (x + GELU_TANH_CUBIC_COEFF * x3);
    return 0.5f * x * (1.0f + tanhf(tanh_arg));
}

// Elementwise BF16 GELU. The grid-stride loop keeps occupancy high for large
// tensors and still handles small tensors without launching multiple kernels.
extern "C" __global__ void gelu_bf16(
    const __nv_bfloat16 *__restrict__ src,
    __nv_bfloat16 *__restrict__ dst,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        dst[i] = __float2bfloat16(fast_gelu(__bfloat162float(src[i])));
    }
}

// Elementwise FP32 GELU using the same approximation as the BF16 path. This is
// useful for logits or temporary activations that are already materialized as F32.
extern "C" __global__ void gelu_f32(
    const float *__restrict__ src,
    float *__restrict__ dst,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        dst[i] = fast_gelu(src[i]);
    }
}
