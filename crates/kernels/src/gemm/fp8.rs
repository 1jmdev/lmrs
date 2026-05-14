/// CUDA module name for SM100 FP8 GEMM.
pub const FP8_GEMM_SM100_MODULE: &str = "lmrs_gemm_fp8_sm100";

/// Exported FP8 E4M3 GEMM symbol.
pub const FP8_GEMM_SM100: &str = "fp8_gemm_sm100";

/// Quantization scales used by the FP8 GEMM kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fp8GemmScales {
    /// Scale applied while dequantizing A.
    pub a_scale: f32,
    /// Scale applied while dequantizing B.
    pub b_scale: f32,
}

impl Fp8GemmScales {
    /// Creates FP8 dequantization scale metadata.
    pub fn new(a_scale: f32, b_scale: f32) -> Self {
        Self { a_scale, b_scale }
    }
}
