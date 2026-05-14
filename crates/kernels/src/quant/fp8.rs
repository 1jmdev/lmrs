/// CUDA module name for FP8 cast kernels.
pub const FP8_CAST_MODULE: &str = "lmrs_quant_fp8_cast";

/// Exported BF16 to FP8 E4M3 cast symbol.
pub const BF16_TO_FP8_E4M3: &str = "bf16_to_fp8_e4m3";

/// Exported FP8 E4M3 to BF16 cast symbol.
pub const FP8_E4M3_TO_BF16: &str = "fp8_e4m3_to_bf16";

/// Scale metadata for FP8 casts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fp8CastScale {
    /// Scale or inverse scale depending on cast direction.
    pub scale: f32,
}

impl Fp8CastScale {
    /// Creates FP8 cast scale metadata.
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }
}
