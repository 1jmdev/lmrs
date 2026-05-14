/// CUDA module name for INT8 cast kernels.
pub const INT8_CAST_MODULE: &str = "lmrs_quant_int8_cast";

/// Exported BF16 to INT8 cast symbol.
pub const BF16_TO_INT8: &str = "bf16_to_int8";

/// Exported INT8 to BF16 cast symbol.
pub const INT8_TO_BF16: &str = "int8_to_bf16";

/// Scale metadata for INT8 casts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Int8CastScale {
    /// Scale or inverse scale depending on cast direction.
    pub scale: f32,
}

impl Int8CastScale {
    /// Creates INT8 cast scale metadata.
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }
}
