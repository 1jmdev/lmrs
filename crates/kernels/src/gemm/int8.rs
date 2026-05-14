/// CUDA module name for SM89 INT8 GEMM.
pub const INT8_GEMM_SM89_MODULE: &str = "lmrs_gemm_int8_sm89";

/// Exported INT8 GEMM symbol.
pub const INT8_GEMM_SM89: &str = "int8_gemm_sm89";

/// Output scale metadata for INT8 GEMM accumulation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Int8GemmScale {
    /// Scale applied to the integer accumulator.
    pub scale: f32,
}

impl Int8GemmScale {
    /// Creates INT8 GEMM scale metadata.
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }
}
