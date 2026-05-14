/// CUDA module name for fused GELU multiply.
pub const FUSED_GELU_MUL_MODULE: &str = "lmrs_activation_gelu";

/// Exported BF16 fused GELU multiply symbol.
pub const FUSED_GELU_MUL_BF16: &str = "fused_gelu_mul_bf16";

/// Describes the row-wise fused activation launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FusedGeluMulLaunch {
    /// Number of packed rows in the input.
    pub rows: usize,
    /// Size of the post-activation row.
    pub intermediate_size: usize,
}

impl FusedGeluMulLaunch {
    /// Creates launch metadata for a fused GELU multiply pass.
    pub fn new(rows: usize, intermediate_size: usize) -> Self {
        Self { rows, intermediate_size }
    }
}
