/// CUDA module name for RMSNorm forward.
pub const RMS_NORM_FWD_MODULE: &str = "lmrs_norm_rms_fwd";

/// CUDA module name for RMSNorm backward.
pub const RMS_NORM_BWD_MODULE: &str = "lmrs_norm_rms_bwd";

/// Exported BF16 RMSNorm forward symbol.
pub const RMS_NORM_FWD_BF16: &str = "rms_norm_fwd_bf16";

/// Exported BF16 RMSNorm backward symbol.
pub const RMS_NORM_BWD_BF16: &str = "rms_norm_bwd_bf16";

/// Row-wise normalization shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormShape {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns per row.
    pub cols: usize,
    /// Numerical stability epsilon.
    pub eps: f32,
}

impl NormShape {
    /// Creates a normalization shape descriptor.
    pub fn new(rows: usize, cols: usize, eps: f32) -> Self {
        Self { rows, cols, eps }
    }
}
