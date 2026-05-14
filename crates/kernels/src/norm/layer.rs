/// CUDA module name for LayerNorm forward.
pub const LAYER_NORM_MODULE: &str = "lmrs_norm_layer";

/// Exported BF16 LayerNorm forward symbol.
pub const LAYER_NORM_FWD_BF16: &str = "layer_norm_fwd_bf16";

/// LayerNorm launch metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerNormLaunch {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns per row.
    pub cols: usize,
    /// Numerical stability epsilon.
    pub eps: f32,
}

impl LayerNormLaunch {
    /// Creates LayerNorm launch metadata.
    pub fn new(rows: usize, cols: usize, eps: f32) -> Self {
        Self { rows, cols, eps }
    }
}
