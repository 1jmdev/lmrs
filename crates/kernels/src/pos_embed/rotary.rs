/// CUDA module name for out-of-place rotary embedding.
pub const ROTARY_FWD_MODULE: &str = "lmrs_pos_embed_rotary_fwd";

/// CUDA module name for in-place rotary embedding.
pub const ROTARY_INPLACE_MODULE: &str = "lmrs_pos_embed_rotary_inplace";

/// Exported BF16 out-of-place rotary symbol.
pub const ROTARY_FWD_BF16: &str = "rotary_fwd_bf16";

/// Exported BF16 in-place rotary symbol.
pub const ROTARY_INPLACE_BF16: &str = "rotary_inplace_bf16";

/// Rotary launch metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RotaryLaunch {
    /// Number of token rows.
    pub tokens: usize,
    /// Rotary dimension.
    pub dim: usize,
}

impl RotaryLaunch {
    /// Creates rotary launch metadata.
    pub fn new(tokens: usize, dim: usize) -> Self {
        Self { tokens, dim }
    }
}
