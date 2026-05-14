use tensor::{Result, Tensor};

/// Scaled dot-product attention configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdpaConfig {
    /// Per-head feature width.
    pub head_dim: usize,
    /// Whether to apply a causal mask before softmax.
    pub causal: bool,
    /// Key-cache offset for causal masking during prefill/decode.
    pub start_pos: usize,
}

/// Runs scaled dot-product attention for query, key, and value states.
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, config: SdpaConfig) -> Result<Tensor> {
    kernels::sdpa(q, k, v, config.head_dim, config.causal, config.start_pos)
}

/// Repeats grouped-query key/value heads to match query heads.
pub fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    kernels::repeat_kv(&x, n_rep)
}
