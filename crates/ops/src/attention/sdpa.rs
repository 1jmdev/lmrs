use candle_core::{D, Result, Tensor};

use crate::attention::apply_causal_mask;

/// Scaled dot-product attention configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdpaConfig {
    /// Per-head feature width.
    pub head_dim: usize,
    /// Whether to apply a causal mask before softmax.
    pub causal: bool,
    /// Position offset for causal masking during prefill/decode.
    pub start_pos: usize,
}

/// Runs scaled dot-product attention for query, key, and value states.
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, config: SdpaConfig) -> Result<Tensor> {
    let scale = 1.0 / (config.head_dim as f64).sqrt();
    let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
    let scores = if config.causal {
        apply_causal_mask(
            &scores,
            q.dim(D::Minus2)?,
            k.dim(D::Minus2)?,
            config.start_pos,
        )?
    } else {
        scores
    };
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    probs.matmul(v)
}

/// Repeats grouped-query key/value heads to match query heads.
pub fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, h, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, h, n_rep, s, d))?
        .reshape((b, h * n_rep, s, d))
}
