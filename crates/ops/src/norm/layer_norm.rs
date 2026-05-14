use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Module, VarBuilder};

/// Configuration for affine layer normalization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerNormConfig {
    /// Width of the final normalized dimension.
    pub hidden_size: usize,
    /// Numerical stability epsilon added to the variance term.
    pub eps: f64,
}

/// Last-dimension LayerNorm operator.
pub struct LayerNormOp {
    inner: LayerNorm,
}

impl LayerNormOp {
    /// Creates a LayerNorm operator from weights in `vb`.
    pub fn new(config: LayerNormConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            inner: candle_nn::layer_norm(config.hidden_size, config.eps, vb)?,
        })
    }

    /// Applies LayerNorm to `x` over its last dimension.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}
