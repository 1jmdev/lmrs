use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

/// Final vocabulary projection for causal language models.
pub struct LmHead {
    inner: Linear,
}

impl LmHead {
    /// Loads an untied projection with shape `[vocab_size, hidden_size]`.
    pub fn new(hidden_size: usize, vocab_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            inner: linear_no_bias(hidden_size, vocab_size, vb)?,
        })
    }

    /// Reuses token embedding weights for tied input/output embeddings.
    pub fn tied(embedding_weight: &Tensor) -> Self {
        Self {
            inner: Linear::new(embedding_weight.clone(), None),
        }
    }

    /// Projects hidden states to logits.
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        self.inner.forward(hidden)
    }
}
