use tensor::{Result, Tensor, TensorError};

/// Final vocabulary projection for causal language models.
pub struct LmHead {
    weight: Tensor,
}

impl LmHead {
    /// Creates an untied projection from a CUDA BF16 weight shaped `[vocab_size, hidden_size]`.
    pub fn new(hidden_size: usize, vocab_size: usize, weight: Tensor) -> Result<Self> {
        if weight.shape().dims() != [vocab_size, hidden_size] {
            return Err(TensorError::ShapeMismatch(format!(
                "lm_head weight must have shape [{vocab_size}, {hidden_size}], got {:?}",
                weight.shape().dims()
            )));
        }
        Ok(Self { weight })
    }

    /// Reuses token embedding weights for tied input/output embeddings.
    pub fn tied(embedding_weight: &Tensor) -> Self {
        Self {
            weight: embedding_weight.clone(),
        }
    }

    /// Wraps an existing CUDA BF16 projection weight.
    pub fn from_weight(weight: Tensor) -> Self {
        Self { weight }
    }

    /// Projects hidden states to logits.
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        kernels::linear(hidden, &self.weight, None)
    }
}
