use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};

/// Token embedding lookup table.
pub struct TokenEmbedding {
    inner: Embedding,
}

impl TokenEmbedding {
    /// Loads an embedding table with shape `[vocab_size, hidden_size]`.
    pub fn new(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            inner: candle_nn::embedding(vocab_size, hidden_size, vb)?,
        })
    }

    /// Wraps an existing embedding table.
    pub fn from_embeddings(embeddings: Tensor) -> Self {
        Self {
            inner: Embedding::new(embeddings, 0),
        }
    }

    /// Returns the backing embedding tensor.
    pub fn embeddings(&self) -> &Tensor {
        self.inner.embeddings()
    }

    /// Looks up token ids and returns hidden states.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.inner.forward(input_ids)
    }
}
