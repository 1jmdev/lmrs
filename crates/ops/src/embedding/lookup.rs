use tensor::{Result, Tensor, TensorError};

/// Token embedding lookup table.
pub struct TokenEmbedding {
    embeddings: Tensor,
}

impl TokenEmbedding {
    /// Creates an embedding table from a CUDA BF16 tensor shaped `[vocab_size, hidden_size]`.
    pub fn new(vocab_size: usize, hidden_size: usize, embeddings: Tensor) -> Result<Self> {
        if embeddings.shape().dims() != [vocab_size, hidden_size] {
            return Err(TensorError::ShapeMismatch(format!(
                "embeddings must have shape [{vocab_size}, {hidden_size}], got {:?}",
                embeddings.shape().dims()
            )));
        }
        Ok(Self { embeddings })
    }

    /// Wraps an existing embedding table.
    pub fn from_embeddings(embeddings: Tensor) -> Self {
        Self { embeddings }
    }

    /// Returns the backing embedding tensor.
    pub fn embeddings(&self) -> &Tensor {
        &self.embeddings
    }

    /// Looks up token ids and returns hidden states.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        kernels::embedding_lookup(input_ids, &self.embeddings)
    }
}
