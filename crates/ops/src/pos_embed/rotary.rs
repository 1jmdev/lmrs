use half::bf16;
use runtime::CudaContext;
use tensor::{DType, Result, Shape, Tensor, TensorError, copy_h2d};

/// Precomputed rotary positional embedding tables.
pub struct RotaryEmbedding {
    context: CudaContext,
    half_dim: usize,
    max_position_embeddings: usize,
    rope_theta: f64,
}

impl RotaryEmbedding {
    /// Builds RoPE cosine and sine tables on `device`.
    pub fn new(
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dtype: DType,
        context: &CudaContext,
    ) -> Result<Self> {
        if dtype != DType::BF16 {
            return Err(TensorError::DTypeMismatch {
                expected: DType::BF16.name(),
                actual: dtype.name(),
            });
        }
        if head_dim % 2 != 0 {
            return Err(TensorError::ShapeMismatch(format!(
                "rotary head_dim must be even, got {head_dim}"
            )));
        }
        let half_dim = head_dim / 2;
        Ok(Self {
            context: context.clone(),
            half_dim,
            max_position_embeddings,
            rope_theta,
        })
    }

    /// Returns the RoPE table slice for the current forward step.
    pub fn get(
        &self,
        total_len: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        if total_len > self.max_position_embeddings {
            return Err(TensorError::InvalidArgument(format!(
                "sequence length {total_len} exceeds max_position_embeddings"
            )));
        }
        let head_dim = self.half_dim * 2;
        let inv: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| (1.0 / self.rope_theta.powf(i as f64 / head_dim as f64)) as f32)
            .collect();
        let mut cos = Vec::with_capacity(seq_len * self.half_dim);
        let mut sin = Vec::with_capacity(seq_len * self.half_dim);
        for position in start_pos..start_pos + seq_len {
            for inv_freq in &inv {
                let freq = position as f32 * inv_freq;
                cos.push(bf16::from_f32(freq.cos()).to_bits());
                sin.push(bf16::from_f32(freq.sin()).to_bits());
            }
        }
        let shape = Shape::new([seq_len, self.half_dim])
            .map_err(|err| TensorError::ShapeMismatch(err.to_string()))?;
        Ok((
            copy_h2d(&self.context, shape.clone(), DType::BF16, &cos)?,
            copy_h2d(&self.context, shape, DType::BF16, &sin)?,
        ))
    }
}

/// Applies RoPE to query or key states shaped `[batch, heads, seq, head_dim]`.
pub fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    kernels::apply_rotary(x, cos, sin)
}
