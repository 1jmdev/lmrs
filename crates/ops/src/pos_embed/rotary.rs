use candle_core::{DType, Device, Result, Tensor};
use candle_nn::rotary_emb::rope;

/// Precomputed rotary positional embedding tables.
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    /// Builds RoPE cosine and sine tables on `device`.
    pub fn new(
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let inv: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| (1.0 / rope_theta.powf(i as f64 / head_dim as f64)) as f32)
            .collect();
        let inv = Tensor::new(inv.as_slice(), device)?;
        let positions: Vec<f32> = (0..max_position_embeddings).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv.unsqueeze(0)?)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?.contiguous()?,
            sin: freqs.sin()?.to_dtype(dtype)?.contiguous()?,
        })
    }

    /// Returns the RoPE table slice for the current forward step.
    pub fn get(
        &self,
        total_len: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        if total_len > self.cos.dim(0)? {
            candle_core::bail!("sequence length {total_len} exceeds max_position_embeddings");
        }
        let cos = self.cos.narrow(0, start_pos, seq_len)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?;
        Ok((cos, sin))
    }
}

/// Applies RoPE to query or key states shaped `[batch, heads, seq, head_dim]`.
pub fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    rope(&x.contiguous()?, cos, sin)
}
