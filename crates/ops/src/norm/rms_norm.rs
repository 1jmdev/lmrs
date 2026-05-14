use tensor::{Result, Tensor, TensorError};

/// Configuration for RMS normalization over the last dimension.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNormConfig {
    /// Width of the final normalized dimension.
    pub hidden_size: usize,
    /// Numerical stability epsilon added to the variance term.
    pub eps: f64,
}

/// Last-dimension RMSNorm operator.
pub struct RmsNormOp {
    weight: Tensor,
    eps: f32,
}

impl RmsNormOp {
    /// Creates an RMSNorm operator from a CUDA BF16 weight tensor.
    pub fn new(config: RmsNormConfig, weight: Tensor) -> Result<Self> {
        if weight.shape().dims() != [config.hidden_size] {
            return Err(TensorError::ShapeMismatch(format!(
                "rms_norm weight must have shape [{}], got {:?}",
                config.hidden_size,
                weight.shape().dims()
            )));
        }
        Ok(Self {
            weight,
            eps: config.eps as f32,
        })
    }

    /// Applies RMSNorm to `x` over its last dimension.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        kernels::rms_norm(x, &self.weight, self.eps)
    }
}
