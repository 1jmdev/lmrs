use tensor::{Result, Tensor, TensorError};

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
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNormOp {
    /// Creates a LayerNorm operator from CUDA BF16 weight and bias tensors.
    pub fn new(config: LayerNormConfig, weight: Tensor, bias: Tensor) -> Result<Self> {
        if weight.shape().dims() != [config.hidden_size]
            || bias.shape().dims() != [config.hidden_size]
        {
            return Err(TensorError::ShapeMismatch(format!(
                "layer_norm weight and bias must have shape [{}], got {:?} and {:?}",
                config.hidden_size,
                weight.shape().dims(),
                bias.shape().dims()
            )));
        }
        Ok(Self {
            weight,
            bias,
            eps: config.eps as f32,
        })
    }

    /// Applies LayerNorm to `x` over its last dimension.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        kernels::layer_norm(x, &self.weight, &self.bias, self.eps)
    }
}
