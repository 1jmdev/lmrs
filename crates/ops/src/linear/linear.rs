use tensor::{Result, Tensor, TensorError};

/// Linear projection configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearConfig {
    /// Input feature width.
    pub in_features: usize,
    /// Output feature width.
    pub out_features: usize,
    /// Whether to load and apply a bias vector.
    pub bias: bool,
}

/// Dense linear projection over the final input dimension.
pub struct LinearOp {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl LinearOp {
    /// Creates a linear projection from CUDA BF16 weight and optional bias tensors.
    pub fn new(config: LinearConfig, weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        validate_shape(config, &weight, bias.as_ref())?;
        Ok(Self { weight, bias })
    }

    /// Wraps existing CUDA BF16 linear projection weights.
    pub fn from_parts(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    /// Applies the projection to `x`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        kernels::linear(x, &self.weight, self.bias.as_ref())
    }

    /// Returns the projection weights.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Returns the optional projection bias.
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

fn validate_shape(config: LinearConfig, weight: &Tensor, bias: Option<&Tensor>) -> Result<()> {
    let dims = weight.shape().dims();
    if dims != [config.out_features, config.in_features] {
        return Err(TensorError::ShapeMismatch(format!(
            "linear weight must have shape [{}, {}], got {:?}",
            config.out_features, config.in_features, dims
        )));
    }
    if let Some(bias) = bias {
        if bias.shape().dims() != [config.out_features] {
            return Err(TensorError::ShapeMismatch(format!(
                "linear bias must have shape [{}], got {:?}",
                config.out_features,
                bias.shape().dims()
            )));
        }
    }
    Ok(())
}
