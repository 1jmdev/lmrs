use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear, linear_no_bias};

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
    inner: Linear,
}

impl LinearOp {
    /// Loads a linear projection from `vb`.
    pub fn new(config: LinearConfig, vb: VarBuilder) -> Result<Self> {
        let inner = if config.bias {
            linear(config.in_features, config.out_features, vb)?
        } else {
            linear_no_bias(config.in_features, config.out_features, vb)?
        };
        Ok(Self { inner })
    }

    /// Wraps an existing Candle linear projection.
    pub fn from_inner(inner: Linear) -> Self {
        Self { inner }
    }

    /// Applies the projection to `x`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }

    /// Returns the projection weights.
    pub fn weight(&self) -> &Tensor {
        self.inner.weight()
    }

    /// Returns the optional projection bias.
    pub fn bias(&self) -> Option<&Tensor> {
        self.inner.bias()
    }
}
