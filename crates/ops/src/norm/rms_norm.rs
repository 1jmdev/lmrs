use candle_core::{Result, Tensor};
use candle_nn::{Module, RmsNorm, VarBuilder};

/// Configuration for RMS normalization over the last dimension.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNormConfig {
    /// Width of the final normalized dimension.
    pub hidden_size: usize,
    /// Numerical stability epsilon added to the variance term.
    pub eps: f64,
}

/// Last-dimension RMSNorm operator.
///
/// # Example
///
/// ```no_run
/// use candle_core::{DType, Device, Tensor};
/// use candle_nn::VarBuilder;
/// use ops::{RmsNormConfig, RmsNormOp};
///
/// # fn main() -> candle_core::Result<()> {
/// let device = Device::new_cuda(0)?;
/// let weights = std::collections::HashMap::new();
/// let vb = VarBuilder::from_tensors(weights, DType::BF16, &device);
/// let norm = RmsNormOp::new(RmsNormConfig { hidden_size: 4, eps: 1e-6 }, vb)?;
/// let x = Tensor::zeros((1, 2, 4), DType::BF16, &device)?;
/// let _y = norm.forward(&x)?;
/// # Ok(())
/// # }
/// ```
pub struct RmsNormOp {
    inner: RmsNorm,
}

impl RmsNormOp {
    /// Creates an RMSNorm operator from weights in `vb`.
    pub fn new(config: RmsNormConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            inner: candle_nn::rms_norm(config.hidden_size, config.eps, vb)?,
        })
    }

    /// Applies RMSNorm to `x` over its last dimension.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}
