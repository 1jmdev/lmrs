use ops::{LinearConfig, LinearOp, silu_mul};
use tensor::{Result, Tensor};

use crate::WeightBuilder;

/// Configuration for the Qwen-style gated SiLU MLP.
///
/// The MLP uses separate CUDA BF16 gate/up/down projections followed by a fused
/// SiLU-multiply kernel. Keeping gate and up separate avoids host-side weight
/// concatenation while the model crate is being migrated off external tensor
/// frameworks.
///
/// # Example
///
/// ```
/// use model::GatedSiluMlpConfig;
///
/// let config = GatedSiluMlpConfig { hidden_size: 128, intermediate_size: 256 };
/// assert_eq!(config.intermediate_size, config.hidden_size * 2);
/// ```
#[derive(Debug, Clone)]
pub struct GatedSiluMlpConfig {
    /// Input and output hidden-state width.
    pub hidden_size: usize,
    /// Intermediate gated projection width.
    pub intermediate_size: usize,
}

/// Qwen-style SwiGLU MLP layer backed only by CUDA kernels.
///
/// # Example
///
/// ```no_run
/// use std::collections::HashMap;
/// use model::{GatedSiluMlp, GatedSiluMlpConfig, WeightBuilder};
/// use runtime::CudaContext;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let weights = WeightBuilder::new(context, HashMap::new());
/// let config = GatedSiluMlpConfig { hidden_size: 128, intermediate_size: 256 };
/// assert!(GatedSiluMlp::new(config, weights).is_err());
/// # Ok(())
/// # }
/// ```
pub struct GatedSiluMlp {
    gate_proj: LinearOp,
    up_proj: LinearOp,
    down_proj: LinearOp,
}

impl GatedSiluMlp {
    /// Builds the MLP from CUDA BF16 checkpoint variables.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::collections::HashMap;
    /// use model::{GatedSiluMlp, GatedSiluMlpConfig, WeightBuilder};
    /// use runtime::CudaContext;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let context = CudaContext::new(0)?;
    /// let weights = WeightBuilder::new(context, HashMap::new());
    /// let config = GatedSiluMlpConfig { hidden_size: 128, intermediate_size: 256 };
    /// assert!(GatedSiluMlp::new(config, weights).is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: GatedSiluMlpConfig, weights: WeightBuilder) -> Result<Self> {
        let linear_config = |out_features| LinearConfig {
            in_features: config.hidden_size,
            out_features,
            bias: false,
        };
        let down_config = LinearConfig {
            in_features: config.intermediate_size,
            out_features: config.hidden_size,
            bias: false,
        };
        Ok(Self {
            gate_proj: LinearOp::new(
                linear_config(config.intermediate_size),
                weights.get("gate_proj.weight")?,
                None,
            )?,
            up_proj: LinearOp::new(
                linear_config(config.intermediate_size),
                weights.get("up_proj.weight")?,
                None,
            )?,
            down_proj: LinearOp::new(down_config, weights.get("down_proj.weight")?, None)?,
        })
    }

    /// Applies the MLP to hidden states shaped `[..., hidden_size]`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use model::GatedSiluMlp;
    /// # use tensor::Tensor;
    /// # fn run(mlp: &GatedSiluMlp, x: &Tensor) -> tensor::Result<Tensor> {
    /// let y = mlp.forward(x)?;
    /// # Ok(y)
    /// # }
    /// ```
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = silu_mul(&gate, &up)?;
        self.down_proj.forward(&activated)
    }
}
