use candle_core::{D, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

/// Configuration for the Qwen-style gated SiLU MLP.
///
/// The MLP uses fused `gate_proj` and `up_proj` weights followed by SiLU gate
/// multiplication and a final `down_proj`.
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

/// Qwen-style SwiGLU MLP layer.
///
/// # Example
///
/// ```no_run
/// use candle_core::{DType, Device, Tensor};
/// use candle_nn::VarBuilder;
/// use model::{GatedSiluMlp, GatedSiluMlpConfig};
///
/// # fn main() -> candle_core::Result<()> {
/// let device = Device::new_cuda(0)?;
/// let tensors = std::collections::HashMap::new();
/// let vb = VarBuilder::from_tensors(tensors, DType::BF16, &device);
/// let mlp = GatedSiluMlp::new(GatedSiluMlpConfig { hidden_size: 128, intermediate_size: 256 }, vb)?;
/// let x = Tensor::zeros((1, 1, 128), DType::BF16, &device)?;
/// let _y = mlp.forward(&x)?;
/// # Ok(())
/// # }
/// ```
pub struct GatedSiluMlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
}

impl GatedSiluMlp {
    /// Builds the MLP from checkpoint variables.
    pub fn new(config: GatedSiluMlpConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;
        let gate_up = Tensor::cat(&[gate_proj.weight(), up_proj.weight()], 0)?;
        let down_proj = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            gate_up_proj: Linear::new(gate_up, None),
            down_proj,
            intermediate_size: config.intermediate_size,
        })
    }

    /// Applies the MLP to hidden states shaped `[..., hidden_size]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?.contiguous()?;
        let gate = gate_up.narrow(D::Minus1, 0, self.intermediate_size)?;
        let up = gate_up.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let activated = (gate.silu()? * up)?;
        self.down_proj.forward(&activated)
    }
}
