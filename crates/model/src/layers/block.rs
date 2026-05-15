use cache::AttentionContext;
use candle_core::{Result, Tensor};
use candle_nn::{Module, RmsNorm, VarBuilder};

use crate::layers::{Attention, AttentionConfig, GatedSiluMlp, GatedSiluMlpConfig};

/// Configuration for one decoder transformer block.
///
/// # Example
///
/// ```
/// use model::{AttentionConfig, DecoderLayerConfig, GatedSiluMlpConfig};
///
/// let config = DecoderLayerConfig {
///     attention: AttentionConfig {
///         hidden_size: 128,
///         num_heads: 4,
///         num_kv_heads: 2,
///         head_dim: 32,
///         attention_bias: false,
///         qk_norm_eps: Some(1e-6),
///     },
///     mlp: GatedSiluMlpConfig { hidden_size: 128, intermediate_size: 256 },
///     hidden_size: 128,
///     rms_norm_eps: 1e-6,
/// };
/// assert_eq!(config.attention.num_heads, 4);
/// ```
#[derive(Debug, Clone)]
pub struct DecoderLayerConfig {
    /// Self-attention configuration.
    pub attention: AttentionConfig,
    /// Feed-forward MLP configuration.
    pub mlp: GatedSiluMlpConfig,
    /// Hidden-state width.
    pub hidden_size: usize,
    /// RMSNorm epsilon for both block norms.
    pub rms_norm_eps: f64,
}

/// Decoder block with attention, residual connections, RMSNorm, and MLP.
///
/// # Example
///
/// ```no_run
/// use candle_core::{DType, Device, Tensor};
/// use candle_nn::VarBuilder;
/// use model::{AttentionConfig, DecoderLayer, DecoderLayerConfig, GatedSiluMlpConfig};
/// use cache::AttentionContext;
///
/// # fn main() -> candle_core::Result<()> {
/// let device = Device::new_cuda(0)?;
/// let tensors = std::collections::HashMap::new();
/// let vb = VarBuilder::from_tensors(tensors, DType::BF16, &device);
/// let config = DecoderLayerConfig {
///     attention: AttentionConfig { hidden_size: 128, num_heads: 4, num_kv_heads: 2, head_dim: 32, attention_bias: false, qk_norm_eps: Some(1e-6) },
///     mlp: GatedSiluMlpConfig { hidden_size: 128, intermediate_size: 256 },
///     hidden_size: 128,
///     rms_norm_eps: 1e-6,
/// };
/// let mut layer = DecoderLayer::new(config, vb)?;
/// let x = Tensor::zeros((1, 1, 128), DType::BF16, &device)?;
/// let cos = Tensor::zeros((1, 16), DType::BF16, &device)?;
/// let sin = Tensor::zeros((1, 16), DType::BF16, &device)?;
/// let _y = layer.forward(&x, &cos, &sin, AttentionContext::Prefill)?;
/// # Ok(())
/// # }
/// ```
pub struct DecoderLayer {
    self_attn: Attention,
    mlp: GatedSiluMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    /// Builds a decoder layer from checkpoint variables.
    pub fn new(config: DecoderLayerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(config.attention, vb.pp("self_attn"))?,
            mlp: GatedSiluMlp::new(config.mlp, vb.pp("mlp"))?,
            input_layernorm: candle_nn::rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: candle_nn::rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    /// Runs a block forward pass for either prefill or decode attention context.
    pub fn forward(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        context: AttentionContext,
    ) -> Result<Tensor> {
        let residual = x;
        let y = self.input_layernorm.forward(x)?;
        let y = self.self_attn.forward(&y, cos, sin, context)?;
        let x = (residual + y)?;
        let residual = &x;
        let y = self.post_attention_layernorm.forward(&x)?;
        let y = self.mlp.forward(&y)?;
        residual + y
    }

    /// Clears the block's attention cache.
    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}
