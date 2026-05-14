use ops::{add, AttentionContext, RmsNormConfig, RmsNormOp};
use tensor::{Result, Tensor};

use crate::WeightBuilder;
use crate::layers::{Attention, AttentionConfig, GatedSiluMlp, GatedSiluMlpConfig};

/// Configuration for one decoder transformer block.
///
/// # Example
///
/// ```
/// use model::{AttentionConfig, DecoderLayerConfig, GatedSiluMlpConfig};
///
/// let config = DecoderLayerConfig {
///     attention: AttentionConfig { hidden_size: 128, num_heads: 4, num_kv_heads: 2, head_dim: 32, attention_bias: false, qk_norm_eps: Some(1e-6) },
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
/// use std::collections::HashMap;
/// use model::{AttentionConfig, DecoderLayer, DecoderLayerConfig, GatedSiluMlpConfig, WeightBuilder};
/// use runtime::CudaContext;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let weights = WeightBuilder::new(context, HashMap::new());
/// let config = DecoderLayerConfig {
///     attention: AttentionConfig { hidden_size: 128, num_heads: 4, num_kv_heads: 2, head_dim: 32, attention_bias: false, qk_norm_eps: Some(1e-6) },
///     mlp: GatedSiluMlpConfig { hidden_size: 128, intermediate_size: 256 },
///     hidden_size: 128,
///     rms_norm_eps: 1e-6,
/// };
/// assert!(DecoderLayer::new(config, weights).is_err());
/// # Ok(())
/// # }
/// ```
pub struct DecoderLayer {
    self_attn: Attention,
    mlp: GatedSiluMlp,
    input_layernorm: RmsNormOp,
    post_attention_layernorm: RmsNormOp,
}

impl DecoderLayer {
    /// Builds a decoder layer from CUDA BF16 checkpoint variables.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::collections::HashMap;
    /// # use model::{AttentionConfig, DecoderLayer, DecoderLayerConfig, GatedSiluMlpConfig, WeightBuilder};
    /// # use runtime::CudaContext;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let context = CudaContext::new(0)?;
    /// let weights = WeightBuilder::new(context, HashMap::new());
    /// let config = DecoderLayerConfig {
    ///     attention: AttentionConfig { hidden_size: 128, num_heads: 4, num_kv_heads: 2, head_dim: 32, attention_bias: false, qk_norm_eps: None },
    ///     mlp: GatedSiluMlpConfig { hidden_size: 128, intermediate_size: 256 },
    ///     hidden_size: 128,
    ///     rms_norm_eps: 1e-6,
    /// };
    /// assert!(DecoderLayer::new(config, weights).is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: DecoderLayerConfig, weights: WeightBuilder) -> Result<Self> {
        let norm_config = RmsNormConfig {
            hidden_size: config.hidden_size,
            eps: config.rms_norm_eps,
        };
        Ok(Self {
            self_attn: Attention::new(config.attention, weights.pp("self_attn"))?,
            mlp: GatedSiluMlp::new(config.mlp, weights.pp("mlp"))?,
            input_layernorm: RmsNormOp::new(
                norm_config,
                weights.get("input_layernorm.weight")?,
            )?,
            post_attention_layernorm: RmsNormOp::new(
                norm_config,
                weights.get("post_attention_layernorm.weight")?,
            )?,
        })
    }

    /// Runs a block forward pass for either prefill or decode attention context.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use model::DecoderLayer;
    /// # use ops::AttentionContext;
    /// # use tensor::Tensor;
    /// # fn run(layer: &mut DecoderLayer, x: &Tensor, cos: &Tensor, sin: &Tensor) -> tensor::Result<Tensor> {
    /// let y = layer.forward(x, cos, sin, AttentionContext::Prefill)?;
    /// # Ok(y)
    /// # }
    /// ```
    pub fn forward(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        context: AttentionContext,
    ) -> Result<Tensor> {
        let y = self.input_layernorm.forward(x)?;
        let y = self.self_attn.forward(&y, cos, sin, context)?;
        let x = add(x, &y)?;
        let y = self.post_attention_layernorm.forward(&x)?;
        let y = self.mlp.forward(&y)?;
        add(&x, &y)
    }

    /// Clears the block's attention cache.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use model::DecoderLayer;
    /// # fn clear(layer: &mut DecoderLayer) {
    /// layer.clear_kv_cache();
    /// # }
    /// ```
    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}
