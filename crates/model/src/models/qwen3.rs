use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, RmsNorm, VarBuilder};
use ops::{AttentionContext, LmHead, RotaryEmbedding, TokenEmbedding};
use serde::Deserialize;

use crate::layers::{AttentionConfig, DecoderLayer, DecoderLayerConfig, GatedSiluMlpConfig};
use crate::traits::{Cacheable, Model, ModelMetadata};

fn default_true() -> bool {
    true
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_max_position_embeddings() -> usize {
    32768
}

/// Typed Qwen3 architecture configuration.
///
/// The shared `ModelConfig` keeps unknown JSON fields intact, then this type
/// performs the Qwen3-specific deserialization used by the model builder.
///
/// # Example
///
/// ```
/// use model::qwen3::Config;
///
/// let json = r#"{
///   "vocab_size": 32000,
///   "hidden_size": 128,
///   "intermediate_size": 256,
///   "num_hidden_layers": 2,
///   "num_attention_heads": 4,
///   "num_key_value_heads": 2,
///   "rms_norm_eps": 0.000001
/// }"#;
/// let config: Config = serde_json::from_str(json).unwrap();
/// assert_eq!(config.head_dim(), 32);
/// assert!(config.tie_word_embeddings);
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Vocabulary size for token embeddings and logits.
    pub vocab_size: usize,
    /// Transformer hidden state width.
    pub hidden_size: usize,
    /// SwiGLU intermediate width.
    pub intermediate_size: usize,
    /// Number of decoder blocks.
    pub num_hidden_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads.
    pub num_key_value_heads: usize,
    /// Optional per-head width. Defaults to `hidden_size / num_attention_heads`.
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Maximum RoPE table length.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// RoPE base theta.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// Whether attention projections include bias terms.
    #[serde(default)]
    pub attention_bias: bool,
    /// Whether to apply Q/K RMSNorm before RoPE.
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,
    /// Whether the LM head reuses token embedding weights.
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
}

impl Config {
    /// Returns the concrete attention head width.
    ///
    /// # Example
    ///
    /// ```
    /// # use model::qwen3::Config;
    /// # let config: Config = serde_json::from_str(r#"{"vocab_size":8,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-6}"#).unwrap();
    /// assert_eq!(config.head_dim(), 16);
    /// ```
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Builds the reusable decoder-layer configuration.
    ///
    /// # Example
    ///
    /// ```
    /// # use model::qwen3::Config;
    /// # let config: Config = serde_json::from_str(r#"{"vocab_size":8,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-6}"#).unwrap();
    /// let layer = config.decoder_layer_config();
    /// assert_eq!(layer.hidden_size, 64);
    /// assert_eq!(layer.attention.head_dim, 16);
    /// ```
    pub fn decoder_layer_config(&self) -> DecoderLayerConfig {
        DecoderLayerConfig {
            attention: AttentionConfig {
                hidden_size: self.hidden_size,
                num_heads: self.num_attention_heads,
                num_kv_heads: self.num_key_value_heads,
                head_dim: self.head_dim(),
                attention_bias: self.attention_bias,
                qk_norm_eps: self.use_qk_norm.then_some(self.rms_norm_eps),
            },
            mlp: GatedSiluMlpConfig {
                hidden_size: self.hidden_size,
                intermediate_size: self.intermediate_size,
            },
            hidden_size: self.hidden_size,
            rms_norm_eps: self.rms_norm_eps,
        }
    }
}

/// Qwen3 causal language model.
///
/// This type owns token embeddings, decoder layers, final norm, LM head, and
/// RoPE tables. It implements both `Model` and `Cacheable` so the engine can
/// run prompt prefill followed by decode steps without depending on concrete
/// Qwen3 internals.
///
/// # Example
///
/// ```no_run
/// use candle_core::{DType, Device};
/// use candle_nn::VarBuilder;
/// use model::qwen3::{Config, ModelForCausalLM};
///
/// # fn main() -> candle_core::Result<()> {
/// let device = Device::new_cuda(0)?;
/// let tensors = std::collections::HashMap::new();
/// let vb = VarBuilder::from_tensors(tensors, DType::BF16, &device);
/// let config: Config = serde_json::from_str(r#"{
///   "vocab_size": 32000,
///   "hidden_size": 128,
///   "intermediate_size": 256,
///   "num_hidden_layers": 2,
///   "num_attention_heads": 4,
///   "num_key_value_heads": 2,
///   "rms_norm_eps": 0.000001
/// }"#).unwrap();
/// let _model = ModelForCausalLM::new(&config, vb)?;
/// # Ok(())
/// # }
/// ```
pub struct ModelForCausalLM {
    config: Config,
    embed_tokens: TokenEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: LmHead,
    rotary: RotaryEmbedding,
}

impl ModelForCausalLM {
    /// Builds a Qwen3 model from checkpoint variables.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use candle_core::{DType, Device};
    /// use candle_nn::VarBuilder;
    /// use model::qwen3::{Config, ModelForCausalLM};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let device = Device::new_cuda(0)?;
    /// let tensors = std::collections::HashMap::new();
    /// let vb = VarBuilder::from_tensors(tensors, DType::BF16, &device);
    /// let config: Config = serde_json::from_str(r#"{"vocab_size":8,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-6}"#).unwrap();
    /// let _model = ModelForCausalLM::new(&config, vb)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        if !vb.device().is_cuda() {
            candle_core::bail!("qwen3 is CUDA-only");
        }
        if vb.dtype() != DType::BF16 {
            candle_core::bail!("qwen3 is BF16-only");
        }
        let model_vb = vb.pp("model");
        let embed_tokens = TokenEmbedding::new(
            config.vocab_size,
            config.hidden_size,
            model_vb.pp("embed_tokens"),
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let layers_vb = model_vb.pp("layers");
        let layer_config = config.decoder_layer_config();
        for idx in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(layer_config.clone(), layers_vb.pp(idx))?);
        }
        let norm =
            candle_nn::rms_norm(config.hidden_size, config.rms_norm_eps, model_vb.pp("norm"))?;
        let lm_head = if config.tie_word_embeddings {
            LmHead::tied(embed_tokens.embeddings())
        } else {
            let linear =
                candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;
            LmHead::from_linear(Linear::new(linear.weight().clone(), None))
        };
        let rotary = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            DType::BF16,
            vb.device(),
        )?;
        Ok(Self {
            config: config.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
        })
    }

    /// Runs a Qwen3 forward pass and returns logits for the last input token.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use candle_core::{DType, Device, Tensor};
    /// use candle_nn::VarBuilder;
    /// use model::qwen3::{Config, ModelForCausalLM};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let device = Device::new_cuda(0)?;
    /// let tensors = std::collections::HashMap::new();
    /// let vb = VarBuilder::from_tensors(tensors, DType::BF16, &device);
    /// let config: Config = serde_json::from_str(r#"{"vocab_size":8,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-6}"#).unwrap();
    /// let mut model = ModelForCausalLM::new(&config, vb)?;
    /// let input_ids = Tensor::new(&[[1u32, 2u32]], &device)?;
    /// let _logits = model.forward(&input_ids, 0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        if !input_ids.device().is_cuda() {
            candle_core::bail!("qwen3 forward requires CUDA input ids");
        }
        let _guard = EventTrackingGuard::disable(input_ids.device());
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let total_len = start_pos + seq_len;
        let (cos, sin) = self.rotary.get(total_len, start_pos, seq_len)?;
        let mut x = self.embed_tokens.forward(input_ids)?;
        let context = if start_pos == 0 {
            AttentionContext::Prefill
        } else {
            AttentionContext::Decode { start_pos }
        };
        for layer in &mut self.layers {
            x = layer.forward(&x, &cos, &sin, context)?;
        }
        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x.narrow(1, seq_len - 1, 1)?)
    }

    /// Clears cached key/value tensors for every decoder layer.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle_core::{DType, Device};
    /// # use candle_nn::VarBuilder;
    /// # use model::qwen3::{Config, ModelForCausalLM};
    /// # fn main() -> candle_core::Result<()> {
    /// # let device = Device::new_cuda(0)?;
    /// # let tensors = std::collections::HashMap::new();
    /// # let vb = VarBuilder::from_tensors(tensors, DType::BF16, &device);
    /// # let config: Config = serde_json::from_str(r#"{"vocab_size":8,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-6}"#).unwrap();
    /// let mut model = ModelForCausalLM::new(&config, vb)?;
    /// model.clear_kv_cache();
    /// # Ok(())
    /// # }
    /// ```
    pub fn clear_kv_cache(&mut self) {
        <Self as Cacheable>::clear_kv_cache(self)
    }
}

impl Model for ModelForCausalLM {
    fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        Self::forward(self, input_ids, start_pos)
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            model_type: "qwen3".to_string(),
            vocab_size: self.config.vocab_size,
            hidden_size: self.config.hidden_size,
            num_hidden_layers: self.config.num_hidden_layers,
        }
    }
}

impl Cacheable for ModelForCausalLM {
    fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

struct EventTrackingGuard;

impl EventTrackingGuard {
    fn disable(device: &Device) -> Self {
        if let Device::Cuda(dev) = device {
            if dev.is_event_tracking() {
                unsafe { dev.disable_event_tracking() };
            }
        }
        Self
    }
}
