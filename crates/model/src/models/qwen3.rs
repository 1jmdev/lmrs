use ops::{narrow_dim1, AttentionContext, LmHead, RmsNormConfig, RmsNormOp, RotaryEmbedding, TokenEmbedding};
use serde::Deserialize;
use tensor::{DType, Result, Tensor, TensorError};

use crate::WeightBuilder;
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
/// # Example
///
/// ```
/// use model::qwen3::Config;
///
/// let json = r#"{"vocab_size":32000,"hidden_size":128,"intermediate_size":256,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":0.000001}"#;
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

/// Qwen3 causal language model backed by local CUDA tensor and kernel crates.
///
/// # Example
///
/// ```no_run
/// use std::collections::HashMap;
/// use model::qwen3::{Config, ModelForCausalLM};
/// use model::WeightBuilder;
/// use runtime::CudaContext;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let weights = WeightBuilder::new(context, HashMap::new());
/// let config: Config = serde_json::from_str(r#"{"vocab_size":8,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-6}"#).unwrap();
/// assert!(ModelForCausalLM::new(&config, weights).is_err());
/// # Ok(())
/// # }
/// ```
pub struct ModelForCausalLM {
    config: Config,
    embed_tokens: TokenEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNormOp,
    lm_head: LmHead,
    rotary: RotaryEmbedding,
}

impl ModelForCausalLM {
    /// Builds a Qwen3 model from CUDA BF16 checkpoint variables.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::collections::HashMap;
    /// # use model::qwen3::{Config, ModelForCausalLM};
    /// # use model::WeightBuilder;
    /// # use runtime::CudaContext;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let context = CudaContext::new(0)?;
    /// let weights = WeightBuilder::new(context, HashMap::new());
    /// let config: Config = serde_json::from_str(r#"{"vocab_size":8,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-6}"#).unwrap();
    /// assert!(ModelForCausalLM::new(&config, weights).is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: &Config, weights: WeightBuilder) -> Result<Self> {
        let model_weights = weights.pp("model");
        let embed_tokens = TokenEmbedding::new(
            config.vocab_size,
            config.hidden_size,
            model_weights.get("embed_tokens.weight")?,
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let layers_weights = model_weights.pp("layers");
        let layer_config = config.decoder_layer_config();
        for idx in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(layer_config.clone(), layers_weights.pp(idx))?);
        }
        let norm = RmsNormOp::new(
            RmsNormConfig {
                hidden_size: config.hidden_size,
                eps: config.rms_norm_eps,
            },
            model_weights.get("norm.weight")?,
        )?;
        let lm_head = if config.tie_word_embeddings {
            LmHead::tied(embed_tokens.embeddings())
        } else {
            LmHead::new(
                config.hidden_size,
                config.vocab_size,
                weights.get("lm_head.weight")?,
            )?
        };
        let rotary = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            DType::BF16,
            weights.context(),
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
    /// # use model::qwen3::ModelForCausalLM;
    /// # use tensor::Tensor;
    /// # fn run(model: &mut ModelForCausalLM, input_ids: &Tensor) -> tensor::Result<Tensor> {
    /// let logits = model.forward(input_ids, 0)?;
    /// # Ok(logits)
    /// # }
    /// ```
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        if input_ids.dtype() != DType::I32 {
            return Err(TensorError::DTypeMismatch {
                expected: DType::I32.name(),
                actual: input_ids.dtype().name(),
            });
        }
        let dims = input_ids.shape().dims();
        if dims.len() != 2 {
            return Err(TensorError::ShapeMismatch(format!(
                "qwen3 input ids must be rank 2, got rank {}",
                dims.len()
            )));
        }
        let seq_len = dims[1];
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
        self.lm_head.forward(&narrow_dim1(&x, seq_len - 1, 1)?)
    }

    /// Clears cached key/value tensors for every decoder layer.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use model::qwen3::ModelForCausalLM;
    /// # fn clear(model: &mut ModelForCausalLM) {
    /// model.clear_kv_cache();
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
