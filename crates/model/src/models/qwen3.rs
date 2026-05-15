use serde::Deserialize;

use crate::layers::{AttentionConfig, DecoderLayerConfig, GatedSiluMlpConfig};
use crate::models::base::{ModelForCausalLM as Base, TransformerConfig};

fn default_true() -> bool { true }
fn default_rope_theta() -> f64 { 1_000_000.0 }
fn default_max_position_embeddings() -> usize { 32768 }

/// Qwen3 architecture configuration deserialized from `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
}

impl Config {
    /// Returns the per-head feature width, defaulting to `hidden_size / num_attention_heads`.
    ///
    /// ```
    /// # use model::qwen3::Config;
    /// # let c: Config = serde_json::from_str(r#"{"vocab_size":8,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-6}"#).unwrap();
    /// assert_eq!(c.head_dim(), 16);
    /// ```
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

impl TransformerConfig for Config {
    const MODEL_TYPE: &'static str = "qwen3";
    fn vocab_size(&self) -> usize { self.vocab_size }
    fn hidden_size(&self) -> usize { self.hidden_size }
    fn num_hidden_layers(&self) -> usize { self.num_hidden_layers }
    fn rms_norm_eps(&self) -> f64 { self.rms_norm_eps }
    fn max_position_embeddings(&self) -> usize { self.max_position_embeddings }
    fn rope_theta(&self) -> f64 { self.rope_theta }
    fn tie_word_embeddings(&self) -> bool { self.tie_word_embeddings }
    fn layer_config(&self) -> DecoderLayerConfig {
        DecoderLayerConfig {
            attention: AttentionConfig {
                hidden_size: self.hidden_size,
                num_heads: self.num_attention_heads,
                num_kv_heads: self.num_key_value_heads,
                head_dim: self.head_dim(),
                attention_bias: self.attention_bias,
                qk_norm_eps: self.use_qk_norm.then_some(self.rms_norm_eps),
            },
            mlp: GatedSiluMlpConfig { hidden_size: self.hidden_size, intermediate_size: self.intermediate_size },
            hidden_size: self.hidden_size,
            rms_norm_eps: self.rms_norm_eps,
        }
    }
}

/// Qwen3 causal language model — generic transformer wired to Qwen3 config.
pub type ModelForCausalLM = Base<Config>;
