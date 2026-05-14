use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Shared hyperparameter fields present in Hugging Face causal LM configs.
///
/// This struct preserves unknown JSON fields in `raw` so the generic loader can
/// parse common metadata first and then hand the original shape to a concrete
/// model family such as Qwen3.
///
/// # Example
///
/// ```
/// use model::ModelConfig;
///
/// let json = r#"{
///   "model_type": "qwen3",
///   "hidden_size": 128,
///   "intermediate_size": 256,
///   "num_attention_heads": 4,
///   "num_hidden_layers": 2,
///   "num_key_value_heads": 2,
///   "vocab_size": 32000,
///   "rms_norm_eps": 0.000001,
///   "rope_theta": 1000000.0
/// }"#;
/// let config: ModelConfig = serde_json::from_str(json).unwrap();
/// assert_eq!(config.model_type(), Some("qwen3"));
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    /// Optional model family identifier.
    #[serde(default)]
    pub model_type: Option<String>,
    /// Hidden state width.
    pub hidden_size: usize,
    /// MLP intermediate width.
    pub intermediate_size: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of transformer blocks.
    pub num_hidden_layers: usize,
    /// Number of KV heads. Defaults are model-specific when omitted.
    pub num_key_value_heads: Option<usize>,
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// Optional RoPE base.
    pub rope_theta: Option<f32>,
    /// Optional maximum RoPE table length.
    pub max_position_embeddings: Option<usize>,
    /// Remaining model-specific fields preserved for typed deserialization.
    #[serde(flatten)]
    pub raw: serde_json::Map<String, Value>,
}

impl ModelConfig {
    /// Returns the model type as a borrowed string.
    ///
    /// # Example
    ///
    /// ```
    /// # let config: model::ModelConfig = serde_json::from_str(r#"{"model_type":"qwen3","hidden_size":1,"intermediate_size":1,"num_attention_heads":1,"num_hidden_layers":1,"vocab_size":1,"rms_norm_eps":1e-6}"#).unwrap();
    /// assert_eq!(config.model_type(), Some("qwen3"));
    /// ```
    pub fn model_type(&self) -> Option<&str> {
        self.model_type.as_deref()
    }

    /// Converts the config back to JSON for model-specific deserialization.
    ///
    /// # Example
    ///
    /// ```
    /// # let config: model::ModelConfig = serde_json::from_str(r#"{"hidden_size":1,"intermediate_size":1,"num_attention_heads":1,"num_hidden_layers":1,"vocab_size":1,"rms_norm_eps":1e-6}"#).unwrap();
    /// let value = config.as_value().unwrap();
    /// assert_eq!(value["hidden_size"], 1);
    /// ```
    pub fn as_value(&self) -> anyhow::Result<Value> {
        Ok(serde_json::to_value(self)?)
    }
}
