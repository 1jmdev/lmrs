use std::{fs, path::Path};

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: Option<f32>,
    pub max_position_embeddings: Option<usize>,
    #[serde(flatten)]
    pub raw: serde_json::Map<String, Value>,
}

impl ModelConfig {
    pub fn from_file(path: &Path) -> crate::error::Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }

    pub fn model_type(&self) -> Option<&str> {
        self.model_type.as_deref()
    }

    pub fn as_value(&self) -> crate::error::Result<Value> {
        Ok(serde_json::to_value(self)?)
    }
}
