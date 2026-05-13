use candle_core::Tensor;
use candle_nn::VarBuilder;
use qwen3::{Config, ModelForCausalLM};

use super::ModelConfig;
use super::models::qwen3;
use crate::error::AppError;

pub struct QwenModel {
    inner: ModelForCausalLM,
}

impl QwenModel {
    pub fn load(config: &ModelConfig, vb: VarBuilder) -> crate::error::Result<Self> {
        match config.model_type() {
            Some("qwen3") => {
                let qwen_config: Config = serde_json::from_value(config.as_value()?)?;
                let model = ModelForCausalLM::new(&qwen_config, vb)?;
                Ok(Self { inner: model })
            }
            Some(model_type) => Err(AppError::BadRequest(format!(
                "unsupported model_type '{model_type}', only qwen3 is supported"
            ))),
            None => Err(AppError::BadRequest(
                "model config is missing model_type; only qwen3 is supported".into(),
            )),
        }
    }

    pub fn forward(&mut self, input: &Tensor, pos: usize) -> crate::error::Result<Tensor> {
        Ok(self.inner.forward(input, pos)?)
    }

    pub fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }
}
