use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::{qwen3, qwen3_moe};

use super::ModelConfig;
use crate::error::AppError;

enum InnerQwenModel {
    Qwen3(qwen3::ModelForCausalLM),
    Qwen3Moe(qwen3_moe::ModelForCausalLM),
}

pub struct QwenModel {
    inner: InnerQwenModel,
}

impl QwenModel {
    pub fn load(config: &ModelConfig, vb: VarBuilder) -> crate::error::Result<Self> {
        match config.model_type() {
            Some("qwen3") => {
                let qwen_config: qwen3::Config = serde_json::from_value(config.as_value()?)?;
                let model = qwen3::ModelForCausalLM::new(&qwen_config, vb)?;
                Ok(Self {
                    inner: InnerQwenModel::Qwen3(model),
                })
            }
            Some("qwen3_moe") | Some("qwen3moe") => {
                let qwen_config: qwen3_moe::Config = serde_json::from_value(config.as_value()?)?;
                let model = qwen3_moe::ModelForCausalLM::new(&qwen_config, vb)?;
                Ok(Self {
                    inner: InnerQwenModel::Qwen3Moe(model),
                })
            }
            Some(model_type) => Err(AppError::BadRequest(format!(
                "unsupported model_type '{model_type}', only qwen3/qwen3_moe are supported"
            ))),
            None => Err(AppError::BadRequest(
                "model config is missing model_type; only qwen3/qwen3_moe are supported".into(),
            )),
        }
    }

    pub fn forward(&mut self, input: &Tensor, pos: usize) -> crate::error::Result<Tensor> {
        match &mut self.inner {
            InnerQwenModel::Qwen3(model) => Ok(model.forward(input, pos)?),
            InnerQwenModel::Qwen3Moe(model) => Ok(model.forward(input, pos)?),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match &mut self.inner {
            InnerQwenModel::Qwen3(model) => model.clear_kv_cache(),
            InnerQwenModel::Qwen3Moe(model) => model.clear_kv_cache(),
        }
    }
}
