use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::{llama::{Cache, Config, Llama}, qwen3};

use super::ModelConfig;

enum InnerModel {
    Llama { model: Llama, cache: Cache },
    Qwen3(qwen3::ModelForCausalLM),
}

pub struct LlamaModel {
    inner: InnerModel,
}

impl LlamaModel {
    pub fn load(config: &ModelConfig, vb: VarBuilder) -> crate::error::Result<Self> {
        if config.model_type() == Some("qwen3") {
            let qwen_config: qwen3::Config = serde_json::from_value(config.as_value()?)?;
            let model = qwen3::ModelForCausalLM::new(&qwen_config, vb)?;
            return Ok(Self { inner: InnerModel::Qwen3(model) });
        }

        let config = Config {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads.unwrap_or(config.num_attention_heads),
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta.unwrap_or(10_000.0),
            max_position_embeddings: config.max_position_embeddings.unwrap_or(4096),
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            tie_word_embeddings: false,
            use_flash_attn: false,
        };
        let cache = Cache::new(false, candle_core::DType::F16, &config, vb.device())?;
        let model = Llama::load(vb, &config)?;
        Ok(Self { inner: InnerModel::Llama { model, cache } })
    }

    pub fn forward(&mut self, input: &Tensor, pos: usize) -> crate::error::Result<Tensor> {
        match &mut self.inner {
            InnerModel::Llama { model, cache } => Ok(model.forward(input, pos, cache)?),
            InnerModel::Qwen3(model) => Ok(model.forward(input, pos)?),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match &mut self.inner {
            InnerModel::Llama { .. } => {}
            InnerModel::Qwen3(model) => model.clear_kv_cache(),
        }
    }
}
