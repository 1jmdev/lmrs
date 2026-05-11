use std::sync::Mutex;

use candle_core::{DType, Device, IndexOp, Tensor};

use crate::{
    config::AppConfig,
    error::Result,
    model::{ModelConfig, QwenModel, load_model},
    sampling::Sampler,
    server::types::{ChatMessage, GenerateParams},
    tokenizer::TokenizerWrapper,
};

pub struct InferenceEngine {
    model_id: String,
    model: Mutex<QwenModel>,
    tokenizer: TokenizerWrapper,
    device: Device,
}

impl InferenceEngine {
    pub fn load(config: &AppConfig) -> Result<Self> {
        let device = match config.device.as_str() {
            "cpu" => Device::Cpu,
            #[cfg(feature = "cuda")]
            "cuda" => Device::new_cuda(0)?,
            #[cfg(feature = "metal")]
            "metal" => Device::new_metal(0)?,
            "auto" => default_device()?,
            other => {
                return Err(crate::error::AppError::BadRequest(format!(
                    "unsupported device: {other}"
                )));
            }
        };
        let loaded = load_model(&config.model, config.revision.as_deref(), &device)?;
        let tokenizer_path = config
            .tokenizer
            .clone()
            .unwrap_or_else(|| loaded.tokenizer_path.clone());
        let tokenizer = TokenizerWrapper::from_file(
            tokenizer_path,
            loaded.tokenizer_config_path,
            loaded.chat_template_path,
        )?;
        let model_config = ModelConfig::from_file(&loaded.config_path)?;
        let model = QwenModel::load(&model_config, loaded.var_builder)?;

        Ok(Self {
            model_id: config.model.clone(),
            model: Mutex::new(model),
            tokenizer,
            device,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self.tokenizer.encode(text)?.len())
    }

    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        self.tokenizer.apply_chat_template(messages)
    }

    pub fn generate(&self, prompt: &str, params: GenerateParams) -> Result<String> {
        Ok(self.generate_tokens(prompt, params)?.concat())
    }

    pub fn generate_tokens(&self, prompt: &str, params: GenerateParams) -> Result<Vec<String>> {
        let token_ids = self.tokenizer.encode(prompt)?;
        let prompt_len = token_ids.len();
        let mut output = Vec::with_capacity(params.max_tokens);
        let mut sampler = Sampler::new(params.temperature, params.top_p, params.top_k);
        let mut model = self.model.lock().expect("model mutex poisoned");
        let mut next_input = 0;

        model.clear_kv_cache();

        for index in 0..params.max_tokens {
            let input = if index == 0 {
                Tensor::new(token_ids.as_slice(), &self.device)?
            } else {
                Tensor::new(&[next_input], &self.device)?
            };
            let input = input.unsqueeze(0)?;
            let logits = model.forward(
                &input,
                if index == 0 {
                    0
                } else {
                    prompt_len + index - 1
                },
            )?;
            let logits = last_token_logits(&logits)?.to_dtype(DType::F32)?;
            let next = sampler.sample(&logits)?;
            next_input = next;
            if self.tokenizer.is_eos(next) {
                break;
            }
            let text = self.tokenizer.decode(&[next])?;
            if params.stop.iter().any(|stop| text.contains(stop)) {
                break;
            }
            output.push(text);
        }
        Ok(output)
    }
}

fn last_token_logits(logits: &Tensor) -> candle_core::Result<Tensor> {
    match logits.dims().len() {
        1 => Ok(logits.clone()),
        2 => logits.i(0),
        3 => logits.i((0, logits.dim(1)? - 1)),
        _ => logits.flatten_all(),
    }
}

fn default_device() -> candle_core::Result<Device> {
    #[cfg(feature = "cuda")]
    {
        return Device::new_cuda(0);
    }
    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        return Device::new_metal(0);
    }
    #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
    {
        Ok(Device::Cpu)
    }
}
