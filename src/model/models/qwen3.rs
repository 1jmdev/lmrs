use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder, linear_no_bias};
use serde::Deserialize;

use crate::model::attention::AttentionConfig;
use crate::model::block::{DecoderLayer, DecoderLayerConfig, GatedSiluMlpConfig};
use crate::model::rotary::RotaryEmbedding;

fn default_true() -> bool {
    true
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

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

fn default_max_position_embeddings() -> usize {
    32768
}

impl Config {
    fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    fn decoder_layer_config(&self) -> DecoderLayerConfig {
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

pub struct ModelForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary: RotaryEmbedding,
}

impl ModelForCausalLM {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        if !vb.device().is_cuda() {
            candle_core::bail!("qwen3 is CUDA-only");
        }
        if vb.dtype() != DType::BF16 {
            candle_core::bail!("qwen3 is BF16-only");
        }
        let model_vb = vb.pp("model");
        let embed_tokens = candle_nn::embedding(
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
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };
        let rotary = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            DType::BF16,
            vb.device(),
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        if !input_ids.device().is_cuda() {
            candle_core::bail!("qwen3 forward requires CUDA input ids");
        }
        let _guard = EventTrackingGuard::disable(input_ids.device());

        let (_b_sz, seq_len) = input_ids.dims2()?;
        let total_len = start_pos + seq_len;
        let (cos, sin) = self.rotary.get(total_len, start_pos, seq_len)?;
        let mut x = self.embed_tokens.forward(input_ids)?;
        let causal = seq_len > 1;
        for layer in &mut self.layers {
            x = layer.forward(&x, &cos, &sin, causal)?;
        }
        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x.narrow(1, seq_len - 1, 1)?)
    }

    pub fn clear_kv_cache(&mut self) {
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
