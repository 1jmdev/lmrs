use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, linear_no_bias};

use crate::model::attention::{Attention, AttentionConfig};
use crate::model::kernels;

#[derive(Debug, Clone)]
pub struct GatedSiluMlpConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

pub struct GatedSiluMlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
}

impl GatedSiluMlp {
    pub fn new(config: GatedSiluMlpConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;
        let gate_up = Tensor::cat(&[gate_proj.weight(), up_proj.weight()], 0)?;
        let down_proj = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            gate_up_proj: Linear::new(gate_up, None),
            down_proj,
            intermediate_size: config.intermediate_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?.contiguous()?;
        let activated = kernels::fused_silu_mul(&gate_up, self.intermediate_size)?;
        self.down_proj.forward(&activated)
    }
}

#[derive(Debug, Clone)]
pub struct DecoderLayerConfig {
    pub attention: AttentionConfig,
    pub mlp: GatedSiluMlpConfig,
    pub hidden_size: usize,
    pub rms_norm_eps: f64,
}

pub struct DecoderLayer {
    self_attn: Attention,
    mlp: GatedSiluMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    pub fn new(config: DecoderLayerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(config.attention, vb.pp("self_attn"))?,
            mlp: GatedSiluMlp::new(config.mlp, vb.pp("mlp"))?,
            input_layernorm: candle_nn::rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: candle_nn::rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    pub fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor, causal: bool) -> Result<Tensor> {
        let residual = x;
        let y = self.input_layernorm.forward(x)?;
        let y = self.self_attn.forward(&y, cos, sin, causal)?;
        let x = (residual + y)?;
        let residual = &x;
        let y = self.post_attention_layernorm.forward(&x)?;
        let y = self.mlp.forward(&y)?;
        residual + y
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}
