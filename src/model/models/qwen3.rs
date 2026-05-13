use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::rotary_emb::rope;
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder, linear, linear_no_bias};
use serde::Deserialize;

use crate::model::kernels;

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
}

struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(config: &Config, device: &Device) -> Result<Self> {
        let dim = config.head_dim();
        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| (1.0 / config.rope_theta.powf(i as f64 / dim as f64)) as f32)
            .collect();
        let inv = Tensor::new(inv.as_slice(), device)?;
        let positions: Vec<f32> = (0..config.max_position_embeddings)
            .map(|i| i as f32)
            .collect();
        let positions = Tensor::new(positions.as_slice(), device)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv.unsqueeze(0)?)?;
        Ok(Self {
            cos: freqs.cos()?.contiguous()?,
            sin: freqs.sin()?.contiguous()?,
        })
    }

    fn get(
        &self,
        total_len: usize,
        start_pos: usize,
        seq_len: usize,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, start_pos, seq_len)?.to_dtype(dtype)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?.to_dtype(dtype)?;
        if total_len > self.cos.dim(0)? {
            candle_core::bail!("sequence length {total_len} exceeds max_position_embeddings");
        }
        Ok((cos, sin))
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    qkv_proj: Option<Linear>,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
    cache_seq_len: usize,
}

impl Attention {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let make_linear = |in_d, out_d, name: &str| {
            if config.attention_bias {
                linear(in_d, out_d, vb.pp(name))
            } else {
                linear_no_bias(in_d, out_d, vb.pp(name))
            }
        };
        let q_proj = make_linear(config.hidden_size, num_heads * head_dim, "q_proj")?;
        let k_proj = make_linear(config.hidden_size, num_kv_heads * head_dim, "k_proj")?;
        let v_proj = make_linear(config.hidden_size, num_kv_heads * head_dim, "v_proj")?;
        let o_proj = make_linear(num_heads * head_dim, config.hidden_size, "o_proj")?;
        let qkv_w = Tensor::cat(&[q_proj.weight(), k_proj.weight(), v_proj.weight()], 0)?;
        let qkv_b = match (q_proj.bias(), k_proj.bias(), v_proj.bias()) {
            (Some(q), Some(k), Some(v)) => Some(Tensor::cat(&[q, k, v], 0)?),
            _ => None,
        };
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let (q_norm, k_norm) = if config.use_qk_norm {
            (
                Some(candle_nn::rms_norm(
                    head_dim,
                    config.rms_norm_eps,
                    vb.pp("q_norm"),
                )?),
                Some(candle_nn::rms_norm(
                    head_dim,
                    config.rms_norm_eps,
                    vb.pp("k_norm"),
                )?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            qkv_proj: Some(Linear::new(qkv_w, qkv_b)),
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            q_dim,
            kv_dim,
            kv_cache: None,
            cache_seq_len: 0,
        })
    }

    fn update_kv_cache(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let new_seq_len = k.dim(2)?;
        match self.kv_cache.take() {
            Some((buf_k, buf_v)) => {
                let new_total = self.cache_seq_len + new_seq_len;
                if new_total <= buf_k.dim(2)? {
                    buf_k.slice_set(&k, 2, self.cache_seq_len)?;
                    buf_v.slice_set(&v, 2, self.cache_seq_len)?;
                    let k_view = buf_k.narrow(2, 0, new_total)?;
                    let v_view = buf_v.narrow(2, 0, new_total)?;
                    self.kv_cache = Some((buf_k, buf_v));
                    self.cache_seq_len = new_total;
                    Ok((k_view, v_view))
                } else {
                    let cur_k = buf_k.narrow(2, 0, self.cache_seq_len)?;
                    let cur_v = buf_v.narrow(2, 0, self.cache_seq_len)?;
                    let full_k = Tensor::cat(&[&cur_k, &k], 2)?;
                    let full_v = Tensor::cat(&[&cur_v, &v], 2)?;
                    let (b, h, s, d) = full_k.dims4()?;
                    let buf_k = Tensor::zeros((b, h, s + 256, d), k.dtype(), k.device())?;
                    let buf_v = Tensor::zeros((b, h, s + 256, d), v.dtype(), v.device())?;
                    buf_k.slice_set(&full_k, 2, 0)?;
                    buf_v.slice_set(&full_v, 2, 0)?;
                    self.kv_cache = Some((buf_k, buf_v));
                    self.cache_seq_len = s;
                    Ok((full_k, full_v))
                }
            }
            None => {
                let (b, h, s, d) = k.dims4()?;
                let buf_k = Tensor::zeros((b, h, s + 256, d), k.dtype(), k.device())?;
                let buf_v = Tensor::zeros((b, h, s + 256, d), v.dtype(), v.device())?;
                buf_k.slice_set(&k, 2, 0)?;
                buf_v.slice_set(&v, 2, 0)?;
                self.kv_cache = Some((buf_k, buf_v));
                self.cache_seq_len = s;
                Ok((k, v))
            }
        }
    }

    fn forward(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let (q, k, v) = if let Some(qkv_proj) = &self.qkv_proj {
            let qkv = qkv_proj.forward(x)?;
            (
                qkv.narrow(D::Minus1, 0, self.q_dim)?,
                qkv.narrow(D::Minus1, self.q_dim, self.kv_dim)?,
                qkv.narrow(D::Minus1, self.q_dim + self.kv_dim, self.kv_dim)?,
            )
        } else {
            (
                self.q_proj.forward(x)?,
                self.k_proj.forward(x)?,
                self.v_proj.forward(x)?,
            )
        };
        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = if let Some(norm) = &self.q_norm {
            norm.forward(&q)?
        } else {
            q
        };
        let k = if let Some(norm) = &self.k_norm {
            norm.forward(&k)?
        } else {
            k
        };
        let q = rope(&q.contiguous()?, cos, sin)?;
        let k = rope(&k.contiguous()?, cos, sin)?;
        let (k, v) = self.update_kv_cache(k, v)?;
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep > 1 && seq_len == 1 {
            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let q = (q.reshape((b_sz, self.num_kv_heads, n_rep, self.head_dim))? * scale)?;
            let attn = q.matmul(&k.transpose(2, 3)?)?;
            let attn = match mask {
                Some(mask) => attn.broadcast_add(mask)?,
                None => attn,
            };
            let attn = candle_nn::ops::softmax_last_dim(&attn)?;
            let y = attn
                .matmul(&v)?
                .reshape((b_sz, self.num_heads, self.head_dim))?
                .reshape((b_sz, 1, self.num_heads * self.head_dim))?;
            return self.o_proj.forward(&y);
        }
        let k = if n_rep > 1 {
            let (b, h, s, d) = k.dims4()?;
            k.unsqueeze(2)?
                .expand((b, h, n_rep, s, d))?
                .reshape((b, h * n_rep, s, d))?
        } else {
            k
        };
        let v = if n_rep > 1 {
            let (b, h, s, d) = v.dims4()?;
            v.unsqueeze(2)?
                .expand((b, h, n_rep, s, d))?
                .reshape((b, h * n_rep, s, d))?
        } else {
            v
        };
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn = match mask {
            Some(mask) => attn.broadcast_add(mask)?,
            None => attn,
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let y = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, seq_len, ()))?;
        self.o_proj.forward(&y)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
        self.cache_seq_len = 0;
    }
}

struct Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
}

impl Mlp {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
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

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?.contiguous()?;
        let activated = kernels::fused_silu_mul(&gate_up, self.intermediate_size)?;
        self.down_proj.forward(&activated)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(config, vb.pp("self_attn"))?,
            mlp: Mlp::new(config, vb.pp("mlp"))?,
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

    fn forward(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = x;
        let y = self.input_layernorm.forward(x)?;
        let y = self.self_attn.forward(&y, cos, sin, mask)?;
        let x = (residual + y)?;
        let residual = &x;
        let y = self.post_attention_layernorm.forward(&x)?;
        let y = self.mlp.forward(&y)?;
        residual + y
    }
}

pub struct ModelForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary: RotaryEmbedding,
    dtype: DType,
}

impl ModelForCausalLM {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let model_vb = vb.pp("model");
        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            model_vb.pp("embed_tokens"),
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let layers_vb = model_vb.pp("layers");
        for idx in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(config, layers_vb.pp(idx))?);
        }
        let norm =
            candle_nn::rms_norm(config.hidden_size, config.rms_norm_eps, model_vb.pp("norm"))?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };
        let rotary = RotaryEmbedding::new(config, vb.device())?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
            dtype,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        let _guard = EventTrackingGuard::disable(input_ids.device());

        let (_b_sz, seq_len) = input_ids.dims2()?;
        let total_len = start_pos + seq_len;
        let (cos, sin) = self.rotary.get(total_len, start_pos, seq_len, self.dtype)?;
        let mut x = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;
        let mask = if seq_len > 1 {
            let mut data = vec![0f32; seq_len * total_len];
            for i in 0..seq_len {
                for j in 0..total_len {
                    if j > start_pos + i {
                        data[i * total_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            Some(
                Tensor::from_vec(data, (seq_len, total_len), input_ids.device())?
                    .to_dtype(self.dtype)?
                    .unsqueeze(0)?
                    .unsqueeze(0)?,
            )
        } else {
            None
        };
        for layer in &mut self.layers {
            x = layer.forward(&x, &cos, &sin, mask.as_ref())?;
        }
        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x.narrow(1, seq_len - 1, 1)?)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.self_attn.clear_kv_cache();
        }
    }
}

#[cfg(feature = "cuda")]
struct EventTrackingGuard;

#[cfg(feature = "cuda")]
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
