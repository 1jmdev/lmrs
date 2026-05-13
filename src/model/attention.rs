use candle_core::{D, Result, Tensor};
use candle_nn::rotary_emb::rope;
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, linear, linear_no_bias};

#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub qk_norm_eps: Option<f64>,
}

pub struct Attention {
    o_proj: Linear,
    qkv_proj: Linear,
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
    pub fn new(config: AttentionConfig, vb: VarBuilder) -> Result<Self> {
        let make_linear = |in_d, out_d, name: &str| {
            if config.attention_bias {
                linear(in_d, out_d, vb.pp(name))
            } else {
                linear_no_bias(in_d, out_d, vb.pp(name))
            }
        };
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let q_proj = make_linear(config.hidden_size, q_dim, "q_proj")?;
        let k_proj = make_linear(config.hidden_size, kv_dim, "k_proj")?;
        let v_proj = make_linear(config.hidden_size, kv_dim, "v_proj")?;
        let o_proj = make_linear(q_dim, config.hidden_size, "o_proj")?;
        let qkv_w = Tensor::cat(&[q_proj.weight(), k_proj.weight(), v_proj.weight()], 0)?;
        let qkv_b = match (q_proj.bias(), k_proj.bias(), v_proj.bias()) {
            (Some(q), Some(k), Some(v)) => Some(Tensor::cat(&[q, k, v], 0)?),
            _ => None,
        };
        let (q_norm, k_norm) = match config.qk_norm_eps {
            Some(eps) => (
                Some(candle_nn::rms_norm(config.head_dim, eps, vb.pp("q_norm"))?),
                Some(candle_nn::rms_norm(config.head_dim, eps, vb.pp("k_norm"))?),
            ),
            None => (None, None),
        };
        Ok(Self {
            o_proj,
            qkv_proj: Linear::new(qkv_w, qkv_b),
            q_norm,
            k_norm,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
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

    pub fn forward(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        causal: bool,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let qkv = self.qkv_proj.forward(x)?;
        let q = qkv.narrow(D::Minus1, 0, self.q_dim)?;
        let k = qkv.narrow(D::Minus1, self.q_dim, self.kv_dim)?;
        let v = qkv.narrow(D::Minus1, self.q_dim + self.kv_dim, self.kv_dim)?;
        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k)?,
            None => k,
        };
        let q = rope(&q.contiguous()?, cos, sin)?;
        let k = rope(&k.contiguous()?, cos, sin)?;
        let (k, v) = self.update_kv_cache(k, v)?;
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep > 1 && seq_len == 1 {
            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let q = (q.reshape((b_sz, self.num_kv_heads, n_rep, self.head_dim))? * scale)?;
            let attn = q.matmul(&k.transpose(2, 3)?)?;
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
        let attn = if causal {
            let total_len = k.dim(2)?;
            let mask = crate::model::kernels::causal_mask(
                seq_len,
                total_len,
                total_len - seq_len,
                x.device(),
            )?;
            attn.broadcast_add(&mask)?
        } else {
            attn
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let y = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, seq_len, ()))?;
        self.o_proj.forward(&y)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
        self.cache_seq_len = 0;
    }
}
