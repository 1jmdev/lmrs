use candle_core::{D, Result, Tensor};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, linear, linear_no_bias};
use ops::{AttentionContext, KvCache, apply_rotary, attention_context};

/// Multi-head self-attention configuration.
///
/// # Example
///
/// ```
/// use model::AttentionConfig;
///
/// let config = AttentionConfig {
///     hidden_size: 128,
///     num_heads: 4,
///     num_kv_heads: 2,
///     head_dim: 32,
///     attention_bias: false,
///     qk_norm_eps: Some(1e-6),
/// };
/// assert_eq!(config.num_heads / config.num_kv_heads, 2);
/// ```
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Input and output hidden-state width.
    pub hidden_size: usize,
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of key/value heads.
    pub num_kv_heads: usize,
    /// Per-head feature width.
    pub head_dim: usize,
    /// Whether Q/K/V/O projections include bias.
    pub attention_bias: bool,
    /// Optional Q/K RMSNorm epsilon.
    pub qk_norm_eps: Option<f64>,
}

/// Fused-QKV self-attention layer with an owned KV cache.
///
/// # Example
///
/// ```no_run
/// use candle_core::{DType, Device, Tensor};
/// use candle_nn::VarBuilder;
/// use model::{Attention, AttentionConfig};
/// use ops::AttentionContext;
///
/// # fn main() -> candle_core::Result<()> {
/// let device = Device::new_cuda(0)?;
/// let tensors = std::collections::HashMap::new();
/// let vb = VarBuilder::from_tensors(tensors, DType::BF16, &device);
/// let mut attention = Attention::new(AttentionConfig { hidden_size: 128, num_heads: 4, num_kv_heads: 2, head_dim: 32, attention_bias: false, qk_norm_eps: Some(1e-6) }, vb)?;
/// let x = Tensor::zeros((1, 1, 128), DType::BF16, &device)?;
/// let cos = Tensor::zeros((1, 16), DType::BF16, &device)?;
/// let sin = Tensor::zeros((1, 16), DType::BF16, &device)?;
/// let _y = attention.forward(&x, &cos, &sin, AttentionContext::Prefill)?;
/// # Ok(())
/// # }
/// ```
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
    cache: KvCache,
}

impl Attention {
    /// Builds attention from Q, K, V, O checkpoint weights.
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
            cache: KvCache::default(),
        })
    }

    /// Runs attention for prefill or decode context.
    pub fn forward(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        context: AttentionContext,
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
        let q = apply_rotary(&q, cos, sin)?;
        let k = apply_rotary(&k, cos, sin)?;
        let y = attention_context(
            &q,
            k,
            v,
            &mut self.cache,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            context,
        )?;
        let y = y
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, seq_len, ()))?;
        self.o_proj.forward(&y)
    }

    /// Clears cached key/value states.
    pub fn clear_kv_cache(&mut self) {
        self.cache.clear();
    }
}
