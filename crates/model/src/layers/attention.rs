use cache::KvCache;
use ops::{apply_rotary, attention_context, reshape, transpose_1_2, AttentionContext, LinearConfig, LinearOp, RmsNormConfig, RmsNormOp};
use tensor::{Result, Tensor, TensorError};

use crate::WeightBuilder;

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

/// CUDA-only self-attention layer with an owned KV cache.
///
/// # Example
///
/// ```no_run
/// use std::collections::HashMap;
/// use model::{Attention, AttentionConfig, WeightBuilder};
/// use runtime::CudaContext;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let weights = WeightBuilder::new(context, HashMap::new());
/// let config = AttentionConfig { hidden_size: 128, num_heads: 4, num_kv_heads: 2, head_dim: 32, attention_bias: false, qk_norm_eps: Some(1e-6) };
/// assert!(Attention::new(config, weights).is_err());
/// # Ok(())
/// # }
/// ```
pub struct Attention {
    q_proj: LinearOp,
    k_proj: LinearOp,
    v_proj: LinearOp,
    o_proj: LinearOp,
    q_norm: Option<RmsNormOp>,
    k_norm: Option<RmsNormOp>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cache: KvCache,
}

impl Attention {
    /// Builds attention from Q, K, V, O checkpoint weights.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::collections::HashMap;
    /// use model::{Attention, AttentionConfig, WeightBuilder};
    /// use runtime::CudaContext;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let context = CudaContext::new(0)?;
    /// let weights = WeightBuilder::new(context, HashMap::new());
    /// let config = AttentionConfig { hidden_size: 128, num_heads: 4, num_kv_heads: 2, head_dim: 32, attention_bias: false, qk_norm_eps: None };
    /// assert!(Attention::new(config, weights).is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: AttentionConfig, weights: WeightBuilder) -> Result<Self> {
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let make_linear = |in_features: usize, out_features: usize, name: &str| {
            let bias = if config.attention_bias {
                Some(weights.get(&format!("{name}.bias"))?)
            } else {
                None
            };
            LinearOp::new(
                LinearConfig {
                    in_features,
                    out_features,
                    bias: config.attention_bias,
                },
                weights.get(&format!("{name}.weight"))?,
                bias,
            )
        };
        let q_norm = match config.qk_norm_eps {
            Some(eps) => Some(RmsNormOp::new(
                RmsNormConfig {
                    hidden_size: config.head_dim,
                    eps,
                },
                weights.get("q_norm.weight")?,
            )?),
            None => None,
        };
        let k_norm = match config.qk_norm_eps {
            Some(eps) => Some(RmsNormOp::new(
                RmsNormConfig {
                    hidden_size: config.head_dim,
                    eps,
                },
                weights.get("k_norm.weight")?,
            )?),
            None => None,
        };
        Ok(Self {
            q_proj: make_linear(config.hidden_size, q_dim, "q_proj")?,
            k_proj: make_linear(config.hidden_size, kv_dim, "k_proj")?,
            v_proj: make_linear(config.hidden_size, kv_dim, "v_proj")?,
            o_proj: make_linear(q_dim, config.hidden_size, "o_proj")?,
            q_norm,
            k_norm,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            cache: KvCache::default(),
        })
    }

    /// Runs attention for prefill or decode context.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use model::Attention;
    /// # use ops::AttentionContext;
    /// # use tensor::Tensor;
    /// # fn run(attention: &mut Attention, x: &Tensor, cos: &Tensor, sin: &Tensor) -> tensor::Result<Tensor> {
    /// let y = attention.forward(x, cos, sin, AttentionContext::Prefill)?;
    /// # Ok(y)
    /// # }
    /// ```
    pub fn forward(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        context: AttentionContext,
    ) -> Result<Tensor> {
        let [batch, seq_len, _hidden] = dims3(x, "attention input")?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let q = reshape(&q, [batch, seq_len, self.num_heads, self.head_dim])?;
        let k = reshape(&k, [batch, seq_len, self.num_kv_heads, self.head_dim])?;
        let v = reshape(&v, [batch, seq_len, self.num_kv_heads, self.head_dim])?;
        let q = transpose_1_2(&q)?;
        let k = transpose_1_2(&k)?;
        let v = transpose_1_2(&v)?;
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
        let y = transpose_1_2(&y)?;
        let y = reshape(&y, [batch, seq_len, self.num_heads * self.head_dim])?;
        self.o_proj.forward(&y)
    }

    /// Clears cached key/value states.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use model::Attention;
    /// # fn clear(attention: &mut Attention) {
    /// attention.clear_kv_cache();
    /// # }
    /// ```
    pub fn clear_kv_cache(&mut self) {
        self.cache.clear();
    }
}

fn dims3(tensor: &Tensor, name: &str) -> Result<[usize; 3]> {
    let dims = tensor.shape().dims();
    if dims.len() != 3 {
        return Err(TensorError::ShapeMismatch(format!(
            "{name} must be rank 3, got rank {}",
            dims.len()
        )));
    }
    Ok([dims[0], dims[1], dims[2]])
}
