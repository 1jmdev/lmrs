use cache::KvCache;
use tensor::{Result, Tensor, TensorError};

use crate::attention::{repeat_kv, sdpa, SdpaConfig};

/// Attention execution mode with an explicit prefill/decode split.
///
/// `Prefill` clears the layer KV cache and writes the full prompt span. `Decode`
/// appends new token states and requires `start_pos` to match the current cache
/// length so callers cannot silently attend over a shifted cache.
///
/// # Example
///
/// ```
/// use ops::AttentionContext;
///
/// let context = AttentionContext::Decode { start_pos: 4 };
/// assert_eq!(context, AttentionContext::Decode { start_pos: 4 });
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionContext {
    /// Processes a prompt span and applies a causal mask across all prompt tokens.
    ///
    /// This mode resets the cache before appending the supplied prompt keys and
    /// values, so it is the correct mode for the first forward pass of a request.
    Prefill,
    /// Processes one or more newly decoded tokens against an existing KV cache.
    ///
    /// `start_pos` must equal the current KV cache length. The attention kernel
    /// then uses it as the key-cache offset for causal indexing.
    Decode { start_pos: usize },
}

/// Runs attention with explicit prefill/decode behavior.
///
/// Queries are shaped `[batch, heads, seq, head_dim]`; keys and values are shaped
/// `[batch, kv_heads, seq, head_dim]`. If grouped-query attention is used, keys
/// and values are repeated with the CUDA `repeat_kv` kernel before scaled-dot
/// product attention runs.
///
/// # Example
///
/// ```no_run
/// use cache::KvCache;
/// use ops::{AttentionContext, attention_context};
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let shape = Shape::new([1, 1, 1, 2]).unwrap();
/// let q = copy_h2d(&context, shape.clone(), DType::BF16, &[0u16; 2])?;
/// let k = copy_h2d(&context, shape.clone(), DType::BF16, &[0u16; 2])?;
/// let v = copy_h2d(&context, shape, DType::BF16, &[0u16; 2])?;
/// let mut cache = KvCache::default();
///
/// let output = attention_context(&q, k, v, &mut cache, 1, 1, 2, AttentionContext::Prefill)?;
/// assert_eq!(output.shape().dims(), &[1, 1, 1, 2]);
/// # Ok(())
/// # }
/// ```
pub fn attention_context(
    q: &Tensor,
    k: Tensor,
    v: Tensor,
    cache: &mut KvCache,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    context: AttentionContext,
) -> Result<Tensor> {
    let seq_len = dim(q, 2, "q")?;
    let (k, v, causal, start_pos) = match context {
        AttentionContext::Prefill => {
            cache.clear();
            let (k, v) = cache.append(k, v)?;
            (k, v, seq_len > 1, 0)
        }
        AttentionContext::Decode { start_pos } => {
            if start_pos != cache.seq_len() {
                return Err(TensorError::InvalidArgument(format!(
                    "decode start_pos {start_pos} does not match cache length {}",
                    cache.seq_len()
                )));
            }
            let (k, v) = cache.append(k, v)?;
            (k, v, false, start_pos)
        }
    };
    let n_rep = num_heads / num_kv_heads;
    let k = repeat_kv(k, n_rep)?;
    let v = repeat_kv(v, n_rep)?;
    sdpa(
        q,
        &k,
        &v,
        SdpaConfig {
            head_dim,
            causal,
            start_pos,
        },
    )
}

fn dim(tensor: &Tensor, index: usize, name: &str) -> Result<usize> {
    tensor.shape().dims().get(index).copied().ok_or_else(|| {
        TensorError::ShapeMismatch(format!(
            "{name} rank {} does not include dim {index}",
            tensor.shape().ndim()
        ))
    })
}
