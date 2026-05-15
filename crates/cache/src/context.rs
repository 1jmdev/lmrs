use tensor::{Result, Tensor, TensorError};

use crate::KvCache;

/// Attention execution mode with an explicit prefill/decode split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionContext {
    /// Processes a prompt span and resets the layer KV cache first.
    Prefill,
    /// Processes decoded tokens against an existing KV cache.
    Decode { start_pos: usize },
}

/// Runs attention with explicit prefill/decode cache behavior.
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
    let k = ops::repeat_kv(k, n_rep)?;
    let v = ops::repeat_kv(v, n_rep)?;
    ops::sdpa(
        q,
        &k,
        &v,
        ops::SdpaConfig {
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
