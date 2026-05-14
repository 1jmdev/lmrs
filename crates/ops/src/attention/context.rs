use tensor::{Result, Tensor, TensorError};

use crate::attention::{SdpaConfig, repeat_kv, sdpa};

/// Attention execution mode with an explicit prefill/decode split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionContext {
    /// Processes a prompt span and applies a causal mask across all prompt tokens.
    Prefill,
    /// Processes one or more newly decoded tokens against an existing KV cache.
    Decode { start_pos: usize },
}

/// Owned key/value cache for a single attention layer.
#[derive(Default)]
pub struct KvCache {
    cache: Option<(Tensor, Tensor)>,
    seq_len: usize,
}

impl KvCache {
    /// Returns the number of valid cached time steps.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Clears all cached key/value states.
    pub fn clear(&mut self) {
        self.cache = None;
        self.seq_len = 0;
    }

    /// Appends new key/value states shaped `[batch, kv_heads, seq, head_dim]`.
    pub fn append(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let new_seq_len = dim(&k, 2, "k")?;
        match self.cache.take() {
            Some((buf_k, buf_v)) => {
                let full_k = kernels::concat_dim2(&buf_k, &k)?;
                let full_v = kernels::concat_dim2(&buf_v, &v)?;
                self.seq_len += new_seq_len;
                self.cache = Some((full_k.clone(), full_v.clone()));
                Ok((full_k, full_v))
            }
            None => self.replace_with_slack(&k, &v),
        }
    }

    fn replace_with_slack(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let s = dim(k, 2, "k")?;
        self.cache = Some((k.clone(), v.clone()));
        self.seq_len = s;
        Ok((k.clone(), v.clone()))
    }
}

/// Runs attention with explicit prefill/decode behavior.
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
