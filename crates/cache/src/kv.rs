use candle_core::{Result, Tensor};
use ops::attention::{SdpaConfig, repeat_kv, sdpa};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SequenceId(u64);

impl SequenceId {
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionContext {
    Prefill,
    Decode { start_pos: usize },
}

#[derive(Default)]
pub struct KvCache {
    cache: Option<(Tensor, Tensor)>,
    seq_len: usize,
}

impl KvCache {
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn clear(&mut self) {
        self.cache = None;
        self.seq_len = 0;
    }

    pub fn append(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let new_seq_len = k.dim(2)?;
        match self.cache.take() {
            Some((buf_k, buf_v)) => {
                let new_total = self.seq_len + new_seq_len;
                if new_total <= buf_k.dim(2)? {
                    buf_k.slice_set(&k, 2, self.seq_len)?;
                    buf_v.slice_set(&v, 2, self.seq_len)?;
                    let k_view = buf_k.narrow(2, 0, new_total)?;
                    let v_view = buf_v.narrow(2, 0, new_total)?;
                    self.cache = Some((buf_k, buf_v));
                    self.seq_len = new_total;
                    Ok((k_view, v_view))
                } else {
                    let cur_k = buf_k.narrow(2, 0, self.seq_len)?;
                    let cur_v = buf_v.narrow(2, 0, self.seq_len)?;
                    let full_k = Tensor::cat(&[&cur_k, &k], 2)?;
                    let full_v = Tensor::cat(&[&cur_v, &v], 2)?;
                    self.replace_with_slack(&full_k, &full_v)
                }
            }
            None => self.replace_with_slack(&k, &v),
        }
    }

    fn replace_with_slack(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, h, s, d) = k.dims4()?;
        let buf_k = Tensor::zeros((b, h, s + 256, d), k.dtype(), k.device())?;
        let buf_v = Tensor::zeros((b, h, s + 256, d), v.dtype(), v.device())?;
        buf_k.slice_set(k, 2, 0)?;
        buf_v.slice_set(v, 2, 0)?;
        self.cache = Some((buf_k, buf_v));
        self.seq_len = s;
        Ok((k.clone(), v.clone()))
    }
}

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
    let seq_len = q.dim(2)?;
    let (k, v, causal, start_pos) = match context {
        AttentionContext::Prefill => {
            cache.clear();
            let (k, v) = cache.append(k, v)?;
            (k, v, seq_len > 1, 0)
        }
        AttentionContext::Decode { start_pos } => {
            if start_pos != cache.seq_len() {
                candle_core::bail!(
                    "decode start_pos {start_pos} does not match cache length {}",
                    cache.seq_len()
                );
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