use tensor::{Result, Tensor, TensorError};

/// Owned CUDA key/value cache for one transformer attention layer.
///
/// The cache stores key and value tensors in `[batch, kv_heads, seq, head_dim]`
/// layout, which matches the attention kernels and avoids extra layout work on
/// every decode step. Appending extends dimension `2`, the sequence dimension.
///
/// # Example
///
/// ```no_run
/// use cache::KvCache;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let shape = Shape::new([1, 2, 1, 4]).unwrap();
/// let key = copy_h2d(&context, shape.clone(), DType::BF16, &[0u16; 8])?;
/// let value = copy_h2d(&context, shape, DType::BF16, &[0u16; 8])?;
///
/// let mut cache = KvCache::default();
/// let (_cached_key, _cached_value) = cache.append(key, value)?;
/// assert_eq!(cache.seq_len(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Default)]
pub struct KvCache {
    /// Cached key/value tensors in `[batch, kv_heads, seq, head_dim]` layout.
    cache: Option<(Tensor, Tensor)>,
    /// Number of valid sequence positions currently stored in `cache`.
    seq_len: usize,
}

impl KvCache {
    /// Returns the number of valid cached sequence positions.
    ///
    /// # Example
    ///
    /// ```
    /// use cache::KvCache;
    ///
    /// let cache = KvCache::default();
    /// assert_eq!(cache.seq_len(), 0);
    /// ```
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Clears all cached key/value tensors and resets the sequence length.
    ///
    /// # Example
    ///
    /// ```
    /// use cache::KvCache;
    ///
    /// let mut cache = KvCache::default();
    /// cache.clear();
    /// assert_eq!(cache.seq_len(), 0);
    /// ```
    pub fn clear(&mut self) {
        self.cache = None;
        self.seq_len = 0;
    }

    /// Appends new key/value tensors and returns the full cached tensors.
    ///
    /// Inputs must be CUDA tensors shaped `[batch, kv_heads, seq, head_dim]`.
    /// On the first append, the tensors are stored directly. Later appends use
    /// the optimized CUDA `concat_dim2` kernel so decode cache growth stays on
    /// device and avoids host copies.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cache::KvCache;
    /// use runtime::CudaContext;
    /// use tensor::{DType, Shape, copy_h2d};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let context = CudaContext::new(0)?;
    /// let shape = Shape::new([1, 1, 1, 2]).unwrap();
    /// let first_key = copy_h2d(&context, shape.clone(), DType::BF16, &[0u16; 2])?;
    /// let first_value = copy_h2d(&context, shape.clone(), DType::BF16, &[0u16; 2])?;
    /// let next_key = copy_h2d(&context, shape.clone(), DType::BF16, &[1u16; 2])?;
    /// let next_value = copy_h2d(&context, shape, DType::BF16, &[1u16; 2])?;
    ///
    /// let mut cache = KvCache::default();
    /// let _ = cache.append(first_key, first_value)?;
    /// let (key, value) = cache.append(next_key, next_value)?;
    ///
    /// assert_eq!(cache.seq_len(), 2);
    /// assert_eq!(key.shape().dims(), &[1, 1, 2, 2]);
    /// assert_eq!(value.shape().dims(), &[1, 1, 2, 2]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn append(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let new_seq_len = dim(&k, 2, "k")?;
        match self.cache.take() {
            Some((buf_k, buf_v)) => {
                let full_k = ops::concat_dim2(&buf_k, &k)?;
                let full_v = ops::concat_dim2(&buf_v, &v)?;
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

fn dim(tensor: &Tensor, index: usize, name: &str) -> Result<usize> {
    tensor.shape().dims().get(index).copied().ok_or_else(|| {
        TensorError::ShapeMismatch(format!(
            "{name} rank {} does not include dim {index}",
            tensor.shape().ndim()
        ))
    })
}
