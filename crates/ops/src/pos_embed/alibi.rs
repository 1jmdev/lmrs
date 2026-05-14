use candle_core::{DType, Device, Result, Tensor};

/// Builds an ALiBi bias tensor placeholder with shape `[1, heads, query, key]`.
pub fn alibi_bias(
    num_heads: usize,
    query_len: usize,
    key_len: usize,
    device: &Device,
) -> Result<Tensor> {
    Tensor::zeros((1, num_heads, query_len, key_len), DType::F32, device)
}
