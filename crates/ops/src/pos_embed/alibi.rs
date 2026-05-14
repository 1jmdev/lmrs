use runtime::CudaContext;
use tensor::{Result, Shape, Tensor, TensorError};

/// Builds an ALiBi bias tensor placeholder with shape `[1, heads, query, key]`.
pub fn alibi_bias(
    num_heads: usize,
    query_len: usize,
    key_len: usize,
    context: &CudaContext,
) -> Result<Tensor> {
    let shape = Shape::new([1, num_heads, query_len, key_len])
        .map_err(|err| TensorError::ShapeMismatch(err.to_string()))?;
    kernels::zeros_f32_like_shape(context, shape)
}
