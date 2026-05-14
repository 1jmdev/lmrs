use tensor::{Result, Shape, Stride, Tensor, TensorError};

/// Adds two CUDA BF16 tensors elementwise.
///
/// This is used for transformer residual paths and requires identical contiguous
/// shapes so the CUDA kernel can run one coalesced pass over both inputs.
///
/// # Example
///
/// ```no_run
/// use ops::add;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let shape = Shape::new([2])?;
/// let left = copy_h2d(&context, shape.clone(), DType::BF16, &[0u16; 2])?;
/// let right = copy_h2d(&context, shape, DType::BF16, &[0u16; 2])?;
/// let out = add(&left, &right)?;
/// assert_eq!(out.shape().dims(), &[2]);
/// # Ok(())
/// # }
/// ```
pub fn add(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    kernels::add(left, right)
}

/// Copies a contiguous slice from dimension 1 of a rank-3 CUDA BF16 tensor.
///
/// The model uses this to select the final token hidden state before the LM head
/// without creating a strided view that later kernels cannot consume.
///
/// # Example
///
/// ```no_run
/// use ops::narrow_dim1;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let x = copy_h2d(&context, Shape::new([1, 2, 4])?, DType::BF16, &[0u16; 8])?;
/// let y = narrow_dim1(&x, 1, 1)?;
/// assert_eq!(y.shape().dims(), &[1, 1, 4]);
/// # Ok(())
/// # }
/// ```
pub fn narrow_dim1(input: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    kernels::narrow_dim1(input, start, len)
}

/// Returns a contiguous metadata reshape over the same CUDA storage.
///
/// This operation does not launch a kernel and does not copy memory; it only
/// validates that the requested shape has the same element count as the input.
///
/// # Example
///
/// ```no_run
/// use ops::reshape;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let x = copy_h2d(&context, Shape::new([2, 4])?, DType::BF16, &[0u16; 8])?;
/// let y = reshape(&x, [1, 2, 4])?;
/// assert_eq!(y.shape().dims(), &[1, 2, 4]);
/// # Ok(())
/// # }
/// ```
pub fn reshape(input: &Tensor, dims: impl Into<Vec<usize>>) -> Result<Tensor> {
    let shape = Shape::new(dims).map_err(|err| TensorError::ShapeMismatch(err.to_string()))?;
    if shape.numel() != input.numel() {
        return Err(TensorError::ShapeMismatch(format!(
            "reshape cannot change element count from {} to {}",
            input.numel(),
            shape.numel()
        )));
    }
    Ok(Tensor::from_storage(
        input.storage().clone(),
        shape.clone(),
        Stride::contiguous(&shape),
        input.dtype(),
    ))
}

/// Transposes dimensions 1 and 2 of a rank-4 CUDA BF16 tensor into contiguous output.
///
/// Attention uses this for `[batch, seq, heads, head_dim]` to
/// `[batch, heads, seq, head_dim]` conversion and the reverse conversion before
/// the output projection.
///
/// # Example
///
/// ```no_run
/// use ops::transpose_1_2;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let x = copy_h2d(&context, Shape::new([1, 2, 3, 4])?, DType::BF16, &[0u16; 24])?;
/// let y = transpose_1_2(&x)?;
/// assert_eq!(y.shape().dims(), &[1, 3, 2, 4]);
/// # Ok(())
/// # }
/// ```
pub fn transpose_1_2(input: &Tensor) -> Result<Tensor> {
    kernels::transpose_1_2(input)
}
