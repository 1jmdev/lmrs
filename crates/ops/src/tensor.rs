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

/// Applies `scale * x + offset` to a BF16 CUDA tensor.
///
/// # Example
///
/// ```no_run
/// use ops::affine;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let x = copy_h2d(&context, Shape::new([4])?, DType::BF16, &[0u16; 4])?;
/// let y = affine(&x, 2.0f32, 1.0f32)?;
/// assert_eq!(y.shape().dims(), &[4]);
/// # Ok(())
/// # }
/// ```
pub fn affine(x: &Tensor, scale: f32, offset: f32) -> Result<Tensor> {
    kernels::affine(x, scale, offset)
}

/// Elementwise `left - right` for CUDA BF16 tensors.
///
/// # Example
///
/// ```no_run
/// use ops::sub;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let left = copy_h2d(&context, Shape::new([2])?, DType::BF16, &[0u16; 2])?;
/// let right = copy_h2d(&context, Shape::new([2])?, DType::BF16, &[0u16; 2])?;
/// let y = sub(&left, &right)?;
/// assert_eq!(y.shape().dims(), &[2]);
/// # Ok(())
/// # }
/// ```
pub fn sub(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    kernels::sub_bf16(left, right)
}

/// Multiplies a BF16 CUDA tensor by a float scalar.
///
/// # Example
///
/// ```no_run
/// use ops::scale;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let x = copy_h2d(&context, Shape::new([2])?, DType::BF16, &[0u16; 2])?;
/// let y = scale(&x, 2.0f32)?;
/// assert_eq!(y.shape().dims(), &[2]);
/// # Ok(())
/// # }
/// ```
pub fn scale(x: &Tensor, scale: f32) -> Result<Tensor> {
    kernels::scale_bf16(x, scale)
}

/// Compares `left >= right` elementwise, producing 1.0/0.0 as BF16.
///
/// # Example
///
/// ```no_run
/// use ops::greater_equal;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let x = copy_h2d(&context, Shape::new([2])?, DType::BF16, &[0u16; 2])?;
/// let y = greater_equal(&x, &x)?;
/// assert_eq!(y.shape().dims(), &[2]);
/// # Ok(())
/// # }
/// ```
pub fn greater_equal(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    kernels::greater_equal_bf16(left, right)
}

/// Selects `true_val[i]` where `cond[i] > 0`, otherwise `false_val[i]`.
///
/// # Example
///
/// ```no_run
/// use ops::where_cond;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let cond = copy_h2d(&context, Shape::new([2])?, DType::BF16, &[0u16, 0x3F80u16])?;
/// let t = copy_h2d(&context, Shape::new([2])?, DType::BF16, &[1u16, 2u16])?;
/// let f = copy_h2d(&context, Shape::new([2])?, DType::BF16, &[3u16, 4u16])?;
/// let y = where_cond(&cond, &t, &f)?;
/// assert_eq!(y.shape().dims(), &[2]);
/// # Ok(())
/// # }
/// ```
pub fn where_cond(cond: &Tensor, true_val: &Tensor, false_val: &Tensor) -> Result<Tensor> {
    kernels::where_cond_bf16(cond, true_val, false_val)
}

/// Casts a BF16 CUDA tensor to host F32 values.
///
/// # Example
///
/// ```no_run
/// use ops::cast_bf16_to_f32;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let x = copy_h2d(&context, Shape::new([2])?, DType::BF16, &[0u16; 2])?;
/// let host = cast_bf16_to_f32(&x)?;
/// assert_eq!(host.len(), 2);
/// # Ok(())
/// # }
/// ```
pub fn cast_bf16_to_f32(x: &Tensor) -> Result<Vec<f32>> {
    kernels::cast_bf16_to_f32(x)
}

/// Multiplies two CUDA BF16 tensors elementwise.
///
/// # Example
///
/// ```no_run
/// use ops::mul;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let shape = Shape::new([2])?;
/// let left = copy_h2d(&context, shape.clone(), DType::BF16, &[0u16; 2])?;
/// let right = copy_h2d(&context, shape, DType::BF16, &[0u16; 2])?;
/// let y = mul(&left, &right)?;
/// assert_eq!(y.shape().dims(), &[2]);
/// # Ok(())
/// # }
/// ```
pub fn mul(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    kernels::mul_bf16(left, right)
}
