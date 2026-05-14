use cudarc::driver::{CudaSlice, DeviceRepr};
use runtime::CudaContext;

use crate::{CudaBuf, DType, Result, Shape, SharedStorage, Stride, Tensor};

/// Host/device copy validation failures.
#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum CopyError {
    /// Host element type does not match the tensor dtype size.
    #[error("host element size {host_size} does not match dtype {dtype} size {dtype_size}")]
    ElementSizeMismatch {
        host_size: usize,
        dtype: DType,
        dtype_size: usize,
    },
    /// Host element count does not match tensor element count.
    #[error("host length {host_len} does not match tensor elements {tensor_len}")]
    LengthMismatch { host_len: usize, tensor_len: usize },
}

/// Copies a typed CUDA slice back to host after validating tensor metadata.
///
/// The current raw storage is byte-oriented, so typed D2H copies are exposed for
/// callers that already hold a typed CUDA slice. Tensor-owned typed views will be
/// added once storage aliasing is finalized.
///
/// # Example
///
/// ```no_run
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, Tensor, copy_dtoh};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let raw = unsafe { context.cudarc().default_stream().alloc::<f32>(4)? };
/// let tensor = Tensor::empty(&context, Shape::new([4]).unwrap(), DType::F32)?;
/// let host = copy_dtoh::<f32>(&context, &tensor, &raw)?;
/// assert_eq!(host.len(), 4);
/// # Ok(())
/// # }
/// ```
pub fn copy_dtoh<T>(context: &CudaContext, tensor: &Tensor, slice: &CudaSlice<T>) -> Result<Vec<T>>
where
    T: DeviceRepr + Default + Clone,
{
    validate_typed::<T>(tensor, slice.len())?;
    Ok(context.cudarc().default_stream().clone_dtoh(slice)?)
}

/// Copies a host buffer into a new contiguous CUDA tensor.
///
/// The host element byte width must match `dtype`; dtype conversion is handled
/// separately by cast/quantization kernels.
///
/// # Example
///
/// ```no_run
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, Tensor, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let tensor = copy_h2d(&context, Shape::new([2]).unwrap(), DType::F32, &[1.0f32, 2.0])?;
/// assert_eq!(tensor.numel(), 2);
/// # Ok(())
/// # }
/// ```
pub fn copy_h2d<T: DeviceRepr>(
    context: &CudaContext,
    shape: Shape,
    dtype: DType,
    host: &[T],
) -> Result<Tensor> {
    validate_parts::<T>(dtype, shape.numel(), host.len())?;
    let bytes = host_as_bytes(host);
    let data = context.cudarc().default_stream().clone_htod(bytes)?;
    let storage = SharedStorage::new(CudaBuf::from_slice(data));
    let stride = Stride::contiguous(&shape);
    Ok(Tensor::from_storage(storage, shape, stride, dtype))
}

fn validate_typed<T>(tensor: &Tensor, len: usize) -> Result<()> {
    validate_parts::<T>(tensor.dtype(), tensor.numel(), len)
}

fn validate_parts<T>(dtype: DType, tensor_len: usize, len: usize) -> Result<()> {
    let host_size = std::mem::size_of::<T>();
    let dtype_size = dtype.size_in_bytes();
    if host_size != dtype_size {
        return Err(CopyError::ElementSizeMismatch {
            host_size,
            dtype,
            dtype_size,
        }
        .into());
    }
    if len != tensor_len {
        return Err(CopyError::LengthMismatch {
            host_len: len,
            tensor_len,
        }
        .into());
    }
    Ok(())
}

fn host_as_bytes<T>(host: &[T]) -> &[u8] {
    let len = std::mem::size_of_val(host);
    unsafe { std::slice::from_raw_parts(host.as_ptr().cast::<u8>(), len) }
}
