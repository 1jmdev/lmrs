use candle_core::cuda_backend::{CudaDevice, WrapErr};
use candle_core::{Result, bail};
use cudarc::driver::{CudaSlice, DeviceRepr};
use runtime::{CudaContext, clone_dtoh};

use crate::{CudaBuf, DType, Shape, SharedStorage, Stride, Tensor};

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
/// use candle_core::Device;
/// use tensor::{DType, Shape, Tensor, copy_dtoh};
///
/// # fn main() -> candle_core::Result<()> {
/// let device = Device::new_cuda(0)?;
/// let cuda = device.as_cuda_device()?;
/// let raw = unsafe { runtime::cuda_alloc::<f32>(cuda, 4)? };
/// let tensor = Tensor::empty(cuda, Shape::new([4]).unwrap(), DType::F32)?;
/// let host = copy_dtoh::<f32>(cuda, &tensor, &raw)?;
/// assert_eq!(host.len(), 4);
/// # Ok(())
/// # }
/// ```
pub fn copy_dtoh<T>(device: &CudaDevice, tensor: &Tensor, slice: &CudaSlice<T>) -> Result<Vec<T>>
where
    T: DeviceRepr + Default + Clone,
{
    validate_typed::<T>(tensor, slice.len())?;
    clone_dtoh(device, slice)
}

/// Copies a host buffer into a new contiguous CUDA tensor.
///
/// The host element byte width must match `dtype`; dtype conversion is handled
/// separately by cast/quantization kernels.
///
/// # Example
///
/// ```no_run
/// use candle_core::Device;
/// use tensor::{DType, Shape, Tensor, copy_h2d};
///
/// # fn main() -> candle_core::Result<()> {
/// let device = Device::new_cuda(0)?;
/// let tensor = copy_h2d(device.as_cuda_device()?, Shape::new([2]).unwrap(), DType::F32, &[1.0f32, 2.0])?;
/// assert_eq!(tensor.numel(), 2);
/// # Ok(())
/// # }
/// ```
pub fn copy_h2d<T: DeviceRepr>(
    device: &CudaDevice,
    shape: Shape,
    dtype: DType,
    host: &[T],
) -> Result<Tensor> {
    validate_parts::<T>(dtype, shape.numel(), host.len())?;
    let bytes = host_as_bytes(host);
    let context = CudaContext::from_candle(device.clone())?;
    let data = context.cudarc().default_stream().clone_htod(bytes).w()?;
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
        bail!(
            "{}",
            CopyError::ElementSizeMismatch {
                host_size,
                dtype,
                dtype_size
            }
        );
    }
    if len != tensor_len {
        bail!(
            "{}",
            CopyError::LengthMismatch {
                host_len: len,
                tensor_len
            }
        );
    }
    Ok(())
}

fn host_as_bytes<T>(host: &[T]) -> &[u8] {
    let len = std::mem::size_of_val(host);
    unsafe { std::slice::from_raw_parts(host.as_ptr().cast::<u8>(), len) }
}
