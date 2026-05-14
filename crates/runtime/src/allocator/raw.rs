use candle_core::Result;
use candle_core::cuda_backend::CudaDevice;
use cudarc::driver::{CudaSlice, DeviceRepr};

/// Allocates uninitialized CUDA device memory on the provided device.
///
/// The allocation is owned by the returned `CudaSlice` and is freed when the
/// slice is dropped.
///
/// # Example
///
/// ```no_run
/// use candle_core::Device;
/// use runtime::cuda_alloc;
///
/// # fn main() -> candle_core::Result<()> {
/// let device = Device::new_cuda(0)?;
/// let cuda = device.as_cuda_device()?;
/// let slice = unsafe { cuda_alloc::<f32>(cuda, 1024)? };
///
/// assert_eq!(slice.len(), 1024);
/// # Ok(())
/// # }
/// ```
///
/// # Safety
///
/// The returned device memory is uninitialized. Callers must write every element
/// before any kernel or D2H transfer reads from it.
pub unsafe fn cuda_alloc<T: DeviceRepr>(device: &CudaDevice, len: usize) -> Result<CudaSlice<T>> {
    unsafe { device.alloc::<T>(len) }
}

/// Copies a CUDA allocation into a host `Vec`.
///
/// The copy is synchronous with respect to the host, so the returned vector is
/// safe to read immediately.
///
/// # Example
///
/// ```no_run
/// use candle_core::Device;
/// use runtime::{clone_dtoh, cuda_alloc};
///
/// # fn main() -> candle_core::Result<()> {
/// let device = Device::new_cuda(0)?;
/// let cuda = device.as_cuda_device()?;
/// let slice = unsafe { cuda_alloc::<f32>(cuda, 4)? };
/// let host = clone_dtoh(cuda, &slice)?;
///
/// assert_eq!(host.len(), 4);
/// # Ok(())
/// # }
/// ```
pub fn clone_dtoh<T: DeviceRepr + Default + Clone>(
    device: &CudaDevice,
    slice: &CudaSlice<T>,
) -> Result<Vec<T>> {
    device.clone_dtoh(slice)
}
