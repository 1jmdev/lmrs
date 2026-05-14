use cudarc::driver::{CudaSlice, DeviceRepr};

use crate::Result;
use crate::device::CudaContext;

/// Allocates uninitialized CUDA device memory on the provided device.
///
/// The allocation is owned by the returned `CudaSlice` and is freed when the
/// slice is dropped.
///
/// # Example
///
/// ```no_run
/// use runtime::{CudaContext, cuda_alloc};
///
/// # fn main() -> runtime::Result<()> {
/// let context = CudaContext::new(0)?;
/// let slice = unsafe { cuda_alloc::<f32>(&context, 1024)? };
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
pub unsafe fn cuda_alloc<T: DeviceRepr>(context: &CudaContext, len: usize) -> Result<CudaSlice<T>> {
    Ok(unsafe { context.cudarc().default_stream().alloc::<T>(len) }?)
}

/// Copies a CUDA allocation into a host `Vec`.
///
/// The copy is synchronous with respect to the host, so the returned vector is
/// safe to read immediately.
///
/// # Example
///
/// ```no_run
/// use runtime::{CudaContext, clone_dtoh, cuda_alloc};
///
/// # fn main() -> runtime::Result<()> {
/// let context = CudaContext::new(0)?;
/// let slice = unsafe { cuda_alloc::<f32>(&context, 4)? };
/// let host = clone_dtoh(&context, &slice)?;
///
/// assert_eq!(host.len(), 4);
/// # Ok(())
/// # }
/// ```
pub fn clone_dtoh<T: DeviceRepr + Default + Clone>(
    context: &CudaContext,
    slice: &CudaSlice<T>,
) -> Result<Vec<T>> {
    Ok(context.cudarc().default_stream().clone_dtoh(slice)?)
}
