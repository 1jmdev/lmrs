use cudarc::driver::{CudaSlice, DeviceRepr};

use crate::Result;
use crate::device::CudaContext;

/// Allocates CUDA memory for one context.
///
/// `MemoryPool` centralizes runtime allocations so higher crates do not call the
/// CUDA backend directly.
///
/// # Example
///
/// ```no_run
/// use runtime::{CudaContext, MemoryPool};
///
/// # fn main() -> runtime::Result<()> {
/// let context = CudaContext::new(0)?;
/// let pool = MemoryPool::new(context);
/// let slice = unsafe { pool.alloc::<f32>(16)? };
///
/// assert_eq!(slice.len(), 16);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct MemoryPool {
    context: CudaContext,
}

impl MemoryPool {
    /// Creates a memory pool bound to `context`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, MemoryPool};
    ///
    /// # fn main() -> runtime::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let pool = MemoryPool::new(context);
    ///
    /// assert_eq!(pool.context().ordinal(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(context: CudaContext) -> Self {
        Self { context }
    }

    /// Allocates uninitialized typed device memory.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, MemoryPool};
    ///
    /// # fn main() -> runtime::Result<()> {
    /// let pool = MemoryPool::new(CudaContext::new(0)?);
    /// let slice = unsafe { pool.alloc::<u8>(256)? };
    ///
    /// assert_eq!(slice.len(), 256);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Safety
    ///
    /// The returned allocation has unspecified contents and must be initialized
    /// before it is read by device or host code.
    pub unsafe fn alloc<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>> {
        let stream = self.context.cudarc().default_stream();
        Ok(unsafe { stream.alloc::<T>(len) }?)
    }

    /// Returns the CUDA context used by this pool.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, MemoryPool};
    ///
    /// # fn main() -> runtime::Result<()> {
    /// let pool = MemoryPool::new(CudaContext::new(0)?);
    /// assert_eq!(pool.context().ordinal(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn context(&self) -> &CudaContext {
        &self.context
    }
}
