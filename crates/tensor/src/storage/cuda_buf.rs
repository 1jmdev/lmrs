use cudarc::driver::CudaSlice;
use runtime::CudaContext;

use crate::Result;

/// Owned raw CUDA allocation measured in bytes.
///
/// `CudaBuf` deliberately stores `u8` elements so tensor dtype handling remains
/// explicit in tensor and copy code. Dropping the buffer releases the CUDA
/// allocation through cudarc.
///
/// # Example
///
/// ```no_run
/// use runtime::CudaContext;
/// use tensor::CudaBuf;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let buf = CudaBuf::new(&context, 4096)?;
/// assert_eq!(buf.len_bytes(), 4096);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct CudaBuf {
    data: CudaSlice<u8>,
}

impl CudaBuf {
    /// Allocates `len_bytes` uninitialized bytes on `device`.
    pub fn new(context: &CudaContext, len_bytes: usize) -> Result<Self> {
        let data = unsafe { context.cudarc().default_stream().alloc::<u8>(len_bytes)? };
        Ok(Self { data })
    }

    /// Wraps an existing byte slice allocation.
    pub fn from_slice(data: CudaSlice<u8>) -> Self {
        Self { data }
    }

    /// Returns the allocation size in bytes.
    pub fn len_bytes(&self) -> usize {
        self.data.len()
    }

    /// Returns whether the allocation has zero bytes.
    pub fn is_empty(&self) -> bool {
        self.data.len() == 0
    }

    /// Returns the underlying cudarc byte slice.
    pub fn as_slice(&self) -> &CudaSlice<u8> {
        &self.data
    }

    /// Returns the underlying cudarc byte slice mutably.
    pub fn as_mut_slice(&mut self) -> &mut CudaSlice<u8> {
        &mut self.data
    }
}
