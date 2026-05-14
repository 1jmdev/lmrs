use std::sync::Arc;

use crate::CudaBuf;

/// Reference-counted CUDA tensor storage.
///
/// Sharing storage lets future views reuse an allocation without copying device
/// memory. Mutation APIs are intentionally absent until view aliasing rules are
/// defined.
///
/// # Example
///
/// ```no_run
/// use candle_core::Device;
/// use tensor::{CudaBuf, SharedStorage};
///
/// # fn main() -> candle_core::Result<()> {
/// let device = Device::new_cuda(0)?;
/// let storage = SharedStorage::new(CudaBuf::new(device.as_cuda_device()?, 16)?);
/// assert_eq!(storage.len_bytes(), 16);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct SharedStorage {
    inner: Arc<CudaBuf>,
}

impl SharedStorage {
    /// Wraps a CUDA buffer in shared storage.
    pub fn new(buffer: CudaBuf) -> Self {
        Self {
            inner: Arc::new(buffer),
        }
    }

    /// Returns the backing buffer size in bytes.
    pub fn len_bytes(&self) -> usize {
        self.inner.len_bytes()
    }

    /// Returns whether the backing allocation is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the backing CUDA buffer.
    pub fn buffer(&self) -> &CudaBuf {
        &self.inner
    }
}
