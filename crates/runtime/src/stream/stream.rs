use std::fmt;
use std::sync::Arc;

use candle_core::Result;
use candle_core::cuda_backend::WrapErr;
use cudarc::driver::CudaStream as CudarcStream;

use crate::device::CudaContext;

/// Scheduling priority requested for a CUDA stream.
///
/// Runtime keeps the requested priority with each stream so schedulers can keep
/// separate pools for latency-sensitive and background work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamPriority {
    Low,
    Normal,
    High,
}

/// Owns a non-default CUDA stream forked from a runtime CUDA context.
///
/// The stream is created through cudarc's safe stream API. On creation, cudarc
/// makes the forked stream wait for already-enqueued default-stream work. On
/// drop, cudarc makes the default stream wait for this stream before destroying
/// it, preserving deterministic ordering between streams.
///
/// # Example
///
/// ```no_run
/// use runtime::{CudaContext, CudaStream, StreamPriority};
///
/// # fn main() -> candle_core::Result<()> {
/// let context = CudaContext::new(0)?;
/// let stream = CudaStream::new(&context, StreamPriority::Normal)?;
/// stream.synchronize()?;
/// # Ok(())
/// # }
/// ```
pub struct CudaStream {
    inner: Arc<CudarcStream>,
    priority: StreamPriority,
}

impl CudaStream {
    /// Creates a non-default CUDA stream associated with `context`.
    ///
    /// The stream is created directly through cudarc for the CUDA context.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, CudaStream, StreamPriority};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let stream = CudaStream::new(&context, StreamPriority::High)?;
    /// assert_eq!(stream.priority(), StreamPriority::High);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(context: &CudaContext, priority: StreamPriority) -> Result<Self> {
        let inner = context.cudarc().new_stream().w()?;
        Ok(Self { inner, priority })
    }

    /// Returns the requested stream priority.
    ///
    /// This value is useful for callers that build separate pools for
    /// low-priority prefill work and high-priority decode work.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, CudaStream, StreamPriority};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let stream = CudaStream::new(&context, StreamPriority::Low)?;
    /// assert_eq!(stream.priority(), StreamPriority::Low);
    /// # Ok(())
    /// # }
    /// ```
    pub fn priority(&self) -> StreamPriority {
        self.priority
    }

    /// Blocks the host until all work currently queued on this stream finishes.
    ///
    /// Use this only at API boundaries, tests, and profiling points. Normal
    /// execution should prefer event dependencies so independent streams can run
    /// concurrently.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, CudaStream, StreamPriority};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let stream = CudaStream::new(&context, StreamPriority::Normal)?;
    /// stream.synchronize()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn synchronize(&self) -> Result<()> {
        self.inner.synchronize().w()
    }

    /// Exposes the raw cudarc stream for low-level runtime modules.
    ///
    /// This is intentionally crate-visible so model and kernel crates do not
    /// depend on cudarc stream internals. Event recording and future graph
    /// capture code can use it inside `runtime`.
    pub(crate) fn inner(&self) -> &Arc<CudarcStream> {
        &self.inner
    }
}

impl fmt::Debug for CudaStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaStream")
            .field("priority", &self.priority)
            .finish_non_exhaustive()
    }
}
