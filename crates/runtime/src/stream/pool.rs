use super::{CudaStream, StreamPriority};
use crate::device::CudaContext;
use candle_core::Result;

/// Fixed-size collection of CUDA streams for one context.
///
/// A stream pool lets runtime code pre-create streams per device and reuse them
/// instead of allocating stream handles inside hot paths. All streams in this
/// pool share the same requested priority.
///
/// # Example
///
/// ```no_run
/// use runtime::{CudaContext, StreamPool, StreamPriority};
///
/// # fn main() -> candle_core::Result<()> {
/// let context = CudaContext::new(0)?;
/// let pool = StreamPool::new(&context, 4, StreamPriority::Normal)?;
/// assert_eq!(pool.len(), 4);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct StreamPool {
    streams: Vec<CudaStream>,
}

impl StreamPool {
    /// Creates `size` streams for `context`.
    ///
    /// Stream creation can fail if CUDA cannot allocate a stream handle, so the
    /// constructor returns Candle's `Result` and leaves the caller with either a
    /// fully-built pool or no pool at all.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, StreamPool, StreamPriority};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let pool = StreamPool::new(&context, 2, StreamPriority::High)?;
    /// assert!(pool.streams().iter().all(|s| s.priority() == StreamPriority::High));
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(context: &CudaContext, size: usize, priority: StreamPriority) -> Result<Self> {
        let streams = (0..size)
            .map(|_| CudaStream::new(context, priority))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { streams })
    }

    /// Returns all streams managed by this pool.
    ///
    /// The slice is immutable so callers can submit work through stream handles
    /// without changing pool membership or invalidating indices held elsewhere.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, StreamPool, StreamPriority};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let pool = StreamPool::new(&context, 1, StreamPriority::Normal)?;
    /// assert_eq!(pool.streams().len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn streams(&self) -> &[CudaStream] {
        &self.streams
    }

    /// Returns the number of streams in the pool.
    ///
    /// This is a convenience wrapper over `streams().len()` used by schedulers
    /// that size per-stream bookkeeping arrays.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, StreamPool, StreamPriority};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let pool = StreamPool::new(&context, 3, StreamPriority::Low)?;
    /// assert_eq!(pool.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn len(&self) -> usize {
        self.streams.len()
    }

    /// Returns `true` when the pool contains no streams.
    ///
    /// Empty pools are valid for configurations that want all work to remain on
    /// the context default stream.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, StreamPool, StreamPriority};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let pool = StreamPool::new(&context, 0, StreamPriority::Normal)?;
    /// assert!(pool.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }
}
