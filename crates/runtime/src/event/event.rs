use std::fmt;

use cudarc::driver::result;
use cudarc::driver::sys;

use crate::Result;
use crate::device::CudaContext;
use crate::stream::CudaStream;

/// Owns a CUDA event handle.
///
/// Events are the synchronization primitive used to order independent streams
/// and measure GPU elapsed time without forcing a full device synchronization.
/// This wrapper owns the raw driver event and destroys it on drop.
///
/// # Example
///
/// ```no_run
/// use runtime::{CudaContext, CudaEvent};
///
/// # fn main() -> runtime::Result<()> {
/// let context = CudaContext::new(0)?;
/// let event = CudaEvent::new()?;
/// event.record_default(&context)?;
/// event.synchronize()?;
/// # Ok(())
/// # }
/// ```
pub struct CudaEvent {
    raw: sys::CUevent,
}

impl CudaEvent {
    /// Creates a timing-capable CUDA event.
    ///
    /// Timing is enabled because `EventTimer` computes elapsed milliseconds from
    /// two events. Use this for both synchronization and profiling until runtime
    /// has a reason to expose a separate low-overhead no-timing event type.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::CudaEvent;
    ///
    /// # fn main() -> runtime::Result<()> {
    /// let _event = CudaEvent::new()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Result<Self> {
        let raw = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)?;
        Ok(Self { raw })
    }

    /// Records this event on the context's default CUDA stream.
    ///
    /// Recording captures all work submitted to the default stream before this
    /// call. `synchronize` can then wait for that work, and `EventTimer` can use
    /// two recorded events to compute GPU time.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, CudaEvent};
    ///
    /// # fn main() -> runtime::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let event = CudaEvent::new()?;
    /// event.record_default(&context)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn record_default(&self, context: &CudaContext) -> Result<()> {
        let stream = context.cudarc().default_stream();
        unsafe { result::event::record(self.raw, stream.cu_stream()) }?;
        Ok(())
    }

    /// Records this event on a runtime-owned non-default stream.
    ///
    /// Use this when work was submitted to a `CudaStream` from a `StreamPool` and
    /// the default stream should not be synchronized unnecessarily.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, CudaEvent, CudaStream, StreamPriority};
    ///
    /// # fn main() -> runtime::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let stream = CudaStream::new(&context, StreamPriority::Normal)?;
    /// let event = CudaEvent::new()?;
    /// event.record_on(&stream)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn record_on(&self, stream: &CudaStream) -> Result<()> {
        unsafe { result::event::record(self.raw, stream.inner().cu_stream()) }?;
        Ok(())
    }

    /// Blocks the host until this event has completed.
    ///
    /// This waits only for work that happens before the event on the stream where
    /// it was recorded. It is therefore narrower than synchronizing the entire
    /// CUDA device.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, CudaEvent};
    ///
    /// # fn main() -> runtime::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let event = CudaEvent::new()?;
    /// event.record_default(&context)?;
    /// event.synchronize()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn synchronize(&self) -> Result<()> {
        unsafe { result::event::synchronize(self.raw) }?;
        Ok(())
    }

    /// Returns the raw CUDA event handle for low-level runtime modules.
    pub(crate) fn raw(&self) -> sys::CUevent {
        self.raw
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { result::event::destroy(self.raw) }.expect("failed to destroy CUDA event");
        }
    }
}

impl fmt::Debug for CudaEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaEvent").finish_non_exhaustive()
    }
}
