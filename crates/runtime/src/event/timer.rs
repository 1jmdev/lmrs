use cudarc::driver::result;

use crate::Result;

use super::CudaEvent;

/// Measures elapsed GPU time between two recorded CUDA events.
///
/// CUDA event timing is reported in milliseconds and reflects device execution
/// time between the two event positions on the stream timeline. The caller must
/// record both events before calling `elapsed_ms`.
///
/// # Example
///
/// ```no_run
/// use runtime::{CudaContext, CudaEvent, EventTimer};
///
/// # fn main() -> runtime::Result<()> {
/// let context = CudaContext::new(0)?;
/// let start = CudaEvent::new()?;
/// let end = CudaEvent::new()?;
/// start.record_default(&context)?;
/// end.record_default(&context)?;
/// end.synchronize()?;
/// let _elapsed_ms = EventTimer::elapsed_ms(&start, &end)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct EventTimer;

impl EventTimer {
    /// Returns elapsed GPU time in milliseconds between two events.
    ///
    /// The events should be recorded on the same stream or on streams whose
    /// dependencies establish a valid order. CUDA returns an error if either
    /// event has not completed or timing is unavailable.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, CudaEvent, EventTimer};
    ///
    /// # fn main() -> runtime::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let start = CudaEvent::new()?;
    /// let end = CudaEvent::new()?;
    /// start.record_default(&context)?;
    /// end.record_default(&context)?;
    /// end.synchronize()?;
    /// let elapsed = EventTimer::elapsed_ms(&start, &end)?;
    /// assert!(elapsed >= 0.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn elapsed_ms(start: &CudaEvent, end: &CudaEvent) -> Result<f32> {
        Ok(unsafe { result::event::elapsed(start.raw(), end.raw()) }?)
    }
}
