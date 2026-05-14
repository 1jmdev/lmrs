use candle_core::Result;

use crate::device::CudaContext;

/// In-progress CUDA graph capture descriptor.
///
/// CUDA graphs are exposed through runtime-owned types so the engine can depend
/// on stable graph capture metadata before graph replay is enabled.
///
/// # Example
///
/// ```no_run
/// use runtime::{CudaContext, GraphCapture};
///
/// # fn main() -> candle_core::Result<()> {
/// let context = CudaContext::new(0)?;
/// let capture = GraphCapture::begin(&context)?;
/// let graph = capture.end()?;
/// assert_eq!(graph.device_ordinal(), 0);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct GraphCapture {
    device_ordinal: usize,
}

impl GraphCapture {
    /// Starts a graph capture descriptor for the provided context.
    ///
    /// This records the device ordinal for the capture session.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, GraphCapture};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let capture = GraphCapture::begin(&context)?;
    /// assert_eq!(capture.device_ordinal(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn begin(context: &CudaContext) -> Result<Self> {
        Ok(Self {
            device_ordinal: context.ordinal(),
        })
    }

    /// Finishes the capture descriptor and returns graph metadata.
    ///
    /// The returned `CapturedGraph` contains the device identity used by graph
    /// execution APIs.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, GraphCapture};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let graph = GraphCapture::begin(&context)?.end()?;
    /// assert_eq!(graph.device_ordinal(), context.ordinal());
    /// # Ok(())
    /// # }
    /// ```
    pub fn end(self) -> Result<CapturedGraph> {
        Ok(CapturedGraph {
            device_ordinal: self.device_ordinal,
        })
    }

    /// Returns the CUDA ordinal associated with this capture descriptor.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, GraphCapture};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let capture = GraphCapture::begin(&context)?;
    /// assert_eq!(capture.device_ordinal(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }
}

/// Metadata for a captured CUDA graph.
///
/// # Example
///
/// ```no_run
/// use runtime::{CudaContext, GraphCapture};
///
/// # fn main() -> candle_core::Result<()> {
/// let context = CudaContext::new(0)?;
/// let graph = GraphCapture::begin(&context)?.end()?;
/// assert_eq!(graph.device_ordinal(), 0);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct CapturedGraph {
    device_ordinal: usize,
}

impl CapturedGraph {
    /// Returns the CUDA ordinal where the graph was captured.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, GraphCapture};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let graph = GraphCapture::begin(&context)?.end()?;
    /// assert_eq!(graph.device_ordinal(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }
}
