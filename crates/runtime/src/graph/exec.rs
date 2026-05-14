use candle_core::Result;

use super::CapturedGraph;

/// Executable CUDA graph metadata.
///
/// `GraphExec` stores the device identity for a captured graph executable.
///
/// # Example
///
/// ```no_run
/// use runtime::{CudaContext, GraphCapture, GraphExec};
///
/// # fn main() -> candle_core::Result<()> {
/// let context = CudaContext::new(0)?;
/// let graph = GraphCapture::begin(&context)?.end()?;
/// let result = GraphExec::instantiate(graph);
/// assert!(result.is_ok());
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct GraphExec {
    device_ordinal: usize,
}

impl GraphExec {
    /// Instantiates a captured graph.
    ///
    /// The executable stores graph metadata and can be passed to launch/update
    /// APIs.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, GraphCapture, GraphExec};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let graph = GraphCapture::begin(&context)?.end()?;
    /// let exec = GraphExec::instantiate(graph)?;
    /// assert_eq!(exec.device_ordinal(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn instantiate(graph: CapturedGraph) -> Result<Self> {
        Ok(Self {
            device_ordinal: graph.device_ordinal(),
        })
    }

    /// Launches an executable graph.
    ///
    /// The method returns a `Result` so CUDA driver launch failures can be
    /// propagated to callers.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, GraphCapture, GraphExec};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let graph = GraphCapture::begin(&context)?.end()?;
    /// let result = GraphExec::instantiate(graph).and_then(|exec| exec.launch());
    /// assert!(result.is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn launch(&self) -> Result<()> {
        let _device_ordinal = self.device_ordinal;
        candle_core::bail!("CUDA graph execution is unavailable")
    }

    /// Returns the CUDA ordinal associated with this executable graph metadata.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, GraphCapture, GraphExec};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let graph = GraphCapture::begin(&context)?.end()?;
    /// let exec = GraphExec::instantiate(graph)?;
    /// assert_eq!(exec.device_ordinal(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }
}
