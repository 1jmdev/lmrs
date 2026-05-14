use std::sync::Arc;

use cudarc::driver::CudaContext as CudarcContext;

use crate::Result;

/// Owns a CUDA device context.
///
/// `CudaContext` is the runtime entry point for CUDA allocation, streams,
/// events, and device property queries.
///
/// # Example
///
/// ```no_run
/// use runtime::CudaContext;
///
/// # fn main() -> runtime::Result<()> {
/// let context = CudaContext::new(0)?;
/// assert_eq!(context.ordinal(), 0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct CudaContext {
    context: Arc<CudarcContext>,
}

impl CudaContext {
    /// Creates a CUDA context for a device ordinal.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::CudaContext;
    ///
    /// # fn main() -> runtime::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// assert_eq!(context.ordinal(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(ordinal: usize) -> Result<Self> {
        let context = CudarcContext::new(ordinal)?;
        Ok(Self { context })
    }

    /// Returns the direct cudarc CUDA context.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::CudaContext;
    ///
    /// # fn main() -> runtime::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// assert_eq!(context.cudarc().ordinal(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn cudarc(&self) -> &Arc<CudarcContext> {
        &self.context
    }

    /// Consumes the context and returns its direct cudarc CUDA context.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::CudaContext;
    ///
    /// # fn main() -> runtime::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let raw = context.into_cudarc();
    /// assert_eq!(raw.ordinal(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn into_cudarc(self) -> Arc<CudarcContext> {
        self.context
    }

    /// Returns the CUDA device ordinal.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::CudaContext;
    ///
    /// # fn main() -> runtime::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// assert_eq!(context.ordinal(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn ordinal(&self) -> usize {
        self.context.ordinal()
    }
}
