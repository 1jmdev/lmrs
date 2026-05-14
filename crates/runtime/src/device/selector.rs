use candle_core::Result;
use candle_core::cuda_backend::WrapErr;
use cudarc::driver::CudaContext as CudarcContext;

use super::CudaContext;

/// Selects a CUDA device by visible ordinal.
///
/// # Example
///
/// ```
/// use runtime::DeviceSelector;
///
/// let selector = DeviceSelector::new(0);
/// assert_eq!(selector.ordinal(), 0);
/// ```
#[derive(Clone, Debug)]
pub struct DeviceSelector {
    ordinal: usize,
}

impl DeviceSelector {
    /// Creates a selector for a specific visible CUDA ordinal.
    ///
    /// # Example
    ///
    /// ```
    /// use runtime::DeviceSelector;
    ///
    /// let selector = DeviceSelector::new(0);
    /// assert_eq!(selector.ordinal(), 0);
    /// ```
    pub fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }

    /// Returns the ordinal this selector will open.
    ///
    /// # Example
    ///
    /// ```
    /// use runtime::DeviceSelector;
    ///
    /// let selector = DeviceSelector::new(2);
    /// assert_eq!(selector.ordinal(), 2);
    /// ```
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Creates a CUDA context for the selected ordinal.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::DeviceSelector;
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = DeviceSelector::new(0).select()?;
    /// assert_eq!(context.ordinal(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn select(&self) -> Result<CudaContext> {
        self.validate()?;
        CudaContext::new(self.ordinal)
    }

    /// Validates that the selected ordinal exists.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::DeviceSelector;
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// DeviceSelector::new(0).validate()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn validate(&self) -> Result<()> {
        let count = CudarcContext::device_count().w()? as usize;
        if self.ordinal >= count {
            candle_core::bail!(
                "CUDA device ordinal {} is out of range; {count} device(s) visible",
                self.ordinal
            );
        }
        Ok(())
    }
}

impl Default for DeviceSelector {
    fn default() -> Self {
        Self::new(0)
    }
}
