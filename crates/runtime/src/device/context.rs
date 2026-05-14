use std::sync::Arc;

use candle_core::backend::BackendDevice;
use candle_core::cuda_backend::CudaDevice as CandleCudaDevice;
use candle_core::cuda_backend::WrapErr;
use candle_core::{Device, DeviceLocation, Result};
use cudarc::driver::CudaContext as CudarcContext;

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
/// # fn main() -> candle_core::Result<()> {
/// let context = CudaContext::new(0)?;
/// assert_eq!(context.ordinal(), 0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct CudaContext {
    context: Arc<CudarcContext>,
    candle_device: Option<CandleCudaDevice>,
}

impl CudaContext {
    /// Creates a CUDA context for a device ordinal.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::CudaContext;
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// assert_eq!(context.ordinal(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(ordinal: usize) -> Result<Self> {
        let context = CudarcContext::new(ordinal).w()?;
        Ok(Self {
            context,
            candle_device: None,
        })
    }

    /// Wraps an existing CUDA device handle.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use candle_core::Device;
    /// use runtime::CudaContext;
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let device = Device::new_cuda(0)?;
    /// let cuda = device.as_cuda_device()?.clone();
    /// let context = CudaContext::from_candle(cuda)?;
    /// assert_eq!(context.ordinal(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_candle(device: CandleCudaDevice) -> Result<Self> {
        let ordinal = match device.location() {
            DeviceLocation::Cuda { gpu_id } => gpu_id,
            _ => unreachable!("CudaDevice returned a non-CUDA location"),
        };
        let context = CudarcContext::new(ordinal).w()?;
        Ok(Self {
            context,
            candle_device: Some(device),
        })
    }

    /// Creates a CUDA context from a generic device.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use candle_core::Device;
    /// use runtime::CudaContext;
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let device = Device::new_cuda(0)?;
    /// let context = CudaContext::from_device(&device)?;
    /// assert_eq!(context.ordinal(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_device(device: &Device) -> Result<Self> {
        Self::from_candle(device.as_cuda_device()?.clone())
    }

    /// Returns the underlying CUDA device handle.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::CudaContext;
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// assert!(context.candle().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn candle(&self) -> Option<&CandleCudaDevice> {
        self.candle_device.as_ref()
    }

    /// Returns the direct cudarc CUDA context.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::CudaContext;
    ///
    /// # fn main() -> candle_core::Result<()> {
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
    /// # fn main() -> candle_core::Result<()> {
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
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// assert_eq!(context.ordinal(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn ordinal(&self) -> usize {
        self.context.ordinal()
    }
}
