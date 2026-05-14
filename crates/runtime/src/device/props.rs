use candle_core::Result;
use candle_core::cuda_backend::WrapErr;
use cudarc::driver::sys::CUdevice_attribute_enum;

use super::CudaContext;

/// CUDA device properties used for scheduling and kernel dispatch.
///
/// # Example
///
/// ```
/// use runtime::DeviceProps;
///
/// let props = DeviceProps {
///     ordinal: 0,
///     name: "Example GPU".to_string(),
///     sm_count: 132,
///     compute_capability_major: 8,
///     compute_capability_minor: 9,
///     total_memory_bytes: 80 * 1024 * 1024 * 1024,
/// };
///
/// assert_eq!(props.sm_arch(), 89);
/// ```
#[derive(Clone, Debug)]
pub struct DeviceProps {
    /// Visible CUDA ordinal.
    pub ordinal: usize,
    /// CUDA device name.
    pub name: String,
    /// Number of streaming multiprocessors.
    pub sm_count: usize,
    /// Major compute capability version.
    pub compute_capability_major: i32,
    /// Minor compute capability version.
    pub compute_capability_minor: i32,
    /// Total global memory in bytes.
    pub total_memory_bytes: usize,
}

impl DeviceProps {
    /// Queries CUDA device properties for `context`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, DeviceProps};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let props = DeviceProps::query(&context)?;
    /// assert!(props.sm_count > 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn query(context: &CudaContext) -> Result<Self> {
        let device = context.cudarc();
        let name = device.name().w()?;
        let sm_count = device
            .attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .w()? as usize;
        let compute_capability_major = device
            .attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .w()?;
        let compute_capability_minor = device
            .attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .w()?;
        let total_memory_bytes = device.total_mem().w()?;

        Ok(Self {
            ordinal: device.ordinal(),
            name,
            sm_count,
            compute_capability_major,
            compute_capability_minor,
            total_memory_bytes,
        })
    }

    /// Returns the packed SM architecture number.
    ///
    /// Compute capability 8.9 returns `89`; compute capability 10.0 returns
    /// `100`.
    ///
    /// # Example
    ///
    /// ```
    /// use runtime::DeviceProps;
    ///
    /// let props = DeviceProps {
    ///     ordinal: 0,
    ///     name: "Example".to_string(),
    ///     sm_count: 1,
    ///     compute_capability_major: 8,
    ///     compute_capability_minor: 9,
    ///     total_memory_bytes: 1,
    /// };
    /// assert_eq!(props.sm_arch(), 89);
    /// ```
    pub fn sm_arch(&self) -> i32 {
        self.compute_capability_major * 10 + self.compute_capability_minor
    }
}
