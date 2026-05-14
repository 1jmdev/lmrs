use kernels::utils::CopyBlocksLaunch;
use thiserror::Error;

use crate::Tensor;

/// Kernel dispatch validation failures.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum DispatchError {
    /// Kernel dispatch requires compact source and destination tensors.
    #[error("kernel dispatch requires contiguous tensors")]
    NonContiguous,
    /// Source and destination logical byte lengths differ.
    #[error("source bytes {src_bytes} do not match destination bytes {dst_bytes}")]
    ByteLengthMismatch { src_bytes: usize, dst_bytes: usize },
}

/// Thin description of a copy kernel launch routed to `kernels`.
///
/// # Example
///
/// ```no_run
/// use candle_core::Device;
/// use tensor::{DType, KernelDispatch, Shape, Tensor};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::new_cuda(0)?;
/// let a = Tensor::empty(device.as_cuda_device()?, Shape::new([4]).unwrap(), DType::U8)?;
/// let b = Tensor::empty(device.as_cuda_device()?, Shape::new([4]).unwrap(), DType::U8)?;
/// let plan = KernelDispatch::copy_plan(&a, &b)?;
/// assert_eq!(plan.launch.block_bytes, 4);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CopyPlan {
    /// The kernels crate symbol selected for this operation.
    pub kernel: CopyKernel,
    /// Launch metadata understood by the kernels crate.
    pub launch: CopyBlocksLaunch,
}

/// Copy kernel symbol selected through the kernels crate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CopyKernel {
    /// Byte-wise block copy.
    BlocksU8,
}

/// Tensor-to-kernel dispatch helpers.
pub struct KernelDispatch;

impl KernelDispatch {
    /// Creates a block-copy launch plan for two contiguous tensors.
    pub fn copy_plan(src: &Tensor, dst: &Tensor) -> Result<CopyPlan, DispatchError> {
        if !src.is_contiguous() || !dst.is_contiguous() {
            return Err(DispatchError::NonContiguous);
        }
        let src_bytes = src.len_bytes();
        let dst_bytes = dst.len_bytes();
        if src_bytes != dst_bytes {
            return Err(DispatchError::ByteLengthMismatch {
                src_bytes,
                dst_bytes,
            });
        }
        Ok(CopyPlan {
            kernel: CopyKernel::BlocksU8,
            launch: CopyBlocksLaunch::new(src_bytes, 1),
        })
    }
}
