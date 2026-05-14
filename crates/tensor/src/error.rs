use std::result::Result as StdResult;

use cudarc::driver::DriverError;

use crate::CopyError;

pub type Result<T> = StdResult<T, TensorError>;

/// Errors returned by the CUDA-only tensor crate.
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    /// CUDA driver operation failed.
    #[error(transparent)]
    Cuda(#[from] DriverError),
    /// Host/device copy validation failed.
    #[error(transparent)]
    Copy(#[from] CopyError),
    /// Tensor dtype did not match the kernel contract.
    #[error("expected dtype {expected}, got {actual}")]
    DTypeMismatch {
        expected: &'static str,
        actual: &'static str,
    },
    /// Tensor shape did not match the kernel contract.
    #[error("{0}")]
    ShapeMismatch(String),
    /// Kernel requires contiguous tensor storage.
    #[error("kernel requires contiguous tensor storage")]
    NonContiguous,
    /// Kernel argument is invalid.
    #[error("{0}")]
    InvalidArgument(String),
}
