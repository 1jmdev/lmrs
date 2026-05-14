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
}
