use std::result::Result as StdResult;

use cudarc::driver::DriverError;

pub type Result<T> = StdResult<T, RuntimeError>;

/// Errors returned by the CUDA-only runtime crate.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// CUDA driver operation failed.
    #[error(transparent)]
    Cuda(#[from] DriverError),
    /// Runtime validation failed before issuing CUDA work.
    #[error("{0}")]
    Message(String),
}

impl RuntimeError {
    /// Creates a validation error with a static or formatted message.
    pub fn msg(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}
