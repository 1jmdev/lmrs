use std::result::Result as StdResult;

use runtime::RuntimeError;
use tensor::{ShapeError, TensorError};

/// Sampling operation result alias.
pub type Result<T> = StdResult<T, SamplingError>;

/// Errors returned by sampling and logits post-processing.
///
/// # Example
///
/// ```
/// use sampling::SamplingError;
///
/// let err = SamplingError::Invalid("temperature must be > 0".to_string());
/// assert!(err.to_string().contains("temperature"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum SamplingError {
    /// Sampling configuration or probability validation failed.
    #[error("{0}")]
    Invalid(String),
    /// CUDA tensor operation failed.
    #[error(transparent)]
    Tensor(#[from] TensorError),
    /// CUDA runtime operation failed.
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    /// Tensor shape construction failed.
    #[error(transparent)]
    Shape(#[from] ShapeError),
}

impl SamplingError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}
