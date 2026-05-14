use std::result::Result as StdResult;

use candle_core::Error;

pub type Result<T> = StdResult<T, SamplingError>;

/// Errors returned by sampling and logits post-processing.
#[derive(Debug, thiserror::Error)]
pub enum SamplingError {
    /// Sampling configuration or probability validation failed.
    #[error("{0}")]
    Invalid(String),
    /// Temporary bridge while logits processors still use Candle tensors.
    #[error(transparent)]
    Candle(#[from] Error),
}

impl SamplingError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}
