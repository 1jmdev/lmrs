use thiserror::Error;

/// Terminal reason for a sequence.
///
/// # Example
///
/// ```
/// use engine::FinishReason;
///
/// assert_eq!(FinishReason::Length.to_string(), "length");
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    /// The configured end-of-sequence token was sampled.
    Eos,
    /// The sequence reached its generated-token limit.
    Length,
    /// The sequence was stopped by the caller.
    Stopped,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eos => f.write_str("eos"),
            Self::Length => f.write_str("length"),
            Self::Stopped => f.write_str("stopped"),
        }
    }
}

/// Lifecycle state for a single autoregressive sequence.
///
/// States are intentionally CUDA-free: the scheduler and executor depend on
/// these transitions before any worker code can safely run.
///
/// # Example
///
/// ```
/// use engine::SequenceStatus;
///
/// assert!(SequenceStatus::Waiting.can_prefill());
/// assert!(!SequenceStatus::Finished.is_runnable());
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SequenceStatus {
    /// Prompt tokens have not been forwarded yet.
    Waiting,
    /// The sequence has cache state and can decode pending generated tokens.
    Running,
    /// Cache blocks are unavailable and the sequence must be resumed later.
    Swapped,
    /// Generation ended normally.
    Finished,
    /// Generation ended because the caller cancelled it.
    Aborted,
}

impl SequenceStatus {
    /// Returns whether this state can enter prompt prefill.
    pub const fn can_prefill(self) -> bool {
        matches!(self, Self::Waiting)
    }

    /// Returns whether this state can run on a worker.
    pub const fn is_runnable(self) -> bool {
        matches!(self, Self::Waiting | Self::Running)
    }

    /// Returns whether this state is terminal.
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Finished | Self::Aborted)
    }
}

/// Error raised when a status transition would violate the state machine.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum SequenceStatusError {
    /// The requested transition is not valid.
    #[error("invalid sequence status transition from {from:?} to {to:?}")]
    InvalidTransition {
        /// Current state.
        from: SequenceStatus,
        /// Requested state.
        to: SequenceStatus,
    },
}

/// Validates a lifecycle transition.
///
/// # Example
///
/// ```
/// use engine::SequenceStatus;
/// use engine::sequence::status::validate_transition;
///
/// validate_transition(SequenceStatus::Waiting, SequenceStatus::Running).unwrap();
/// assert!(validate_transition(SequenceStatus::Finished, SequenceStatus::Running).is_err());
/// ```
pub fn validate_transition(
    from: SequenceStatus,
    to: SequenceStatus,
) -> Result<(), SequenceStatusError> {
    let valid = matches!(
        (from, to),
        (SequenceStatus::Waiting, SequenceStatus::Running)
            | (SequenceStatus::Waiting, SequenceStatus::Finished)
            | (SequenceStatus::Waiting, SequenceStatus::Aborted)
            | (SequenceStatus::Running, SequenceStatus::Swapped)
            | (SequenceStatus::Running, SequenceStatus::Finished)
            | (SequenceStatus::Running, SequenceStatus::Aborted)
            | (SequenceStatus::Swapped, SequenceStatus::Running)
            | (SequenceStatus::Swapped, SequenceStatus::Aborted)
    ) || from == to;

    if valid {
        Ok(())
    } else {
        Err(SequenceStatusError::InvalidTransition { from, to })
    }
}
