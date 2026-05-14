use cache::SequenceId;
use sampling::SampleOutput;
use thiserror::Error;

use super::status::{FinishReason, SequenceStatus, SequenceStatusError, validate_transition};

/// Errors returned by sequence state operations.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SequenceError {
    /// A prompt must have at least one token so decode has a previous token.
    #[error("sequence prompt cannot be empty")]
    EmptyPrompt,
    /// The requested status transition is invalid.
    #[error(transparent)]
    Status(#[from] SequenceStatusError),
    /// The caller tried to process more tokens than the sequence contains.
    #[error("processed token count {processed} exceeds total token count {total}")]
    ProcessedPastEnd {
        /// Requested processed length.
        processed: usize,
        /// Available logical tokens.
        total: usize,
    },
    /// The sequence cannot be advanced because it is terminal.
    #[error("sequence {0} is terminal")]
    Terminal(u64),
}

/// Token buffer and lifecycle state for one autoregressive request.
///
/// `processed_len` tracks tokens already forwarded into model/cache state. A
/// generated token is appended before it is decoded; the next scheduler pass
/// forwards that single pending token with `start_pos = processed_len`.
///
/// # Example
///
/// ```
/// use cache::SequenceId;
/// use engine::{Sequence, SequenceStatus};
/// use sampling::SampleOutput;
///
/// let mut seq = Sequence::new(SequenceId::new(7), vec![10, 11], 2, Some(99)).unwrap();
/// assert_eq!(seq.status(), SequenceStatus::Waiting);
/// seq.mark_processed(2).unwrap();
/// seq.append_sample(SampleOutput::new(12, 0.0, 4.0)).unwrap();
/// assert_eq!(seq.pending_tokens(), &[12]);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Sequence {
    id: SequenceId,
    prompt_tokens: Vec<u32>,
    generated: Vec<SampleOutput>,
    max_new_tokens: usize,
    eos_token: Option<u32>,
    processed_len: usize,
    status: SequenceStatus,
    finish_reason: Option<FinishReason>,
}

impl Sequence {
    /// Creates a waiting sequence from prompt tokens and generation limits.
    pub fn new(
        id: SequenceId,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        eos_token: Option<u32>,
    ) -> Result<Self, SequenceError> {
        if prompt_tokens.is_empty() {
            return Err(SequenceError::EmptyPrompt);
        }
        Ok(Self {
            id,
            prompt_tokens,
            generated: Vec::new(),
            max_new_tokens,
            eos_token,
            processed_len: 0,
            status: SequenceStatus::Waiting,
            finish_reason: None,
        })
    }

    /// Returns the stable cache/engine sequence id.
    pub const fn id(&self) -> SequenceId {
        self.id
    }

    /// Returns the current lifecycle state.
    pub const fn status(&self) -> SequenceStatus {
        self.status
    }

    /// Returns the terminal reason, if one is available.
    pub const fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason
    }

    /// Returns prompt tokens.
    pub fn prompt_tokens(&self) -> &[u32] {
        &self.prompt_tokens
    }

    /// Returns sampled outputs.
    pub fn generated(&self) -> &[SampleOutput] {
        &self.generated
    }

    /// Returns generated token ids.
    pub fn generated_token_ids(&self) -> Vec<u32> {
        self.generated.iter().map(SampleOutput::token_id).collect()
    }

    /// Returns the logical number of prompt plus generated tokens.
    pub fn total_len(&self) -> usize {
        self.prompt_tokens.len() + self.generated.len()
    }

    /// Returns the number of tokens already forwarded through the model.
    pub const fn processed_len(&self) -> usize {
        self.processed_len
    }

    /// Returns tokens waiting to be forwarded.
    pub fn pending_tokens(&self) -> Vec<u32> {
        self.token_slice(self.processed_len, self.total_len())
    }

    /// Returns whether this sequence has unprocessed model input.
    pub fn has_pending_model_input(&self) -> bool {
        self.status.is_runnable() && self.processed_len < self.total_len()
    }

    /// Moves the sequence to swapped state.
    pub fn swap_out(&mut self) -> Result<(), SequenceError> {
        self.set_status(SequenceStatus::Swapped)
    }

    /// Moves the sequence from swapped back to running.
    pub fn resume(&mut self) -> Result<(), SequenceError> {
        self.set_status(SequenceStatus::Running)
    }

    /// Marks a number of newly forwarded tokens as processed.
    pub fn mark_processed(&mut self, token_count: usize) -> Result<(), SequenceError> {
        if self.status.is_terminal() {
            return Err(SequenceError::Terminal(self.id.get()));
        }
        let processed = self.processed_len + token_count;
        let total = self.total_len();
        if processed > total {
            return Err(SequenceError::ProcessedPastEnd { processed, total });
        }
        self.processed_len = processed;
        if self.status == SequenceStatus::Waiting {
            self.set_status(SequenceStatus::Running)?;
        }
        Ok(())
    }

    /// Appends a sampled token and finalizes the sequence when stop criteria hit.
    pub fn append_sample(&mut self, sample: SampleOutput) -> Result<(), SequenceError> {
        if self.status.is_terminal() {
            return Err(SequenceError::Terminal(self.id.get()));
        }
        self.generated.push(sample);
        if self.eos_token == Some(sample.token_id()) {
            self.finish(FinishReason::Eos)?;
        } else if self.generated.len() >= self.max_new_tokens {
            self.finish(FinishReason::Length)?;
        }
        Ok(())
    }

    /// Stops the sequence with a caller-visible stop reason.
    pub fn stop(&mut self) -> Result<(), SequenceError> {
        self.finish(FinishReason::Stopped)
    }

    /// Aborts the sequence without a normal finish reason.
    pub fn abort(&mut self) -> Result<(), SequenceError> {
        self.finish_reason = None;
        self.set_status(SequenceStatus::Aborted)
    }

    fn finish(&mut self, reason: FinishReason) -> Result<(), SequenceError> {
        self.finish_reason = Some(reason);
        self.set_status(SequenceStatus::Finished)
    }

    fn set_status(&mut self, status: SequenceStatus) -> Result<(), SequenceError> {
        validate_transition(self.status, status)?;
        self.status = status;
        Ok(())
    }

    pub(crate) fn token_slice(&self, start: usize, end: usize) -> Vec<u32> {
        (start..end)
            .map(|index| {
                if index < self.prompt_tokens.len() {
                    self.prompt_tokens[index]
                } else {
                    self.generated[index - self.prompt_tokens.len()].token_id()
                }
            })
            .collect()
    }
}
