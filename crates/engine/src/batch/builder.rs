use cache::SequenceId;
use thiserror::Error;

use crate::sequence::{Sequence, SequenceStatus};

use super::{PaddedBatch, VarLenBatch};

/// Batch execution mode.
///
/// # Example
///
/// ```
/// use engine::BatchMode;
///
/// assert!(BatchMode::Prefill.is_prefill());
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchMode {
    /// Runs prompt tokens for sequences with no cache state.
    Prefill,
    /// Runs one pending token per sequence with existing cache state.
    Decode,
}

impl BatchMode {
    /// Returns whether this is the prompt prefill path.
    pub const fn is_prefill(self) -> bool {
        matches!(self, Self::Prefill)
    }
}

/// One sequence row inside an execution batch.
///
/// # Example
///
/// ```
/// use cache::SequenceId;
/// use engine::{BatchEntry, BatchMode};
///
/// let entry = BatchEntry::new(SequenceId::new(1), BatchMode::Decode, vec![7], 3);
/// assert_eq!(entry.start_pos(), 3);
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchEntry {
    sequence_id: SequenceId,
    mode: BatchMode,
    tokens: Vec<u32>,
    start_pos: usize,
}

impl BatchEntry {
    /// Creates a batch entry.
    pub fn new(
        sequence_id: SequenceId,
        mode: BatchMode,
        tokens: Vec<u32>,
        start_pos: usize,
    ) -> Self {
        Self {
            sequence_id,
            mode,
            tokens,
            start_pos,
        }
    }

    /// Returns sequence id.
    pub const fn sequence_id(&self) -> SequenceId {
        self.sequence_id
    }

    /// Returns execution mode.
    pub const fn mode(&self) -> BatchMode {
        self.mode
    }

    /// Returns input tokens.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Returns model start position.
    pub const fn start_pos(&self) -> usize {
        self.start_pos
    }
}

/// Complete CPU-side batch description.
///
/// It carries both padded and variable-length layouts so workers can choose the
/// correct tensorization path without rebuilding scheduler metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionBatch {
    entries: Vec<BatchEntry>,
    padded: PaddedBatch,
    varlen: VarLenBatch,
}

impl ExecutionBatch {
    /// Creates a batch from entries and materialized layouts.
    pub fn new(entries: Vec<BatchEntry>, padded: PaddedBatch, varlen: VarLenBatch) -> Self {
        Self {
            entries,
            padded,
            varlen,
        }
    }

    /// Returns batch entries.
    pub fn entries(&self) -> &[BatchEntry] {
        &self.entries
    }

    /// Returns padded layout.
    pub const fn padded(&self) -> &PaddedBatch {
        &self.padded
    }

    /// Returns variable-length layout.
    pub const fn varlen(&self) -> &VarLenBatch {
        &self.varlen
    }

    /// Returns whether no entries are present.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Errors returned by `BatchBuilder`.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum BatchBuildError {
    /// Waiting and running states cannot be mixed in the same model call.
    #[error("cannot mix prefill and decode entries in one batch")]
    MixedModes,
}

/// Builds execution batches from runnable sequence state.
///
/// # Example
///
/// ```
/// use cache::SequenceId;
/// use engine::{BatchBuilder, BatchMode, Sequence};
///
/// let seq = Sequence::new(SequenceId::new(1), vec![4, 5], 1, None).unwrap();
/// let batch = BatchBuilder::new(0).build(std::slice::from_ref(&seq)).unwrap();
/// assert_eq!(batch.entries()[0].mode(), BatchMode::Prefill);
/// ```
#[derive(Clone, Debug)]
pub struct BatchBuilder {
    pad_token: u32,
}

impl BatchBuilder {
    /// Creates a builder using `pad_token` for padded rows.
    pub const fn new(pad_token: u32) -> Self {
        Self { pad_token }
    }

    /// Builds a homogeneous prefill or decode batch.
    pub fn build(&self, sequences: &[Sequence]) -> Result<ExecutionBatch, BatchBuildError> {
        let mut mode = None;
        let mut entries = Vec::new();
        for seq in sequences.iter().filter(|seq| seq.has_pending_model_input()) {
            let entry_mode = if seq.status() == SequenceStatus::Waiting {
                BatchMode::Prefill
            } else {
                BatchMode::Decode
            };
            if mode.is_some_and(|current| current != entry_mode) {
                return Err(BatchBuildError::MixedModes);
            }
            mode = Some(entry_mode);
            entries.push(BatchEntry::new(
                seq.id(),
                entry_mode,
                seq.pending_tokens(),
                seq.processed_len(),
            ));
        }

        let max_len = entries
            .iter()
            .map(|entry| entry.tokens.len())
            .max()
            .unwrap_or(0);
        let mut padded_rows = Vec::with_capacity(entries.len());
        let mut lengths = Vec::with_capacity(entries.len());
        let mut varlen_tokens = Vec::new();
        let mut cu_seqlens = Vec::with_capacity(entries.len() + 1);
        cu_seqlens.push(0);
        for entry in &entries {
            let mut row = entry.tokens.clone();
            lengths.push(row.len());
            varlen_tokens.extend(row.iter().copied());
            row.resize(max_len, self.pad_token);
            padded_rows.push(row);
            cu_seqlens.push(varlen_tokens.len());
        }

        Ok(ExecutionBatch::new(
            entries,
            PaddedBatch::new(padded_rows, lengths),
            VarLenBatch::new(varlen_tokens, cu_seqlens),
        ))
    }
}
