use cache::SequenceId;
use sampling::SampleOutput;

/// Sampled output for one sequence in a scheduler step.
///
/// # Example
///
/// ```
/// use cache::SequenceId;
/// use engine::StepOutput;
/// use sampling::SampleOutput;
///
/// let out = StepOutput::new(SequenceId::new(3), SampleOutput::new(9, 0.0, 1.0));
/// assert_eq!(out.sequence_id(), SequenceId::new(3));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepOutput {
    sequence_id: SequenceId,
    sample: SampleOutput,
}

impl StepOutput {
    /// Creates a step output.
    pub const fn new(sequence_id: SequenceId, sample: SampleOutput) -> Self {
        Self {
            sequence_id,
            sample,
        }
    }

    /// Returns sequence id.
    pub const fn sequence_id(self) -> SequenceId {
        self.sequence_id
    }

    /// Returns sampled token metadata.
    pub const fn sample(self) -> SampleOutput {
        self.sample
    }
}
