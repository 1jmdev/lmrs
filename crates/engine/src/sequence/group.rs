use cache::SequenceId;

use super::Sequence;

/// A scheduling unit containing one or more related sequences.
///
/// Beam search and parallel sampling can attach multiple sequences to a group;
/// simple greedy generation uses exactly one sequence.
///
/// # Example
///
/// ```
/// use cache::SequenceId;
/// use engine::{Sequence, SequenceGroup};
///
/// let seq = Sequence::new(SequenceId::new(1), vec![4, 5], 1, None).unwrap();
/// let group = SequenceGroup::new(9, vec![seq]).unwrap();
/// assert_eq!(group.id(), 9);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceGroup {
    id: u64,
    sequences: Vec<Sequence>,
    priority: i64,
}

impl SequenceGroup {
    /// Creates a group with default priority zero.
    pub fn new(id: u64, sequences: Vec<Sequence>) -> Result<Self, &'static str> {
        if sequences.is_empty() {
            return Err("sequence group cannot be empty");
        }
        Ok(Self {
            id,
            sequences,
            priority: 0,
        })
    }

    /// Sets scheduler priority. Higher values run first under priority policy.
    pub fn with_priority(mut self, priority: i64) -> Self {
        self.priority = priority;
        self
    }

    /// Returns the group id.
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Returns group priority.
    pub const fn priority(&self) -> i64 {
        self.priority
    }

    /// Returns immutable sequences.
    pub fn sequences(&self) -> &[Sequence] {
        &self.sequences
    }

    /// Returns mutable sequences.
    pub fn sequences_mut(&mut self) -> &mut [Sequence] {
        &mut self.sequences
    }

    /// Returns whether all child sequences are terminal.
    pub fn is_finished(&self) -> bool {
        self.sequences.iter().all(|seq| seq.status().is_terminal())
    }

    /// Finds a mutable sequence by id.
    pub fn sequence_mut(&mut self, id: SequenceId) -> Option<&mut Sequence> {
        self.sequences.iter_mut().find(|seq| seq.id() == id)
    }
}
