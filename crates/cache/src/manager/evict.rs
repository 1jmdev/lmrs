use crate::manager::SequenceId;

/// Selects a sequence for eviction from cache residency metadata.
///
/// The policy is deterministic: the least recently touched sequence has the
/// smallest tick and is selected first.
///
/// # Example
///
/// ```
/// use cache::{LruEvictionPolicy, SequenceId};
///
/// let mut policy = LruEvictionPolicy::default();
/// policy.touch(SequenceId::new(1));
/// policy.touch(SequenceId::new(2));
/// assert_eq!(policy.victim(), Some(SequenceId::new(1)));
/// policy.remove(SequenceId::new(1));
/// assert_eq!(policy.victim(), Some(SequenceId::new(2)));
/// ```
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LruEvictionPolicy {
    tick: u64,
    entries: Vec<(SequenceId, u64)>,
}

impl LruEvictionPolicy {
    /// Records access for `sequence`.
    pub fn touch(&mut self, sequence: SequenceId) {
        self.tick = self.tick.saturating_add(1);
        if let Some((_, seen)) = self.entries.iter_mut().find(|(id, _)| *id == sequence) {
            *seen = self.tick;
        } else {
            self.entries.push((sequence, self.tick));
        }
    }

    /// Removes a sequence from eviction tracking.
    pub fn remove(&mut self, sequence: SequenceId) {
        self.entries.retain(|(id, _)| *id != sequence);
    }

    /// Returns the least recently touched sequence.
    pub fn victim(&self) -> Option<SequenceId> {
        self.entries
            .iter()
            .min_by_key(|(_, seen)| *seen)
            .map(|(id, _)| *id)
    }

    /// Returns the number of tracked sequences.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether no sequences are tracked.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
