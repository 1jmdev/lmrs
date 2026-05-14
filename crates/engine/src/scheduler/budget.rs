/// Limits applied when selecting a model execution batch.
///
/// # Example
///
/// ```
/// use engine::SchedulerBudget;
///
/// let budget = SchedulerBudget::new(8, 128);
/// assert!(budget.can_add(2, 16));
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulerBudget {
    max_sequences: usize,
    max_tokens: usize,
}

impl SchedulerBudget {
    /// Creates a scheduler budget.
    pub const fn new(max_sequences: usize, max_tokens: usize) -> Self {
        Self {
            max_sequences,
            max_tokens,
        }
    }

    /// Returns maximum sequence rows per batch.
    pub const fn max_sequences(self) -> usize {
        self.max_sequences
    }

    /// Returns maximum tokens per batch.
    pub const fn max_tokens(self) -> usize {
        self.max_tokens
    }

    /// Returns whether adding `sequences` and `tokens` remains in budget.
    pub const fn can_add(self, sequences: usize, tokens: usize) -> bool {
        sequences <= self.max_sequences && tokens <= self.max_tokens
    }
}

impl Default for SchedulerBudget {
    fn default() -> Self {
        Self::new(32, 4096)
    }
}
