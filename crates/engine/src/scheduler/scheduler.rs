use std::collections::VecDeque;

use cache::SequenceId;

use crate::batch::{BatchBuilder, ExecutionBatch};
use crate::sequence::{Sequence, SequenceError, SequenceGroup};

use super::{SchedulePolicy, SchedulerBudget};

/// Batch and sequence ids selected by a scheduler pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduleResult {
    batch: ExecutionBatch,
    sequence_ids: Vec<SequenceId>,
}

impl ScheduleResult {
    /// Creates a schedule result.
    pub fn new(batch: ExecutionBatch, sequence_ids: Vec<SequenceId>) -> Self {
        Self {
            batch,
            sequence_ids,
        }
    }

    /// Returns the execution batch.
    pub const fn batch(&self) -> &ExecutionBatch {
        &self.batch
    }

    /// Consumes the result and returns the execution batch.
    pub fn into_batch(self) -> ExecutionBatch {
        self.batch
    }

    /// Returns selected sequence ids.
    pub fn sequence_ids(&self) -> &[SequenceId] {
        &self.sequence_ids
    }
}

/// Stateful request scheduler for prefill and decode steps.
///
/// The scheduler prioritizes prefill work over decode work to match the existing
/// single-engine behavior: new prompts run as full prompt tensors, then running
/// sequences decode one pending token per step.
///
/// # Example
///
/// ```
/// use cache::SequenceId;
/// use engine::{Scheduler, SchedulerBudget, Sequence, SequenceGroup};
///
/// let seq = Sequence::new(SequenceId::new(1), vec![1, 2], 1, None).unwrap();
/// let mut scheduler = Scheduler::new(SchedulerBudget::new(4, 16));
/// scheduler.add_group(SequenceGroup::new(1, vec![seq]).unwrap());
/// assert!(scheduler.schedule().unwrap().is_some());
/// ```
#[derive(Clone, Debug)]
pub struct Scheduler {
    groups: VecDeque<SequenceGroup>,
    budget: SchedulerBudget,
    policy: SchedulePolicy,
    builder: BatchBuilder,
}

impl Scheduler {
    /// Creates an FCFS scheduler.
    pub fn new(budget: SchedulerBudget) -> Self {
        Self {
            groups: VecDeque::new(),
            budget,
            policy: SchedulePolicy::Fcfs,
            builder: BatchBuilder::new(0),
        }
    }

    /// Sets the scheduling policy.
    pub fn with_policy(mut self, policy: SchedulePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Adds a sequence group.
    pub fn add_group(&mut self, group: SequenceGroup) {
        self.groups.push_back(group);
    }

    /// Returns immutable groups.
    pub fn groups(&self) -> &VecDeque<SequenceGroup> {
        &self.groups
    }

    /// Returns mutable groups.
    pub fn groups_mut(&mut self) -> &mut VecDeque<SequenceGroup> {
        &mut self.groups
    }

    /// Finds a mutable sequence by id.
    pub fn sequence_mut(&mut self, id: SequenceId) -> Option<&mut Sequence> {
        self.groups
            .iter_mut()
            .find_map(|group| group.sequence_mut(id))
    }

    /// Marks selected batch inputs as processed after a worker forward pass.
    pub fn mark_batch_processed(&mut self, batch: &ExecutionBatch) -> Result<(), SequenceError> {
        for entry in batch.entries() {
            if let Some(seq) = self.sequence_mut(entry.sequence_id()) {
                seq.mark_processed(entry.tokens().len())?;
            }
        }
        Ok(())
    }

    /// Removes finished groups from the active queue and returns them.
    pub fn drain_finished(&mut self) -> Vec<SequenceGroup> {
        let mut finished = Vec::new();
        let mut active = VecDeque::with_capacity(self.groups.len());
        while let Some(group) = self.groups.pop_front() {
            if group.is_finished() {
                finished.push(group);
            } else {
                active.push_back(group);
            }
        }
        self.groups = active;
        finished
    }

    /// Selects the next homogeneous prefill or decode batch.
    pub fn schedule(
        &self,
    ) -> Result<Option<ScheduleResult>, crate::batch::builder::BatchBuildError> {
        let ordered = self.ordered_groups();
        let prefill = self.collect_candidates(&ordered, true);
        let selected = if prefill.is_empty() {
            self.collect_candidates(&ordered, false)
        } else {
            prefill
        };
        if selected.is_empty() {
            return Ok(None);
        }
        let ids = selected.iter().map(|seq| seq.id()).collect();
        let batch = self.builder.build(&selected)?;
        Ok(Some(ScheduleResult::new(batch, ids)))
    }

    fn ordered_groups(&self) -> Vec<&SequenceGroup> {
        let mut groups: Vec<_> = self.groups.iter().collect();
        if self.policy == SchedulePolicy::Priority {
            groups.sort_by_key(|group| std::cmp::Reverse(group.priority()));
        }
        groups
    }

    fn collect_candidates<'a>(&self, groups: &[&'a SequenceGroup], prefill: bool) -> Vec<Sequence> {
        let mut selected = Vec::new();
        let mut tokens = 0;
        for group in groups {
            for seq in group.sequences() {
                if !seq.has_pending_model_input() || (seq.status().can_prefill() != prefill) {
                    continue;
                }
                let candidate_tokens = seq.pending_tokens().len();
                let next_sequences = selected.len() + 1;
                let next_tokens = tokens + candidate_tokens;
                if !self.budget.can_add(next_sequences, next_tokens) {
                    return selected;
                }
                selected.push(seq.clone());
                tokens = next_tokens;
            }
        }
        selected
    }
}
