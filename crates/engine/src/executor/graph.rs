use crate::batch::ExecutionBatch;

/// CUDA graph capture state for a stable batch shape.
///
/// Graph replay is deliberately explicit and disabled until the normal worker
/// path succeeds for a shape. This prevents the graph path from hiding scheduler
/// or cache-state bugs.
///
/// # Example
///
/// ```
/// use engine::{ExecutionBatch, GraphCaptureState, PaddedBatch, VarLenBatch};
///
/// let batch = ExecutionBatch::new(vec![], PaddedBatch::new(vec![], vec![]), VarLenBatch::new(vec![], vec![0]));
/// let mut state = GraphCaptureState::default();
/// assert!(!state.can_replay(&batch));
/// state.capture_shape(&batch);
/// assert!(state.can_replay(&batch));
/// ```
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GraphCaptureState {
    captured: Option<GraphReplayPlan>,
}

impl GraphCaptureState {
    /// Records a replayable shape after a successful non-graph execution.
    pub fn capture_shape(&mut self, batch: &ExecutionBatch) {
        self.captured = Some(GraphReplayPlan::from_batch(batch));
    }

    /// Returns whether a captured graph can replay this batch shape.
    pub fn can_replay(&self, batch: &ExecutionBatch) -> bool {
        self.captured
            .as_ref()
            .is_some_and(|plan| plan.matches(batch))
    }

    /// Returns the captured plan.
    pub const fn captured(&self) -> Option<&GraphReplayPlan> {
        self.captured.as_ref()
    }
}

/// Stable shape metadata used to decide graph replay eligibility.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphReplayPlan {
    rows: usize,
    max_len: usize,
    total_tokens: usize,
}

impl GraphReplayPlan {
    /// Creates a replay plan from an execution batch.
    pub fn from_batch(batch: &ExecutionBatch) -> Self {
        Self {
            rows: batch.entries().len(),
            max_len: batch.padded().max_len(),
            total_tokens: batch.varlen().tokens().len(),
        }
    }

    /// Returns whether the batch has the same replay shape.
    pub fn matches(&self, batch: &ExecutionBatch) -> bool {
        self == &Self::from_batch(batch)
    }
}
