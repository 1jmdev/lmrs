use crate::batch::ExecutionBatch;
use crate::executor::StepOutput;

/// Messages sent from the engine to a worker thread.
///
/// # Example
///
/// ```
/// use engine::{EngineRequest, ExecutionBatch, PaddedBatch, VarLenBatch};
///
/// let batch = ExecutionBatch::new(vec![], PaddedBatch::new(vec![], vec![]), VarLenBatch::new(vec![], vec![0]));
/// let request = EngineRequest::RunBatch { step_id: 1, batch };
/// assert!(matches!(request, EngineRequest::RunBatch { step_id: 1, .. }));
/// ```
#[derive(Clone, Debug)]
pub enum EngineRequest {
    /// Runs one model step for a scheduled batch.
    RunBatch {
        /// Monotonic executor step id.
        step_id: u64,
        /// Batch metadata and CPU token rows.
        batch: ExecutionBatch,
    },
    /// Stops the worker loop after all earlier messages are processed.
    Shutdown,
}

/// Messages sent from a worker thread to the engine.
#[derive(Clone, Debug)]
pub enum WorkerResponse {
    /// Batch completed successfully.
    BatchComplete {
        /// Step id copied from the request.
        step_id: u64,
        /// One sampled token per sequence entry.
        outputs: Vec<StepOutput>,
    },
    /// Worker failed while running a step.
    Failed {
        /// Step id when available.
        step_id: Option<u64>,
        /// Human-readable error.
        error: String,
    },
    /// Worker acknowledged shutdown.
    ShutdownAck,
}
