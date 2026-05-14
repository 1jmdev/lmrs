use crate::batch::BatchBuildError;
use crate::scheduler::Scheduler;
use crate::sequence::SequenceError;
use crate::worker::{EngineRequest, WorkerError, WorkerHandle, WorkerResponse};
use thiserror::Error;

use super::{GraphCaptureState, StepOutput};

/// Errors returned by executor orchestration.
#[derive(Debug, Error)]
pub enum ExecutorError {
    /// Worker failed or disconnected.
    #[error(transparent)]
    Worker(#[from] WorkerError),
    /// Sequence state update failed.
    #[error(transparent)]
    Sequence(#[from] SequenceError),
    /// Batch construction failed.
    #[error(transparent)]
    Batch(#[from] BatchBuildError),
    /// Worker response did not match the requested step.
    #[error("unexpected worker response for step {expected}")]
    UnexpectedResponse {
        /// Expected step id.
        expected: u64,
    },
}

/// Coordinates scheduler, worker execution, and output application.
///
/// # Example
///
/// ```no_run
/// # use engine::{EngineExecutor, Scheduler, SchedulerBudget, WorkerHandle};
/// # fn build(handle: WorkerHandle) {
/// let scheduler = Scheduler::new(SchedulerBudget::new(4, 32));
/// let _executor = EngineExecutor::new(scheduler, handle);
/// # }
/// ```
pub struct EngineExecutor {
    scheduler: Scheduler,
    worker: WorkerHandle,
    graph: GraphCaptureState,
    next_step_id: u64,
}

impl EngineExecutor {
    /// Creates an executor from a scheduler and threaded worker handle.
    pub fn new(scheduler: Scheduler, worker: WorkerHandle) -> Self {
        Self {
            scheduler,
            worker,
            graph: GraphCaptureState::default(),
            next_step_id: 0,
        }
    }

    /// Returns scheduler state.
    pub const fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Returns mutable scheduler state.
    pub fn scheduler_mut(&mut self) -> &mut Scheduler {
        &mut self.scheduler
    }

    /// Runs one prefill or decode step, applying outputs back to sequences.
    pub fn step(&mut self) -> Result<Option<Vec<StepOutput>>, ExecutorError> {
        let Some(schedule) = self.scheduler.schedule()? else {
            return Ok(None);
        };
        let batch = schedule.into_batch();
        let step_id = self.next_step_id;
        self.next_step_id += 1;
        self.worker.send(EngineRequest::RunBatch {
            step_id,
            batch: batch.clone(),
        })?;
        let outputs = match self.worker.recv()? {
            WorkerResponse::BatchComplete {
                step_id: response_step,
                outputs,
            } if response_step == step_id => outputs,
            WorkerResponse::Failed { error, .. } => return Err(WorkerError::Failed(error).into()),
            _ => return Err(ExecutorError::UnexpectedResponse { expected: step_id }),
        };
        self.scheduler.mark_batch_processed(&batch)?;
        for output in &outputs {
            if let Some(seq) = self.scheduler.sequence_mut(output.sequence_id()) {
                seq.append_sample(output.sample())?;
            }
        }
        self.graph.capture_shape(&batch);
        self.scheduler.drain_finished();
        Ok(Some(outputs))
    }

    /// Runs until no sequence has pending model input.
    pub fn run_until_idle(&mut self) -> Result<Vec<StepOutput>, ExecutorError> {
        let mut all = Vec::new();
        while let Some(outputs) = self.step()? {
            all.extend(outputs);
        }
        Ok(all)
    }

    /// Shuts down the owned worker thread.
    pub fn shutdown(self) -> Result<(), WorkerError> {
        self.worker.shutdown()
    }
}
