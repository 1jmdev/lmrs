use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::{self, JoinHandle};

use candle_core::{Device, IndexOp, Tensor};
use model::Model;
use sampling::Sampler;
use thiserror::Error;

use crate::batch::{BatchEntry, ExecutionBatch};
use crate::executor::StepOutput;

use super::{EngineRequest, WorkerResponse};

/// Errors returned by worker execution.
#[derive(Debug, Error)]
pub enum WorkerError {
    /// Candle model/tensor operation failed.
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    /// A worker channel disconnected.
    #[error("worker channel disconnected")]
    ChannelDisconnected,
    /// The worker returned an error response.
    #[error("worker failed: {0}")]
    Failed(String),
}

/// Per-device worker owning a model instance and sampler.
///
/// # Example
///
/// ```
/// use engine::Worker;
///
/// struct Dummy;
/// impl model::Model for Dummy {
///     fn forward(&mut self, input: &candle_core::Tensor, _start_pos: usize) -> candle_core::Result<candle_core::Tensor> {
///         Ok(input.clone())
///     }
///     fn metadata(&self) -> model::ModelMetadata {
///         model::ModelMetadata { model_type: "dummy".into(), vocab_size: 1, hidden_size: 1, num_hidden_layers: 1 }
///     }
/// }
/// ```
pub struct Worker<M> {
    model: M,
    sampler: Sampler,
    device: Device,
}

impl<M> Worker<M>
where
    M: Model,
{
    /// Creates a worker over an owned model, sampler, and device.
    pub fn new(model: M, sampler: Sampler, device: Device) -> Self {
        Self {
            model,
            sampler,
            device,
        }
    }

    /// Runs one scheduled batch and samples one next token per entry.
    pub fn run_batch(&mut self, batch: &ExecutionBatch) -> Result<Vec<StepOutput>, WorkerError> {
        let mut outputs = Vec::with_capacity(batch.entries().len());
        for entry in batch.entries() {
            outputs.push(self.run_entry(entry)?);
        }
        Ok(outputs)
    }

    fn run_entry(&mut self, entry: &BatchEntry) -> Result<StepOutput, WorkerError> {
        let input = Tensor::new(entry.tokens(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, entry.start_pos())?;
        let logits = last_token_logits(&logits)?;
        let sample = self.sampler.sample(&logits, &[])?;
        Ok(StepOutput::new(entry.sequence_id(), sample))
    }
}

/// Threaded worker handle used by the executor.
pub struct WorkerHandle {
    request_tx: Sender<EngineRequest>,
    response_rx: Receiver<WorkerResponse>,
    join: Option<JoinHandle<()>>,
}

impl WorkerHandle {
    /// Spawns a worker loop on its own OS thread.
    pub fn spawn<M>(mut worker: Worker<M>) -> Self
    where
        M: Model + Send + 'static,
    {
        let (request_tx, request_rx) = channel();
        let (response_tx, response_rx) = channel();
        let join = thread::spawn(move || worker_loop(&mut worker, request_rx, response_tx));
        Self {
            request_tx,
            response_rx,
            join: Some(join),
        }
    }

    /// Sends a request to the worker.
    pub fn send(&self, request: EngineRequest) -> Result<(), WorkerError> {
        self.request_tx
            .send(request)
            .map_err(|_| WorkerError::ChannelDisconnected)
    }

    /// Receives the next worker response.
    pub fn recv(&self) -> Result<WorkerResponse, WorkerError> {
        self.response_rx
            .recv()
            .map_err(|_| WorkerError::ChannelDisconnected)
    }

    /// Requests worker shutdown and joins the thread.
    pub fn shutdown(mut self) -> Result<(), WorkerError> {
        self.send(EngineRequest::Shutdown)?;
        match self.recv()? {
            WorkerResponse::ShutdownAck => {}
            WorkerResponse::Failed { error, .. } => return Err(WorkerError::Failed(error)),
            WorkerResponse::BatchComplete { .. } => {}
        }
        if let Some(join) = self.join.take() {
            join.join().map_err(|_| WorkerError::ChannelDisconnected)?;
        }
        Ok(())
    }
}

fn worker_loop<M>(
    worker: &mut Worker<M>,
    request_rx: Receiver<EngineRequest>,
    response_tx: Sender<WorkerResponse>,
) where
    M: Model,
{
    while let Ok(request) = request_rx.recv() {
        match request {
            EngineRequest::RunBatch { step_id, batch } => {
                let response = match worker.run_batch(&batch) {
                    Ok(outputs) => WorkerResponse::BatchComplete { step_id, outputs },
                    Err(error) => WorkerResponse::Failed {
                        step_id: Some(step_id),
                        error: error.to_string(),
                    },
                };
                if response_tx.send(response).is_err() {
                    return;
                }
            }
            EngineRequest::Shutdown => {
                let _ = response_tx.send(WorkerResponse::ShutdownAck);
                return;
            }
        }
    }
}

fn last_token_logits(logits: &Tensor) -> candle_core::Result<Tensor> {
    match logits.dims().len() {
        1 => Ok(logits.clone()),
        2 => logits.i(0),
        3 => logits.i((0, logits.dim(1)? - 1)),
        _ => logits.flatten_all(),
    }
}
