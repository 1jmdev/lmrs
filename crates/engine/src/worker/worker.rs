use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::{self, JoinHandle};

use cache::CacheManager;
use model::Model;
use ops::{narrow_dim1, reshape};
use runtime::CudaContext;
use sampling::{Sampler, SamplingError};
use tensor::{DType, Shape, Tensor, copy_h2d};
use thiserror::Error;

use crate::batch::{BatchEntry, ExecutionBatch};
use crate::executor::StepOutput;

use super::{EngineRequest, WorkerResponse};

/// Errors returned by worker execution.
///
/// # Example
///
/// ```
/// use engine::WorkerError;
///
/// assert!(WorkerError::Failed("test".to_string()).to_string().contains("test"));
/// ```
#[derive(Debug, Error)]
pub enum WorkerError {
    /// Model/tensor operation failed.
    #[error(transparent)]
    Tensor(#[from] tensor::TensorError),
    /// Sampling failed after model execution.
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    /// A worker channel disconnected.
    #[error("worker channel disconnected")]
    ChannelDisconnected,
    /// The worker returned an error response.
    #[error("worker failed: {0}")]
    Failed(String),
}

/// Per-device worker owning a model instance and cache manager.
///
/// # Example
///
/// ```
/// use cache::{BlockPool, CacheManager, SlotLayout};
/// use engine::Worker;
///
/// let pool = BlockPool::new(2, SlotLayout::new(4, 8, 1)).unwrap();
/// let manager = CacheManager::new(pool);
/// let _ = manager.sequence_count();
/// ```
pub struct Worker<M> {
    model: M,
    sampler: Sampler,
    cache: CacheManager,
    context: CudaContext,
}

impl<M> Worker<M>
where
    M: Model,
{
    /// Creates a worker over an owned model, sampler, cache manager, and CUDA context.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cache::{BlockPool, CacheManager, SlotLayout};
    /// use engine::Worker;
    /// use runtime::CudaContext;
    /// use sampling::Sampler;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let context = CudaContext::new(0)?;
    /// let pool = BlockPool::new(2, SlotLayout::new(4, 8, 1)).unwrap();
    /// let manager = CacheManager::new(pool);
    /// let sampler = Sampler::default();
    /// struct DummyModel;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        model: M,
        sampler: Sampler,
        cache: CacheManager,
        context: CudaContext,
    ) -> Self {
        Self {
            model,
            sampler,
            cache,
            context,
        }
    }

    /// Returns the cache manager owned by this worker.
    ///
    /// # Example
    ///
    /// ```
    /// use cache::{BlockPool, CacheManager, SlotLayout};
    /// use engine::Worker;
    ///
    /// let pool = BlockPool::new(2, SlotLayout::new(4, 8, 1)).unwrap();
    /// let manager = CacheManager::new(pool);
    /// ```
    pub const fn cache(&self) -> &CacheManager {
        &self.cache
    }

    /// Runs one scheduled batch and samples one next token per entry.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cache::{BlockPool, CacheManager, SlotLayout};
    /// # use engine::Worker;
    /// # use model::ModelMetadata;
    /// # use runtime::CudaContext;
    /// # use sampling::Sampler;
    /// # use tensor::Tensor;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let context = CudaContext::new(0)?;
    /// # let pool = BlockPool::new(2, SlotLayout::new(4, 8, 1)).unwrap();
    /// # let manager = CacheManager::new(pool);
    /// # struct DummyModel;
    /// # impl model::Model for DummyModel {
    /// #     fn forward(&mut self, _: &Tensor, _: usize) -> tensor::Result<Tensor> { panic!() }
    /// #     fn metadata(&self) -> ModelMetadata { ModelMetadata { model_type: "".into(), vocab_size: 1, hidden_size: 1, num_hidden_layers: 0 } }
    /// # }
    /// let mut worker = Worker::new(DummyModel, Sampler::default(), manager, context);
    /// # Ok(())
    /// # }
    /// ```
    pub fn run_batch(
        &mut self,
        batch: &ExecutionBatch,
    ) -> Result<Vec<StepOutput>, WorkerError> {
        let mut outputs = Vec::with_capacity(batch.entries().len());
        for entry in batch.entries() {
            outputs.push(self.run_entry(entry)?);
        }
        Ok(outputs)
    }

    fn run_entry(&mut self, entry: &BatchEntry) -> Result<StepOutput, WorkerError> {
        let tokens = entry.tokens();
        let shape = Shape::new([1, tokens.len()])
            .map_err(|err| tensor::TensorError::ShapeMismatch(err.to_string()))?;
        let input = copy_h2d(&self.context, shape, DType::I32, tokens)?;
        let logits = self.model.forward(&input, entry.start_pos())?;
        let logits = last_token_logits(&logits)?;
        let sample = self.sampler.sample(&logits, &[])?;
        Ok(StepOutput::new(entry.sequence_id(), sample))
    }
}

/// Threaded worker handle used by the executor.
///
/// # Example
///
/// ```
/// use engine::WorkerHandle;
/// ```
pub struct WorkerHandle {
    request_tx: Sender<EngineRequest>,
    response_rx: Receiver<WorkerResponse>,
    join: Option<JoinHandle<()>>,
}

impl WorkerHandle {
    /// Spawns a worker loop on its own OS thread.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cache::{BlockPool, CacheManager, SlotLayout};
    /// # use engine::{Worker, WorkerHandle};
    /// # use model::{Model, ModelMetadata};
    /// # use runtime::CudaContext;
    /// # use sampling::Sampler;
    /// # use tensor::Tensor;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let context = CudaContext::new(0)?;
    /// # let pool = BlockPool::new(2, SlotLayout::new(4, 8, 1)).unwrap();
    /// # let manager = CacheManager::new(pool);
    /// # struct DummyModel;
    /// # impl Model for DummyModel {
    /// #     fn forward(&mut self, _: &Tensor, _: usize) -> tensor::Result<Tensor> { panic!() }
    /// #     fn metadata(&self) -> ModelMetadata { ModelMetadata { model_type: "".into(), vocab_size: 1, hidden_size: 1, num_hidden_layers: 0 } }
    /// # }
    /// let worker = Worker::new(DummyModel, Sampler::default(), manager, context);
    /// let handle = WorkerHandle::spawn(worker);
    /// handle.shutdown().unwrap();
    /// # Ok(())
    /// # }
    /// ```
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
    ///
    /// # Example
    ///
    /// ```text
    /// use engine::{EngineRequest, ExecutionBatch, PaddedBatch, VarLenBatch};
    /// let batch = ExecutionBatch::new(vec![], PaddedBatch::new(vec![], vec![]), VarLenBatch::new(vec![], vec![0]));
    /// let request = EngineRequest::RunBatch { step_id: 1, batch };
    /// assert!(handle.send(request).is_ok());
    /// ```
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
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cache::{BlockPool, CacheManager, SlotLayout};
    /// # use engine::{Worker, WorkerHandle};
    /// # use model::{Model, ModelMetadata};
    /// # use runtime::CudaContext;
    /// # use sampling::Sampler;
    /// # use tensor::Tensor;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let context = CudaContext::new(0)?;
    /// # let pool = BlockPool::new(2, SlotLayout::new(4, 8, 1)).unwrap();
    /// # let manager = CacheManager::new(pool);
    /// # struct DummyModel;
    /// # impl Model for DummyModel {
    /// #     fn forward(&mut self, _: &Tensor, _: usize) -> tensor::Result<Tensor> { panic!() }
    /// #     fn metadata(&self) -> ModelMetadata { ModelMetadata { model_type: "".into(), vocab_size: 1, hidden_size: 1, num_hidden_layers: 0 } }
    /// # }
    /// let worker = Worker::new(DummyModel, Sampler::default(), manager, context);
    /// let handle = WorkerHandle::spawn(worker);
    /// handle.shutdown().unwrap();
    /// # Ok(())
    /// # }
    /// ```
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

/// Extracts the last-token logit row from a model output tensor.
///
/// The model returns `[batch, seq, vocab]` logits. This helper selects the final
/// sequence position and removes batch/seq dimensions so the sampler receives a
/// flat `[vocab]` tensor.
///
/// # Example
///
/// ```
/// use engine::last_token_logits;
/// use ops::reshape;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let logits = copy_h2d(
///     &context,
///     Shape::new([1, 2, 4])?,
///     DType::BF16,
///     &[0u16; 8],
/// )?;
/// let flat = last_token_logits(&logits)?;
/// assert_eq!(flat.shape().dims(), &[4]);
/// # Ok(())
/// # }
/// ```
pub fn last_token_logits(logits: &Tensor) -> tensor::Result<Tensor> {
    let dims = logits.shape().dims().to_vec();
    match dims.len() {
        1 => Ok(logits.clone()),
        2 => {
            let x = reshape(logits, [dims[0], 1, dims[1]])?;
            let x = narrow_dim1(&x, 0, 1)?;
            reshape(&x, [dims[1]])
        }
        3 => {
            if dims[1] == 0 {
                return Err(tensor::TensorError::InvalidArgument(
                    "cannot extract last token from empty sequence dim".to_string(),
                ));
            }
            let x = narrow_dim1(logits, dims[1] - 1, 1)?;
            reshape(&x, [dims[2]])
        }
        _ => reshape(logits, [logits.numel()]),
    }
}
