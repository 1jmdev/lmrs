use cache::{BlockPool, CacheManager, SlotLayout};
use engine::{
    EngineExecutor, Scheduler, SchedulerBudget, Sequence, SequenceGroup, Worker, WorkerHandle,
};
use half::bf16;
use model::{Model, ModelMetadata};
use runtime::CudaContext;
use sampling::Sampler;
use tensor::{DType, Result, Shape, Tensor, copy_h2d};

struct ScriptModel {
    tokens: Vec<u32>,
    calls: usize,
    vocab_size: usize,
    context: CudaContext,
}

impl ScriptModel {
    fn new(tokens: Vec<u32>, vocab_size: usize, context: CudaContext) -> Self {
        Self {
            tokens,
            calls: 0,
            vocab_size,
            context,
        }
    }
}

impl Model for ScriptModel {
    fn forward(&mut self, input_ids: &Tensor, _start_pos: usize) -> Result<Tensor> {
        let dims = input_ids.shape().dims();
        let seq_len = *dims.last().unwrap_or(&1);
        let token = self.tokens[self.calls];
        self.calls += 1;

        let total = seq_len * self.vocab_size;
        let mut logits = vec![0u16; total];
        let row = seq_len - 1;
        let idx = row * self.vocab_size + token as usize;
        logits[idx] = bf16::from_f32(100.0).to_bits();
        copy_h2d(
            &self.context,
            Shape::new([1, seq_len, self.vocab_size])
                .map_err(|e| tensor::TensorError::ShapeMismatch(e.to_string()))?,
            DType::BF16,
            &logits,
        )
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            model_type: "script".to_string(),
            vocab_size: self.vocab_size,
            hidden_size: 1,
            num_hidden_layers: 0,
        }
    }
}

#[test]
fn full_prefill_and_decode_loop_runs_without_server() {
    let context = CudaContext::new(0).unwrap();
    let pool = BlockPool::new(8, SlotLayout::new(4, 16, 1)).unwrap();
    let cache = CacheManager::new(pool);
    let model = ScriptModel::new(vec![3, 4, 5], 8, context.clone());
    let worker = Worker::new(model, Sampler::default(), cache, context);
    let handle = WorkerHandle::spawn(worker);

    let sequence = Sequence::new(cache::SequenceId::new(1), vec![1, 2], 3, Some(5)).unwrap();
    let group = SequenceGroup::new(1, vec![sequence]).unwrap();
    let mut scheduler = Scheduler::new(SchedulerBudget::new(4, 16));
    scheduler.add_group(group);
    let mut executor = EngineExecutor::new(scheduler, handle);

    let outputs = executor.run_until_idle().unwrap();
    let token_ids: Vec<_> = outputs
        .iter()
        .map(|output| output.sample().token_id())
        .collect();
    assert_eq!(token_ids, vec![3, 4, 5]);
    assert!(executor.scheduler().groups().is_empty());

    executor.shutdown().unwrap();
}
