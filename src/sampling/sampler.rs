use candle_core::Tensor;

use crate::model::kernels;

pub struct Sampler;

impl Sampler {
    pub fn new(_temperature: f32, _top_p: f32, _top_k: Option<usize>) -> Self {
        Self
    }

    pub fn sample(&mut self, logits: &Tensor) -> crate::error::Result<u32> {
        Ok(kernels::gpu_argmax(logits)?)
    }
}
