use candle_core::Tensor;

pub struct Sampler;

impl Sampler {
    pub fn new(_temperature: f32, _top_p: f32, _top_k: Option<usize>) -> Self {
        Self
    }

    pub fn sample(&mut self, logits: &Tensor) -> crate::error::Result<u32> {
        Ok(argmax(&logits.to_vec1::<f32>()?) as u32)
    }
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(id, _)| id)
        .unwrap_or(0)
}
