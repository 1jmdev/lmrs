use candle_core::Tensor;

pub struct Sampler {
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, top_k: Option<usize>) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> crate::error::Result<u32> {
        let mut logits = logits.to_vec1::<f32>()?;
        if self.temperature <= 0.0 {
            return Ok(argmax(&logits) as u32);
        }
        for logit in &mut logits {
            *logit /= self.temperature;
        }
        let mut scored = logits.into_iter().enumerate().collect::<Vec<_>>();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        if let Some(top_k) = self.top_k {
            scored.truncate(top_k.max(1));
        }
        let max = scored.first().map(|(_, v)| *v).unwrap_or(0.0);
        let mut probs = scored
            .into_iter()
            .map(|(id, v)| (id, (v - max).exp()))
            .collect::<Vec<_>>();
        let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
        if sum > 0.0 {
            for (_, p) in &mut probs {
                *p /= sum;
            }
        }
        probs.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut kept = Vec::new();
        let mut cumulative = 0.0;
        for item in probs {
            cumulative += item.1;
            kept.push(item);
            if cumulative >= self.top_p {
                break;
            }
        }
        Ok(kept.first().map(|(id, _)| *id as u32).unwrap_or(0))
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
