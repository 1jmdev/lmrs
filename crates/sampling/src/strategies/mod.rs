pub mod greedy;
pub mod top_k;
pub mod top_p;

pub use greedy::Greedy;
pub use top_k::TopK;
pub use top_p::TopP;

use candle_core::Result;

use crate::sampler::SampleOutput;

pub trait SamplingStrategy: Send + Sync {
    fn sample(&self, logits: &[f32], rng: &mut u64) -> Result<SampleOutput>;
}

pub(crate) fn sorted_logits(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut pairs: Vec<_> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs
}

pub(crate) fn softmax_choice(candidates: &[(usize, f32)], rng: &mut u64) -> Result<SampleOutput> {
    if candidates.is_empty() {
        candle_core::bail!("cannot sample from an empty candidate set")
    }
    let max = candidates[0].1;
    let mut weights = Vec::with_capacity(candidates.len());
    let mut sum = 0.0_f64;
    for &(_, logit) in candidates {
        let weight = f64::from(logit - max).exp();
        sum += weight;
        weights.push(weight);
    }
    if !sum.is_finite() || sum <= 0.0 {
        candle_core::bail!("logits produced an invalid probability distribution")
    }
    let mut target = next_unit(rng) * sum;
    for ((token, logit), weight) in candidates.iter().zip(weights) {
        if target <= weight {
            let prob = weight / sum;
            return Ok(SampleOutput::new(*token as u32, prob.ln() as f32, *logit));
        }
        target -= weight;
    }
    let (token, logit) = candidates[candidates.len() - 1];
    let prob = f64::from(logit - max).exp() / sum;
    Ok(SampleOutput::new(token as u32, prob.ln() as f32, logit))
}

pub(crate) fn next_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let bits = (*state >> 11) | 0x3ff0_0000_0000_0000;
    f64::from_bits(bits) - 1.0
}