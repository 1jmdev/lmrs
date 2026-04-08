use crate::api::{Sampling, TemperatureSampling};
use crate::llama::Token;

pub fn sample_token(logits: &[f32], sampling: &Sampling, rng: &mut XorShift64) -> Option<Token> {
    match sampling {
        Sampling::Greedy => greedy(logits),
        Sampling::Temperature(settings) => temperature_sample(logits, settings, rng),
    }
}

fn greedy(logits: &[f32]) -> Option<Token> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(idx, _)| idx as Token)
}

fn temperature_sample(
    logits: &[f32],
    settings: &TemperatureSampling,
    rng: &mut XorShift64,
) -> Option<Token> {
    if settings.temperature <= 0.0 {
        return greedy(logits);
    }

    let mut candidates = logits
        .iter()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .map(|(idx, logit)| (idx as Token, *logit))
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
    if settings.top_k > 0 && candidates.len() > settings.top_k {
        candidates.truncate(settings.top_k);
    }

    let max_logit = candidates
        .iter()
        .map(|(_, logit)| *logit)
        .max_by(|left, right| left.total_cmp(right))?;

    let mut weights = Vec::with_capacity(candidates.len());
    let mut total = 0.0f64;
    for (_, logit) in &candidates {
        let scaled = ((*logit - max_logit) / settings.temperature) as f64;
        let weight = scaled.exp();
        total += weight;
        weights.push(weight);
    }
    if total <= f64::EPSILON {
        return candidates.first().map(|(token, _)| *token);
    }

    let clamped_top_p = settings.top_p.clamp(0.0, 1.0) as f64;
    let mut cumulative = 0.0f64;
    let mut selected = candidates.len();
    if clamped_top_p > 0.0 && clamped_top_p < 1.0 {
        for (idx, weight) in weights.iter().enumerate() {
            cumulative += *weight / total;
            if cumulative >= clamped_top_p {
                selected = idx + 1;
                break;
            }
        }
    }

    if selected == 0 {
        return candidates.first().map(|(token, _)| *token);
    }

    let truncated_weights = &weights[..selected];
    let truncated_candidates = &candidates[..selected];
    let weight_sum: f64 = truncated_weights.iter().sum();
    if weight_sum <= f64::EPSILON {
        return truncated_candidates.first().map(|(token, _)| *token);
    }

    let mut threshold = rng.next_f64() * weight_sum;
    for (idx, weight) in truncated_weights.iter().enumerate() {
        threshold -= *weight;
        if threshold <= 0.0 {
            return truncated_candidates.get(idx).map(|(token, _)| *token);
        }
    }
    truncated_candidates.last().map(|(token, _)| *token)
}

#[derive(Clone, Debug)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x2545_F491_4F6C_DD1D
        } else {
            seed
        };
        Self { state }
    }

    pub fn reseed(&mut self, seed: u64) {
        self.state = if seed == 0 {
            0x2545_F491_4F6C_DD1D
        } else {
            seed
        };
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        (value as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}
