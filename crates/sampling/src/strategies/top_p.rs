use candle_core::Result;

use crate::sampler::SampleOutput;

use super::{SamplingStrategy, softmax_choice, sorted_logits};

/// Samples from the smallest high-probability nucleus above probability `p`.
///
/// # Example
///
/// ```
/// use sampling::{SamplingStrategy, TopP};
///
/// # fn main() -> candle_core::Result<()> {
/// let mut rng = 9;
/// let out = TopP::new(0.9)?.sample(&[0.0, 10.0, 9.0], &mut rng)?;
/// assert!(out.token_id() == 1 || out.token_id() == 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TopP {
    p: f64,
}

impl TopP {
    /// Creates a nucleus sampling strategy.
    pub fn new(p: f64) -> Result<Self> {
        if !p.is_finite() || p <= 0.0 || p > 1.0 {
            candle_core::bail!("top_p must be in (0, 1]")
        }
        Ok(Self { p })
    }

    /// Returns the nucleus probability threshold.
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl SamplingStrategy for TopP {
    fn sample(&self, logits: &[f32], rng: &mut u64) -> Result<SampleOutput> {
        let sorted = sorted_logits(logits);
        if sorted.is_empty() {
            candle_core::bail!("cannot sample from empty logits")
        }
        let max = sorted[0].1;
        let denom: f64 = sorted.iter().map(|(_, logit)| f64::from(*logit - max).exp()).sum();
        let mut acc = 0.0;
        let mut keep = 0;
        for (_, logit) in &sorted {
            acc += f64::from(*logit - max).exp() / denom;
            keep += 1;
            if acc >= self.p {
                break;
            }
        }
        softmax_choice(&sorted[..keep], rng)
    }
}
