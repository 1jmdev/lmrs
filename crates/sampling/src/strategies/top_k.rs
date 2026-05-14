use candle_core::Result;

use crate::sampler::SampleOutput;

use super::{SamplingStrategy, softmax_choice, sorted_logits};

/// Samples from the `k` highest-logit tokens.
///
/// # Example
///
/// ```
/// use sampling::{SamplingStrategy, TopK};
///
/// # fn main() -> candle_core::Result<()> {
/// let mut rng = 42;
/// let out = TopK::new(2)?.sample(&[0.0, 10.0, 9.0], &mut rng)?;
/// assert!(out.token_id() == 1 || out.token_id() == 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopK {
    k: usize,
}

impl TopK {
    /// Creates a top-k strategy.
    pub fn new(k: usize) -> Result<Self> {
        if k == 0 {
            candle_core::bail!("top_k must be greater than zero")
        }
        Ok(Self { k })
    }

    /// Returns the candidate count.
    pub fn k(&self) -> usize {
        self.k
    }
}

impl SamplingStrategy for TopK {
    fn sample(&self, logits: &[f32], rng: &mut u64) -> Result<SampleOutput> {
        let sorted = sorted_logits(logits);
        let end = self.k.min(sorted.len());
        softmax_choice(&sorted[..end], rng)
    }
}
