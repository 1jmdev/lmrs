use crate::{Result, SampleOutput, SamplingError};

use super::{SamplingStrategy, softmax_choice, sorted_logits};

/// Samples tokens whose probability is at least `min_p * max_probability`.
///
/// # Example
///
/// ```
/// use sampling::{MinP, SamplingStrategy};
///
/// # fn main() -> sampling::Result<()> {
/// let mut rng = 3;
/// let out = MinP::new(0.1)?.sample(&[0.0, 10.0, 9.0], &mut rng)?;
/// assert!(out.token_id() == 1 || out.token_id() == 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinP {
    min_p: f64,
}

impl MinP {
    /// Creates a min-p sampling strategy.
    pub fn new(min_p: f64) -> Result<Self> {
        if !min_p.is_finite() || min_p < 0.0 || min_p > 1.0 {
            return Err(SamplingError::invalid("min_p must be in [0, 1]"));
        }
        Ok(Self { min_p })
    }

    /// Returns the minimum relative probability.
    pub fn min_p(&self) -> f64 {
        self.min_p
    }
}

impl SamplingStrategy for MinP {
    fn sample(&self, logits: &[f32], rng: &mut u64) -> Result<SampleOutput> {
        let sorted = sorted_logits(logits);
        if sorted.is_empty() {
            return Err(SamplingError::invalid("cannot sample from empty logits"));
        }
        let max = sorted[0].1;
        let threshold = f64::from(max).exp() * self.min_p;
        let mut keep = Vec::new();
        for pair in sorted {
            if f64::from(pair.1).exp() >= threshold {
                keep.push(pair);
            }
        }
        softmax_choice(&keep, rng)
    }
}
