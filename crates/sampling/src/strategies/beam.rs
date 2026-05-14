use candle_core::Result;

use crate::sampler::SampleOutput;

use super::{SamplingStrategy, sorted_logits};

/// Returns the best beam candidate for single-step beam expansion.
///
/// Full beam bookkeeping belongs in the engine sequence scheduler; this
/// strategy provides the deterministic per-row candidate selection surface used
/// by that scheduler.
///
/// # Example
///
/// ```
/// use sampling::{BeamSearch, SamplingStrategy};
///
/// # fn main() -> candle_core::Result<()> {
/// let mut rng = 0;
/// let out = BeamSearch::new(4)?.sample(&[1.0, 3.0, 2.0], &mut rng)?;
/// assert_eq!(out.token_id(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeamSearch {
    width: usize,
}

impl BeamSearch {
    /// Creates a beam-search step strategy.
    pub fn new(width: usize) -> Result<Self> {
        if width == 0 {
            candle_core::bail!("beam width must be greater than zero")
        }
        Ok(Self { width })
    }

    /// Returns the beam width.
    pub fn width(&self) -> usize {
        self.width
    }
}

impl SamplingStrategy for BeamSearch {
    fn sample(&self, logits: &[f32], _rng: &mut u64) -> Result<SampleOutput> {
        let Some((token, logit)) = sorted_logits(logits).into_iter().take(self.width).next() else {
            candle_core::bail!("cannot sample from empty logits")
        };
        Ok(SampleOutput::new(token as u32, 0.0, logit))
    }
}
