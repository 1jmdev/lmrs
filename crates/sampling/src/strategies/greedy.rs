use crate::{Result, SampleOutput, SamplingError};

use super::{SamplingStrategy, sorted_logits};

/// Selects the highest-logit token.
///
/// # Example
///
/// ```
/// use sampling::{Greedy, SamplingStrategy};
///
/// # fn main() -> sampling::Result<()> {
/// let mut rng = 7;
/// let out = Greedy.sample(&[0.0, 2.0, 1.0], &mut rng)?;
/// assert_eq!(out.token_id(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Greedy;

impl SamplingStrategy for Greedy {
    fn sample(&self, logits: &[f32], _rng: &mut u64) -> Result<SampleOutput> {
        let Some((token, logit)) = sorted_logits(logits).into_iter().next() else {
            return Err(SamplingError::invalid("cannot sample from empty logits"));
        };
        Ok(SampleOutput::new(token as u32, 0.0, logit))
    }
}
