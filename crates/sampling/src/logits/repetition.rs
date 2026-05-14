use candle_core::{Result, Tensor};

use super::LogitsProcessor;

/// Applies the standard repetition penalty to tokens already generated.
///
/// Positive logits for repeated tokens are divided by the penalty while
/// negative logits are multiplied by it, matching the Hugging Face generation
/// convention.
///
/// # Example
///
/// ```
/// use candle_core::{Device, Tensor};
/// use sampling::{LogitsProcessor, RepetitionPenalty};
///
/// # fn main() -> candle_core::Result<()> {
/// let logits = Tensor::from_vec(vec![2.0_f32, -2.0, 1.0], 3, &Device::Cpu)?;
/// let out = RepetitionPenalty::new(2.0)?.process(&logits, &[0, 1])?;
/// assert_eq!(out.to_vec1::<f32>()?, vec![1.0, -4.0, 1.0]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RepetitionPenalty {
    penalty: f32,
}

impl RepetitionPenalty {
    /// Creates a repetition penalty processor.
    pub fn new(penalty: f32) -> Result<Self> {
        if !penalty.is_finite() || penalty <= 0.0 {
            candle_core::bail!("repetition penalty must be finite and greater than zero")
        }
        Ok(Self { penalty })
    }

    /// Returns the configured penalty.
    pub fn penalty(&self) -> f32 {
        self.penalty
    }
}

impl LogitsProcessor for RepetitionPenalty {
    fn process(&self, logits: &Tensor, history: &[u32]) -> Result<Tensor> {
        if self.penalty == 1.0 || history.is_empty() {
            return Ok(logits.clone());
        }
        let device = logits.device().clone();
        let dims = logits.dims().to_vec();
        let mut values = logits.flatten_all()?.to_vec1::<f32>()?;
        for &token in history {
            if let Some(value) = values.get_mut(token as usize) {
                if *value >= 0.0 {
                    *value /= self.penalty;
                } else {
                    *value *= self.penalty;
                }
            }
        }
        Tensor::from_vec(values, dims, &device)
    }
}
