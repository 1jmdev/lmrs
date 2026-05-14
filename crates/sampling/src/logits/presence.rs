use std::collections::HashMap;

use candle_core::{Result, Tensor};

use super::LogitsProcessor;

/// Applies OpenAI-style presence and frequency penalties.
///
/// Presence penalty subtracts a fixed amount once a token has appeared.
/// Frequency penalty subtracts `frequency * count` for each generated token.
///
/// # Example
///
/// ```
/// use candle_core::{Device, Tensor};
/// use sampling::{FrequencyPresencePenalty, LogitsProcessor};
///
/// # fn main() -> candle_core::Result<()> {
/// let logits = Tensor::from_vec(vec![3.0_f32, 3.0, 3.0], 3, &Device::Cpu)?;
/// let penalty = FrequencyPresencePenalty::new(0.5, 0.25)?;
/// let out = penalty.process(&logits, &[1, 1])?;
/// assert_eq!(out.to_vec1::<f32>()?, vec![3.0, 2.0, 3.0]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrequencyPresencePenalty {
    presence: f32,
    frequency: f32,
}

impl FrequencyPresencePenalty {
    /// Creates a combined presence and frequency penalty processor.
    pub fn new(presence: f32, frequency: f32) -> Result<Self> {
        if !presence.is_finite() || !frequency.is_finite() {
            candle_core::bail!("presence and frequency penalties must be finite")
        }
        Ok(Self {
            presence,
            frequency,
        })
    }

    /// Returns the presence penalty.
    pub fn presence(&self) -> f32 {
        self.presence
    }

    /// Returns the frequency penalty.
    pub fn frequency(&self) -> f32 {
        self.frequency
    }
}

impl LogitsProcessor for FrequencyPresencePenalty {
    fn process(&self, logits: &Tensor, history: &[u32]) -> Result<Tensor> {
        if history.is_empty() || (self.presence == 0.0 && self.frequency == 0.0) {
            return Ok(logits.clone());
        }
        let device = logits.device().clone();
        let dims = logits.dims().to_vec();
        let mut values = logits.flatten_all()?.to_vec1::<f32>()?;
        let mut counts = HashMap::<u32, u32>::new();
        for &token in history {
            *counts.entry(token).or_default() += 1;
        }
        for (token, count) in counts {
            if let Some(value) = values.get_mut(token as usize) {
                *value -= self.presence + self.frequency * count as f32;
            }
        }
        Tensor::from_vec(values, dims, &device)
    }
}
