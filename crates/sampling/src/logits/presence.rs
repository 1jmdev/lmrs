use std::collections::HashMap;

use half::bf16;
use ops::sub;
use tensor::{DType, Shape, Tensor, copy_h2d};

use crate::{Result, SamplingError};

use super::LogitsProcessor;

/// Applies OpenAI-style presence and frequency penalties.
///
/// Presence penalty subtracts a fixed amount once a token has appeared.
/// Frequency penalty subtracts `frequency * count` for each generated token.
///
/// # Example
///
/// ```no_run
/// use half::bf16;
/// use ops::cast_bf16_to_f32;
/// use runtime::CudaContext;
/// use sampling::{FrequencyPresencePenalty, LogitsProcessor};
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> sampling::Result<()> {
/// let context = CudaContext::new(0)?;
/// let logits = copy_h2d(&context, Shape::new([3])?, DType::BF16, &[
///     bf16::from_f32(3.0).to_bits(),
///     bf16::from_f32(3.0).to_bits(),
///     bf16::from_f32(3.0).to_bits(),
/// ])?;
/// let penalty = FrequencyPresencePenalty::new(0.5, 0.25)?;
/// let out = penalty.process(&logits, &[1, 1])?;
/// let values = cast_bf16_to_f32(&out)?;
/// assert!((values[0] - 3.0).abs() < 0.01);
/// assert!((values[1] - 2.0).abs() < 0.01);
/// assert!((values[2] - 3.0).abs() < 0.01);
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
    ///
    /// # Example
    ///
    /// ```
    /// use sampling::FrequencyPresencePenalty;
    ///
    /// # fn main() -> sampling::Result<()> {
    /// assert!(FrequencyPresencePenalty::new(f32::NAN, 0.0).is_err());
    /// assert!(FrequencyPresencePenalty::new(0.5, 0.25).is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(presence: f32, frequency: f32) -> Result<Self> {
        if !presence.is_finite() || !frequency.is_finite() {
            return Err(SamplingError::invalid(
                "presence and frequency penalties must be finite",
            ));
        }
        Ok(Self {
            presence,
            frequency,
        })
    }

    /// Returns the presence penalty.
    ///
    /// # Example
    ///
    /// ```
    /// # use sampling::FrequencyPresencePenalty;
    /// # let fp = FrequencyPresencePenalty::new(0.5, 0.25).unwrap();
    /// assert_eq!(fp.presence(), 0.5);
    /// ```
    pub fn presence(&self) -> f32 {
        self.presence
    }

    /// Returns the frequency penalty.
    ///
    /// # Example
    ///
    /// ```
    /// # use sampling::FrequencyPresencePenalty;
    /// # let fp = FrequencyPresencePenalty::new(0.5, 0.25).unwrap();
    /// assert_eq!(fp.frequency(), 0.25);
    /// ```
    pub fn frequency(&self) -> f32 {
        self.frequency
    }
}

impl LogitsProcessor for FrequencyPresencePenalty {
    fn process(&self, logits: &Tensor, history: &[u32]) -> Result<Tensor> {
        if history.is_empty() || (self.presence == 0.0 && self.frequency == 0.0) {
            return Ok(logits.clone());
        }
        let len = logits.numel();
        let mut penalties_host = vec![0.0_f32; len];
        let mut counts = HashMap::<u32, u32>::new();
        for &token in history {
            *counts.entry(token).or_default() += 1;
        }
        for (token, count) in counts {
            let index = token as usize;
            if index < len {
                penalties_host[index] = self.presence + self.frequency * count as f32;
            }
        }

        let dims = logits.shape().dims().to_vec();
        let stream = logits.storage().buffer().as_slice().stream();
        let ctx = stream.context();
        let shape = Shape::new(dims.iter().copied().collect::<Vec<_>>())
            .map_err(|e| SamplingError::invalid(e.to_string()))?;
        let penalty_bits: Vec<u16> = penalties_host
            .iter()
            .map(|&f| bf16::from_f32(f).to_bits())
            .collect();
        let cuda_ctx = runtime::CudaContext::from_cudarc(ctx.clone());
        let penalty_tensor = copy_h2d(&cuda_ctx, shape, DType::BF16, &penalty_bits)?;
        Ok(sub(logits, &penalty_tensor)?)
    }
}
