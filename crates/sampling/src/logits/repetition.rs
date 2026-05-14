use half::bf16;
use ops::{greater_equal, mul, where_cond};
use tensor::{DType, Shape, Tensor, copy_h2d};

use crate::{Result, SamplingError};

use super::LogitsProcessor;

/// Applies the standard repetition penalty to tokens already generated.
///
/// Positive logits for repeated tokens are divided by the penalty while
/// negative logits are multiplied by it, matching the Hugging Face generation
/// convention.
///
/// # Example
///
/// ```no_run
/// use half::bf16;
/// use ops::cast_bf16_to_f32;
/// use runtime::CudaContext;
/// use sampling::{LogitsProcessor, RepetitionPenalty};
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> sampling::Result<()> {
/// let context = CudaContext::new(0)?;
/// let logits = copy_h2d(&context, Shape::new([3])?, DType::BF16, &[
///     bf16::from_f32(2.0).to_bits(),
///     bf16::from_f32(-2.0).to_bits(),
///     bf16::from_f32(1.0).to_bits(),
/// ])?;
/// let out = RepetitionPenalty::new(2.0)?.process(&logits, &[0, 1])?;
/// let values = cast_bf16_to_f32(&out)?;
/// assert!((values[0] - 1.0).abs() < 0.01);
/// assert!((values[2] - 1.0).abs() < 0.01);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RepetitionPenalty {
    penalty: f32,
}

impl RepetitionPenalty {
    /// Creates a repetition penalty processor.
    ///
    /// # Example
    ///
    /// ```
    /// use sampling::RepetitionPenalty;
    ///
    /// # fn main() -> sampling::Result<()> {
    /// assert!(RepetitionPenalty::new(-1.0).is_err());
    /// assert!(RepetitionPenalty::new(2.0).is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(penalty: f32) -> Result<Self> {
        if !penalty.is_finite() || penalty <= 0.0 {
            return Err(SamplingError::invalid(
                "repetition penalty must be finite and greater than zero",
            ));
        }
        Ok(Self { penalty })
    }

    /// Returns the configured penalty.
    ///
    /// # Example
    ///
    /// ```
    /// # use sampling::RepetitionPenalty;
    /// # let rp = RepetitionPenalty::new(2.0).unwrap();
    /// assert_eq!(rp.penalty(), 2.0);
    /// ```
    pub fn penalty(&self) -> f32 {
        self.penalty
    }
}

impl LogitsProcessor for RepetitionPenalty {
    fn process(&self, logits: &Tensor, history: &[u32]) -> Result<Tensor> {
        if self.penalty == 1.0 || history.is_empty() {
            return Ok(logits.clone());
        }
        let len = logits.numel();
        let dims = logits.shape().dims().to_vec();

        let inv_penalty = 1.0_f32 / self.penalty;
        let mut pos_factors = vec![1.0_f32; len];
        let mut neg_factors = vec![1.0_f32; len];
        for &token in history {
            let index = token as usize;
            if index < len {
                pos_factors[index] = inv_penalty;
                neg_factors[index] = self.penalty;
            }
        }

        let stream = logits.storage().buffer().as_slice().stream();
        let ctx = stream.context();
        let shape = Shape::new(dims.iter().copied().collect::<Vec<_>>())
            .map_err(|e| SamplingError::invalid(e.to_string()))?;

        let pos_bits: Vec<u16> = pos_factors.iter().map(|&f| bf16::from_f32(f).to_bits()).collect();
        let neg_bits: Vec<u16> = neg_factors.iter().map(|&f| bf16::from_f32(f).to_bits()).collect();
        let zero_bits = vec![bf16::from_f32(0.0_f32).to_bits(); len];

        let cuda_ctx = runtime::CudaContext::from_cudarc(ctx.clone());
        let pos_tensor = copy_h2d(&cuda_ctx, shape.clone(), DType::BF16, &pos_bits)?;
        let neg_tensor = copy_h2d(&cuda_ctx, shape.clone(), DType::BF16, &neg_bits)?;
        let zero_tensor = copy_h2d(&cuda_ctx, shape, DType::BF16, &zero_bits)?;

        let positive = mul(logits, &pos_tensor)?;
        let negative = mul(logits, &neg_tensor)?;
        let cond = greater_equal(logits, &zero_tensor)?;
        Ok(where_cond(&cond, &positive, &negative)?)
    }
}
