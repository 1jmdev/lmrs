use candle_core::{Result, Tensor};

use super::LogitsProcessor;

/// Divides logits by a strictly positive sampling temperature.
///
/// Values below `1.0` sharpen the distribution and values above `1.0` flatten
/// it. A temperature of exactly `1.0` returns a clone of the input tensor.
///
/// # Example
///
/// ```
/// use candle_core::{Device, Tensor};
/// use sampling::{LogitsProcessor, Temperature};
///
/// # fn main() -> candle_core::Result<()> {
/// let logits = Tensor::from_vec(vec![2.0_f32, 4.0], 2, &Device::Cpu)?;
/// let out = Temperature::new(2.0)?.process(&logits, &[])?;
/// assert_eq!(out.to_vec1::<f32>()?, vec![1.0, 2.0]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Temperature {
    value: f64,
}

impl Temperature {
    /// Creates a temperature processor.
    pub fn new(value: f64) -> Result<Self> {
        if !value.is_finite() || value <= 0.0 {
            candle_core::bail!("temperature must be finite and greater than zero")
        }
        Ok(Self { value })
    }

    /// Returns the configured temperature value.
    pub fn value(&self) -> f64 {
        self.value
    }
}

impl LogitsProcessor for Temperature {
    fn process(&self, logits: &Tensor, _history: &[u32]) -> Result<Tensor> {
        if self.value == 1.0 {
            return Ok(logits.clone());
        }
        logits.affine(1.0 / self.value, 0.0)
    }
}
