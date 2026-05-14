use ops::affine;
use tensor::Tensor;

use crate::{Result, SamplingError};

use super::LogitsProcessor;

/// Divides logits by a strictly positive sampling temperature.
///
/// Values below `1.0` sharpen the distribution and values above `1.0` flatten
/// it. A temperature of exactly `1.0` returns a clone of the input tensor.
///
/// # Example
///
/// ```
/// use sampling::{LogitsProcessor, Temperature};
///
/// # fn main() -> sampling::Result<()> {
/// let processor = Temperature::new(2.0)?;
/// assert_eq!(processor.value(), 2.0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Temperature {
    value: f64,
}

impl Temperature {
    /// Creates a temperature processor.
    ///
    /// # Example
    ///
    /// ```
    /// use sampling::Temperature;
    ///
    /// # fn main() -> sampling::Result<()> {
    /// assert!(Temperature::new(0.0).is_err());
    /// assert!(Temperature::new(2.0).is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(value: f64) -> Result<Self> {
        if !value.is_finite() || value <= 0.0 {
            return Err(SamplingError::invalid(
                "temperature must be finite and greater than zero",
            ));
        }
        Ok(Self { value })
    }

    /// Returns the configured temperature value.
    ///
    /// # Example
    ///
    /// ```
    /// # use sampling::Temperature;
    /// # let temp = Temperature::new(0.5).unwrap();
    /// assert_eq!(temp.value(), 0.5);
    /// ```
    pub fn value(&self) -> f64 {
        self.value
    }
}

impl LogitsProcessor for Temperature {
    fn process(&self, logits: &Tensor, _history: &[u32]) -> Result<Tensor> {
        if self.value == 1.0 {
            return Ok(logits.clone());
        }
        Ok(affine(logits, (1.0 / self.value) as f32, 0.0f32)?)
    }
}
