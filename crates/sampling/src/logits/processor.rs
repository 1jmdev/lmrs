use tensor::Tensor;

use crate::Result;

/// Transforms a logits vector before token selection.
///
/// Processors are chained by `Sampler` and receive the token history for the
/// sequence being decoded. All processor math runs on CUDA via the `ops` and
/// `kernels` crates.
///
/// # Example
///
/// ```
/// use sampling::{LogitsProcessor, Temperature};
///
/// # fn main() -> sampling::Result<()> {
/// let _processor = Temperature::new(2.0)?;
/// # Ok(())
/// # }
/// ```
pub trait LogitsProcessor: Send + Sync {
    /// Applies this processor to a one-dimensional logits tensor.
    fn process(&self, logits: &Tensor, history: &[u32]) -> Result<Tensor>;
}
