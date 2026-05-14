use candle_core::{Result, Tensor};

/// Transforms a logits vector before token selection.
///
/// Processors are chained by `Sampler` and receive the token history for the
/// sequence being decoded. Implementations should preserve the input device;
/// when the input tensor is CUDA-backed, returned tensors should also be
/// CUDA-backed.
///
/// # Example
///
/// ```
/// use candle_core::{Device, Tensor};
/// use sampling::{LogitsProcessor, Temperature};
///
/// # fn main() -> candle_core::Result<()> {
/// let logits = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], 3, &Device::Cpu)?;
/// let scaled = Temperature::new(2.0)?.process(&logits, &[])?;
/// assert_eq!(scaled.to_vec1::<f32>()?, vec![0.5, 1.0, 1.5]);
/// # Ok(())
/// # }
/// ```
pub trait LogitsProcessor: Send + Sync {
    /// Applies this processor to a one-dimensional logits tensor.
    fn process(&self, logits: &Tensor, history: &[u32]) -> Result<Tensor>;
}
