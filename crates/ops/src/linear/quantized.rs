use candle_core::{Result, Tensor};

/// Placeholder surface for quantized linear projections.
pub struct QuantizedLinear;

impl QuantizedLinear {
    /// Quantized linear execution is not wired until packed weight formats move into `tensor`.
    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        candle_core::bail!("quantized linear is not implemented yet")
    }
}
