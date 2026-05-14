use candle_core::{Result, Tensor};

/// Applies GELU to the input tensor.
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    x.gelu()
}
