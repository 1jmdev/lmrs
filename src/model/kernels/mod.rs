use candle_core::{Module, Result, Tensor};

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::fused_silu_mul;

#[cfg(not(feature = "cuda"))]
pub fn fused_silu_mul(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    fallback_silu_mul(gate_up, intermediate_size)
}

pub fn fallback_silu_mul(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    let gate = gate_up.narrow(candle_core::D::Minus1, 0, intermediate_size)?;
    let up = gate_up.narrow(candle_core::D::Minus1, intermediate_size, intermediate_size)?;
    let gate = candle_nn::Activation::Silu.forward(&gate)?;
    gate * up
}
