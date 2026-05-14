use candle_core::{Result, Tensor};

/// Applies the fused SiLU-and-multiply activation used by gated MLPs.
///
/// The input last dimension must be `intermediate_size * 2`, with gate values
/// followed by up-projection values. CUDA BF16 inputs route to `kernels`.
pub fn fused_silu_mul(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    kernels::fused_silu_mul(gate_up, intermediate_size)
}
