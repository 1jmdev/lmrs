use candle_core::{D, Result, Tensor};

/// Applies the fused SiLU-and-multiply activation used by gated MLPs.
///
/// The input last dimension must be `intermediate_size * 2`, with gate values
/// in the first half and up-projection values in the second half.
pub fn fused_silu_mul(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    let gate = gate_up.narrow(D::Minus1, 0, intermediate_size)?;
    let up = gate_up.narrow(D::Minus1, intermediate_size, intermediate_size)?;
    let silu_gate = gate.silu()?;
    silu_gate.broadcast_mul(&up)
}