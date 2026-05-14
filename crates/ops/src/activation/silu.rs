use tensor::{Result, Tensor};

/// Applies the fused SiLU-and-multiply activation used by gated MLPs.
///
/// The input last dimension must be `intermediate_size * 2`, with gate values
/// followed by up-projection values. CUDA BF16 inputs route to `kernels`.
pub fn fused_silu_mul(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    kernels::fused_silu_mul(gate_up, intermediate_size)
}

/// Applies SiLU to `gate` and multiplies by `up` elementwise.
///
/// # Example
///
/// ```no_run
/// use ops::silu_mul;
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, copy_h2d};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let shape = Shape::new([1, 4])?;
/// let gate = copy_h2d(&context, shape.clone(), DType::BF16, &[0u16; 4])?;
/// let up = copy_h2d(&context, shape, DType::BF16, &[0u16; 4])?;
/// let out = silu_mul(&gate, &up)?;
/// assert_eq!(out.shape().dims(), &[1, 4]);
/// # Ok(())
/// # }
/// ```
pub fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    kernels::silu_mul(gate, up)
}
