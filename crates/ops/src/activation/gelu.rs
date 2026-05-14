use tensor::{Result, Tensor};

/// Applies GELU to the input tensor.
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    kernels::gelu(x)
}
