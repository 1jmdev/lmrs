use tensor::{Result, Tensor};

/// Adds a bias vector to the final dimension of `x`.
pub fn add_bias(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    kernels::add_bias(x, bias)
}
