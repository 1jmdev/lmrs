use tensor::{Result, Tensor};

/// Adds a causal mask to attention scores.
pub fn apply_causal_mask(
    scores: &Tensor,
    query_len: usize,
    key_len: usize,
    start_pos: usize,
) -> Result<Tensor> {
    kernels::apply_causal_mask(scores, query_len, key_len, start_pos)
}
