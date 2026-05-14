use candle_core::{Result, Tensor};

/// Adds a causal mask to attention scores.
pub fn apply_causal_mask(
    scores: &Tensor,
    query_len: usize,
    key_len: usize,
    start_pos: usize,
) -> Result<Tensor> {
    let mask = Tensor::zeros((query_len, key_len), scores.dtype(), scores.device())?
        .where_cond(
            &Tensor::arange(0u32, query_len as u32, scores.device())?
                .unsqueeze(1)?
                .broadcast_add(&Tensor::new(&[start_pos as u32], scores.device())?)?
                .broadcast_ge(
                    &Tensor::arange(0u32, key_len as u32, scores.device())?.unsqueeze(0)?,
                )?,
            &Tensor::full(f32::NEG_INFINITY, (query_len, key_len), scores.device())?,
        )?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    scores.broadcast_add(&mask)
}
