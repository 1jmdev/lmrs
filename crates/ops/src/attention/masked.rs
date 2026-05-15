use candle_core::{DType, Device, Result, Tensor};

/// Adds a causal mask to attention scores.
///
/// Constructs a BF16 upper-triangular mask of shape `[1, 1, query_len, key_len]`
/// where positions that should not attend get a large negative value.
pub fn apply_causal_mask(
    scores: &Tensor,
    query_len: usize,
    key_len: usize,
    start_pos: usize,
) -> Result<Tensor> {
    let device = scores.device();
    let mask = causal_mask(query_len, key_len, start_pos, device)?;
    scores.broadcast_add(&mask)
}

fn causal_mask(query_len: usize, key_len: usize, start_pos: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..query_len)
        .flat_map(|i| {
            let row_pos = start_pos + i;
            (0..key_len).map(move |j| if j > row_pos { f32::NEG_INFINITY } else { 0.0 })
        })
        .collect();
    Tensor::from_vec(mask, (query_len, key_len), device)?
        .to_dtype(DType::BF16)?
        .unsqueeze(0)?
        .unsqueeze(0)
}