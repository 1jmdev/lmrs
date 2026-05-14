use candle_core::{Result, Tensor};

/// Paged attention entry point reserved for the cache crate integration.
pub fn paged_attention(_q: &Tensor, _k_cache: &Tensor, _v_cache: &Tensor) -> Result<Tensor> {
    candle_core::bail!("paged attention requires cache block tables and is not implemented yet")
}
