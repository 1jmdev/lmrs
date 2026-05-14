/// CUDA module name for the MLA forward kernel object.
pub const MLA_FWD_MODULE: &str = "lmrs_attention_mla_fwd";

/// Exported BF16 Multi-Latent Attention projection symbol.
pub const MLA_FWD_BF16: &str = "mla_fwd_bf16";

/// Shape metadata for the MLA latent projection kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlaLaunch {
    /// Number of input rows.
    pub rows: usize,
    /// Latent hidden dimension.
    pub latent_dim: usize,
    /// Output hidden dimension.
    pub out_dim: usize,
}

impl MlaLaunch {
    /// Creates shape metadata for the MLA forward projection.
    pub fn new(rows: usize, latent_dim: usize, out_dim: usize) -> Self {
        Self { rows, latent_dim, out_dim }
    }
}
