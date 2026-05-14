/// CUDA module name for the flash MHA forward kernel object.
pub const FLASH_MHA_FWD_MODULE: &str = "lmrs_attention_flash_mha_fwd";

/// CUDA module name for the flash MHA backward kernel object.
pub const FLASH_MHA_BWD_MODULE: &str = "lmrs_attention_flash_mha_bwd";

/// Exported forward kernel symbol for BF16 flash attention.
pub const FLASH_MHA_FWD_BF16: &str = "flash_mha_fwd_bf16";

/// Exported backward kernel symbol for the BF16 gradient copy path.
pub const FLASH_MHA_BWD_COPY_BF16: &str = "flash_mha_bwd_copy_bf16";

/// Launch geometry used by the correctness-first flash MHA kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FlashMhaLaunch {
    /// Number of query rows to evaluate.
    pub rows: usize,
    /// Sequence length read by each row.
    pub seq_len: usize,
    /// Per-head hidden dimension.
    pub head_dim: usize,
}

impl FlashMhaLaunch {
    /// Creates launch metadata for one CUDA block per query row.
    pub fn new(rows: usize, seq_len: usize, head_dim: usize) -> Self {
        Self { rows, seq_len, head_dim }
    }
}
