/// CUDA module name for block copy kernels.
pub const COPY_BLOCKS_MODULE: &str = "lmrs_utils_copy_blocks";

/// CUDA module name for gather/scatter kernels.
pub const GATHER_SCATTER_MODULE: &str = "lmrs_utils_gather_scatter";

/// Exported byte block copy symbol.
pub const COPY_BLOCKS_U8: &str = "copy_blocks_u8";

/// Exported BF16 gather symbol.
pub const GATHER_BF16: &str = "gather_bf16";

/// Exported BF16 scatter symbol.
pub const SCATTER_BF16: &str = "scatter_bf16";
