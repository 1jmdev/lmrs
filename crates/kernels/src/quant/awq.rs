/// CUDA module name for AWQ GEMM.
pub const AWQ_GEMM_MODULE: &str = "lmrs_quant_awq_gemm";

/// Exported BF16 AWQ GEMM symbol.
pub const AWQ_GEMM_BF16: &str = "awq_gemm_bf16";

/// AWQ matrix shape metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AwqGemmShape {
    /// Rows in activation matrix.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
}

impl AwqGemmShape {
    /// Creates AWQ GEMM shape metadata.
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k }
    }
}
