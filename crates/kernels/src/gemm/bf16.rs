/// Exported BF16 GEMM kernel symbols.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bf16GemmSymbol {
    /// SM 8.9 correctness-first BF16 GEMM.
    Sm89,
    /// SM 10.0 correctness-first BF16 GEMM.
    Sm100,
}

impl Bf16GemmSymbol {
    /// Returns the CUDA symbol name for this BF16 GEMM variant.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sm89 => "bf16_gemm_sm89",
            Self::Sm100 => "bf16_gemm_sm100",
        }
    }
}

/// Matrix dimensions consumed by BF16 GEMM kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GemmShape {
    /// Rows in A and C.
    pub m: usize,
    /// Columns in B and C.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
}

impl GemmShape {
    /// Creates a GEMM shape descriptor.
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k }
    }
}
