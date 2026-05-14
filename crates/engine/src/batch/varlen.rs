/// Variable-length batch layout compatible with flash-attention style kernels.
///
/// # Example
///
/// ```
/// use engine::VarLenBatch;
///
/// let batch = VarLenBatch::new(vec![1, 2, 3], vec![0, 2, 3]);
/// assert_eq!(batch.cu_seqlens(), &[0, 2, 3]);
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VarLenBatch {
    tokens: Vec<u32>,
    cu_seqlens: Vec<usize>,
}

impl VarLenBatch {
    /// Creates a variable-length token batch.
    pub fn new(tokens: Vec<u32>, cu_seqlens: Vec<usize>) -> Self {
        Self { tokens, cu_seqlens }
    }

    /// Returns concatenated tokens.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Returns cumulative sequence lengths with leading zero.
    pub fn cu_seqlens(&self) -> &[usize] {
        &self.cu_seqlens
    }
}
