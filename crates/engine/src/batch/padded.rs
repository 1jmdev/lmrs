/// CPU-side padded token matrix used to materialize model input tensors.
///
/// # Example
///
/// ```
/// use engine::PaddedBatch;
///
/// let batch = PaddedBatch::new(vec![vec![1, 2], vec![3, 0]], vec![2, 1]);
/// assert_eq!(batch.max_len(), 2);
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PaddedBatch {
    tokens: Vec<Vec<u32>>,
    lengths: Vec<usize>,
}

impl PaddedBatch {
    /// Creates a padded batch from rows and original lengths.
    pub fn new(tokens: Vec<Vec<u32>>, lengths: Vec<usize>) -> Self {
        Self { tokens, lengths }
    }

    /// Returns padded token rows.
    pub fn tokens(&self) -> &[Vec<u32>] {
        &self.tokens
    }

    /// Returns unpadded row lengths.
    pub fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    /// Returns maximum row length.
    pub fn max_len(&self) -> usize {
        self.tokens.first().map_or(0, Vec::len)
    }
}
