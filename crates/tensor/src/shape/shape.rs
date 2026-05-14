use thiserror::Error;

/// Shape construction and arithmetic failures.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ShapeError {
    /// A dimension product overflowed `usize`.
    #[error("shape element count overflowed usize")]
    NumelOverflow,
    /// A requested stride does not match shape rank.
    #[error("stride rank {stride_rank} does not match shape rank {shape_rank}")]
    RankMismatch {
        shape_rank: usize,
        stride_rank: usize,
    },
}

/// Tensor extents in row-major dimension order.
///
/// `Shape` owns its dimensions and validates element-count overflow at
/// construction time. Empty rank represents a scalar and has one element.
///
/// # Example
///
/// ```
/// use tensor::Shape;
///
/// let shape = Shape::new([2, 3, 4])?;
/// assert_eq!(shape.ndim(), 3);
/// assert_eq!(shape.numel(), 24);
/// # Ok::<(), tensor::ShapeError>(())
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Shape {
    dims: Vec<usize>,
    numel: usize,
}

impl Shape {
    /// Creates a shape from row-major dimensions.
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self, ShapeError> {
        let dims = dims.into();
        let mut numel = 1usize;
        for dim in &dims {
            numel = numel.checked_mul(*dim).ok_or(ShapeError::NumelOverflow)?;
        }
        Ok(Self { dims, numel })
    }

    /// Creates a scalar shape.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Shape;
    ///
    /// assert_eq!(Shape::scalar().numel(), 1);
    /// ```
    pub fn scalar() -> Self {
        Self {
            dims: Vec::new(),
            numel: 1,
        }
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total element count.
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Returns dimensions as a slice.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns whether any dimension is zero.
    pub fn is_empty(&self) -> bool {
        self.numel == 0
    }
}
