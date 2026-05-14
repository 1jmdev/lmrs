use crate::{Shape, ShapeError};

/// Element strides for a tensor view.
///
/// Strides are measured in elements, not bytes. Contiguous row-major tensors use
/// the last dimension stride of one.
///
/// # Example
///
/// ```
/// use tensor::{Shape, Stride};
///
/// let shape = Shape::new([2, 3, 4])?;
/// let stride = Stride::contiguous(&shape);
/// assert_eq!(stride.as_slice(), &[12, 4, 1]);
/// assert!(stride.is_contiguous(&shape));
/// # Ok::<(), tensor::ShapeError>(())
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Stride {
    strides: Vec<usize>,
}

impl Stride {
    /// Computes row-major contiguous strides for `shape`.
    pub fn contiguous(shape: &Shape) -> Self {
        let mut strides = vec![1; shape.ndim()];
        let mut running = 1usize;
        for (idx, dim) in shape.dims().iter().enumerate().rev() {
            strides[idx] = running;
            running = running.saturating_mul(*dim);
        }
        Self { strides }
    }

    /// Creates a stride after checking its rank against `shape`.
    pub fn new(shape: &Shape, strides: impl Into<Vec<usize>>) -> Result<Self, ShapeError> {
        let strides = strides.into();
        if strides.len() != shape.ndim() {
            return Err(ShapeError::RankMismatch { shape_rank: shape.ndim(), stride_rank: strides.len() });
        }
        Ok(Self { strides })
    }

    /// Returns strides as a slice.
    pub fn as_slice(&self) -> &[usize] {
        &self.strides
    }

    /// Returns whether this is the row-major contiguous stride for `shape`.
    pub fn is_contiguous(&self, shape: &Shape) -> bool {
        self.strides == Self::contiguous(shape).strides
    }
}
