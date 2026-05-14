use runtime::CudaContext;

use crate::{CudaBuf, DType, Result, Shape, SharedStorage, Stride};

/// Core CUDA tensor metadata and storage handle.
///
/// This type owns no operation semantics yet; it records dtype, shape, stride,
/// and byte storage so higher crates can agree on a stable tensor API.
///
/// # Example
///
/// ```no_run
/// use runtime::CudaContext;
/// use tensor::{DType, Shape, Tensor};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let tensor = Tensor::empty(&context, Shape::new([2, 3]).unwrap(), DType::F32)?;
/// assert_eq!(tensor.len_bytes(), 24);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Tensor {
    shape: Shape,
    stride: Stride,
    storage: SharedStorage,
    dtype: DType,
}

impl Tensor {
    /// Allocates an uninitialized contiguous tensor.
    pub fn empty(context: &CudaContext, shape: Shape, dtype: DType) -> Result<Self> {
        let len_bytes = shape.numel() * dtype.size_in_bytes();
        let storage = SharedStorage::new(CudaBuf::new(context, len_bytes)?);
        let stride = Stride::contiguous(&shape);
        Ok(Self {
            shape,
            stride,
            storage,
            dtype,
        })
    }

    /// Builds a tensor from existing shared storage and explicit metadata.
    pub fn from_storage(
        storage: SharedStorage,
        shape: Shape,
        stride: Stride,
        dtype: DType,
    ) -> Self {
        Self {
            shape,
            stride,
            storage,
            dtype,
        }
    }

    /// Returns tensor shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns tensor stride.
    pub fn stride(&self) -> &Stride {
        &self.stride
    }

    /// Returns shared CUDA storage.
    pub fn storage(&self) -> &SharedStorage {
        &self.storage
    }

    /// Returns element dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the total element count.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Returns the logical byte length.
    pub fn len_bytes(&self) -> usize {
        self.numel() * self.dtype.size_in_bytes()
    }

    /// Returns whether this tensor has contiguous row-major layout.
    pub fn is_contiguous(&self) -> bool {
        self.stride.is_contiguous(&self.shape)
    }
}
