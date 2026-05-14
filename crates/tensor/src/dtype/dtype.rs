use std::fmt;

/// Element type stored by a tensor.
///
/// `DType` is intentionally small and copyable because shape, storage, copy,
/// and kernel dispatch code all use it to validate byte sizes before touching
/// CUDA memory.
///
/// # Example
///
/// ```
/// use tensor::DType;
///
/// assert_eq!(DType::F32.size_in_bytes(), 4);
/// assert_eq!(DType::BF16.name(), "bf16");
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DType {
    /// IEEE 32-bit floating point.
    F32,
    /// IEEE 16-bit floating point.
    F16,
    /// BFloat16 floating point.
    BF16,
    /// Float8 E4M3 byte storage.
    F8E4M3,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer.
    U32,
}

impl DType {
    /// Returns the element size in bytes.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::DType;
    ///
    /// assert_eq!(DType::F16.size_in_bytes(), 2);
    /// ```
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F8E4M3 | Self::I8 | Self::U8 => 1,
        }
    }

    /// Returns the minimum useful host alignment for the dtype.
    ///
    /// CUDA allocations are more strictly aligned than this value; this query is
    /// for validating host buffers and kernel metadata.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::DType;
    ///
    /// assert_eq!(DType::BF16.align_in_bytes(), 2);
    /// ```
    pub const fn align_in_bytes(self) -> usize {
        self.size_in_bytes()
    }

    /// Returns the canonical lowercase dtype name.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::DType;
    ///
    /// assert_eq!(DType::F8E4M3.name(), "f8e4m3");
    /// ```
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F8E4M3 => "f8e4m3",
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I32 => "i32",
            Self::U32 => "u32",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
