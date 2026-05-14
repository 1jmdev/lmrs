use crate::DType;

/// Safety category for converting between two tensor dtypes.
///
/// This records policy only. Actual conversion kernels live in higher-level ops
/// once quantization and cast kernels are available.
///
/// # Example
///
/// ```
/// use tensor::{CastKind, DType};
///
/// assert_eq!(CastKind::between(DType::F32, DType::BF16), CastKind::Narrowing);
/// assert!(CastKind::between(DType::U8, DType::I32).is_allowed());
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CastKind {
    /// Source and destination dtypes are identical.
    Identity,
    /// Destination can represent at least the source byte width.
    Widening,
    /// Destination loses precision or range.
    Narrowing,
    /// Conversion needs an explicit quantization/dequantization policy.
    Quantized,
}

impl CastKind {
    /// Classifies a cast between two dtypes.
    pub const fn between(src: DType, dst: DType) -> Self {
        if src as u8 == dst as u8 {
            Self::Identity
        } else if matches!(src, DType::F8E4M3) || matches!(dst, DType::F8E4M3) {
            Self::Quantized
        } else if dst.size_in_bytes() >= src.size_in_bytes() {
            Self::Widening
        } else {
            Self::Narrowing
        }
    }

    /// Returns whether this cast can be requested without extra scale metadata.
    pub const fn is_allowed(self) -> bool {
        !matches!(self, Self::Quantized)
    }
}
