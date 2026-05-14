use crate::block::BlockId;

/// Direction for moving cache blocks between GPU and CPU storage.
///
/// # Example
///
/// ```
/// use cache::SwapDirection;
///
/// assert_eq!(SwapDirection::GpuToCpu.reverse(), SwapDirection::CpuToGpu);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SwapDirection {
    /// Move a resident GPU cache block into host storage.
    GpuToCpu,
    /// Restore a host cache block into GPU storage.
    CpuToGpu,
}

impl SwapDirection {
    /// Returns the opposite transfer direction.
    pub const fn reverse(self) -> Self {
        match self {
            Self::GpuToCpu => Self::CpuToGpu,
            Self::CpuToGpu => Self::GpuToCpu,
        }
    }
}

/// Description of a block swap operation.
///
/// `SwapPlan` is intentionally data-only so an engine worker can translate it
/// into copy kernels or asynchronous runtime copies without mutating manager
/// state while planning.
///
/// # Example
///
/// ```
/// use cache::{BlockId, SwapDirection, SwapPlan};
///
/// let plan = SwapPlan::new(vec![BlockId::new(0)], SwapDirection::GpuToCpu);
/// assert_eq!(plan.blocks(), &[BlockId::new(0)]);
/// assert_eq!(plan.direction(), SwapDirection::GpuToCpu);
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SwapPlan {
    blocks: Vec<BlockId>,
    direction: SwapDirection,
}

impl SwapPlan {
    /// Creates a swap plan for `blocks` in `direction`.
    pub fn new(blocks: Vec<BlockId>, direction: SwapDirection) -> Self {
        Self { blocks, direction }
    }

    /// Returns the planned block ids.
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Returns the transfer direction.
    pub const fn direction(&self) -> SwapDirection {
        self.direction
    }
}
