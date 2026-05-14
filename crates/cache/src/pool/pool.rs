use std::collections::HashSet;

use runtime::CudaContext;
use tensor::CudaBuf;
use thiserror::Error;

use crate::block::{BlockId, CacheBlock, SlotLayout};
use crate::pool::BlockPoolStats;

/// Errors returned by `BlockPool` allocation and free operations.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum BlockPoolError {
    /// The requested allocation cannot fit in the remaining free blocks.
    #[error("block pool exhausted: requested {requested} blocks, only {available} free")]
    Exhausted { requested: usize, available: usize },
    /// The caller attempted to free a block id outside the pool range.
    #[error("block id {block} is outside pool capacity {capacity}")]
    OutOfRange { block: usize, capacity: usize },
    /// The caller attempted to free a block that is already free.
    #[error("block id {block} is already free")]
    DoubleFree { block: usize },
    /// The layout has no usable bytes per block.
    #[error("block layout must contain at least one byte")]
    EmptyLayout,
    /// CUDA backing allocation failed.
    #[error("CUDA backing allocation failed: {0}")]
    Tensor(String),
}

/// Free-list allocator over fixed-size cache blocks.
///
/// `BlockPool` tracks block ownership and can optionally own a single contiguous
/// CUDA allocation large enough for all blocks. Allocation and free operations are
/// O(number of requested blocks) and deterministic: the lowest free ids are
/// handed out first after construction.
///
/// # Example
///
/// ```
/// use cache::{BlockId, BlockPool, SlotLayout};
///
/// let layout = SlotLayout::new(16, 128, 2);
/// let mut pool = BlockPool::new(4, layout).unwrap();
/// let blocks = pool.allocate(2).unwrap();
/// assert_eq!(blocks, vec![BlockId::new(0), BlockId::new(1)]);
/// pool.free_many(blocks).unwrap();
/// assert!(!pool.has_leaks());
/// ```
#[derive(Debug)]
pub struct BlockPool {
    layout: SlotLayout,
    free: Vec<BlockId>,
    allocated: HashSet<BlockId>,
    backing: Option<CudaBuf>,
    capacity: usize,
}

impl BlockPool {
    /// Creates a metadata-only pool with `capacity` fixed-size blocks.
    pub fn new(capacity: usize, layout: SlotLayout) -> Result<Self, BlockPoolError> {
        Self::validate_layout(layout)?;
        let free = (0..capacity).rev().map(BlockId::new).collect();
        Ok(Self {
            layout,
            free,
            allocated: HashSet::with_capacity(capacity),
            backing: None,
            capacity,
        })
    }

    /// Creates a block pool backed by one contiguous CUDA byte allocation.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::CudaContext;
    /// use cache::{BlockPool, SlotLayout};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let context = CudaContext::new(0)?;
    /// let pool = BlockPool::new_cuda(&context, 8, SlotLayout::new(16, 128, 2))?;
    /// assert_eq!(pool.capacity(), 8);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_cuda(
        context: &CudaContext,
        capacity: usize,
        layout: SlotLayout,
    ) -> Result<Self, BlockPoolError> {
        let mut pool = Self::new(capacity, layout)?;
        let bytes = capacity * layout.block_size_bytes();
        pool.backing = Some(
            CudaBuf::new(context, bytes).map_err(|err| BlockPoolError::Tensor(err.to_string()))?,
        );
        Ok(pool)
    }

    /// Allocates `count` blocks from the free list.
    pub fn allocate(&mut self, count: usize) -> Result<Vec<BlockId>, BlockPoolError> {
        if count > self.free.len() {
            return Err(BlockPoolError::Exhausted {
                requested: count,
                available: self.free.len(),
            });
        }

        let mut blocks = Vec::with_capacity(count);
        for _ in 0..count {
            let block = self.free.pop().expect("free length checked before pop");
            self.allocated.insert(block);
            blocks.push(block);
        }
        Ok(blocks)
    }

    /// Frees one allocated block back into the free list.
    pub fn free(&mut self, block: BlockId) -> Result<(), BlockPoolError> {
        if block.index() >= self.capacity {
            return Err(BlockPoolError::OutOfRange {
                block: block.index(),
                capacity: self.capacity,
            });
        }
        if !self.allocated.remove(&block) {
            return Err(BlockPoolError::DoubleFree {
                block: block.index(),
            });
        }
        self.free.push(block);
        Ok(())
    }

    /// Frees multiple blocks. If any block is invalid, earlier blocks remain freed.
    pub fn free_many<I>(&mut self, blocks: I) -> Result<(), BlockPoolError>
    where
        I: IntoIterator<Item = BlockId>,
    {
        for block in blocks {
            self.free(block)?;
        }
        Ok(())
    }

    /// Returns metadata for a block id in this pool.
    pub fn block(&self, id: BlockId) -> Option<CacheBlock> {
        (id.index() < self.capacity).then_some(CacheBlock::new(id, self.layout))
    }

    /// Returns this pool's slot layout.
    pub const fn layout(&self) -> SlotLayout {
        self.layout
    }

    /// Returns the total number of blocks managed by this pool.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of free blocks.
    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    /// Returns the number of allocated blocks.
    pub fn allocated_blocks(&self) -> usize {
        self.allocated.len()
    }

    /// Returns true when every block has been returned to the free list.
    pub fn has_leaks(&self) -> bool {
        !self.allocated.is_empty()
    }

    /// Returns point-in-time pool utilization metrics.
    pub fn stats(&self) -> BlockPoolStats {
        BlockPoolStats::new(
            self.capacity,
            self.free.len(),
            self.layout.block_size_bytes(),
        )
    }

    /// Returns the optional contiguous CUDA backing allocation.
    pub fn backing(&self) -> Option<&CudaBuf> {
        self.backing.as_ref()
    }

    fn validate_layout(layout: SlotLayout) -> Result<(), BlockPoolError> {
        if layout.block_size_bytes() == 0 {
            return Err(BlockPoolError::EmptyLayout);
        }
        Ok(())
    }
}
