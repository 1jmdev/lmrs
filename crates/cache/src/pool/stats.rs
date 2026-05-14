/// Point-in-time utilization metrics for a block pool.
///
/// # Example
///
/// ```
/// use cache::BlockPoolStats;
///
/// let stats = BlockPoolStats::new(10, 4, 1024);
/// assert_eq!(stats.used_blocks(), 6);
/// assert_eq!(stats.free_blocks(), 4);
/// assert_eq!(stats.used_bytes(), 6144);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockPoolStats {
    total_blocks: usize,
    free_blocks: usize,
    block_size_bytes: usize,
}

impl BlockPoolStats {
    /// Creates pool metrics from total blocks, free blocks, and block size.
    pub const fn new(total_blocks: usize, free_blocks: usize, block_size_bytes: usize) -> Self {
        Self {
            total_blocks,
            free_blocks,
            block_size_bytes,
        }
    }

    /// Returns the total number of blocks in the pool.
    pub const fn total_blocks(self) -> usize {
        self.total_blocks
    }

    /// Returns the number of currently free blocks.
    pub const fn free_blocks(self) -> usize {
        self.free_blocks
    }

    /// Returns the number of currently allocated blocks.
    pub const fn used_blocks(self) -> usize {
        self.total_blocks - self.free_blocks
    }

    /// Returns the fixed byte size of one block.
    pub const fn block_size_bytes(self) -> usize {
        self.block_size_bytes
    }

    /// Returns total backing capacity in bytes.
    pub const fn total_bytes(self) -> usize {
        self.total_blocks * self.block_size_bytes
    }

    /// Returns allocated backing capacity in bytes.
    pub const fn used_bytes(self) -> usize {
        self.used_blocks() * self.block_size_bytes
    }

    /// Returns free backing capacity in bytes.
    pub const fn free_bytes(self) -> usize {
        self.free_blocks * self.block_size_bytes
    }
}
