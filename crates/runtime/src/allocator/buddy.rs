/// A byte range inside a larger allocation.
///
/// # Example
///
/// ```
/// use runtime::BuddyBlock;
///
/// let block = BuddyBlock { offset: 128, size: 64 };
/// assert_eq!(block.offset, 128);
/// assert_eq!(block.size, 64);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuddyBlock {
    /// Byte offset from the start of the backing allocation.
    pub offset: usize,
    /// Size of the block in bytes.
    pub size: usize,
}

/// Manages power-of-two byte ranges inside one slab.
///
/// `BuddyAllocator` does not own CUDA memory. It only returns offsets and sizes
/// that a caller can apply to a separately owned allocation.
///
/// # Example
///
/// ```
/// use runtime::BuddyAllocator;
///
/// let mut allocator = BuddyAllocator::new(1024, 64);
/// let block = allocator.allocate(100).unwrap();
///
/// assert_eq!(block.size, 128);
/// ```
#[derive(Debug)]
pub struct BuddyAllocator {
    slab_size: usize,
    min_block_size: usize,
    free: Vec<BuddyBlock>,
}

impl BuddyAllocator {
    /// Creates a buddy allocator for one slab.
    ///
    /// `slab_size` and `min_block_size` are rounded up to powers of two.
    ///
    /// # Example
    ///
    /// ```
    /// use runtime::BuddyAllocator;
    ///
    /// let allocator = BuddyAllocator::new(1000, 24);
    /// assert_eq!(allocator.free_blocks()[0].size, 1024);
    /// ```
    pub fn new(slab_size: usize, min_block_size: usize) -> Self {
        let min_block_size = min_block_size.max(1).next_power_of_two();
        let slab_size = slab_size.max(min_block_size).next_power_of_two();
        Self {
            slab_size,
            min_block_size,
            free: vec![BuddyBlock {
                offset: 0,
                size: slab_size,
            }],
        }
    }

    /// Allocates a block that can hold at least `size` bytes.
    ///
    /// The returned block size is rounded up to the allocator's power-of-two
    /// class.
    ///
    /// # Example
    ///
    /// ```
    /// use runtime::BuddyAllocator;
    ///
    /// let mut allocator = BuddyAllocator::new(1024, 64);
    /// let block = allocator.allocate(65).unwrap();
    ///
    /// assert_eq!(block.size, 128);
    /// ```
    pub fn allocate(&mut self, size: usize) -> Option<BuddyBlock> {
        let target = size.max(self.min_block_size).next_power_of_two();
        let index = self.free.iter().position(|block| block.size >= target)?;
        let block = self.free.swap_remove(index);
        let mut current = block;

        while current.size / 2 >= target {
            current.size /= 2;
            self.free.push(BuddyBlock {
                offset: current.offset + current.size,
                size: current.size,
            });
        }

        Some(current)
    }

    /// Frees a block and merges it with free buddies when possible.
    ///
    /// # Example
    ///
    /// ```
    /// use runtime::BuddyAllocator;
    ///
    /// let mut allocator = BuddyAllocator::new(1024, 64);
    /// let block = allocator.allocate(64).unwrap();
    /// allocator.free(block);
    ///
    /// assert_eq!(allocator.free_blocks()[0].size, 1024);
    /// ```
    pub fn free(&mut self, mut block: BuddyBlock) {
        loop {
            let buddy_offset = block.offset ^ block.size;
            let Some(index) = self.free.iter().position(|candidate| {
                candidate.offset == buddy_offset && candidate.size == block.size
            }) else {
                self.free.push(block);
                return;
            };

            self.free.swap_remove(index);
            block.offset = block.offset.min(buddy_offset);
            block.size *= 2;

            if block.size > self.slab_size {
                self.free.push(BuddyBlock {
                    offset: 0,
                    size: self.slab_size,
                });
                return;
            }
        }
    }

    /// Returns the currently free blocks.
    ///
    /// # Example
    ///
    /// ```
    /// use runtime::BuddyAllocator;
    ///
    /// let allocator = BuddyAllocator::new(512, 64);
    /// assert_eq!(allocator.free_blocks().len(), 1);
    /// ```
    pub fn free_blocks(&self) -> &[BuddyBlock] {
        &self.free
    }
}
