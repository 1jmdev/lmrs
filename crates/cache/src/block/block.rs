/// Stable identifier for one fixed-size KV-cache block in a pool.
///
/// Block identifiers are dense indexes into `BlockPool`. They are cheap to copy
/// and safe to store in per-sequence block tables.
///
/// # Example
///
/// ```
/// use cache::BlockId;
///
/// let id = BlockId::new(7);
/// assert_eq!(id.index(), 7);
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BlockId(usize);

impl BlockId {
    /// Creates a block id from a dense pool index.
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the dense pool index for this block.
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Byte layout of one cache block inside the pool backing allocation.
///
/// `offset_bytes` is relative to the start of the backing allocation. Kernels can
/// derive the address of a block as `base + offset_bytes` and then address token
/// slots using `slot_stride_bytes`.
///
/// # Example
///
/// ```
/// use cache::SlotLayout;
///
/// let layout = SlotLayout::new(16, 128, 2);
/// assert_eq!(layout.block_size_bytes(), 4096);
/// assert_eq!(layout.slot_offset_bytes(3), Some(384));
/// assert_eq!(layout.slot_offset_bytes(16), None);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SlotLayout {
    slots_per_block: usize,
    slot_stride_bytes: usize,
    planes: usize,
}

impl SlotLayout {
    /// Creates a slot layout.
    ///
    /// `planes` is normally `2` for key and value planes. It is kept explicit so
    /// grouped or latent-cache layouts can reserve different plane counts.
    pub const fn new(slots_per_block: usize, slot_stride_bytes: usize, planes: usize) -> Self {
        Self {
            slots_per_block,
            slot_stride_bytes,
            planes,
        }
    }

    /// Returns how many token slots fit in one block.
    pub const fn slots_per_block(self) -> usize {
        self.slots_per_block
    }

    /// Returns the byte stride between two adjacent token slots in one plane.
    pub const fn slot_stride_bytes(self) -> usize {
        self.slot_stride_bytes
    }

    /// Returns how many contiguous planes are stored per block.
    pub const fn planes(self) -> usize {
        self.planes
    }

    /// Returns the total byte size of one cache block.
    pub const fn block_size_bytes(self) -> usize {
        self.slots_per_block * self.slot_stride_bytes * self.planes
    }

    /// Returns the byte offset of `slot` inside one plane of the block.
    pub fn slot_offset_bytes(self, slot: usize) -> Option<usize> {
        (slot < self.slots_per_block).then_some(slot * self.slot_stride_bytes)
    }
}

/// Metadata for one allocated cache block.
///
/// A `CacheBlock` does not own device memory. It describes which region of the
/// pool backing allocation belongs to a sequence table.
///
/// # Example
///
/// ```
/// use cache::{BlockId, CacheBlock, SlotLayout};
///
/// let block = CacheBlock::new(BlockId::new(2), SlotLayout::new(8, 64, 2));
/// assert_eq!(block.offset_bytes(), 2048);
/// assert_eq!(block.len_bytes(), 1024);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheBlock {
    id: BlockId,
    layout: SlotLayout,
}

impl CacheBlock {
    /// Creates metadata for `id` using `layout`.
    pub const fn new(id: BlockId, layout: SlotLayout) -> Self {
        Self { id, layout }
    }

    /// Returns this block's identifier.
    pub const fn id(self) -> BlockId {
        self.id
    }

    /// Returns this block's slot layout.
    pub const fn layout(self) -> SlotLayout {
        self.layout
    }

    /// Returns this block's byte offset in the pool backing allocation.
    pub const fn offset_bytes(self) -> usize {
        self.id.index() * self.layout.block_size_bytes()
    }

    /// Returns this block's byte length.
    pub const fn len_bytes(self) -> usize {
        self.layout.block_size_bytes()
    }

    /// Returns whether this block has zero bytes.
    pub const fn is_empty(self) -> bool {
        self.len_bytes() == 0
    }
}
