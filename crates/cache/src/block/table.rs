use crate::block::{BlockId, CacheBlock, SlotLayout};

/// Per-sequence mapping from logical token positions to cache blocks.
///
/// The table owns only block ids. Blocks remain owned by `BlockPool`; callers free
/// the ids after removing a sequence from the manager.
///
/// # Example
///
/// ```
/// use cache::{BlockId, BlockTable, SlotLayout};
///
/// let mut table = BlockTable::new(SlotLayout::new(4, 16, 2));
/// table.push(BlockId::new(9));
/// assert_eq!(table.block_for_token(3), Some(BlockId::new(9)));
/// assert_eq!(table.block_for_token(4), None);
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockTable {
    layout: SlotLayout,
    blocks: Vec<BlockId>,
}

impl BlockTable {
    /// Creates an empty table using `layout` for token-to-block addressing.
    pub fn new(layout: SlotLayout) -> Self {
        Self {
            layout,
            blocks: Vec::new(),
        }
    }

    /// Creates a table from already allocated block ids.
    pub fn from_blocks(layout: SlotLayout, blocks: Vec<BlockId>) -> Self {
        Self { layout, blocks }
    }

    /// Appends one block id to the logical end of the sequence table.
    pub fn push(&mut self, block: BlockId) {
        self.blocks.push(block);
    }

    /// Extends the table with allocated block ids.
    pub fn extend<I>(&mut self, blocks: I)
    where
        I: IntoIterator<Item = BlockId>,
    {
        self.blocks.extend(blocks);
    }

    /// Removes and returns all blocks from the table.
    pub fn drain(&mut self) -> Vec<BlockId> {
        self.blocks.drain(..).collect()
    }

    /// Returns the layout used by this table.
    pub const fn layout(&self) -> SlotLayout {
        self.layout
    }

    /// Returns the number of blocks held by this table.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Returns whether the table holds no blocks.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Returns the block id at table index `index`.
    pub fn get(&self, index: usize) -> Option<BlockId> {
        self.blocks.get(index).copied()
    }

    /// Returns an immutable view of all block ids.
    pub fn as_slice(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Returns block metadata for `index`.
    pub fn cache_block(&self, index: usize) -> Option<CacheBlock> {
        self.get(index).map(|id| CacheBlock::new(id, self.layout))
    }

    /// Returns the block containing a logical token position.
    pub fn block_for_token(&self, token_position: usize) -> Option<BlockId> {
        let block_index = token_position / self.layout.slots_per_block();
        self.get(block_index)
    }

    /// Returns how many token slots this table can address.
    pub fn capacity_tokens(&self) -> usize {
        self.blocks.len() * self.layout.slots_per_block()
    }
}
