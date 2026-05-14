use std::collections::HashMap;

use thiserror::Error;

use crate::block::{BlockId, BlockTable};
use crate::pool::{BlockPool, BlockPoolError};

/// Stable identifier for a sequence tracked by `CacheManager`.
///
/// # Example
///
/// ```
/// use cache::SequenceId;
///
/// let seq = SequenceId::new(42);
/// assert_eq!(seq.get(), 42);
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SequenceId(u64);

impl SequenceId {
    /// Creates a sequence id from an engine-level integer id.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw integer id.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Result of assigning new blocks to a sequence.
///
/// # Example
///
/// ```
/// use cache::{BlockAssignment, BlockId, SequenceId};
///
/// let assignment = BlockAssignment::new(SequenceId::new(1), vec![BlockId::new(0)]);
/// assert_eq!(assignment.sequence(), SequenceId::new(1));
/// assert_eq!(assignment.blocks(), &[BlockId::new(0)]);
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockAssignment {
    sequence: SequenceId,
    blocks: Vec<BlockId>,
}

impl BlockAssignment {
    /// Creates an assignment record.
    pub fn new(sequence: SequenceId, blocks: Vec<BlockId>) -> Self {
        Self { sequence, blocks }
    }

    /// Returns the sequence that received blocks.
    pub const fn sequence(&self) -> SequenceId {
        self.sequence
    }

    /// Returns the newly assigned block ids.
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Consumes the assignment and returns the block ids.
    pub fn into_blocks(self) -> Vec<BlockId> {
        self.blocks
    }
}

/// Errors returned by `CacheManager`.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum CacheManagerError {
    /// The sequence is already tracked.
    #[error("sequence {0} is already registered")]
    SequenceExists(u64),
    /// The sequence is not tracked.
    #[error("sequence {0} is not registered")]
    UnknownSequence(u64),
    /// The underlying block pool rejected an operation.
    #[error(transparent)]
    Pool(#[from] BlockPoolError),
}

/// Owns sequence-to-block-table mappings and allocates from a block pool.
///
/// `CacheManager` is the correctness layer above `BlockPool`: it ensures a block
/// allocated for a sequence is recorded exactly once and that freeing a sequence
/// returns every block to the pool.
///
/// # Example
///
/// ```
/// use cache::{BlockPool, CacheManager, SequenceId, SlotLayout};
///
/// let pool = BlockPool::new(4, SlotLayout::new(16, 128, 2)).unwrap();
/// let mut manager = CacheManager::new(pool);
/// let seq = SequenceId::new(5);
/// manager.register_sequence(seq).unwrap();
/// manager.assign_blocks(seq, 2).unwrap();
/// assert_eq!(manager.table(seq).unwrap().len(), 2);
/// let freed = manager.free_sequence(seq).unwrap();
/// assert_eq!(freed.len(), 2);
/// assert!(!manager.pool().has_leaks());
/// ```
#[derive(Debug)]
pub struct CacheManager {
    pool: BlockPool,
    tables: HashMap<SequenceId, BlockTable>,
}

impl CacheManager {
    /// Creates a manager over an existing block pool.
    pub fn new(pool: BlockPool) -> Self {
        Self {
            pool,
            tables: HashMap::new(),
        }
    }

    /// Registers a sequence with an empty block table.
    pub fn register_sequence(&mut self, sequence: SequenceId) -> Result<(), CacheManagerError> {
        if self.tables.contains_key(&sequence) {
            return Err(CacheManagerError::SequenceExists(sequence.get()));
        }
        self.tables
            .insert(sequence, BlockTable::new(self.pool.layout()));
        Ok(())
    }

    /// Allocates `count` blocks and appends them to the sequence table.
    pub fn assign_blocks(
        &mut self,
        sequence: SequenceId,
        count: usize,
    ) -> Result<BlockAssignment, CacheManagerError> {
        if !self.tables.contains_key(&sequence) {
            return Err(CacheManagerError::UnknownSequence(sequence.get()));
        }
        let blocks = self.pool.allocate(count)?;
        let table = self
            .tables
            .get_mut(&sequence)
            .expect("table existence checked before allocation");
        table.extend(blocks.iter().copied());
        Ok(BlockAssignment::new(sequence, blocks))
    }

    /// Frees every block assigned to `sequence` and removes its table.
    pub fn free_sequence(
        &mut self,
        sequence: SequenceId,
    ) -> Result<Vec<BlockId>, CacheManagerError> {
        let mut table = self
            .tables
            .remove(&sequence)
            .ok_or(CacheManagerError::UnknownSequence(sequence.get()))?;
        let blocks = table.drain();
        self.pool.free_many(blocks.iter().copied())?;
        Ok(blocks)
    }

    /// Returns a sequence block table.
    pub fn table(&self, sequence: SequenceId) -> Option<&BlockTable> {
        self.tables.get(&sequence)
    }

    /// Returns all tracked sequence ids.
    pub fn sequences(&self) -> impl Iterator<Item = SequenceId> + '_ {
        self.tables.keys().copied()
    }

    /// Returns the number of tracked sequences.
    pub fn sequence_count(&self) -> usize {
        self.tables.len()
    }

    /// Returns the underlying block pool.
    pub const fn pool(&self) -> &BlockPool {
        &self.pool
    }

    /// Returns the underlying block pool mutably.
    pub fn pool_mut(&mut self) -> &mut BlockPool {
        &mut self.pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SlotLayout;

    #[test]
    fn assign_and_free_sequence_without_leaks() {
        let pool = BlockPool::new(3, SlotLayout::new(4, 8, 2)).unwrap();
        let mut manager = CacheManager::new(pool);
        let sequence = SequenceId::new(11);

        manager.register_sequence(sequence).unwrap();
        let assignment = manager.assign_blocks(sequence, 2).unwrap();

        assert_eq!(assignment.blocks(), &[BlockId::new(0), BlockId::new(1)]);
        assert_eq!(manager.table(sequence).unwrap().capacity_tokens(), 8);
        assert_eq!(manager.pool().allocated_blocks(), 2);

        let freed = manager.free_sequence(sequence).unwrap();
        assert_eq!(freed, vec![BlockId::new(0), BlockId::new(1)]);
        assert_eq!(manager.pool().free_blocks(), 3);
        assert!(!manager.pool().has_leaks());
    }
}
