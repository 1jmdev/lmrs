use std::collections::HashMap;

use crate::block::BlockId;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Node {
    blocks: Vec<BlockId>,
    children: HashMap<u32, Node>,
}

/// Token radix tree for prefix cache lookup.
///
/// The tree stores block ids at exact prefixes and returns the longest matching
/// stored prefix for an incoming token sequence.
///
/// # Example
///
/// ```
/// use cache::{BlockId, RadixPrefixCache};
///
/// let mut cache = RadixPrefixCache::new();
/// cache.insert([10, 20, 30], vec![BlockId::new(4)]);
/// let hit = cache.longest_match([10, 20, 30, 40]).unwrap();
/// assert_eq!(hit.matched_tokens(), 3);
/// assert_eq!(hit.blocks(), &[BlockId::new(4)]);
/// ```
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RadixPrefixCache {
    root: Node,
}

impl RadixPrefixCache {
    /// Creates an empty prefix cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces the block ids for an exact token prefix.
    pub fn insert<I>(&mut self, tokens: I, blocks: Vec<BlockId>)
    where
        I: IntoIterator<Item = u32>,
    {
        let mut node = &mut self.root;
        for token in tokens {
            node = node.children.entry(token).or_default();
        }
        node.blocks = blocks;
    }

    /// Returns the longest stored prefix that matches `tokens`.
    pub fn longest_match<I>(&self, tokens: I) -> Option<PrefixMatch>
    where
        I: IntoIterator<Item = u32>,
    {
        let mut node = &self.root;
        let mut best = (!node.blocks.is_empty()).then(|| PrefixMatch::new(0, node.blocks.clone()));

        for (index, token) in tokens.into_iter().enumerate() {
            let Some(next) = node.children.get(&token) else {
                break;
            };
            node = next;
            if !node.blocks.is_empty() {
                best = Some(PrefixMatch::new(index + 1, node.blocks.clone()));
            }
        }
        best
    }

    /// Returns whether the prefix cache has no stored block mappings.
    pub fn is_empty(&self) -> bool {
        self.root.children.is_empty() && self.root.blocks.is_empty()
    }
}

/// Result of a longest-prefix lookup.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixMatch {
    matched_tokens: usize,
    blocks: Vec<BlockId>,
}

impl PrefixMatch {
    /// Creates a prefix match record.
    pub fn new(matched_tokens: usize, blocks: Vec<BlockId>) -> Self {
        Self {
            matched_tokens,
            blocks,
        }
    }

    /// Returns the number of matched prefix tokens.
    pub const fn matched_tokens(&self) -> usize {
        self.matched_tokens
    }

    /// Returns the block ids associated with the prefix.
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }
}
