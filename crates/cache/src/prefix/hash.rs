use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Hash value for a token prefix.
///
/// # Example
///
/// ```
/// use cache::{PrefixHash, PrefixHasher};
///
/// let hash = PrefixHasher::hash_tokens([1_u32, 2, 3]);
/// assert_ne!(hash, PrefixHash::EMPTY);
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PrefixHash(u64);

impl PrefixHash {
    /// Hash for the empty prefix.
    pub const EMPTY: Self = Self(0);

    /// Creates a prefix hash from a raw value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw hash value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Stable token-prefix hashing helper.
///
/// The hasher includes token order and length, so `[1, 23]` and `[12, 3]` do not
/// collide due to naive concatenation.
pub struct PrefixHasher;

impl PrefixHasher {
    /// Hashes an ordered token prefix.
    pub fn hash_tokens<I, T>(tokens: I) -> PrefixHash
    where
        I: IntoIterator<Item = T>,
        T: Hash,
    {
        let mut hasher = DefaultHasher::new();
        let mut len = 0_usize;
        for token in tokens {
            token.hash(&mut hasher);
            len += 1;
        }
        len.hash(&mut hasher);
        PrefixHash::new(hasher.finish())
    }
}
