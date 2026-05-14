/// Trait for models that maintain a reusable key/value cache across decode calls.
///
/// The engine can call this when a sequence finishes or when a request switches
/// from one prompt to another.
///
/// # Example
///
/// ```
/// use model::Cacheable;
///
/// struct DummyCache { cleared: bool }
///
/// impl Cacheable for DummyCache {
///     fn clear_kv_cache(&mut self) {
///         self.cleared = true;
///     }
/// }
///
/// let mut cache = DummyCache { cleared: false };
/// cache.clear_kv_cache();
/// assert!(cache.cleared);
/// ```
pub trait Cacheable {
    /// Clears all layer KV caches.
    fn clear_kv_cache(&mut self);
}
