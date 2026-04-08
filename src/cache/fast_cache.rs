use std::sync::Arc;

use moka::sync::Cache;

pub struct FastCache<V> {
    entries: Cache<u64, Arc<V>>,
}

impl<V> FastCache<V>
where
    V: Send + Sync + 'static,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Cache::new(capacity as u64),
        }
    }

    pub fn get(&self, key: u64) -> Option<Arc<V>> {
        self.entries.get(&key)
    }

    pub fn insert(&self, key: u64, value: V) -> Arc<V> {
        let value = Arc::new(value);
        self.entries.insert(key, value.clone());
        value
    }
}
