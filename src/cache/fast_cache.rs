use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub struct FastCache<V> {
    capacity: usize,
    entries: RwLock<HashMap<u64, Arc<V>>>,
}

impl<V> FastCache<V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: RwLock::new(HashMap::new()),
        }
    }

    pub fn get(&self, key: u64) -> Option<Arc<V>> {
        self.entries
            .read()
            .ok()
            .and_then(|entries| entries.get(&key).cloned())
    }

    pub fn insert(&self, key: u64, value: V) -> Arc<V> {
        let value = Arc::new(value);
        if let Ok(mut entries) = self.entries.write() {
            if self.capacity > 0 && entries.len() >= self.capacity {
                if let Some(key_to_drop) = entries.keys().next().copied() {
                    entries.remove(&key_to_drop);
                }
            }
            entries.insert(key, value.clone());
        }
        value
    }
}
