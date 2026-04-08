pub use crate::llama::LoadConfig;
use crate::model::ModelResolverConfig;

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub load: LoadConfig,
    pub resolver: ModelResolverConfig,
    pub prompt_cache_size: usize,
    pub piece_cache_size: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            load: LoadConfig::default(),
            resolver: ModelResolverConfig::default(),
            prompt_cache_size: 1024,
            piece_cache_size: 4096,
        }
    }
}
