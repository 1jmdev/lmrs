use crate::api::Sampling;

#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub enable_thinking: bool,
    pub sampling: Sampling,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 32_768,
            enable_thinking: true,
            sampling: Sampling::Greedy,
        }
    }
}
