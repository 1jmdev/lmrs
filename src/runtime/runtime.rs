use std::sync::{Arc, RwLock};

use crate::api::{GenerationConfig, Message, Sampling};
use crate::cache::{FastCache, hash_text};
use crate::error::LmrsError;
use crate::llama::{LlamaHandle, Token};
use crate::model::{ModelArtifact, ModelResolver, ModelSource};
use crate::runtime::config::RuntimeConfig;
use crate::runtime::sampler::{XorShift64, sample_token};

pub struct Runtime {
    handle: LlamaHandle,
    artifact: ModelArtifact,
    prompt_cache: FastCache<Vec<Token>>,
    piece_cache: FastCache<Vec<u8>>,
    rng: RwLock<XorShift64>,
}

impl Runtime {
    pub fn from_source(source: ModelSource) -> Result<Self, LmrsError> {
        Self::from_source_with_config(source, RuntimeConfig::default())
    }

    pub fn from_source_with_config(
        source: ModelSource,
        config: RuntimeConfig,
    ) -> Result<Self, LmrsError> {
        let resolver = ModelResolver::new(config.resolver.clone());
        let artifact = resolver.resolve(&source)?;
        let handle = LlamaHandle::load(&artifact.gguf_path, config.load)?;

        Ok(Self {
            handle,
            artifact,
            prompt_cache: FastCache::new(config.prompt_cache_size),
            piece_cache: FastCache::new(config.piece_cache_size),
            rng: RwLock::new(XorShift64::new(0x4D59_444F_4D31_4E47)),
        })
    }

    pub fn artifact(&self) -> &ModelArtifact {
        &self.artifact
    }

    pub fn generate(
        &self,
        messages: &[Message],
        config: GenerationConfig,
    ) -> Result<String, LmrsError> {
        self.generate_with_callback(messages, config, |_| {})
    }

    pub fn generate_stream<F>(
        &self,
        messages: &[Message],
        config: GenerationConfig,
        on_chunk: F,
    ) -> Result<String, LmrsError>
    where
        F: FnMut(&[u8]),
    {
        self.generate_with_callback(messages, config, on_chunk)
    }

    pub fn count_tokens(&self, text: &str) -> Result<usize, LmrsError> {
        self.handle.count_tokens(text)
    }

    fn generate_with_callback<F>(
        &self,
        messages: &[Message],
        config: GenerationConfig,
        mut on_chunk: F,
    ) -> Result<String, LmrsError>
    where
        F: FnMut(&[u8]),
    {
        let prompt = self.handle.apply_chat_template(messages)?;
        let prompt_key = hash_text(&prompt);
        let prompt_tokens = match self.prompt_cache.get(prompt_key) {
            Some(tokens) => tokens,
            None => self
                .prompt_cache
                .insert(prompt_key, self.handle.tokenize(&prompt)?),
        };

        if prompt_tokens.is_empty() {
            return Ok(String::new());
        }

        let mut ingest = prompt_tokens.as_slice().to_vec();
        self.handle.decode(&mut ingest)?;

        if let Sampling::Temperature(sampling) = &config.sampling {
            if let Ok(mut rng) = self.rng.write() {
                rng.reseed(sampling.seed);
            }
        }

        let mut output_bytes = Vec::with_capacity(config.max_tokens.saturating_mul(4));
        for _ in 0..config.max_tokens {
            let token = self.sample_next_token(&config.sampling)?;
            if self.handle.token_is_eog(token) {
                break;
            }

            let piece = self.token_piece(token)?;
            on_chunk(&piece);
            output_bytes.extend_from_slice(&piece);

            let mut next = [token];
            self.handle.decode(&mut next)?;
        }

        Ok(String::from_utf8_lossy(&output_bytes).trim().to_string())
    }

    fn sample_next_token(&self, sampling: &Sampling) -> Result<Token, LmrsError> {
        let logits = self.handle.logits()?;
        let token = if let Ok(mut rng) = self.rng.write() {
            sample_token(logits, sampling, &mut rng)
        } else {
            None
        };
        token.ok_or(LmrsError::NoLogits)
    }

    fn token_piece(&self, token: Token) -> Result<Arc<Vec<u8>>, LmrsError> {
        let key = token as u64;
        if let Some(piece) = self.piece_cache.get(key) {
            return Ok(piece);
        }

        let piece = self.handle.token_to_piece_bytes(token)?;
        Ok(self.piece_cache.insert(key, piece))
    }
}
