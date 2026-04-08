use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::api::{GenerationConfig, Message, Sampling};
use crate::cache::{hash_text, FastCache};
use crate::error::LmrsError;
use crate::llama::{LlamaHandle, Token};
use crate::model::{ModelArtifact, ModelResolver, ModelSource};
use crate::runtime::config::RuntimeConfig;
use crate::runtime::sampler::{greedy_token, sample_token, XorShift64};
use crate::runtime::thinking_parser::ThinkingParser;
use crate::runtime::StreamChunk;
use tracing::{debug, instrument};

pub struct Runtime {
    handle: LlamaHandle,
    artifact: ModelArtifact,
    prompt_cache: FastCache<Vec<Token>>,
    piece_cache: FastCache<Vec<u8>>,
    rng: RwLock<XorShift64>,
}

impl Runtime {
    #[instrument(skip(source))]
    pub fn from_source(source: ModelSource) -> Result<Self, LmrsError> {
        Self::from_source_with_config(source, RuntimeConfig::default())
    }

    #[instrument(skip(source, config))]
    pub fn from_source_with_config(
        source: ModelSource,
        config: RuntimeConfig,
    ) -> Result<Self, LmrsError> {
        let started = Instant::now();
        let resolver = ModelResolver::new(config.resolver.clone());
        let artifact = resolver.resolve(&source)?;
        let handle = LlamaHandle::load(&artifact.gguf_path, config.load)?;

        let runtime = Self {
            handle,
            artifact,
            prompt_cache: FastCache::new(config.prompt_cache_size),
            piece_cache: FastCache::new(config.piece_cache_size),
            rng: RwLock::new(XorShift64::new(0x4D59_444F_4D31_4E47)),
        };
        debug!(
            load_ms = started.elapsed().as_millis() as u64,
            "runtime initialized"
        );
        Ok(runtime)
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
        mut on_chunk: F,
    ) -> Result<String, LmrsError>
    where
        F: FnMut(&[u8]),
    {
        self.generate_stream_events(messages, config, |chunk| {
            if let StreamChunk::Content(content) = chunk {
                on_chunk(&content);
            }
        })
    }

    pub fn generate_stream_events<F>(
        &self,
        messages: &[Message],
        config: GenerationConfig,
        on_chunk: F,
    ) -> Result<String, LmrsError>
    where
        F: FnMut(StreamChunk),
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
        F: FnMut(StreamChunk),
    {
        let started = Instant::now();
        let prompt = self
            .handle
            .apply_chat_template(messages, config.enable_thinking)?;
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

        let context_size = self.handle.context_size();
        let available_generation_tokens = context_size.saturating_sub(prompt_tokens.len());
        let max_tokens = config.max_tokens.min(available_generation_tokens);

        if max_tokens == 0 {
            return Ok(String::new());
        }

        let mut ingest = prompt_tokens.as_slice().to_vec();
        self.handle.decode(&mut ingest)?;

        if let Sampling::Temperature(sampling) = &config.sampling {
            if let Ok(mut rng) = self.rng.write() {
                rng.reseed(sampling.seed);
            }
        }

        let mut parser = ThinkingParser::new();
        let mut output_bytes = Vec::with_capacity(max_tokens.saturating_mul(4));
        let mut generated = 0usize;
        for _ in 0..max_tokens {
            let token = self.sample_next_token(&config.sampling)?;
            if self.handle.token_is_eog(token) {
                break;
            }

            let piece = self.token_piece(token)?;
            parser.push(&piece, &mut output_bytes, &mut on_chunk);

            let mut next = [token];
            self.handle.decode(&mut next)?;
            generated += 1;
        }

        parser.finish(&mut output_bytes, &mut on_chunk);

        let elapsed = started.elapsed();
        if generated > 0 {
            debug!(
                generated,
                elapsed_ms = elapsed.as_millis() as u64,
                tok_per_sec = (generated as f64 / elapsed.as_secs_f64()),
                "generation completed"
            );
        }

        Ok(String::from_utf8_lossy(&output_bytes).trim().to_string())
    }

    fn sample_next_token(&self, sampling: &Sampling) -> Result<Token, LmrsError> {
        let logits = self.handle.logits()?;
        let token = match sampling {
            Sampling::Greedy => greedy_token(logits),
            Sampling::Temperature(_) => {
                if let Ok(mut rng) = self.rng.write() {
                    sample_token(logits, sampling, &mut rng)
                } else {
                    None
                }
            }
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
