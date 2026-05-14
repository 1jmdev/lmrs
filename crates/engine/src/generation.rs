use std::{path::PathBuf, sync::Mutex};

use candle_core::{Device, Tensor};
use model::AutoModelForCausalLM;
use sampling::{Greedy, Sampler, SamplerConfig, Temperature, TopK, TopP};

/// Paths for tokenizer assets resolved while loading the generation engine.
///
/// The engine does not tokenize text itself; callers can load these files in the
/// tokenizer crate and pass token ids to `generate`.
///
/// # Example
///
/// ```
/// use std::path::PathBuf;
/// use engine::TokenizerAssets;
///
/// let assets = TokenizerAssets::new(PathBuf::from("tokenizer.json"), None, None);
/// assert!(assets.tokenizer_path().ends_with("tokenizer.json"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenizerAssets {
    tokenizer_path: PathBuf,
    tokenizer_config_path: Option<PathBuf>,
    chat_template_path: Option<PathBuf>,
}

impl TokenizerAssets {
    /// Creates tokenizer asset paths.
    pub fn new(
        tokenizer_path: PathBuf,
        tokenizer_config_path: Option<PathBuf>,
        chat_template_path: Option<PathBuf>,
    ) -> Self {
        Self {
            tokenizer_path,
            tokenizer_config_path,
            chat_template_path,
        }
    }

    /// Returns the `tokenizer.json` path.
    pub fn tokenizer_path(&self) -> &PathBuf {
        &self.tokenizer_path
    }

    /// Returns the optional `tokenizer_config.json` path.
    pub fn tokenizer_config_path(&self) -> Option<&PathBuf> {
        self.tokenizer_config_path.as_ref()
    }

    /// Returns the optional chat template path.
    pub fn chat_template_path(&self) -> Option<&PathBuf> {
        self.chat_template_path.as_ref()
    }
}

/// Sampling controls for autoregressive generation.
///
/// # Example
///
/// ```
/// use engine::GenerationConfig;
///
/// let config = GenerationConfig::new(32, 0.8, 0.95, Some(40)).unwrap();
/// assert_eq!(config.max_tokens(), 32);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationConfig {
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    top_k: Option<usize>,
}

impl GenerationConfig {
    /// Creates validated generation parameters.
    pub fn new(
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
        top_k: Option<usize>,
    ) -> anyhow::Result<Self> {
        if temperature <= 0.0 {
            anyhow::bail!("temperature must be greater than zero");
        }
        if !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
            anyhow::bail!("top_p must be in the range (0, 1]");
        }
        Ok(Self {
            max_tokens,
            temperature,
            top_p,
            top_k,
        })
    }

    /// Returns the maximum number of tokens to generate.
    pub const fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Returns sampling temperature.
    pub const fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Returns nucleus sampling threshold.
    pub const fn top_p(&self) -> f64 {
        self.top_p
    }

    /// Returns optional top-k sampling cutoff.
    pub const fn top_k(&self) -> Option<usize> {
        self.top_k
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
        }
    }
}

/// Token ids produced by the generation engine.
///
/// # Example
///
/// ```
/// use engine::GenerationOutput;
///
/// let output = GenerationOutput::new(vec![1, 2, 3]);
/// assert_eq!(output.tokens(), 3);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationOutput {
    token_ids: Vec<u32>,
}

impl GenerationOutput {
    /// Creates output from generated token ids.
    pub fn new(token_ids: Vec<u32>) -> Self {
        Self { token_ids }
    }

    /// Returns generated token ids.
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    /// Consumes output and returns generated token ids.
    pub fn into_token_ids(self) -> Vec<u32> {
        self.token_ids
    }

    /// Returns the number of generated tokens.
    pub fn tokens(&self) -> usize {
        self.token_ids.len()
    }
}

/// Architecture-neutral generation engine.
///
/// This type owns the routed model and sampling loop. It accepts already-tokenized
/// prompts so the server/tokenizer layer can remain responsible for chat
/// templates, token counting, decoding, and stop strings.
///
/// # Example
///
/// ```no_run
/// use candle_core::Device;
/// use engine::{GenerationConfig, GenerationEngine};
///
/// # fn main() -> anyhow::Result<()> {
/// let engine = GenerationEngine::load("Qwen/Qwen3-0.6B", None, Device::new_cuda(0)?)?;
/// let output = engine.generate(&[1, 2, 3], GenerationConfig::default(), |token| token == 2)?;
/// assert!(output.tokens() <= 128);
/// # Ok(())
/// # }
/// ```
pub struct GenerationEngine {
    model_id: String,
    model: Mutex<AutoModelForCausalLM>,
    device: Device,
    tokenizer_assets: TokenizerAssets,
}

impl GenerationEngine {
    /// Loads a model on CUDA device 0.
    pub fn load_cuda0(model_id: impl Into<String>, revision: Option<&str>) -> anyhow::Result<Self> {
        Self::load(model_id, revision, Device::new_cuda(0)?)
    }

    /// Loads a local directory or Hugging Face model id and routes architecture automatically.
    pub fn load(
        model_id: impl Into<String>,
        revision: Option<&str>,
        device: Device,
    ) -> anyhow::Result<Self> {
        let model_id = model_id.into();
        let loaded = AutoModelForCausalLM::load(&model_id, revision, &device)?;
        Ok(Self {
            model_id,
            model: Mutex::new(loaded.model),
            device,
            tokenizer_assets: TokenizerAssets::new(
                loaded.tokenizer_path,
                loaded.tokenizer_config_path,
                loaded.chat_template_path,
            ),
        })
    }

    /// Creates an engine from a prebuilt auto-routed model.
    pub fn from_model(
        model_id: impl Into<String>,
        model: AutoModelForCausalLM,
        device: Device,
        tokenizer_assets: TokenizerAssets,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            model: Mutex::new(model),
            device,
            tokenizer_assets,
        }
    }

    /// Returns the public model id associated with this engine.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Returns tokenizer assets resolved with the model checkpoint.
    pub fn tokenizer_assets(&self) -> &TokenizerAssets {
        &self.tokenizer_assets
    }

    /// Generates token ids from an already-tokenized prompt.
    pub fn generate(
        &self,
        input_ids: &[u32],
        config: GenerationConfig,
        mut is_eos: impl FnMut(u32) -> bool,
    ) -> anyhow::Result<GenerationOutput> {
        if input_ids.is_empty() {
            anyhow::bail!("prompt encoded to zero tokens");
        }

        let mut model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
        model.clear_kv_cache();

        let mut sampler = sampler_for(&config)?;
        let mut context = input_ids.to_vec();
        let mut generated = Vec::with_capacity(config.max_tokens());
        let mut start_pos = 0;

        for step in 0..config.max_tokens() {
            let forward_tokens: Vec<u32> = if step == 0 {
                context.clone()
            } else {
                vec![*context.last().expect("context is never empty")]
            };
            let input = Tensor::new(forward_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = model.forward(&input, start_pos)?.flatten_all()?;
            let token = sampler.sample(&logits, &generated)?.token_id();

            if is_eos(token) {
                break;
            }

            generated.push(token);
            context.push(token);
            start_pos += forward_tokens.len();
        }

        Ok(GenerationOutput::new(generated))
    }
}

fn sampler_for(config: &GenerationConfig) -> anyhow::Result<Sampler> {
    let strategy: Box<dyn sampling::SamplingStrategy> = if let Some(top_k) = config.top_k() {
        Box::new(TopK::new(top_k)?)
    } else if config.top_p() < 1.0 {
        Box::new(TopP::new(config.top_p())?)
    } else {
        Box::new(Greedy)
    };
    let mut sampler = Sampler::new(SamplerConfig::new(strategy));
    if config.temperature() != 1.0 {
        sampler.push_processor(Temperature::new(config.temperature())?);
    }
    Ok(sampler)
}
