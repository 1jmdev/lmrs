use std::path::{Path, PathBuf};

use runtime::CudaContext;
use tensor::{Result, Tensor};

use crate::{
    Cacheable, LoadedModel, Model, ModelConfig, ModelMetadata, load_config, load_model, qwen3,
};

/// A loaded causal LM plus tokenizer-side assets resolved from the same model id.
///
/// `model` owns the architecture-specific implementation selected from
/// `config.json`. The asset paths let higher layers keep tokenization outside the
/// model crate while still using the exact files that were resolved with the
/// checkpoint.
///
/// # Example
///
/// ```no_run
/// use model::AutoModelForCausalLM;
/// use runtime::CudaContext;
///
/// # fn main() -> anyhow::Result<()> {
/// let context = CudaContext::new(0)?;
/// let loaded = AutoModelForCausalLM::load("Qwen/Qwen3-0.6B", None, &context)?;
/// assert_eq!(loaded.model.metadata().model_type, "qwen3");
/// assert!(loaded.tokenizer_path.ends_with("tokenizer.json"));
/// # Ok(())
/// # }
/// ```
pub struct AutoLoadedModel {
    /// Automatically routed causal language model.
    pub model: AutoModelForCausalLM,
    /// Path to `config.json`.
    pub config_path: PathBuf,
    /// Path to `tokenizer.json`.
    pub tokenizer_path: PathBuf,
    /// Optional `tokenizer_config.json` path.
    pub tokenizer_config_path: Option<PathBuf>,
    /// Optional chat template path.
    pub chat_template_path: Option<PathBuf>,
}

/// Trait object bound used by the automatic causal LM router.
pub trait CausalLM: Model + Cacheable + Send {}

impl<T> CausalLM for T where T: Model + Cacheable + Send {}

/// Architecture-neutral causal language model wrapper.
///
/// The wrapper inspects Hugging Face-style `config.json` metadata and builds the
/// matching concrete implementation. Engines should depend on this type instead
/// of matching model families themselves.
///
/// # Example
///
/// ```no_run
/// use model::AutoModelForCausalLM;
/// use runtime::CudaContext;
///
/// # fn main() -> anyhow::Result<()> {
/// let context = CudaContext::new(0)?;
/// let loaded = AutoModelForCausalLM::load("/models/qwen3", None, &context)?;
/// let metadata = loaded.model.metadata();
/// assert_eq!(metadata.model_type, "qwen3");
/// # Ok(())
/// # }
/// ```
pub struct AutoModelForCausalLM {
    inner: Box<dyn CausalLM>,
}

impl AutoModelForCausalLM {
    /// Loads a model id or local directory and routes to the correct architecture.
    pub fn load(
        model: &str,
        revision: Option<&str>,
        context: &CudaContext,
    ) -> anyhow::Result<AutoLoadedModel> {
        let loaded = load_model(model, revision, context)?;
        Self::from_loaded(loaded)
    }

    /// Builds an auto model from already resolved checkpoint files.
    pub fn from_loaded(loaded: LoadedModel) -> anyhow::Result<AutoLoadedModel> {
        let config = load_config(&loaded.config_path)?;
        let model = Self::from_config_path(&config, &loaded.config_path, loaded.weights)?;
        Ok(AutoLoadedModel {
            model,
            config_path: loaded.config_path,
            tokenizer_path: loaded.tokenizer_path,
            tokenizer_config_path: loaded.tokenizer_config_path,
            chat_template_path: loaded.chat_template_path,
        })
    }

    /// Builds an auto model from a parsed config and variable builder.
    pub fn from_config_path(
        config: &ModelConfig,
        config_path: &Path,
        weights: crate::WeightBuilder,
    ) -> anyhow::Result<Self> {
        match model_family(config).as_deref() {
            Some("qwen3") => {
                let typed: qwen3::Config = serde_json::from_value(config.as_value()?)?;
                Ok(Self {
                    inner: Box::new(qwen3::ModelForCausalLM::new(&typed, weights)?),
                })
            }
            Some(model_type) => anyhow::bail!("unsupported model architecture `{model_type}`"),
            None => anyhow::bail!(
                "could not infer model architecture from {}",
                config_path.display()
            ),
        }
    }

    /// Runs a forward pass and returns logits for the last token position.
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        self.inner.forward(input_ids, start_pos)
    }

    /// Clears all model-owned KV cache state.
    pub fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }

    /// Returns architecture metadata for the selected model.
    pub fn metadata(&self) -> ModelMetadata {
        self.inner.metadata()
    }
}

fn model_family(config: &ModelConfig) -> Option<String> {
    config
        .model_type()
        .map(normalize_architecture)
        .or_else(|| architecture_family(config))
}

fn architecture_family(config: &ModelConfig) -> Option<String> {
    let architectures = config.raw.get("architectures")?.as_array()?;
    architectures
        .iter()
        .filter_map(|value| value.as_str())
        .map(normalize_architecture)
        .find(|family| family == "qwen3")
}

fn normalize_architecture(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    if lower.contains("qwen3") {
        "qwen3".to_string()
    } else {
        lower
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_qwen3_from_model_type() {
        let config: ModelConfig = serde_json::from_str(
            r#"{"model_type":"qwen3","hidden_size":1,"intermediate_size":1,"num_attention_heads":1,"num_hidden_layers":1,"vocab_size":1,"rms_norm_eps":1e-6}"#,
        )
        .unwrap();

        assert_eq!(model_family(&config).as_deref(), Some("qwen3"));
    }

    #[test]
    fn routes_qwen3_from_architectures() {
        let config: ModelConfig = serde_json::from_str(
            r#"{"architectures":["Qwen3ForCausalLM"],"hidden_size":1,"intermediate_size":1,"num_attention_heads":1,"num_hidden_layers":1,"vocab_size":1,"rms_norm_eps":1e-6}"#,
        )
        .unwrap();

        assert_eq!(model_family(&config).as_deref(), Some("qwen3"));
    }
}
