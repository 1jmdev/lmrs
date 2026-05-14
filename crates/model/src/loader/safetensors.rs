use std::{
    fs,
    path::{Path, PathBuf},
};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{Repo, RepoType, api::sync::Api};
use safetensors::SafeTensors;

/// Files and weight builder needed to construct a model.
///
/// The loader keeps tokenizer and chat-template paths beside the checkpoint
/// builder so server and tokenizer crates can consume the same resolved model
/// directory without re-running discovery.
///
/// # Example
///
/// ```no_run
/// use candle_core::Device;
/// use model::load_model;
///
/// # fn main() -> anyhow::Result<()> {
/// let device = Device::new_cuda(0)?;
/// let loaded = load_model("/models/qwen3", None, &device)?;
/// assert!(loaded.config_path.ends_with("config.json"));
/// # Ok(())
/// # }
/// ```
pub struct LoadedModel {
    /// Path to `config.json`.
    pub config_path: PathBuf,
    /// Path to `tokenizer.json`.
    pub tokenizer_path: PathBuf,
    /// Optional `tokenizer_config.json` path.
    pub tokenizer_config_path: Option<PathBuf>,
    /// Optional chat template path.
    pub chat_template_path: Option<PathBuf>,
    /// Memory-mapped safetensors variable builder.
    pub var_builder: VarBuilder<'static>,
}

/// Loads a local directory or Hugging Face repo into a BF16 variable builder.
///
/// Local directories must contain `config.json`, `tokenizer.json`, and
/// `model.safetensors`. The safetensors file is validated with the official
/// `safetensors` crate before Candle receives the mmap path.
///
/// # Example
///
/// ```no_run
/// use candle_core::Device;
/// use model::load_model;
///
/// # fn main() -> anyhow::Result<()> {
/// let device = Device::new_cuda(0)?;
/// let loaded = load_model("Qwen/Qwen3-0.6B", Some("main"), &device)?;
/// assert!(loaded.tokenizer_path.ends_with("tokenizer.json"));
/// # Ok(())
/// # }
/// ```
pub fn load_model(
    model: &str,
    revision: Option<&str>,
    device: &Device,
) -> anyhow::Result<LoadedModel> {
    let root = Path::new(model);
    if root.exists() {
        return load_from_dir(root, device);
    }
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model.to_owned(),
        RepoType::Model,
        revision.unwrap_or("main").to_owned(),
    ));
    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer_config_path = repo.get("tokenizer_config.json").ok();
    let chat_template_path = repo.get("chat_template.jinja").ok();
    let weights = repo
        .get("model.safetensors")
        .or_else(|_| repo.get("pytorch_model.bin"))?;
    let var_builder =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::BF16, device)? };
    Ok(LoadedModel {
        config_path,
        tokenizer_path,
        tokenizer_config_path,
        chat_template_path,
        var_builder,
    })
}

fn load_from_dir(root: &Path, device: &Device) -> anyhow::Result<LoadedModel> {
    let config_path = root.join("config.json");
    let tokenizer_path = root.join("tokenizer.json");
    let tokenizer_config_path = root.join("tokenizer_config.json");
    let chat_template_path = root.join("chat_template.jinja");
    if !config_path.exists() || !tokenizer_path.exists() {
        anyhow::bail!(
            "model directory must contain config.json and tokenizer.json: {}",
            root.display()
        );
    }
    let weights = vec![root.join("model.safetensors")];
    for weight in &weights {
        let bytes = fs::read(weight)?;
        SafeTensors::deserialize(&bytes)?;
    }
    let var_builder =
        unsafe { VarBuilder::from_mmaped_safetensors(&weights, DType::BF16, device)? };
    Ok(LoadedModel {
        config_path,
        tokenizer_path,
        tokenizer_config_path: tokenizer_config_path
            .exists()
            .then_some(tokenizer_config_path),
        chat_template_path: chat_template_path.exists().then_some(chat_template_path),
        var_builder,
    })
}
