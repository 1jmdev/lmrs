use std::{
    fs,
    path::{Path, PathBuf},
};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{Repo, RepoType, api::sync::Api};

use crate::error::{AppError, Result};

pub struct LoadedModel {
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub tokenizer_config_path: Option<PathBuf>,
    pub chat_template_path: Option<PathBuf>,
    pub var_builder: VarBuilder<'static>,
}

pub fn load_model(model: &str, revision: Option<&str>, device: &Device) -> Result<LoadedModel> {
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
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F16, device)? };
    Ok(LoadedModel {
        config_path,
        tokenizer_path,
        tokenizer_config_path,
        chat_template_path,
        var_builder: vb,
    })
}

fn load_from_dir(root: &Path, device: &Device) -> Result<LoadedModel> {
    let config_path = root.join("config.json");
    let tokenizer_path = root.join("tokenizer.json");
    let tokenizer_config_path = root.join("tokenizer_config.json");
    let chat_template_path = root.join("chat_template.jinja");
    let weights = root.join("model.safetensors");
    if !config_path.exists() || !tokenizer_path.exists() || !weights.exists() {
        return Err(AppError::BadRequest(format!(
            "model directory must contain config.json, tokenizer.json, and model.safetensors: {}",
            root.display()
        )));
    }
    fs::metadata(&weights)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F16, device)? };
    Ok(LoadedModel {
        config_path,
        tokenizer_path,
        tokenizer_config_path: tokenizer_config_path
            .exists()
            .then_some(tokenizer_config_path),
        chat_template_path: chat_template_path.exists().then_some(chat_template_path),
        var_builder: vb,
    })
}
