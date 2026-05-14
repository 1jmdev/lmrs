use std::{fs, path::Path};

use crate::config::ModelConfig;

/// Loads a Hugging Face-style `config.json` from disk.
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use model::load_config;
///
/// # fn main() -> anyhow::Result<()> {
/// let config = load_config(Path::new("/models/qwen3/config.json"))?;
/// assert!(config.hidden_size > 0);
/// # Ok(())
/// # }
/// ```
pub fn load_config(path: &Path) -> anyhow::Result<ModelConfig> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}
