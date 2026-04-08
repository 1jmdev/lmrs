use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::LmrsError;
use crate::model::{ModelArtifact, ModelSource};

#[derive(Clone, Debug)]
pub struct ModelResolverConfig {
    pub cache_dir: PathBuf,
    pub allow_download: bool,
    pub allow_conversion: bool,
    pub python_bin: String,
    pub hf_cli_bin: String,
}

impl Default for ModelResolverConfig {
    fn default() -> Self {
        Self {
            cache_dir: default_cache_dir(),
            allow_download: true,
            allow_conversion: true,
            python_bin: "python3".to_string(),
            hf_cli_bin: "huggingface-cli".to_string(),
        }
    }
}

pub struct ModelResolver {
    config: ModelResolverConfig,
}

impl ModelResolver {
    pub fn new(config: ModelResolverConfig) -> Self {
        Self { config }
    }

    pub fn resolve(&self, source: &ModelSource) -> Result<ModelArtifact, LmrsError> {
        match source {
            ModelSource::LocalPath(path) => self.resolve_local(path),
            ModelSource::HuggingFace {
                repo,
                revision,
                file,
            } => self.resolve_hugging_face(repo, revision.as_deref(), file.as_deref()),
        }
    }

    fn resolve_local(&self, path: &Path) -> Result<ModelArtifact, LmrsError> {
        if !path.exists() {
            return Err(LmrsError::ModelNotFound(path.to_path_buf()));
        }

        if path.is_file() {
            if has_gguf_extension(path) {
                return Ok(ModelArtifact {
                    gguf_path: path.to_path_buf(),
                    source_label: "local-gguf".to_string(),
                    cache_hit: true,
                });
            }

            let parent = path
                .parent()
                .ok_or_else(|| LmrsError::UnsupportedModelSource(path.display().to_string()))?;
            return self.convert_hf_directory(parent, "local-converted");
        }

        if let Some(gguf) = find_gguf(path)? {
            return Ok(ModelArtifact {
                gguf_path: gguf,
                source_label: "local-dir-gguf".to_string(),
                cache_hit: true,
            });
        }

        self.convert_hf_directory(path, "local-converted")
    }

    fn resolve_hugging_face(
        &self,
        repo: &str,
        revision: Option<&str>,
        file: Option<&str>,
    ) -> Result<ModelArtifact, LmrsError> {
        if !self.config.allow_download {
            return Err(LmrsError::UnsupportedModelSource(
                "download is disabled in resolver config".to_string(),
            ));
        }

        let repo_dir = self.hf_repo_dir(repo, revision);
        fs::create_dir_all(&repo_dir)?;

        if dir_is_empty(&repo_dir)? {
            let mut args = vec!["download".to_string(), repo.to_string()];
            if let Some(revision) = revision {
                args.push("--revision".to_string());
                args.push(revision.to_string());
            }
            if let Some(file) = file {
                args.push(file.to_string());
            }
            args.push("--local-dir".to_string());
            args.push(repo_dir.to_string_lossy().into_owned());
            run_command(&self.config.hf_cli_bin, &args)?;
        }

        if let Some(file_name) = file {
            let candidate = repo_dir.join(file_name);
            if candidate.exists() && has_gguf_extension(&candidate) {
                return Ok(ModelArtifact {
                    gguf_path: candidate,
                    source_label: "huggingface-gguf".to_string(),
                    cache_hit: true,
                });
            }
        }

        if let Some(gguf) = find_gguf(&repo_dir)? {
            return Ok(ModelArtifact {
                gguf_path: gguf,
                source_label: "huggingface-gguf".to_string(),
                cache_hit: true,
            });
        }

        self.convert_hf_directory(&repo_dir, "huggingface-converted")
    }

    fn convert_hf_directory(
        &self,
        source_dir: &Path,
        source_label: &str,
    ) -> Result<ModelArtifact, LmrsError> {
        if !self.config.allow_conversion {
            return Err(LmrsError::UnsupportedModelSource(
                "conversion is disabled in resolver config".to_string(),
            ));
        }

        let convert_script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("vendor")
            .join("llama.cpp")
            .join("convert_hf_to_gguf.py");
        if !convert_script.exists() {
            return Err(LmrsError::ModelNotFound(convert_script));
        }

        let output_file = self.conversion_output(source_dir);
        if output_file.exists() {
            return Ok(ModelArtifact {
                gguf_path: output_file,
                source_label: source_label.to_string(),
                cache_hit: true,
            });
        }

        if !contains_convertible_hf_model(source_dir)? {
            return Err(LmrsError::UnsupportedModelSource(format!(
                "directory does not contain a convertible HF model: {}",
                source_dir.display()
            )));
        }

        if let Some(parent) = output_file.parent() {
            fs::create_dir_all(parent)?;
        }

        let args = vec![
            convert_script.to_string_lossy().into_owned(),
            source_dir.to_string_lossy().into_owned(),
            "--outfile".to_string(),
            output_file.to_string_lossy().into_owned(),
        ];
        run_command(&self.config.python_bin, &args)?;

        if !output_file.exists() {
            return Err(LmrsError::ModelNotFound(output_file));
        }

        Ok(ModelArtifact {
            gguf_path: output_file,
            source_label: source_label.to_string(),
            cache_hit: false,
        })
    }

    fn hf_repo_dir(&self, repo: &str, revision: Option<&str>) -> PathBuf {
        let mut name = repo.replace('/', "--");
        if let Some(revision) = revision {
            name.push_str("--");
            name.push_str(&revision.replace('/', "_"));
        }
        self.config.cache_dir.join("hf").join(name)
    }

    fn conversion_output(&self, source_dir: &Path) -> PathBuf {
        let source_name = source_dir
            .file_name()
            .and_then(OsStr::to_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| "model".to_string());
        self.config
            .cache_dir
            .join("converted")
            .join(format!("{source_name}.f16.gguf"))
    }
}

fn has_gguf_extension(path: &Path) -> bool {
    path.extension().and_then(OsStr::to_str) == Some("gguf")
}

fn default_cache_dir() -> PathBuf {
    match env::var_os("LMRS_CACHE_DIR") {
        Some(path) => PathBuf::from(path),
        None => match env::var_os("HOME") {
            Some(home) => PathBuf::from(home).join(".cache").join("lmrs"),
            None => PathBuf::from(".lmrs-cache"),
        },
    }
}

fn contains_convertible_hf_model(path: &Path) -> Result<bool, LmrsError> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_file() {
            continue;
        }
        if let Some(extension) = entry.path().extension().and_then(OsStr::to_str) {
            if matches!(extension, "safetensors" | "bin" | "pt" | "pth") {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn find_gguf(path: &Path) -> Result<Option<PathBuf>, LmrsError> {
    let mut candidate = None;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        if entry.file_type()?.is_file() && has_gguf_extension(&entry_path) {
            candidate = Some(entry_path);
            break;
        }
    }
    Ok(candidate)
}

fn run_command(program: &str, args: &[String]) -> Result<(), LmrsError> {
    let output = Command::new(program).args(args).output()?;
    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    Err(LmrsError::CommandFailed {
        program: program.to_string(),
        args: args.to_vec(),
        stderr,
    })
}

fn dir_is_empty(path: &Path) -> Result<bool, LmrsError> {
    Ok(fs::read_dir(path)?.next().is_none())
}
