use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use runtime::CudaContext;
use safetensors::Dtype;
use safetensors::SafeTensors;
use tensor::copy_h2d;
use tensor::{DType, Shape, Tensor, TensorError};

/// CUDA checkpoint tensor lookup scoped by model prefix.
///
/// `WeightBuilder` replaces framework-specific variable builders in the model
/// crate. It owns a CUDA context plus BF16 tensors loaded from safetensors and
/// provides `pp` prefixing that matches Hugging Face checkpoint names.
///
/// # Example
///
/// ```no_run
/// use std::collections::HashMap;
/// use model::WeightBuilder;
/// use runtime::CudaContext;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let context = CudaContext::new(0)?;
/// let weights = WeightBuilder::new(context, HashMap::new());
/// let layer_weights = weights.pp("model").pp("layers").pp(0);
/// let _ = layer_weights;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct WeightBuilder {
    context: CudaContext,
    tensors: Arc<HashMap<String, Tensor>>,
    prefix: String,
}

impl WeightBuilder {
    /// Creates a root weight builder from CUDA BF16 checkpoint tensors.
    ///
    /// Tensor names must use the exact checkpoint keys, for example
    /// `model.embed_tokens.weight`. The builder does not copy tensors; callers
    /// should pass tensors that are already resident on CUDA.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::collections::HashMap;
    /// use model::WeightBuilder;
    /// use runtime::CudaContext;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let context = CudaContext::new(0)?;
    /// let weights = WeightBuilder::new(context, HashMap::new());
    /// assert!(weights.get("missing.weight").is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(context: CudaContext, tensors: HashMap<String, Tensor>) -> Self {
        Self {
            context,
            tensors: Arc::new(tensors),
            prefix: String::new(),
        }
    }

    /// Returns the CUDA context used by all loaded tensors.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::collections::HashMap;
    /// use model::WeightBuilder;
    /// use runtime::CudaContext;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let context = CudaContext::new(0)?;
    /// let weights = WeightBuilder::new(context, HashMap::new());
    /// let _cuda = weights.context();
    /// # Ok(())
    /// # }
    /// ```
    pub fn context(&self) -> &CudaContext {
        &self.context
    }

    /// Returns a CUDA tensor by name under the current prefix.
    ///
    /// The returned tensor is a cheap clone of shared CUDA storage. Missing
    /// weights are reported as tensor errors so model constructors can fail fast
    /// with the exact checkpoint key.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::collections::HashMap;
    /// use model::WeightBuilder;
    /// use runtime::CudaContext;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let context = CudaContext::new(0)?;
    /// let weights = WeightBuilder::new(context, HashMap::new()).pp("model");
    /// let err = weights.get("embed_tokens.weight").unwrap_err();
    /// assert!(err.to_string().contains("model.embed_tokens.weight"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn get(&self, name: &str) -> tensor::Result<Tensor> {
        let key = self.key(name);
        self.tensors.get(&key).cloned().ok_or_else(|| {
            TensorError::InvalidArgument(format!("missing checkpoint tensor {key}"))
        })
    }

    /// Returns a child builder with one path segment appended.
    ///
    /// Prefixing is allocation-light and keeps tensor storage shared through an
    /// `Arc`, so constructors can freely pass scoped builders into submodules.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::collections::HashMap;
    /// use model::WeightBuilder;
    /// use runtime::CudaContext;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let context = CudaContext::new(0)?;
    /// let weights = WeightBuilder::new(context, HashMap::new());
    /// let first_layer = weights.pp("model").pp("layers").pp(0);
    /// assert!(first_layer.get("self_attn.q_proj.weight").is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn pp(&self, path: impl ToString) -> Self {
        let segment = path.to_string();
        let prefix = if self.prefix.is_empty() {
            segment
        } else {
            format!("{}.{}", self.prefix, segment)
        };
        Self {
            context: self.context.clone(),
            tensors: self.tensors.clone(),
            prefix,
        }
    }

    fn key(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        }
    }
}

/// Files and CUDA BF16 weights needed to construct a model.
///
/// The loader resolves tokenizer/config sidecar files and eagerly copies all
/// safetensors weights to CUDA. Model crates then build directly from
/// `WeightBuilder` without depending on external tensor frameworks.
///
/// # Example
///
/// ```no_run
/// use model::load_model;
/// use runtime::CudaContext;
///
/// # fn main() -> anyhow::Result<()> {
/// let context = CudaContext::new(0)?;
/// let loaded = load_model("/models/qwen3", None, &context)?;
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
    /// CUDA BF16 checkpoint tensors keyed by safetensors name.
    pub weights: WeightBuilder,
}

/// Loads a local model directory or Hugging Face repo into CUDA BF16 tensors.
///
/// Local directories must contain `config.json`, `tokenizer.json`, and
/// `model.safetensors`. Remote repositories are resolved through `hf-hub` and
/// currently require `model.safetensors` because this loader is CUDA-only and
/// does not implement PyTorch `.bin` conversion.
///
/// # Example
///
/// ```no_run
/// use model::load_model;
/// use runtime::CudaContext;
///
/// # fn main() -> anyhow::Result<()> {
/// let context = CudaContext::new(0)?;
/// let loaded = load_model("Qwen/Qwen3-0.6B", Some("main"), &context)?;
/// assert!(loaded.tokenizer_path.ends_with("tokenizer.json"));
/// # Ok(())
/// # }
/// ```
pub fn load_model(
    model: &str,
    revision: Option<&str>,
    context: &CudaContext,
) -> anyhow::Result<LoadedModel> {
    let root = Path::new(model);
    if root.exists() {
        return load_from_dir(root, context);
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
    let weights_path = repo.get("model.safetensors")?;
    let weights = load_weights(&[weights_path], context)?;
    Ok(LoadedModel {
        config_path,
        tokenizer_path,
        tokenizer_config_path,
        chat_template_path,
        weights,
    })
}

fn load_from_dir(root: &Path, context: &CudaContext) -> anyhow::Result<LoadedModel> {
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
    let weights = load_weights(&[root.join("model.safetensors")], context)?;
    Ok(LoadedModel {
        config_path,
        tokenizer_path,
        tokenizer_config_path: tokenizer_config_path
            .exists()
            .then_some(tokenizer_config_path),
        chat_template_path: chat_template_path.exists().then_some(chat_template_path),
        weights,
    })
}

fn load_weights(paths: &[PathBuf], context: &CudaContext) -> anyhow::Result<WeightBuilder> {
    let mut tensors = HashMap::new();
    for path in paths {
        let bytes = fs::read(path)?;
        let safetensors = SafeTensors::deserialize(&bytes)?;
        for name in safetensors.names() {
            let view = safetensors.tensor(name)?;
            if view.dtype() != Dtype::BF16 {
                anyhow::bail!("{name} has dtype {:?}; model weights must be BF16", view.dtype());
            }
            let shape = Shape::new(view.shape().to_vec())?;
            let data = view.data();
            let values = as_u16_slice(data)?;
            tensors.insert(name.to_string(), copy_h2d(context, shape, DType::BF16, values)?);
        }
    }
    Ok(WeightBuilder::new(context.clone(), tensors))
}

fn as_u16_slice(data: &[u8]) -> anyhow::Result<&[u16]> {
    let (prefix, values, suffix) = unsafe { data.align_to::<u16>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        anyhow::bail!("BF16 safetensor data is not u16-aligned");
    }
    Ok(values)
}
