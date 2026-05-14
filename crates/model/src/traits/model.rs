use tensor::{Result, Tensor};

/// Minimal metadata exposed by a loaded model implementation.
///
/// Metadata is kept independent from checkpoint loading so schedulers and
/// serving layers can inspect a model without knowing its concrete family.
///
/// # Example
///
/// ```
/// use model::ModelMetadata;
///
/// let metadata = ModelMetadata {
///     model_type: "qwen3".to_string(),
///     vocab_size: 32000,
///     hidden_size: 4096,
///     num_hidden_layers: 32,
/// };
/// assert_eq!(metadata.model_type, "qwen3");
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelMetadata {
    /// Stable model family identifier, for example `qwen3`.
    pub model_type: String,
    /// Vocabulary size used by the input embedding and LM head.
    pub vocab_size: usize,
    /// Hidden state width.
    pub hidden_size: usize,
    /// Number of transformer blocks.
    pub num_hidden_layers: usize,
}

/// Common forward contract for autoregressive model implementations.
///
/// Implementations accept token ids plus the logical start position. A prompt
/// prefill call usually passes `start_pos = 0`; decode calls pass the current
/// cache length.
///
/// # Example
///
/// ```no_run
/// use model::{Model, ModelMetadata};
/// use tensor::{Result, Tensor};
///
/// struct Dummy;
///
/// impl Model for Dummy {
///     fn forward(&mut self, input_ids: &Tensor, _start_pos: usize) -> Result<Tensor> {
///         Ok(input_ids.clone())
///     }
///
///     fn metadata(&self) -> ModelMetadata {
///         ModelMetadata {
///             model_type: "dummy".to_string(),
///             vocab_size: 1,
///             hidden_size: 1,
///             num_hidden_layers: 0,
///         }
///     }
/// }
/// ```
pub trait Model {
    /// Runs a forward pass for token ids starting at `start_pos` in the sequence.
    fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor>;

    /// Returns static model metadata.
    fn metadata(&self) -> ModelMetadata;
}
