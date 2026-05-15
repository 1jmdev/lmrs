use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder, linear_no_bias, rms_norm};
use cache::AttentionContext;
use ops::RotaryEmbedding;

use crate::layers::{DecoderLayer, DecoderLayerConfig};
use crate::traits::{Cacheable, Model, ModelMetadata};

/// Construction contract every decoder-only architecture must satisfy.
///
/// Implement this on your `Config` struct and you get a fully functional
/// `ModelForCausalLM<YourConfig>` with no extra boilerplate.
pub trait TransformerConfig: Clone + Send + 'static {
    /// Stable identifier returned in `ModelMetadata`, e.g. `"qwen3"`.
    const MODEL_TYPE: &'static str;
    fn vocab_size(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_hidden_layers(&self) -> usize;
    fn rms_norm_eps(&self) -> f64;
    fn max_position_embeddings(&self) -> usize;
    fn rope_theta(&self) -> f64;
    fn tie_word_embeddings(&self) -> bool;
    /// Full per-layer configuration including attention, MLP, and norms.
    fn layer_config(&self) -> DecoderLayerConfig;
}

/// Generic CUDA-only BF16 causal language model for decoder-only architectures.
///
/// Each model family defines a `Config` that implements `TransformerConfig`, then
/// aliases this type: `pub type ModelForCausalLM = base::ModelForCausalLM<Config>;`
pub struct ModelForCausalLM<C: TransformerConfig> {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary: RotaryEmbedding,
    config: C,
}

impl<C: TransformerConfig> ModelForCausalLM<C> {
    /// Builds the model from checkpoint weights. Requires CUDA + BF16.
    ///
    /// ```no_run
    /// use candle_core::{DType, Device};
    /// use candle_nn::VarBuilder;
    /// use model::qwen3::{Config, ModelForCausalLM};
    /// # fn main() -> candle_core::Result<()> {
    /// let device = Device::new_cuda(0)?;
    /// let vb = VarBuilder::from_tensors(std::collections::HashMap::new(), DType::BF16, &device);
    /// let cfg: Config = serde_json::from_str(r#"{"vocab_size":8,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-6}"#).unwrap();
    /// let _model = ModelForCausalLM::new(&cfg, vb)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: &C, vb: VarBuilder) -> Result<Self> {
        if !vb.device().is_cuda() { candle_core::bail!("model is CUDA-only"); }
        if vb.dtype() != DType::BF16 { candle_core::bail!("model is BF16-only"); }
        // Disable CUDA event tracking once at init — removes per-op CPU sync overhead.
        if let Device::Cuda(dev) = vb.device() {
            if dev.is_event_tracking() { unsafe { dev.disable_event_tracking() }; }
        }
        let m = vb.pp("model");
        let embed = candle_nn::embedding(config.vocab_size(), config.hidden_size(), m.pp("embed_tokens"))?;
        let lcfg = config.layer_config();
        let head_dim = lcfg.attention.head_dim;
        let lv = m.pp("layers");
        let layers = (0..config.num_hidden_layers())
            .map(|i| DecoderLayer::new(lcfg.clone(), lv.pp(i)))
            .collect::<Result<_>>()?;
        let norm = rms_norm(config.hidden_size(), config.rms_norm_eps(), m.pp("norm"))?;
        let lm_head = if config.tie_word_embeddings() {
            Linear::new(embed.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size(), config.vocab_size(), vb.pp("lm_head"))?
        };
        let rotary = RotaryEmbedding::new(head_dim, config.max_position_embeddings(), config.rope_theta(), DType::BF16, vb.device())?;
        Ok(Self { embed_tokens: embed, layers, norm, lm_head, rotary, config: config.clone() })
    }

    /// Forward pass returning logits shaped `[1, 1, vocab_size]` for the last input token.
    ///
    /// ```no_run
    /// # use candle_core::{DType, Device, Tensor};
    /// # use candle_nn::VarBuilder;
    /// # use model::qwen3::{Config, ModelForCausalLM};
    /// # fn main() -> candle_core::Result<()> {
    /// # let device = Device::new_cuda(0)?;
    /// # let vb = VarBuilder::from_tensors(std::collections::HashMap::new(), DType::BF16, &device);
    /// # let cfg: Config = serde_json::from_str(r#"{"vocab_size":8,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-6}"#).unwrap();
    /// let mut model = ModelForCausalLM::new(&cfg, vb)?;
    /// let ids = Tensor::new(&[[1u32, 2u32]], &device)?;
    /// let logits = model.forward(&ids, 0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let (cos, sin) = self.rotary.get(start_pos + seq_len, start_pos, seq_len)?;
        let mut x = self.embed_tokens.forward(input_ids)?;
        let ctx = if start_pos == 0 { AttentionContext::Prefill } else { AttentionContext::Decode { start_pos } };
        for layer in &mut self.layers { x = layer.forward(&x, &cos, &sin, ctx)?; }
        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x.narrow(1, seq_len - 1, 1)?)
    }
}

impl<C: TransformerConfig> Model for ModelForCausalLM<C> {
    fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> { Self::forward(self, input_ids, start_pos) }
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata { model_type: C::MODEL_TYPE.to_string(), vocab_size: self.config.vocab_size(), hidden_size: self.config.hidden_size(), num_hidden_layers: self.config.num_hidden_layers() }
    }
}

impl<C: TransformerConfig> Cacheable for ModelForCausalLM<C> {
    fn clear_kv_cache(&mut self) { for layer in &mut self.layers { layer.clear_kv_cache(); } }
}
