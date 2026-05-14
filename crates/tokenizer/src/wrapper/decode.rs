use super::TokenizerWrapper;

/// Options for token decoding.
///
/// # Example
///
/// ```
/// use tokenizer::wrapper::DecodeOptions;
///
/// assert!(DecodeOptions::default().skip_special_tokens());
/// assert!(!DecodeOptions::including_special_tokens().skip_special_tokens());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeOptions {
    skip_special_tokens: bool,
}

impl DecodeOptions {
    /// Creates decoding options.
    pub fn new(skip_special_tokens: bool) -> Self {
        Self { skip_special_tokens }
    }

    /// Creates options that preserve special tokens in decoded text.
    pub fn including_special_tokens() -> Self {
        Self::new(false)
    }

    /// Returns whether special tokens are omitted.
    pub fn skip_special_tokens(&self) -> bool {
        self.skip_special_tokens
    }
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self::new(true)
    }
}

impl TokenizerWrapper {
    /// Decodes tokens with explicit options.
    ///
    /// # Example
    ///
    /// ```
    /// use ahash::AHashMap;
    /// use tokenizers::{Tokenizer, models::wordlevel::WordLevel};
    /// use tokenizer::{TokenizerWrapper, wrapper::DecodeOptions};
    ///
    /// # fn main() -> tokenizer::Result<()> {
    /// let tokenizer = Tokenizer::new(WordLevel::builder()
    ///     .vocab(AHashMap::from_iter([
    ///         ("hello".to_string(), 0_u32),
    ///         ("[UNK]".to_string(), 1_u32),
    ///     ]))
    ///     .unk_token("[UNK]".to_string())
    ///     .build()?);
    /// let wrapper = TokenizerWrapper::from_tokenizer(tokenizer);
    /// assert_eq!(wrapper.decode_with_options(&[0], DecodeOptions::default())?, "hello");
    /// # Ok(())
    /// # }
    /// ```
    pub fn decode_with_options(
        &self,
        tokens: &[u32],
        options: DecodeOptions,
    ) -> crate::Result<String> {
        Ok(self.tokenizer().decode(tokens, options.skip_special_tokens())?)
    }

    /// Decodes tokens on Tokio's blocking pool.
    pub async fn decode_async(&self, tokens: Vec<u32>) -> crate::Result<String> {
        let this = self.clone();
        tokio::task::spawn_blocking(move || this.decode(&tokens)).await?
    }
}
