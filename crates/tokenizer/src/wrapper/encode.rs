use std::{fs, path::Path, sync::Arc};

use minijinja::{Environment, context};
use serde::Serialize;
use tokenizers::Tokenizer;

use super::{DecodeOptions, SpecialTokens};

/// A chat message accepted by tokenizer chat templates.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ChatMessage {
    role: String,
    content: String,
}

impl ChatMessage {
    /// Creates a chat message.
    ///
    /// ```
    /// use tokenizer::ChatMessage;
    /// let msg = ChatMessage::new("user", "hello");
    /// assert_eq!(msg.role(), "user");
    /// ```
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    /// Returns the message role.
    pub fn role(&self) -> &str {
        &self.role
    }

    /// Returns the message content.
    pub fn content(&self) -> &str {
        &self.content
    }
}

/// Options for text encoding.
///
/// # Example
///
/// ```
/// use tokenizer::wrapper::EncodeOptions;
///
/// assert!(EncodeOptions::default().add_special_tokens());
/// assert!(!EncodeOptions::without_special_tokens().add_special_tokens());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeOptions {
    add_special_tokens: bool,
}

impl EncodeOptions {
    /// Creates encoding options.
    pub fn new(add_special_tokens: bool) -> Self {
        Self { add_special_tokens }
    }

    /// Creates options that do not insert tokenizer special tokens.
    pub fn without_special_tokens() -> Self {
        Self::new(false)
    }

    /// Returns whether special tokens are inserted.
    pub fn add_special_tokens(&self) -> bool {
        self.add_special_tokens
    }
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self::new(true)
    }
}

/// Thread-safe tokenizer wrapper with sync and async encode/decode helpers.
///
/// # Example
///
/// ```
/// use ahash::AHashMap;
/// use tokenizers::{Tokenizer, models::wordlevel::WordLevel};
/// use tokenizer::TokenizerWrapper;
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
/// assert_eq!(wrapper.encode("hello")?, vec![0]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct TokenizerWrapper {
    tokenizer: Arc<Tokenizer>,
    special_tokens: SpecialTokens,
    chat_template: Option<Arc<str>>,
}

impl TokenizerWrapper {
    /// Creates a wrapper around an existing tokenizer.
    pub fn from_tokenizer(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
            special_tokens: SpecialTokens::default(),
            chat_template: None,
        }
    }

    /// Loads a tokenizer from `tokenizer.json` with optional config/template files.
    pub fn from_file(
        tokenizer_path: impl AsRef<Path>,
        tokenizer_config_path: Option<impl AsRef<Path>>,
        chat_template_path: Option<impl AsRef<Path>>,
    ) -> crate::Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())?;
        let special_tokens = match tokenizer_config_path {
            Some(path) => SpecialTokens::from_config_file(&tokenizer, path)?,
            None => SpecialTokens::default(),
        };
        let chat_template = match chat_template_path {
            Some(path) => Some(Arc::<str>::from(fs::read_to_string(path)?)),
            None => None,
        };
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            special_tokens,
            chat_template,
        })
    }

    /// Encodes text with default options.
    pub fn encode(&self, text: &str) -> crate::Result<Vec<u32>> {
        self.encode_with_options(text, EncodeOptions::default())
    }

    /// Encodes text with explicit options.
    pub fn encode_with_options(
        &self,
        text: &str,
        options: EncodeOptions,
    ) -> crate::Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, options.add_special_tokens())?
            .get_ids()
            .to_vec())
    }

    /// Encodes many texts in one tokenizer call.
    pub fn encode_batch(&self, texts: &[String], options: EncodeOptions) -> crate::Result<Vec<Vec<u32>>> {
        Ok(self
            .tokenizer
            .encode_batch(texts.to_vec(), options.add_special_tokens())?
            .into_iter()
            .map(|encoding| encoding.get_ids().to_vec())
            .collect())
    }

    /// Encodes text on Tokio's blocking pool.
    pub async fn encode_async(&self, text: String) -> crate::Result<Vec<u32>> {
        let this = self.clone();
        tokio::task::spawn_blocking(move || this.encode(&text)).await?
    }

    /// Returns the number of tokens produced without adding special tokens.
    pub fn count_text_tokens(&self, text: &str) -> crate::Result<usize> {
        Ok(self.tokenizer.encode(text, false)?.get_ids().len())
    }

    /// Applies the configured Jinja chat template.
    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> crate::Result<String> {
        let template = self.chat_template.as_ref().ok_or_else(|| {
            crate::Error::Template("tokenizer does not provide a chat template".into())
        })?;
        let mut env = Environment::new();
        env.add_template("chat", template)?;
        Ok(env.get_template("chat")?.render(context! {
            messages => messages,
            add_generation_prompt => true,
            enable_thinking => false,
        })?)
    }

    /// Returns true when `token` is an end-of-sequence token.
    pub fn is_eos(&self, token: u32) -> bool {
        self.special_tokens.is_eos(token)
    }

    /// Returns the special-token registry.
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Returns the underlying Hugging Face tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Decodes tokens with default options.
    pub fn decode(&self, tokens: &[u32]) -> crate::Result<String> {
        self.decode_with_options(tokens, DecodeOptions::default())
    }
}

#[cfg(test)]
mod tests {
    use ahash::AHashMap;
    use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

    use super::{ChatMessage, EncodeOptions, TokenizerWrapper};

    fn wrapper() -> crate::Result<TokenizerWrapper> {
        let tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(AHashMap::from_iter([
                    ("hello".to_string(), 0_u32),
                    ("world".to_string(), 1_u32),
                    ("[UNK]".to_string(), 2_u32),
                ]))
                .unk_token("[UNK]".to_string())
                .build()?,
        );
        Ok(TokenizerWrapper::from_tokenizer(tokenizer))
    }

    #[test]
    fn encodes_single_and_batch_text() -> crate::Result<()> {
        let wrapper = wrapper()?;
        assert_eq!(wrapper.encode("hello")?, vec![0]);
        let batch = wrapper.encode_batch(&["hello".into(), "world".into()], EncodeOptions::default())?;
        assert_eq!(batch, vec![vec![0], vec![1]]);
        Ok(())
    }

    #[test]
    fn renders_chat_template() -> crate::Result<()> {
        let dir = tempfile::tempdir()?;
        let tokenizer_path = dir.path().join("tokenizer.json");
        let template_path = dir.path().join("chat.jinja");
        wrapper()?.tokenizer().save(&tokenizer_path, false)?;
        std::fs::write(
            &template_path,
            "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}",
        )?;
        let wrapper = TokenizerWrapper::from_file(
            &tokenizer_path,
            None::<&std::path::Path>,
            Some(&template_path),
        )?;
        let text = wrapper.apply_chat_template(&[ChatMessage::new("user", "hello")])?;
        assert_eq!(text, "user: hello\n");
        Ok(())
    }
}
