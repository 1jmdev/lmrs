use std::{collections::HashMap, path::PathBuf, sync::Arc};

use tokio::sync::RwLock;

use crate::TokenizerWrapper;

/// Async tokenizer cache keyed by model id.
///
/// # Example
///
/// ```
/// use tokenizer::TokenizerRegistry;
///
/// let registry = TokenizerRegistry::new();
/// assert_eq!(registry.len_blocking(), 0);
/// ```
#[derive(Clone, Default)]
pub struct TokenizerRegistry {
    tokenizers: Arc<RwLock<HashMap<String, TokenizerWrapper>>>,
}

impl TokenizerRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads a tokenizer for `model_id`, returning the cached wrapper on later calls.
    ///
    /// # Example
    ///
    /// ```
    /// use ahash::AHashMap;
    /// use tokenizers::{Tokenizer, models::wordlevel::WordLevel};
    /// use tokenizer::TokenizerRegistry;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> tokenizer::Result<()> {
    /// let dir = tempfile::tempdir()?;
    /// let path = dir.path().join("tokenizer.json");
    /// Tokenizer::new(WordLevel::builder()
    ///     .vocab(AHashMap::from_iter([
    ///         ("hello".to_string(), 0_u32),
    ///         ("[UNK]".to_string(), 1_u32),
    ///     ]))
    ///     .unk_token("[UNK]".to_string())
    ///     .build()?)
    ///     .save(&path, false)?;
    /// let registry = TokenizerRegistry::new();
    /// registry.load("tiny", &path, None::<&std::path::Path>, None::<&std::path::Path>).await?;
    /// assert!(registry.get("tiny").await.is_some());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn load(
        &self,
        model_id: impl Into<String>,
        tokenizer_path: impl Into<PathBuf>,
        tokenizer_config_path: Option<impl Into<PathBuf>>,
        chat_template_path: Option<impl Into<PathBuf>>,
    ) -> crate::Result<TokenizerWrapper> {
        let model_id = model_id.into();
        if let Some(tokenizer) = self.get(&model_id).await {
            return Ok(tokenizer);
        }
        let wrapper = TokenizerWrapper::from_file(
            tokenizer_path.into(),
            tokenizer_config_path.map(Into::into),
            chat_template_path.map(Into::into),
        )?;
        self.tokenizers
            .write()
            .await
            .insert(model_id, wrapper.clone());
        Ok(wrapper)
    }

    /// Returns a cached tokenizer by model id.
    pub async fn get(&self, model_id: &str) -> Option<TokenizerWrapper> {
        self.tokenizers.read().await.get(model_id).cloned()
    }

    /// Inserts an already constructed tokenizer.
    pub async fn insert(&self, model_id: impl Into<String>, tokenizer: TokenizerWrapper) {
        self.tokenizers
            .write()
            .await
            .insert(model_id.into(), tokenizer);
    }

    /// Returns the number of cached tokenizers.
    pub async fn len(&self) -> usize {
        self.tokenizers.read().await.len()
    }

    /// Returns true when the registry is empty.
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// Returns the number of cached tokenizers from synchronous contexts.
    pub fn len_blocking(&self) -> usize {
        self.tokenizers.blocking_read().len()
    }
}

#[cfg(test)]
mod tests {
    use ahash::AHashMap;
    use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

    use super::TokenizerRegistry;

    #[tokio::test]
    async fn loads_and_caches_tokenizers() -> crate::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("tokenizer.json");
        Tokenizer::new(
            WordLevel::builder()
                .vocab(AHashMap::from_iter([
                    ("hello".to_string(), 0_u32),
                    ("[UNK]".to_string(), 1_u32),
                ]))
                .unk_token("[UNK]".to_string())
                .build()?,
        )
        .save(&path, false)?;

        let registry = TokenizerRegistry::new();
        let first = registry
            .load(
                "tiny",
                &path,
                None::<&std::path::Path>,
                None::<&std::path::Path>,
            )
            .await?;
        let second = registry
            .load(
                "tiny",
                &path,
                None::<&std::path::Path>,
                None::<&std::path::Path>,
            )
            .await?;
        assert_eq!(first.encode("hello")?, second.encode("hello")?);
        assert_eq!(registry.len().await, 1);
        Ok(())
    }
}
