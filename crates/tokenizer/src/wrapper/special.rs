use std::{fs, path::Path};

use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

/// Special token ids extracted from tokenizer metadata.
///
/// # Example
///
/// ```
/// use tokenizer::SpecialTokens;
///
/// let specials = SpecialTokens::new(vec![2, 3]);
/// assert!(specials.is_eos(2));
/// assert!(!specials.is_eos(9));
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpecialTokens {
    eos: Vec<u32>,
}

impl SpecialTokens {
    /// Creates a special-token registry from end-of-sequence token ids.
    pub fn new(mut eos: Vec<u32>) -> Self {
        eos.sort_unstable();
        eos.dedup();
        Self { eos }
    }

    /// Loads EOS token ids from a Hugging Face `tokenizer_config.json` file.
    pub fn from_config_file(tokenizer: &Tokenizer, path: impl AsRef<Path>) -> crate::Result<Self> {
        let config: TokenizerConfig = serde_json::from_str(&fs::read_to_string(path)?)?;
        let Some(eos_token) = config.eos_token else {
            return Ok(Self::default());
        };
        let mut tokens = Vec::new();
        collect_eos_tokens(&eos_token, &mut tokens);
        let vocab = tokenizer.get_vocab(true);
        Ok(Self::new(
            tokens
                .into_iter()
                .filter_map(|token| vocab.get(&token).copied())
                .collect(),
        ))
    }

    /// Returns true when `token` is an end-of-sequence id.
    pub fn is_eos(&self, token: u32) -> bool {
        self.eos.binary_search(&token).is_ok()
    }

    /// Returns all known end-of-sequence token ids.
    pub fn eos(&self) -> &[u32] {
        &self.eos
    }
}

#[derive(Deserialize)]
struct TokenizerConfig {
    eos_token: Option<Value>,
}

fn collect_eos_tokens(value: &Value, tokens: &mut Vec<String>) {
    match value {
        Value::String(token) => tokens.push(token.clone()),
        Value::Object(map) => {
            if let Some(Value::String(content)) = map.get("content") {
                tokens.push(content.clone());
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_eos_tokens(value, tokens);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

    use super::SpecialTokens;

    #[test]
    fn loads_string_and_object_eos_tokens() -> crate::Result<()> {
        let tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(
                    [("</s>".to_string(), 2), ("<|end|>".to_string(), 3)]
                        .into_iter()
                        .collect(),
                )
                .unk_token("</s>".to_string())
                .build()?,
        );
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("tokenizer_config.json");
        fs::write(
            &path,
            r#"{"eos_token":["</s>",{"content":"<|end|>"}]}"#,
        )?;
        let specials = SpecialTokens::from_config_file(&tokenizer, &path)?;
        assert_eq!(specials.eos(), &[2, 3]);
        Ok(())
    }
}
