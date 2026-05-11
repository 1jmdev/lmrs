use std::{fs, path::PathBuf};

use minijinja::{Environment, context};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    error::{AppError, Result},
    server::types::ChatMessage,
};

#[derive(Clone)]
pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    eos: Vec<u32>,
    chat_template: Option<String>,
}

impl TokenizerWrapper {
    pub fn from_file(
        path: PathBuf,
        tokenizer_config_path: Option<PathBuf>,
        chat_template_path: Option<PathBuf>,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)?;
        let eos = load_eos_tokens(&tokenizer, tokenizer_config_path)?;
        let chat_template = chat_template_path.map(fs::read_to_string).transpose()?;
        Ok(Self {
            tokenizer,
            eos,
            chat_template,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(text, true)?.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        Ok(self.tokenizer.decode(tokens, true)?)
    }

    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        let template = self.chat_template.as_ref().ok_or_else(|| {
            AppError::BadRequest("model does not provide chat_template.jinja".into())
        })?;
        let mut env = Environment::new();
        env.add_template("chat", template)?;
        Ok(env.get_template("chat")?.render(context! {
            messages => messages,
            add_generation_prompt => true,
            enable_thinking => false,
        })?)
    }

    pub fn is_eos(&self, token: u32) -> bool {
        self.eos.contains(&token)
    }
}

#[derive(Deserialize)]
struct TokenizerConfig {
    eos_token: Option<Value>,
}

fn load_eos_tokens(tokenizer: &Tokenizer, path: Option<PathBuf>) -> Result<Vec<u32>> {
    let Some(path) = path else {
        return Ok(Vec::new());
    };
    let config: TokenizerConfig = serde_json::from_str(&fs::read_to_string(path)?)?;
    let Some(eos_token) = config.eos_token else {
        return Ok(Vec::new());
    };
    let mut tokens = Vec::new();
    collect_eos_tokens(&eos_token, &mut tokens);
    let vocab = tokenizer.get_vocab(true);
    Ok(tokens
        .into_iter()
        .filter_map(|token| vocab.get(&token).copied())
        .collect())
}

fn collect_eos_tokens(value: &Value, tokens: &mut Vec<String>) {
    match value {
        Value::String(token) => tokens.push(token.clone()),
        Value::Object(map) => {
            if let Some(Value::String(content)) = map.get("content") {
                tokens.push(content.clone());
            }
        }
        Value::Array(values) => values
            .iter()
            .for_each(|value| collect_eos_tokens(value, tokens)),
        _ => {}
    }
}
