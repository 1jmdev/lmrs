use std::{fs, path::PathBuf};

use minijinja::{Environment, context};
use tokenizers::Tokenizer;

use crate::{error::{AppError, Result}, server::types::ChatMessage};

#[derive(Clone)]
pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    eos: Option<u32>,
    chat_template: Option<String>,
}

impl TokenizerWrapper {
    pub fn from_file(path: PathBuf, chat_template_path: Option<PathBuf>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)?;
        let eos = tokenizer.get_vocab(true).get("</s>").copied().or_else(|| tokenizer.get_vocab(true).get("<|endoftext|>").copied());
        let chat_template = chat_template_path.map(fs::read_to_string).transpose()?;
        Ok(Self { tokenizer, eos, chat_template })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(text, true)?.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        Ok(self.tokenizer.decode(tokens, true)?)
    }

    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        let template = self.chat_template.as_ref().ok_or_else(|| AppError::BadRequest("model does not provide chat_template.jinja".into()))?;
        let mut env = Environment::new();
        env.add_template("chat", template)?;
        Ok(env.get_template("chat")?.render(context! {
            messages => messages,
            add_generation_prompt => true,
            enable_thinking => false,
        })?)
    }

    pub fn is_eos(&self, token: u32) -> bool { self.eos == Some(token) }
}
