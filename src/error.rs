use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum LmrsError {
    ModelPathNotUtf8,
    CStringNul,
    ModelLoadFailed,
    ContextInitFailed,
    TemplateApplyFailed,
    DecodeFailed(i32),
    TokenizeFailed(i32),
    TokenToPieceFailed(i32),
    NoLogits,
}

impl Display for LmrsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelPathNotUtf8 => f.write_str("model path is not valid UTF-8"),
            Self::CStringNul => f.write_str("input contains an interior NUL byte"),
            Self::ModelLoadFailed => f.write_str("llama.cpp failed to load the GGUF model"),
            Self::ContextInitFailed => f.write_str("llama.cpp failed to create a context"),
            Self::TemplateApplyFailed => f.write_str("llama.cpp failed to apply a chat template"),
            Self::DecodeFailed(code) => write!(f, "llama.cpp decode failed with code {code}"),
            Self::TokenizeFailed(code) => write!(f, "llama.cpp tokenize failed with code {code}"),
            Self::TokenToPieceFailed(code) => {
                write!(f, "llama.cpp token_to_piece failed with code {code}")
            }
            Self::NoLogits => f.write_str("llama.cpp did not return logits"),
        }
    }
}

impl Error for LmrsError {}
