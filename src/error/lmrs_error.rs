use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io;
use std::path::PathBuf;

#[derive(Debug)]
pub enum LmrsError {
    ModelPathNotUtf8(PathBuf),
    ModelNotFound(PathBuf),
    UnsupportedModelSource(String),
    ModelLoadFailed,
    ContextInitFailed,
    CStringNul,
    TemplateApplyFailed,
    DecodeFailed(i32),
    TokenizeFailed(i32),
    TokenToPieceFailed(i32),
    NoLogits,
    Io(io::Error),
    CommandFailed {
        program: String,
        args: Vec<String>,
        stderr: String,
    },
}

impl Display for LmrsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelPathNotUtf8(path) => {
                write!(f, "model path is not valid UTF-8: {}", path.display())
            }
            Self::ModelNotFound(path) => write!(f, "model artifact not found: {}", path.display()),
            Self::UnsupportedModelSource(source) => {
                write!(f, "unsupported model source: {source}")
            }
            Self::ModelLoadFailed => f.write_str("llama.cpp failed to load model"),
            Self::ContextInitFailed => f.write_str("llama.cpp failed to create context"),
            Self::CStringNul => f.write_str("input contains an interior NUL byte"),
            Self::TemplateApplyFailed => f.write_str("llama.cpp failed to apply chat template"),
            Self::DecodeFailed(code) => write!(f, "llama.cpp decode failed with code {code}"),
            Self::TokenizeFailed(code) => write!(f, "llama.cpp tokenize failed with code {code}"),
            Self::TokenToPieceFailed(code) => {
                write!(f, "llama.cpp token_to_piece failed with code {code}")
            }
            Self::NoLogits => f.write_str("llama.cpp did not return logits"),
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::CommandFailed {
                program,
                args,
                stderr,
            } => {
                write!(f, "command failed: {program} {}", args.join(" "))?;
                if !stderr.is_empty() {
                    write!(f, ": {stderr}")?;
                }
                Ok(())
            }
        }
    }
}

impl Error for LmrsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for LmrsError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}
