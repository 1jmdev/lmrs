/// Tokenizer crate error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// File-system error while loading tokenizer assets.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Hugging Face tokenizer error.
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    /// JSON parse error for tokenizer metadata.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// Jinja template error.
    #[error("template error: {0}")]
    MiniJinja(#[from] minijinja::Error),
    /// Tokio task join error.
    #[error("task join error: {0}")]
    Join(#[from] tokio::task::JoinError),
    /// Invalid tokenizer template configuration.
    #[error("template error: {0}")]
    Template(String),
}

/// Result alias for tokenizer operations.
pub type Result<T> = std::result::Result<T, Error>;
