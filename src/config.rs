use std::{net::SocketAddr, path::PathBuf};

use clap::Parser;

#[derive(Debug, Clone, Parser)]
#[command(author, version, about)]
pub struct AppConfig {
    #[arg(long, env = "VLLM_RS_HOST", default_value = "127.0.0.1")]
    pub host: String,
    #[arg(long, env = "VLLM_RS_PORT", default_value_t = 8000)]
    pub port: u16,
    #[arg(long, env = "VLLM_RS_MODEL")]
    pub model: String,
    #[arg(long, env = "VLLM_RS_REVISION")]
    pub revision: Option<String>,
    #[arg(long, env = "VLLM_RS_TOKENIZER")]
    pub tokenizer: Option<PathBuf>,
    #[arg(long, env = "VLLM_RS_DEVICE", default_value = "auto")]
    pub device: String,
}

impl AppConfig {
    pub fn from_env() -> Self {
        Self::parse()
    }

    pub fn bind_addr(&self) -> SocketAddr {
        format!("{}:{}", self.host, self.port)
            .parse()
            .expect("validated socket address")
    }
}
