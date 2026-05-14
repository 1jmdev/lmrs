use std::net::SocketAddr;

use clap::Parser;

/// Runtime configuration for the HTTP server binary.
///
/// Values can be provided as CLI flags or environment variables. The defaults
/// bind the server to localhost and expose an OpenAI-compatible mock model that
/// exercises the full request, response, and SSE path without CUDA setup.
///
/// # Example
///
/// ```
/// use server::ServerConfig;
///
/// let config = ServerConfig::default();
/// assert_eq!(config.bind_addr().to_string(), "127.0.0.1:8000");
/// assert_eq!(config.model, "local-dev");
/// ```
#[derive(Clone, Debug, Parser)]
#[command(author, version, about = "OpenAI-compatible vLLM Rust server")]
pub struct ServerConfig {
    /// Host interface to bind.
    #[arg(long, env = "VLLM_RS_HOST", default_value = "127.0.0.1")]
    pub host: String,
    /// TCP port to bind.
    #[arg(long, env = "VLLM_RS_PORT", default_value_t = 8000)]
    pub port: u16,
    /// Public model id returned by OpenAI-compatible responses.
    #[arg(long, env = "VLLM_RS_MODEL", default_value = "local-dev")]
    pub model: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8000,
            model: "local-dev".into(),
        }
    }
}

impl ServerConfig {
    /// Parses configuration from process arguments and environment variables.
    pub fn from_env() -> Self {
        Self::parse()
    }

    /// Returns the socket address used by the listener.
    ///
    /// # Example
    ///
    /// ```
    /// use server::ServerConfig;
    ///
    /// let config = ServerConfig { host: "0.0.0.0".into(), port: 3000, ..Default::default() };
    /// assert_eq!(config.bind_addr().to_string(), "0.0.0.0:3000");
    /// ```
    pub fn bind_addr(&self) -> SocketAddr {
        format!("{}:{}", self.host, self.port)
            .parse()
            .expect("host and port must form a socket address")
    }
}
