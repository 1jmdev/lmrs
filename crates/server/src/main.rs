use axum::serve;
use server::{AppState, ServerConfig, ServerEngine, build_router};
use tokio::net::TcpListener;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = ServerConfig::from_env();
    let engine = ServerEngine::load(&config)?;
    let state = AppState::new(engine);
    let app = build_router(state);
    let listener = TcpListener::bind(config.bind_addr()).await?;

    info!(addr = %config.bind_addr(), model = %config.model, "server listening");
    serve(listener, app).await?;
    Ok(())
}
