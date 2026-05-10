mod config;
mod engine;
mod error;
mod model;
mod sampling;
mod server;
mod tokenizer;

use std::sync::Arc;

use axum::serve;
use tokio::net::TcpListener;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;

use crate::{config::AppConfig, engine::InferenceEngine, server::router::build_router};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = AppConfig::from_env();
    let engine = Arc::new(InferenceEngine::load(&config)?);
    let app = build_router(engine)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    let listener = TcpListener::bind(config.bind_addr()).await?;
    info!(addr = %config.bind_addr(), "vllm-rs listening");
    serve(listener, app).await?;

    Ok(())
}
