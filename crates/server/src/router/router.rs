use axum::{Router, routing::get, routing::post};
use tower_http::cors::CorsLayer;

use crate::{
    handlers::{chat_completions, completions, health, metrics, models},
    middleware::trace_layer,
    router::AppState,
};

/// Builds the OpenAI-compatible HTTP router.
///
/// # Example
///
/// ```no_run
/// use server::{AppState, ServerConfig, ServerEngine, build_router};
///
/// # fn main() -> anyhow::Result<()> {
/// let config = ServerConfig::from_env();
/// let state = AppState::new(ServerEngine::load(&config)?);
/// let _router = build_router(state);
/// # Ok(())
/// # }
/// ```
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
        .layer(CorsLayer::permissive())
        .layer(trace_layer())
}
