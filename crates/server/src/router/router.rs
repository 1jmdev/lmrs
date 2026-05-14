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
/// ```
/// use server::{AppState, EchoEngine, build_router};
///
/// let state = AppState::new(EchoEngine::new("demo"));
/// let _router = build_router(state);
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
