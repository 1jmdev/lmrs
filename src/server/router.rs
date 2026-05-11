use std::sync::Arc;

use axum::{
    Router,
    routing::{get, post},
};

use crate::engine::InferenceEngine;

use super::handlers::{chat_completions, completions, health};

pub fn build_router(engine: Arc<InferenceEngine>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(engine)
}
