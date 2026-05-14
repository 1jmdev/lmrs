use axum::{Json, extract::State, response::IntoResponse};

use crate::{
    router::AppState,
    types::{CompletionChoice, CompletionRequest, CompletionResponse, GenerateParams, Usage},
};

use super::chat::now;
use crate::types::completion_sse;

/// Handles `POST /v1/completions`.
pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> crate::Result<impl IntoResponse> {
    let engine = state.engine_handle();
    let params = GenerateParams::from(&req);
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| engine.model_id().to_owned());

    if req.stream.unwrap_or(false) {
        return Ok(completion_sse(engine, req.prompt, params).into_response());
    }

    let output = engine.generate_with_usage(&req.prompt, params)?;
    let prompt_tokens = engine.count_tokens(&req.prompt)?;
    let completion_tokens = output.tokens();
    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".into(),
        created: now(),
        model,
        choices: vec![CompletionChoice {
            index: 0,
            text: output.into_text(),
            finish_reason: "stop".into(),
        }],
        usage: Usage::new(prompt_tokens, completion_tokens),
    })
    .into_response())
}
