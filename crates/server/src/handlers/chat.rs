use std::time::{SystemTime, UNIX_EPOCH};

use axum::{Json, extract::State, response::IntoResponse};

use crate::{
    router::AppState,
    types::{
        ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, GenerateParams,
        Usage, chat_sse,
    },
};

/// Handles `POST /v1/chat/completions`.
///
/// Streaming requests return Server-Sent Events ending with `[DONE]`, matching
/// the OpenAI-compatible client contract.
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> crate::Result<impl IntoResponse> {
    let engine = state.engine_handle();
    let prompt_tokens = engine.count_message_tokens(&req.messages)?;
    let prompt = engine.apply_chat_template(&req.messages)?;
    let params = GenerateParams::from(&req);
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| engine.model_id().to_owned());

    if req.stream.unwrap_or(false) {
        return Ok(chat_sse(engine, prompt, params).into_response());
    }

    let output = engine.generate_with_usage(&prompt, params)?;
    let completion_tokens = output.tokens();
    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".into(),
        created: now(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage::assistant(output.into_text()),
            finish_reason: "stop".into(),
        }],
        usage: Usage::new(prompt_tokens, completion_tokens),
    })
    .into_response())
}

/// Returns the current Unix timestamp in seconds.
pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}
