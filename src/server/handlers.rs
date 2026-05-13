use std::{
    convert::Infallible,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Sse, sse::Event},
};
use futures::{Stream, StreamExt};
use tokio_stream::iter;
use uuid::Uuid;

use crate::{engine::InferenceEngine, error::Result};

use super::types::*;

pub async fn health() -> &'static str {
    "ok"
}

pub async fn completions(
    State(engine): State<Arc<InferenceEngine>>,
    Json(req): Json<CompletionRequest>,
) -> Result<impl IntoResponse> {
    let params = GenerateParams::from(&req);
    let response_model = req
        .model
        .clone()
        .unwrap_or_else(|| engine.model_id().to_owned());
    if req.stream.unwrap_or(false) {
        return Ok(stream_completion(engine, req.prompt, params).into_response());
    }
    let output = engine.generate_with_usage(&req.prompt, params)?;
    let prompt_tokens = engine.count_tokens(&req.prompt)?;
    let completion_tokens = output.tokens;
    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: "text_completion".into(),
        created: now(),
        model: response_model,
        choices: vec![CompletionChoice {
            index: 0,
            text: output.text,
            finish_reason: "stop".into(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response())
}

pub async fn chat_completions(
    State(engine): State<Arc<InferenceEngine>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse> {
    let prompt_tokens = engine.count_message_tokens(&req.messages)?;
    let prompt = engine.apply_chat_template(&req.messages)?;
    let params = GenerateParams::from(&req);
    let response_model = req
        .model
        .clone()
        .unwrap_or_else(|| engine.model_id().to_owned());
    if req.stream.unwrap_or(false) {
        return Ok(stream_chat_completion(engine, prompt, params).into_response());
    }
    let output = engine.generate_with_usage(&prompt, params)?;
    let completion_tokens = output.tokens;
    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".into(),
        created: now(),
        model: response_model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: output.text,
            },
            finish_reason: "stop".into(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response())
}

fn stream_completion(
    engine: Arc<InferenceEngine>,
    prompt: String,
    params: GenerateParams,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
    let id = format!("cmpl-{}", Uuid::new_v4());
    let created = now();
    let model = engine.model_id().to_owned();
    let chunks = match engine.generate_tokens(&prompt, params) {
        Ok(tokens) => tokens
            .into_iter()
            .map(move |text| {
                serialize_sse(CompletionChunk {
                    id: id.clone(),
                    object: "text_completion.chunk".into(),
                    created,
                    model: model.clone(),
                    choices: vec![CompletionChunkChoice {
                        index: 0,
                        text,
                        finish_reason: None,
                    }],
                })
            })
            .collect::<Vec<_>>(),
        Err(err) => vec![Ok(Event::default().event("error").data(err.to_string()))],
    };
    Sse::new(iter(chunks).chain(iter([Ok(Event::default().data("[DONE]"))])))
}

fn stream_chat_completion(
    engine: Arc<InferenceEngine>,
    prompt: String,
    params: GenerateParams,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = now();
    let model = engine.model_id().to_owned();
    let chunks = match engine.generate_tokens(&prompt, params) {
        Ok(tokens) => tokens
            .into_iter()
            .map(move |content| {
                serialize_sse(ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: Some(content),
                        },
                        finish_reason: None,
                    }],
                })
            })
            .collect::<Vec<_>>(),
        Err(err) => vec![Ok(Event::default().event("error").data(err.to_string()))],
    };
    Sse::new(iter(chunks).chain(iter([Ok(Event::default().data("[DONE]"))])))
}

fn serialize_sse<T: serde::Serialize>(value: T) -> std::result::Result<Event, Infallible> {
    Ok(Event::default().data(serde_json::to_string(&value).unwrap_or_else(|err| err.to_string())))
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}
