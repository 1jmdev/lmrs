use std::convert::Infallible;

use axum::response::{Sse, sse::Event};
use futures::{Stream, StreamExt};
use serde::Serialize;
use tokio_stream::iter;

use crate::{engine::SharedEngine, handlers::chat::now};

use super::GenerateParams;

/// Text completion SSE chunk.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
}

/// Choice payload for a text completion SSE chunk.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CompletionChunkChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
}

/// Chat completion SSE chunk.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

/// Choice payload for a chat completion SSE chunk.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

/// Incremental chat delta.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ChatDelta {
    pub role: Option<String>,
    pub content: Option<String>,
}

/// Streams text completion chunks as Server-Sent Events.
pub fn completion_sse(
    engine: SharedEngine,
    prompt: String,
    params: GenerateParams,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let id = format!("cmpl-{}", uuid::Uuid::new_v4());
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
        Err(error) => vec![Ok(Event::default().event("error").data(error.to_string()))],
    };
    Sse::new(iter(chunks).chain(iter([Ok(Event::default().data("[DONE]"))])))
}

/// Streams chat completion chunks as Server-Sent Events.
pub fn chat_sse(
    engine: SharedEngine,
    prompt: String,
    params: GenerateParams,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
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
        Err(error) => vec![Ok(Event::default().event("error").data(error.to_string()))],
    };
    Sse::new(iter(chunks).chain(iter([Ok(Event::default().data("[DONE]"))])))
}

/// Serializes an SSE data event.
pub fn serialize_sse<T: Serialize>(value: T) -> Result<Event, Infallible> {
    Ok(Event::default().data(serde_json::to_string(&value).unwrap_or_else(|err| err.to_string())))
}
