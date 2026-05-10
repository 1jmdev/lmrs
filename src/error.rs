use axum::{Json, http::StatusCode, response::{IntoResponse, Response}};
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error(transparent)]
    Tokenizer(#[from] tokenizers::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Hub(#[from] hf_hub::api::sync::ApiError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Template(#[from] minijinja::Error),
    #[error("{0}")]
    BadRequest(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, AppError>;

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: ErrorMessage<'a>,
}

#[derive(Serialize)]
struct ErrorMessage<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    kind: &'a str,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match self {
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let message = self.to_string();
        (status, Json(ErrorBody { error: ErrorMessage { message: &message, kind: "lmrs_error" } })).into_response()
    }
}
