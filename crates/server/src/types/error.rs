use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

/// Server API error type.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    /// Request validation failed.
    #[error("{0}")]
    BadRequest(String),
    /// JSON serialization or parsing failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Catch-all internal error.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Convenient result alias for handlers and middleware.
pub type Result<T> = std::result::Result<T, ApiError>;

/// OpenAI-compatible error response body.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ErrorBody {
    /// Error payload.
    pub error: ErrorMessage,
}

/// OpenAI-compatible error payload.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ErrorMessage {
    /// Human-readable error message.
    pub message: String,
    /// Machine-readable error category.
    #[serde(rename = "type")]
    pub kind: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self {
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Json(_) | Self::Other(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let message = self.to_string();
        (
            status,
            Json(ErrorBody {
                error: ErrorMessage {
                    message,
                    kind: "server_error".into(),
                },
            }),
        )
            .into_response()
    }
}
