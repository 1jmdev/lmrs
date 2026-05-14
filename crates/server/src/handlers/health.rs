use axum::{Json, response::IntoResponse};
use serde::Serialize;

/// Health response returned by liveness endpoints.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct HealthResponse {
    /// Stable machine-readable status.
    pub status: &'static str,
}

/// Returns HTTP 200 when the server is alive.
///
/// # Example
///
/// ```
/// use axum::response::IntoResponse;
///
/// # async fn run() {
/// let response = server::health().await.into_response();
/// assert_eq!(response.status(), axum::http::StatusCode::OK);
/// # }
/// ```
pub async fn health() -> impl IntoResponse {
    Json(HealthResponse { status: "ok" })
}

/// Returns simple process metrics in Prometheus text format.
pub async fn metrics() -> impl IntoResponse {
    "server_up 1\n"
}
