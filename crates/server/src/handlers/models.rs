use axum::{Json, extract::State};

use crate::{router::AppState, types::ModelListResponse};

/// Lists models served by this process.
///
/// # Example
///
/// ```
/// use server::ModelListResponse;
///
/// let response = ModelListResponse::single("demo");
/// assert_eq!(response.data[0].id, "demo");
/// ```
pub async fn models(State(state): State<AppState>) -> Json<ModelListResponse> {
    Json(ModelListResponse::single(state.engine().model_id()))
}
