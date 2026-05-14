use tower_http::{
    classify::{ServerErrorsAsFailures, SharedClassifier},
    trace::TraceLayer,
};

/// Builds HTTP tracing middleware.
///
/// # Example
///
/// ```
/// let _layer = server::trace_layer();
/// ```
pub fn trace_layer() -> TraceLayer<SharedClassifier<ServerErrorsAsFailures>> {
    TraceLayer::new_for_http()
}
