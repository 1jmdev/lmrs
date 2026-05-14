use crate::Result;
use crate::RuntimeError;

use super::GraphExec;

/// Describes an update to CUDA graph kernel parameters.
///
/// Graph node updates describe changes that can be applied to an executable
/// graph before launch.
///
/// # Example
///
/// ```no_run
/// use runtime::GraphUpdate;
///
/// let update = GraphUpdate::new();
/// assert!(format!("{update:?}").contains("GraphUpdate"));
/// ```
#[derive(Debug)]
pub struct GraphUpdate;

impl GraphUpdate {
    /// Creates an empty graph update descriptor.
    ///
    /// This is useful for plumbing scheduler/executor code before concrete node
    /// parameter update payloads exist.
    ///
    /// # Example
    ///
    /// ```
    /// use runtime::GraphUpdate;
    ///
    /// let _update = GraphUpdate::new();
    /// ```
    pub fn new() -> Self {
        Self
    }

    /// Applies this update to an executable graph.
    ///
    /// The current runtime reports graph updates as unavailable with a normal
    /// error value.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use runtime::{CudaContext, GraphCapture, GraphExec, GraphUpdate};
    ///
    /// # fn main() -> runtime::Result<()> {
    /// let context = CudaContext::new(0)?;
    /// let graph = GraphCapture::begin(&context)?.end()?;
    /// let result = GraphExec::instantiate(graph).and_then(|mut exec| GraphUpdate::new().apply(&mut exec));
    /// assert!(result.is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn apply(&self, _exec: &mut GraphExec) -> Result<()> {
        Err(RuntimeError::msg("CUDA graph updates are unavailable"))
    }
}

impl Default for GraphUpdate {
    fn default() -> Self {
        Self::new()
    }
}
