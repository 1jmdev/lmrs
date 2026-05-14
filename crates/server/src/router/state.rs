use std::sync::Arc;

use engine::SchedulerBudget;

use crate::engine::{EngineHandle, SharedEngine};

/// Shared Axum application state.
///
/// `AppState` owns the engine handle used by all generation routes.
///
/// # Example
///
/// ```
/// use server::{AppState, EchoEngine};
///
/// let state = AppState::new(EchoEngine::new("demo"));
/// assert_eq!(state.engine().model_id(), "demo");
/// ```
#[derive(Clone)]
pub struct AppState {
    engine: SharedEngine,
    scheduler_budget: SchedulerBudget,
}

impl AppState {
    /// Creates state from any engine implementation.
    pub fn new<E>(engine: E) -> Self
    where
        E: EngineHandle,
    {
        Self {
            engine: Arc::new(engine),
            scheduler_budget: SchedulerBudget::new(64, 4096),
        }
    }

    /// Creates state from an already shared engine pointer.
    pub fn from_shared_engine(engine: SharedEngine) -> Self {
        Self {
            engine,
            scheduler_budget: SchedulerBudget::new(64, 4096),
        }
    }

    /// Returns the shared inference engine.
    pub fn engine(&self) -> &dyn EngineHandle {
        self.engine.as_ref()
    }

    /// Returns a clone of the shared engine pointer.
    pub fn engine_handle(&self) -> SharedEngine {
        self.engine.clone()
    }

    /// Returns the engine scheduler budget associated with this server.
    pub const fn scheduler_budget(&self) -> &SchedulerBudget {
        &self.scheduler_budget
    }
}
