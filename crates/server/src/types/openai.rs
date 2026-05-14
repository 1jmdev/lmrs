use serde::{Deserialize, Serialize};

/// OpenAI-compatible text completion request.
///
/// # Example
///
/// ```
/// let json = r#"{"prompt":"hello","stream":true}"#;
/// let req: server::CompletionRequest = serde_json::from_str(json).unwrap();
/// assert_eq!(req.prompt, "hello");
/// assert_eq!(req.stream, Some(true));
/// ```
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct CompletionRequest {
    /// Optional model override.
    pub model: Option<String>,
    /// Prompt text to complete.
    pub prompt: String,
    /// Maximum generated tokens.
    pub max_tokens: Option<usize>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f32>,
    /// Top-k sampling cutoff.
    pub top_k: Option<usize>,
    /// Whether to stream chunks as SSE.
    pub stream: Option<bool>,
    /// Stop strings.
    pub stop: Option<Vec<String>>,
}

/// OpenAI-compatible chat message.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ChatMessage {
    /// Message role such as `system`, `user`, or `assistant`.
    pub role: String,
    /// Message content.
    pub content: String,
}

impl ChatMessage {
    /// Creates a chat message.
    ///
    /// # Example
    ///
    /// ```
    /// let msg = server::ChatMessage::new("user", "hello");
    /// assert_eq!(msg.role(), "user");
    /// ```
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    /// Creates an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }

    /// Returns the message role.
    pub fn role(&self) -> &str {
        &self.role
    }

    /// Returns the message content.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Converts into the tokenizer crate's chat message type.
    pub fn to_tokenizer_message(&self) -> tokenizer::ChatMessage {
        tokenizer::ChatMessage::new(self.role.clone(), self.content.clone())
    }
}

/// OpenAI-compatible chat completion request.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ChatCompletionRequest {
    /// Optional model override.
    pub model: Option<String>,
    /// Conversation messages.
    pub messages: Vec<ChatMessage>,
    /// Maximum generated tokens.
    pub max_tokens: Option<usize>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f32>,
    /// Top-k sampling cutoff.
    pub top_k: Option<usize>,
    /// Whether to stream chunks as SSE.
    pub stream: Option<bool>,
    /// Stop strings.
    pub stop: Option<Vec<String>>,
}

/// Generation parameters normalized from API requests.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerateParams {
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    stop: Vec<String>,
}

impl GenerateParams {
    /// Creates generation parameters.
    pub fn new(
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        stop: Vec<String>,
    ) -> Self {
        Self {
            max_tokens,
            temperature,
            top_p,
            top_k,
            stop,
        }
    }

    /// Returns maximum generated tokens.
    pub const fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Returns temperature.
    pub const fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Returns top-p.
    pub const fn top_p(&self) -> f32 {
        self.top_p
    }

    /// Returns top-k.
    pub const fn top_k(&self) -> Option<usize> {
        self.top_k
    }

    /// Returns stop strings.
    pub fn stop(&self) -> &[String] {
        &self.stop
    }
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self::new(128, 1.0, 1.0, None, Vec::new())
    }
}

impl From<&CompletionRequest> for GenerateParams {
    fn from(req: &CompletionRequest) -> Self {
        Self::new(
            req.max_tokens.unwrap_or(128),
            req.temperature.unwrap_or(1.0),
            req.top_p.unwrap_or(1.0),
            req.top_k,
            req.stop.clone().unwrap_or_default(),
        )
    }
}

impl From<&ChatCompletionRequest> for GenerateParams {
    fn from(req: &ChatCompletionRequest) -> Self {
        Self::new(
            req.max_tokens.unwrap_or(128),
            req.temperature.unwrap_or(1.0),
            req.top_p.unwrap_or(1.0),
            req.top_k,
            req.stop.clone().unwrap_or_default(),
        )
    }
}

/// Token usage metadata.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Usage {
    /// Prompt token count.
    pub prompt_tokens: usize,
    /// Completion token count.
    pub completion_tokens: usize,
    /// Total token count.
    pub total_tokens: usize,
}

impl Usage {
    /// Creates usage metadata.
    pub const fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// Text completion response.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

/// One text completion choice.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
}

/// Chat completion response.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// One chat completion choice.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Model list response.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

impl ModelListResponse {
    /// Creates a model list containing one model id.
    pub fn single(id: impl Into<String>) -> Self {
        Self {
            object: "list".into(),
            data: vec![ModelObject {
                id: id.into(),
                object: "model".into(),
                owned_by: "local".into(),
            }],
        }
    }
}

/// Model descriptor returned by `/v1/models`.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}
