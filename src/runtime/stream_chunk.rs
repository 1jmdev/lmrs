#[derive(Debug)]
pub enum StreamChunk {
    Content(Vec<u8>),
    ThinkingStarted,
    Thinking(Vec<u8>),
    ThinkingFinished,
}
