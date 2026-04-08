use crate::runtime::StreamChunk;

const THINK_START: &[u8] = b"<think>";
const THINK_END: &[u8] = b"</think>";

pub struct ThinkingParser {
    pending: Vec<u8>,
    in_thinking: bool,
}

impl ThinkingParser {
    pub fn new() -> Self {
        Self {
            pending: Vec::with_capacity(64),
            in_thinking: false,
        }
    }

    pub fn push<F>(&mut self, chunk: &[u8], output: &mut Vec<u8>, on_chunk: &mut F)
    where
        F: FnMut(StreamChunk),
    {
        self.pending.extend_from_slice(chunk);

        loop {
            if self.in_thinking {
                if let Some(index) = find_subslice(&self.pending, THINK_END) {
                    if index > 0 {
                        let thinking = self.pending[..index].to_vec();
                        on_chunk(StreamChunk::Thinking(thinking));
                    }
                    self.pending.drain(..index + THINK_END.len());
                    self.in_thinking = false;
                    on_chunk(StreamChunk::ThinkingFinished);
                    continue;
                }

                let keep = THINK_END.len().saturating_sub(1);
                let writable = self.pending.len().saturating_sub(keep);
                if writable > 0 {
                    let thinking = self.pending[..writable].to_vec();
                    on_chunk(StreamChunk::Thinking(thinking));
                    self.pending.drain(..writable);
                }
                break;
            }

            if let Some(index) = find_subslice(&self.pending, THINK_START) {
                if index > 0 {
                    let content = self.pending[..index].to_vec();
                    output.extend_from_slice(&content);
                    on_chunk(StreamChunk::Content(content));
                }
                self.pending.drain(..index + THINK_START.len());
                self.in_thinking = true;
                on_chunk(StreamChunk::ThinkingStarted);
                continue;
            }

            let keep = THINK_START.len().saturating_sub(1);
            let writable = self.pending.len().saturating_sub(keep);
            if writable > 0 {
                let content = self.pending[..writable].to_vec();
                output.extend_from_slice(&content);
                on_chunk(StreamChunk::Content(content));
                self.pending.drain(..writable);
            }
            break;
        }
    }

    pub fn finish<F>(&mut self, output: &mut Vec<u8>, on_chunk: &mut F)
    where
        F: FnMut(StreamChunk),
    {
        if !self.pending.is_empty() {
            let chunk = std::mem::take(&mut self.pending);
            if self.in_thinking {
                on_chunk(StreamChunk::Thinking(chunk));
            } else {
                output.extend_from_slice(&chunk);
                on_chunk(StreamChunk::Content(chunk));
            }
        }

        if self.in_thinking {
            self.in_thinking = false;
            on_chunk(StreamChunk::ThinkingFinished);
        }
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }

    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
