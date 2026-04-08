use lmrs::{GenerationConfig, Message, ModelSource, Runtime, StreamChunk};
use std::io::{self, BufRead, Write};
use std::time::Instant;

const ANSI_RESET: &str = "\x1b[0m";
const ANSI_GRAY: &str = "\x1b[90m";
const ANSI_BOLD_GRAY: &str = "\x1b[1;90m";

fn main() -> Result<(), lmrs::LmrsError> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .compact()
        .with_target(false)
        .try_init();

    let runtime = Runtime::from_source(ModelSource::local(
        "/home/maty/llms/rust/backup/models/google_gemma-4-E4B-it-Q4_K_M.gguf",
    ))?;

    let mut messages = vec![Message::system(
        "You are a helpfull assistant. Reply in plain text only.",
    )];

    let stdout = io::stdout();
    let stdin = io::stdin();

    println!("Chat started. Type /exit to quit.\n");

    loop {
        print!("You: ");
        let _ = stdout.lock().flush();

        let mut input = String::new();
        let _ = stdin.lock().read_line(&mut input);
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "/exit" {
            break;
        }

        messages.push(Message::user(input));

        print!("Assistant: ");
        let _ = stdout.lock().flush();

        let started_at = Instant::now();
        let mut first_token_at = None;

        let output =
            runtime.generate_stream_events(&messages, GenerationConfig::default(), |chunk| {
                if first_token_at.is_none() {
                    first_token_at = Some(started_at.elapsed());
                }

                let mut lock = stdout.lock();
                match chunk {
                    StreamChunk::Content(content) => {
                        let _ = lock.write_all(&content);
                    }
                    StreamChunk::ThinkingStarted => {
                        let _ = lock.write_all(b"\n");
                        let _ = lock.write_all(ANSI_BOLD_GRAY.as_bytes());
                        let _ = lock.write_all(b"Thinking...");
                        let _ = lock.write_all(ANSI_RESET.as_bytes());
                        let _ = lock.write_all(b"\n");
                    }
                    StreamChunk::Thinking(thinking) => {
                        let _ = lock.write_all(ANSI_GRAY.as_bytes());
                        let _ = lock.write_all(&thinking);
                        let _ = lock.write_all(ANSI_RESET.as_bytes());
                    }
                    StreamChunk::ThinkingFinished => {
                        let _ = lock.write_all(ANSI_RESET.as_bytes());
                    }
                }
                let _ = lock.flush();
            })?;

        let completed_at = started_at.elapsed();
        let generated_tokens = runtime.count_tokens(&output)?;
        let time_to_first_token = first_token_at.unwrap_or(completed_at);
        let generation_window = completed_at.saturating_sub(time_to_first_token);
        let tokens_per_second = if generation_window.as_secs_f64() > 0.0 {
            generated_tokens as f64 / generation_window.as_secs_f64()
        } else {
            0.0
        };

        println!("\n");
        println!(
            "[{:.2} tok/s · {} tokens · {:.3}s]",
            tokens_per_second,
            generated_tokens,
            completed_at.as_secs_f64()
        );
        println!();

        messages.push(Message::assistant(&output));
    }

    Ok(())
}
