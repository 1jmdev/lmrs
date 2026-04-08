use lmrs::{GenerationConfig, Message, ModelSource, Runtime};
use std::io::{self, BufRead, Write};
use std::time::Instant;

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
        stdout.lock().flush().unwrap();

        let mut input = String::new();
        stdin.lock().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "/exit" {
            break;
        }

        messages.push(Message::user(input));

        print!("Assistant: ");
        stdout.lock().flush().unwrap();

        let started_at = Instant::now();
        let mut first_token_at = None;

        let output = runtime.generate_stream(&messages, GenerationConfig::default(), |chunk| {
            if first_token_at.is_none() && !chunk.is_empty() {
                first_token_at = Some(started_at.elapsed());
            }
            let _ = stdout.lock().write_all(chunk);
            let _ = stdout.lock().flush();
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
