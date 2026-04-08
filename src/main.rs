use lmrs::{GenerationConfig, Message, ModelSource, Runtime};
use std::io::{self, Write};
use std::time::Instant;
use tracing::info;

fn main() -> Result<(), lmrs::LmrsError> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .compact()
        .with_target(false)
        .try_init();

    let model_path = std::env::var("LMRS_MODEL").map_err(|_| {
        lmrs::LmrsError::UnsupportedModelSource("set LMRS_MODEL to a local model path".to_string())
    })?;

    info!(model = %model_path, "creating runtime");
    let runtime = Runtime::from_source(ModelSource::local(model_path))?;

    let started_at = Instant::now();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    let mut first_token_at = None;

    let output = runtime.generate_stream(
        &[
            Message::system("You are a concise Rust assistant. Reply in plain text only."),
            Message::user("Explain Rust ownership in exactly three short sentences."),
        ],
        GenerationConfig::default(),
        |chunk| {
            if first_token_at.is_none() && !chunk.is_empty() {
                first_token_at = Some(started_at.elapsed());
            }
            let _ = stdout.write_all(chunk);
            let _ = stdout.flush();
        },
    )?;

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
        "model: {} ({})",
        runtime.artifact().gguf_path.display(),
        runtime.artifact().source_label
    );
    println!(
        "time to first token: {:.3}s",
        time_to_first_token.as_secs_f64()
    );
    println!("generation time: {:.3}s", generation_window.as_secs_f64());
    println!("total time: {:.3}s", completed_at.as_secs_f64());
    println!("generated tokens: {generated_tokens}");
    println!("tok/s: {:.2}", tokens_per_second);

    Ok(())
}
