use lmrs::{GenerationConfig, LlamaRuntime, Message};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = LlamaRuntime::load_with_config(
        "../backup/models/Llama-3.2-1B-Instruct-f16.gguf",
        GenerationConfig {
            context_size: 2048,
            ..GenerationConfig::default()
        },
    )?;

    let started_at = Instant::now();
    let output = runtime.generate(
        &[
            Message::system("You are a concise Rust assistant. Reply in plain text only."),
            Message::user(
                "Explain Rust ownership in exactly three short sentences with no code blocks.",
            ),
        ],
        GenerationConfig {
            max_tokens: 72,
            ..GenerationConfig::default()
        },
    )?;
    let elapsed = started_at.elapsed();
    let generated_tokens = runtime.count_tokens(&output)?;
    let tokens_per_second = if elapsed.as_secs_f64() > 0.0 {
        generated_tokens as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!("{output}\n");
    println!("time: {:.3}s", elapsed.as_secs_f64());
    println!("generated tokens: {generated_tokens}");
    println!("tok/s: {:.2}", tokens_per_second);
    Ok(())
}
