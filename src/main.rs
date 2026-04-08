use lmrs::{GenerationConfig, LlamaRuntime, Message};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = LlamaRuntime::load_with_config(
        "../backup/models/Llama-3.2-1B-Instruct-f16.gguf",
        GenerationConfig {
            context_size: 2048,
            ..GenerationConfig::default()
        },
    )?;

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

    println!("{output}");
    Ok(())
}
