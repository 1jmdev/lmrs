# lmrs

`lmrs` is a lightweight Rust runtime for local GGUF language models, powered by `llama.cpp`.

It provides a small API for loading a model, building chat prompts, and generating text with streaming support.

## Highlights

- Local GGUF model loading via `ModelSource`
- Chat-style prompt handling with `Message`
- Streaming generation callbacks
- Greedy and temperature-based sampling
- Prompt and token-piece caching for lower overhead

## Build Requirements

This project builds `llama.cpp` from `vendor/llama.cpp` during `cargo build`.

Required tools:

- Rust (edition 2024 toolchain)
- CMake
- Ninja
- A C/C++ toolchain

Optional acceleration is enabled automatically when detected:

- CUDA (`nvcc`)
- ROCm/HIP (`hipcc` + ROCm clang)
- Vulkan (`VULKAN_SDK` or system Vulkan headers/tools)
- Metal (macOS)

## Usage (Library)

```rust
use lmrs::{GenerationConfig, Message, ModelSource, Runtime};

fn main() -> Result<(), lmrs::LmrsError> {
    let runtime = Runtime::from_source(ModelSource::local("/path/to/model.gguf"))?;

    let messages = [
        Message::system("You are a concise assistant."),
        Message::user("Write one sentence about Rust."),
    ];

    let output = runtime.generate(&messages, GenerationConfig::default())?;
    println!("{output}");

    Ok(())
}
```

## Usage (Current CLI Example)

This repository includes a simple interactive chat example in `src/main.rs`.

Before running it, update the model path in `src/main.rs` to a valid local `.gguf` file.

```bash
cargo run
```

Then type prompts; use `/exit` to quit.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
