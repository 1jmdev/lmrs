# lmrs

Rust inference server built with Axum, Candle, and Hugging Face tokenizers.

## Features

- OpenAI-compatible `/v1/completions` endpoint
- OpenAI-compatible `/v1/chat/completions` endpoint
- Server-sent events for streaming responses
- Local model directory or Hugging Face Hub model loading
- CUDA-first runtime with cuDNN and custom CUDA kernels enabled by default
- Greedy, top-k, and top-p sampling support

## Layout

```text
src/
├── main.rs
├── config.rs
├── error.rs
├── server/
├── engine/
├── model/
├── sampling/
└── tokenizer/
```

## Configuration

The server can be configured with CLI flags.

| CLI flag | Default |
| --- | --- |
| `--host` | `127.0.0.1` |
| `--port` | `8000` |
| `--model` | required |
| `--revision` | none |
| `--tokenizer` | model tokenizer |

## Run

```bash
CUDA_COMPUTE_CAP=89 cargo run --release -- --model /path/to/model
```

For a Hugging Face Hub model:

```bash
CUDA_COMPUTE_CAP=89 cargo run --release -- --model organization/model-name
```

For other NVIDIA GPUs, set the matching CUDA architecture before building:

```bash
CUDA_COMPUTE_CAP=80 cargo build --release   # A100
CUDA_COMPUTE_CAP=89 cargo build --release   # RTX 4060
CUDA_COMPUTE_CAP=100 cargo build --release  # B200, CUDA 12.8+ recommended
```

## API

Completion request:

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"local","prompt":"Hello","max_tokens":32}'
```

Chat request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

Streaming request:

```bash
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"local","prompt":"Hello","max_tokens":32,"stream":true}'
```

## Local Model Files

A local model directory is expected to contain:

- `config.json`
- `tokenizer.json`
- `model.safetensors`

## License

MIT
