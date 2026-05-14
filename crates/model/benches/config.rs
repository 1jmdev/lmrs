use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use model::{ModelConfig, ModelMetadata};

fn bench_model_config_from_json(c: &mut Criterion) {
    let json = r#"{
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "vocab_size": 151936,
        "rms_norm_eps": 0.000001,
        "rope_theta": 1000000.0
    }"#;
    c.bench_function("model_config_deserialize", |b| {
        b.iter(|| serde_json::from_str::<ModelConfig>(black_box(json)))
    });
}

fn bench_model_config_model_type(c: &mut Criterion) {
    let json = r#"{
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "vocab_size": 151936,
        "rms_norm_eps": 0.000001
    }"#;
    let config: ModelConfig = serde_json::from_str(json).unwrap();
    c.bench_function("model_config_model_type", |b| {
        b.iter(|| black_box(&config).model_type())
    });
}

fn bench_model_config_as_value(c: &mut Criterion) {
    let json = r#"{"hidden_size": 32, "intermediate_size": 64, "num_attention_heads": 4, "num_hidden_layers": 2, "vocab_size": 1000, "rms_norm_eps": 0.000001}"#;
    let config: ModelConfig = serde_json::from_str(json).unwrap();
    c.bench_function("model_config_as_value", |b| {
        b.iter(|| black_box(&config).as_value())
    });
}

fn bench_model_config_clone(c: &mut Criterion) {
    let json = r#"{"hidden_size": 32, "intermediate_size": 64, "num_attention_heads": 4, "num_hidden_layers": 2, "vocab_size": 1000, "rms_norm_eps": 0.000001}"#;
    let config: ModelConfig = serde_json::from_str(json).unwrap();
    c.bench_function("model_config_clone", |b| {
        b.iter(|| black_box(&config).clone())
    });
}

fn bench_model_metadata_new(c: &mut Criterion) {
    c.bench_function("model_metadata_new", |b| {
        b.iter(|| ModelMetadata {
            model_type: black_box("qwen3".to_string()),
            vocab_size: black_box(151936),
            hidden_size: black_box(4096),
            num_hidden_layers: black_box(32),
        })
    });
}

fn bench_model_metadata_clone(c: &mut Criterion) {
    let metadata = ModelMetadata {
        model_type: "qwen3".to_string(),
        vocab_size: 151936,
        hidden_size: 4096,
        num_hidden_layers: 32,
    };
    c.bench_function("model_metadata_clone", |b| {
        b.iter(|| black_box(&metadata).clone())
    });
}

criterion_group!(
    benches,
    bench_model_config_from_json,
    bench_model_config_model_type,
    bench_model_config_as_value,
    bench_model_config_clone,
    bench_model_metadata_new,
    bench_model_metadata_clone,
);
criterion_main!(benches);
