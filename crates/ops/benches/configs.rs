use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use ops::{LayerNormConfig, LinearConfig, RmsNormConfig, SdpaConfig};

fn bench_rms_norm_config(c: &mut Criterion) {
    c.bench_function("rms_norm_config_new", |b| {
        b.iter(|| RmsNormConfig {
            hidden_size: black_box(4096),
            eps: black_box(1e-6),
        })
    });
    c.bench_function("rms_norm_config_clone", |b| {
        let cfg = RmsNormConfig {
            hidden_size: 4096,
            eps: 1e-6,
        };
        b.iter(|| black_box(&cfg).clone())
    });
}

fn bench_layer_norm_config(c: &mut Criterion) {
    c.bench_function("layer_norm_config_new", |b| {
        b.iter(|| LayerNormConfig {
            hidden_size: black_box(4096),
            eps: black_box(1e-5),
        })
    });
    c.bench_function("layer_norm_config_clone", |b| {
        let cfg = LayerNormConfig {
            hidden_size: 4096,
            eps: 1e-5,
        };
        b.iter(|| black_box(&cfg).clone())
    });
}

fn bench_linear_config(c: &mut Criterion) {
    c.bench_function("linear_config_new", |b| {
        b.iter(|| LinearConfig {
            in_features: black_box(4096),
            out_features: black_box(11008),
            bias: black_box(false),
        })
    });
    c.bench_function("linear_config_with_bias", |b| {
        b.iter(|| LinearConfig {
            in_features: black_box(4096),
            out_features: black_box(4096),
            bias: black_box(true),
        })
    });
    c.bench_function("linear_config_clone", |b| {
        let cfg = LinearConfig {
            in_features: 4096,
            out_features: 4096,
            bias: false,
        };
        b.iter(|| black_box(&cfg).clone())
    });
}

fn bench_sdpa_config(c: &mut Criterion) {
    c.bench_function("sdpa_config_new", |b| {
        b.iter(|| SdpaConfig {
            head_dim: black_box(128),
            causal: black_box(true),
            start_pos: black_box(0),
        })
    });
    c.bench_function("sdpa_config_clone", |b| {
        let cfg = SdpaConfig {
            head_dim: 128,
            causal: true,
            start_pos: 0,
        };
        b.iter(|| black_box(&cfg).clone())
    });
}

criterion_group!(
    benches,
    bench_rms_norm_config,
    bench_layer_norm_config,
    bench_linear_config,
    bench_sdpa_config,
);
criterion_main!(benches);
