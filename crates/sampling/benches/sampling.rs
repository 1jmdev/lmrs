use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use sampling::{
    BeamSearch, FrequencyPresencePenalty, Greedy, MinP, RepetitionPenalty, SampleOutput,
    Sampler, SamplerConfig, SamplingStrategy, Temperature, TopK, TopP,
};

fn bench_temperature_new(c: &mut Criterion) {
    c.bench_function("temperature_new_valid", |b| {
        b.iter(|| Temperature::new(black_box(0.8)))
    });
    c.bench_function("temperature_new_invalid", |b| {
        b.iter(|| Temperature::new(black_box(0.0)))
    });
}

fn bench_temperature_value(c: &mut Criterion) {
    let temp = Temperature::new(0.8).unwrap();
    c.bench_function("temperature_value", |b| {
        b.iter(|| black_box(&temp).value())
    });
}

fn bench_top_k_new(c: &mut Criterion) {
    c.bench_function("top_k_new_valid", |b| {
        b.iter(|| TopK::new(black_box(50)))
    });
}

fn bench_top_p_new(c: &mut Criterion) {
    c.bench_function("top_p_new_valid", |b| {
        b.iter(|| TopP::new(black_box(0.9)))
    });
}

fn bench_min_p_new(c: &mut Criterion) {
    c.bench_function("min_p_new_valid", |b| {
        b.iter(|| MinP::new(black_box(0.1)))
    });
}

fn bench_beam_search_new(c: &mut Criterion) {
    c.bench_function("beam_search_new", |b| {
        b.iter(|| BeamSearch::new(black_box(4)))
    });
}

fn bench_greedy_sample(c: &mut Criterion) {
    let sizes = [16, 1024, 16384, 65536, 151936];
    for &size in &sizes {
        let logits: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
        c.bench_with_input(
            BenchmarkId::new("greedy_sample", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut rng = 42_u64;
                    Greedy.sample(black_box(&logits), black_box(&mut rng))
                })
            },
        );
    }
}

fn bench_top_k_sample(c: &mut Criterion) {
    let sizes = [1024, 16384, 65536];
    for &size in &sizes {
        let logits: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
        c.bench_with_input(
            BenchmarkId::new("top_k_sample", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut rng = 42_u64;
                    TopK::new(50).unwrap().sample(black_box(&logits), black_box(&mut rng))
                })
            },
        );
    }
}

fn bench_top_p_sample(c: &mut Criterion) {
    let sizes = [1024, 16384, 65536];
    for &size in &sizes {
        let logits: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
        c.bench_with_input(
            BenchmarkId::new("top_p_sample", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut rng = 42_u64;
                    TopP::new(0.9).unwrap().sample(black_box(&logits), black_box(&mut rng))
                })
            },
        );
    }
}

fn bench_min_p_sample(c: &mut Criterion) {
    let sizes = [1024, 16384, 65536];
    for &size in &sizes {
        let logits: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
        c.bench_with_input(
            BenchmarkId::new("min_p_sample", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut rng = 42_u64;
                    MinP::new(0.1).unwrap().sample(black_box(&logits), black_box(&mut rng))
                })
            },
        );
    }
}

fn bench_beam_search_sample(c: &mut Criterion) {
    let sizes = [1024, 16384, 65536];
    for &size in &sizes {
        let logits: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
        c.bench_with_input(
            BenchmarkId::new("beam_search_sample", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut rng = 42_u64;
                    BeamSearch::new(4)
                        .unwrap()
                        .sample(black_box(&logits), black_box(&mut rng))
                })
            },
        );
    }
}

fn bench_sample_output_new(c: &mut Criterion) {
    c.bench_function("sample_output_new", |b| {
        b.iter(|| SampleOutput::new(black_box(42), black_box(-1.5_f32), black_box(8.3_f32)))
    });
}

fn bench_sample_output_accessors(c: &mut Criterion) {
    let out = SampleOutput::new(42, -1.5, 8.3);
    c.bench_function("sample_output_token_id", |b| {
        b.iter(|| black_box(&out).token_id())
    });
    c.bench_function("sample_output_log_prob", |b| {
        b.iter(|| black_box(&out).log_prob())
    });
    c.bench_function("sample_output_logit", |b| {
        b.iter(|| black_box(&out).logit())
    });
}

fn bench_sampler_config_new(c: &mut Criterion) {
    c.bench_function("sampler_config_new", |b| {
        b.iter(|| SamplerConfig::new(black_box(Box::new(Greedy))))
    });
}

fn bench_sampler_new(c: &mut Criterion) {
    c.bench_function("sampler_new", |b| {
        b.iter(|| Sampler::new(black_box(SamplerConfig::new(Box::new(Greedy)).with_seed(42))))
    });
}

fn bench_sampler_default(c: &mut Criterion) {
    c.bench_function("sampler_default", |b| {
        b.iter(|| Sampler::default())
    });
}

fn bench_repetition_penalty_new(c: &mut Criterion) {
    c.bench_function("repetition_penalty_new", |b| {
        b.iter(|| RepetitionPenalty::new(black_box(1.1)))
    });
}

fn bench_repetition_penalty_value(c: &mut Criterion) {
    let rp = RepetitionPenalty::new(2.0).unwrap();
    c.bench_function("repetition_penalty_penalty", |b| {
        b.iter(|| black_box(&rp).penalty())
    });
}

fn bench_frequency_presence_penalty_new(c: &mut Criterion) {
    c.bench_function("frequency_presence_penalty_new", |b| {
        b.iter(|| FrequencyPresencePenalty::new(black_box(0.3), black_box(0.3)))
    });
}

fn bench_frequency_presence_penalty_accessors(c: &mut Criterion) {
    let fp = FrequencyPresencePenalty::new(0.5, 0.25).unwrap();
    c.bench_function("frequency_presence_penalty_presence", |b| {
        b.iter(|| black_box(&fp).presence())
    });
    c.bench_function("frequency_presence_penalty_frequency", |b| {
        b.iter(|| black_box(&fp).frequency())
    });
}

criterion_group!(
    benches,
    bench_temperature_new,
    bench_temperature_value,
    bench_top_k_new,
    bench_top_p_new,
    bench_min_p_new,
    bench_beam_search_new,
    bench_greedy_sample,
    bench_top_k_sample,
    bench_top_p_sample,
    bench_min_p_sample,
    bench_beam_search_sample,
    bench_sample_output_new,
    bench_sample_output_accessors,
    bench_sampler_config_new,
    bench_sampler_new,
    bench_sampler_default,
    bench_repetition_penalty_new,
    bench_repetition_penalty_value,
    bench_frequency_presence_penalty_new,
    bench_frequency_presence_penalty_accessors,
);
criterion_main!(benches);
