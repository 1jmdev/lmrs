use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use server::{ApiError, ErrorBody, ErrorMessage, ServerConfig};

fn bench_api_error_creation(c: &mut Criterion) {
    c.bench_function("api_error_bad_request", |b| {
        b.iter(|| ApiError::BadRequest(black_box("test error".to_string())))
    });
}

fn bench_api_error_display(c: &mut Criterion) {
    let err = ApiError::BadRequest("invalid model name".to_string());
    c.bench_function("api_error_display", |b| {
        b.iter(|| black_box(&err).to_string())
    });
}

fn bench_error_body_creation(c: &mut Criterion) {
    c.bench_function("error_body_new", |b| {
        b.iter(|| {
            ErrorBody {
                error: ErrorMessage {
                    message: black_box("test error".to_string()),
                    kind: black_box("server_error".to_string()),
                },
            }
        })
    });
}

fn bench_error_body_serialize(c: &mut Criterion) {
    let body = ErrorBody {
        error: ErrorMessage {
            message: "test error message".to_string(),
            kind: "server_error".to_string(),
        },
    };
    c.bench_function("error_body_serialize", |b| {
        b.iter(|| serde_json::to_string(black_box(&body)))
    });
}

fn bench_server_config_default(c: &mut Criterion) {
    c.bench_function("server_config_default", |b| {
        b.iter(|| ServerConfig::default())
    });
}

fn bench_server_config_accessors(c: &mut Criterion) {
    let config = ServerConfig {
        host: "127.0.0.1".into(),
        port: 8000,
        model: "local-dev".into(),
    };
    c.bench_function("server_config_bind_addr", |b| {
        b.iter(|| black_box(&config).bind_addr())
    });
    c.bench_function("server_config_host", |b| {
        b.iter(|| black_box(&config).host.clone())
    });
    c.bench_function("server_config_port", |b| {
        b.iter(|| black_box(config.port))
    });
}

fn bench_uuid_generation(c: &mut Criterion) {
    c.bench_function("uuid_v4_generation", |b| {
        b.iter(|| uuid::Uuid::new_v4())
    });
}

criterion_group!(
    benches,
    bench_api_error_creation,
    bench_api_error_display,
    bench_error_body_creation,
    bench_error_body_serialize,
    bench_server_config_default,
    bench_server_config_accessors,
    bench_uuid_generation,
);
criterion_main!(benches);
