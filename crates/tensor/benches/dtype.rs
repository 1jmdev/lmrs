use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use tensor::DType;

fn bench_dtype_size_in_bytes(c: &mut Criterion) {
    c.bench_function("dtype_size_in_bytes_f32", |b| {
        b.iter(|| black_box(DType::F32).size_in_bytes())
    });
    c.bench_function("dtype_size_in_bytes_bf16", |b| {
        b.iter(|| black_box(DType::BF16).size_in_bytes())
    });
}

fn bench_dtype_name(c: &mut Criterion) {
    c.bench_function("dtype_name_f32", |b| {
        b.iter(|| black_box(DType::F32).name())
    });
    c.bench_function("dtype_name_bf16", |b| {
        b.iter(|| black_box(DType::BF16).name())
    });
}

fn bench_dtype_align_in_bytes(c: &mut Criterion) {
    c.bench_function("dtype_align_in_bytes_f32", |b| {
        b.iter(|| black_box(DType::F32).align_in_bytes())
    });
}

fn bench_dtype_display(c: &mut Criterion) {
    c.bench_function("dtype_display_f32", |b| {
        b.iter(|| black_box(DType::F32).to_string())
    });
}

criterion_group!(
    benches,
    bench_dtype_size_in_bytes,
    bench_dtype_name,
    bench_dtype_align_in_bytes,
    bench_dtype_display,
);
criterion_main!(benches);
