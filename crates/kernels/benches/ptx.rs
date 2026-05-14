use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use kernels::ptx::BASIC;

fn bench_ptx_basic(c: &mut Criterion) {
    c.bench_function("ptx_basic_len", |b| {
        b.iter(|| black_box(BASIC).len())
    });
}

fn bench_ptx_basic_is_empty(c: &mut Criterion) {
    c.bench_function("ptx_basic_is_empty", |b| {
        b.iter(|| black_box(BASIC).is_empty())
    });
}

fn bench_ptx_module_clone_str(c: &mut Criterion) {
    c.bench_function("ptx_basic_to_string", |b| {
        b.iter(|| black_box(BASIC).to_string())
    });
}

criterion_group!(benches, bench_ptx_basic, bench_ptx_basic_is_empty, bench_ptx_module_clone_str);
criterion_main!(benches);
