use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use tensor::Shape;

fn bench_shape_new_small(c: &mut Criterion) {
    c.bench_function("shape_new_3d", |b| {
        b.iter(|| Shape::new(black_box([64, 128, 256])))
    });
}

fn bench_shape_new_scalar(c: &mut Criterion) {
    c.bench_function("shape_new_scalar", |b| {
        b.iter(|| Shape::scalar())
    });
}

fn bench_shape_numel(c: &mut Criterion) {
    let shape = Shape::new([1024, 4096]).unwrap();
    c.bench_function("shape_numel", |b| {
        b.iter(|| black_box(&shape).numel())
    });
}

fn bench_shape_ndim(c: &mut Criterion) {
    let shape = Shape::new([16, 32, 64, 128]).unwrap();
    c.bench_function("shape_ndim", |b| {
        b.iter(|| black_box(&shape).ndim())
    });
}

fn bench_shape_dims(c: &mut Criterion) {
    let shape = Shape::new([16, 32, 64, 128]).unwrap();
    c.bench_function("shape_dims", |b| {
        b.iter(|| black_box(&shape).dims())
    });
}

fn bench_shape_is_empty(c: &mut Criterion) {
    let shape = Shape::new([16, 32, 64, 128]).unwrap();
    c.bench_function("shape_is_empty", |b| {
        b.iter(|| black_box(&shape).is_empty())
    });
}

fn bench_shape_clone(c: &mut Criterion) {
    let shape = Shape::new([16, 32, 64, 128]).unwrap();
    c.bench_function("shape_clone", |b| {
        b.iter(|| black_box(&shape).clone())
    });
}

criterion_group!(
    benches,
    bench_shape_new_small,
    bench_shape_new_scalar,
    bench_shape_numel,
    bench_shape_ndim,
    bench_shape_dims,
    bench_shape_is_empty,
    bench_shape_clone,
);
criterion_main!(benches);
