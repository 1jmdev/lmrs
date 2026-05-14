use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use tensor::DType;

fn bench_tensor_error_creation(c: &mut Criterion) {
    c.bench_function("tensor_error_shape_mismatch", |b| {
        b.iter(|| tensor::TensorError::ShapeMismatch(black_box("test error".to_string())))
    });
    c.bench_function("tensor_error_dtype_mismatch", |b| {
        b.iter(|| tensor::TensorError::DTypeMismatch {
            expected: black_box("f32"),
            actual: black_box("bf16"),
        })
    });
    c.bench_function("tensor_error_display", |b| {
        let err = tensor::TensorError::ShapeMismatch("test message".to_string());
        b.iter(|| black_box(&err).to_string())
    });
}

fn bench_shape_error_creation(c: &mut Criterion) {
    c.bench_function("shape_error_numel_overflow", |b| {
        b.iter(|| tensor::ShapeError::NumelOverflow)
    });
    c.bench_function("shape_error_rank_mismatch", |b| {
        b.iter(|| tensor::ShapeError::RankMismatch {
            shape_rank: black_box(3),
            stride_rank: black_box(4),
        })
    });
}

fn bench_copy_error_creation(c: &mut Criterion) {
    c.bench_function("copy_error_element_size_mismatch", |b| {
        b.iter(|| tensor::CopyError::ElementSizeMismatch {
            host_size: black_box(4),
            dtype: black_box(DType::BF16),
            dtype_size: black_box(2),
        })
    });
    c.bench_function("copy_error_length_mismatch", |b| {
        b.iter(|| tensor::CopyError::LengthMismatch {
            host_len: black_box(8),
            tensor_len: black_box(16),
        })
    });
}

fn bench_cast_kind_between(c: &mut Criterion) {
    c.bench_function("cast_kind_between", |b| {
        b.iter(|| tensor::CastKind::between(black_box(DType::BF16), black_box(DType::F32)))
    });
}

criterion_group!(
    benches,
    bench_tensor_error_creation,
    bench_shape_error_creation,
    bench_copy_error_creation,
    bench_cast_kind_between,
);
criterion_main!(benches);
