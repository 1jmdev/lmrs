use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use cache::SlotLayout;

fn bench_slot_layout_new(c: &mut Criterion) {
    c.bench_function("slot_layout_new", |b| {
        b.iter(|| SlotLayout::new(black_box(16), black_box(128), black_box(2)))
    });
}

fn bench_slot_layout_accessors(c: &mut Criterion) {
    let layout = SlotLayout::new(16, 128, 2);
    c.bench_function("slot_layout_slots_per_block", |b| {
        b.iter(|| black_box(layout).slots_per_block())
    });
    c.bench_function("slot_layout_slot_stride_bytes", |b| {
        b.iter(|| black_box(layout).slot_stride_bytes())
    });
    c.bench_function("slot_layout_planes", |b| {
        b.iter(|| black_box(layout).planes())
    });
    c.bench_function("slot_layout_block_size_bytes", |b| {
        b.iter(|| black_box(layout).block_size_bytes())
    });
}

fn bench_slot_layout_slot_offset(c: &mut Criterion) {
    let layout = SlotLayout::new(16, 128, 2);
    c.bench_function("slot_layout_slot_offset_valid", |b| {
        b.iter(|| black_box(layout).slot_offset_bytes(black_box(7)))
    });
    c.bench_function("slot_layout_slot_offset_invalid", |b| {
        b.iter(|| black_box(layout).slot_offset_bytes(black_box(32)))
    });
}

criterion_group!(
    benches,
    bench_slot_layout_new,
    bench_slot_layout_accessors,
    bench_slot_layout_slot_offset,
);
criterion_main!(benches);
