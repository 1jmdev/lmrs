use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion, BatchSize, BenchmarkId};
use runtime::BuddyAllocator;

fn bench_buddy_new(c: &mut Criterion) {
    c.bench_function("buddy_allocator_new_1mb", |b| {
        b.iter(|| BuddyAllocator::new(black_box(1024 * 1024), black_box(64)))
    });
    c.bench_function("buddy_allocator_new_64mb", |b| {
        b.iter(|| BuddyAllocator::new(black_box(64 * 1024 * 1024), black_box(256)))
    });
}

fn bench_buddy_allocate(c: &mut Criterion) {
    let sizes = [64, 128, 256, 1024, 4096, 16384, 65536];
    for &size in &sizes {
        c.bench_with_input(
            BenchmarkId::new("buddy_allocate", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || BuddyAllocator::new(1024 * 1024, 64),
                    |mut allocator| allocator.allocate(black_box(size)),
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

fn bench_buddy_allocate_many(c: &mut Criterion) {
    c.bench_function("buddy_allocate_many_64b", |b| {
        b.iter_batched(
            || BuddyAllocator::new(1024 * 1024, 64),
            |mut allocator| {
                for _ in 0..64 {
                    black_box(allocator.allocate(64));
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_buddy_free(c: &mut Criterion) {
    c.bench_function("buddy_free", |b| {
        b.iter_batched(
            || {
                let mut allocator = BuddyAllocator::new(1024 * 1024, 64);
                let block = allocator.allocate(128).unwrap();
                (allocator, block)
            },
            |(mut allocator, block)| allocator.free(black_box(block)),
            BatchSize::SmallInput,
        )
    });
}

fn bench_buddy_allocate_free_cycle(c: &mut Criterion) {
    c.bench_function("buddy_alloc_free_cycle", |b| {
        b.iter_batched(
            || BuddyAllocator::new(1024 * 1024, 128),
            |mut allocator| {
                let block = allocator.allocate(256).unwrap();
                allocator.free(block);
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_buddy_free_blocks(c: &mut Criterion) {
    let allocator = BuddyAllocator::new(1024 * 1024, 64);
    c.bench_function("buddy_free_blocks", |b| {
        b.iter(|| black_box(&allocator).free_blocks())
    });
}

fn bench_buddy_exhaustion(c: &mut Criterion) {
    c.bench_function("buddy_exhaustion", |b| {
        b.iter_batched(
            || BuddyAllocator::new(4096, 64),
            |mut allocator| {
                while allocator.allocate(64).is_some() {}
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    benches,
    bench_buddy_new,
    bench_buddy_allocate,
    bench_buddy_allocate_many,
    bench_buddy_free,
    bench_buddy_allocate_free_cycle,
    bench_buddy_free_blocks,
    bench_buddy_exhaustion,
);
criterion_main!(benches);
