use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion, BatchSize, BenchmarkId};
use cache::{BlockPool, SlotLayout};

fn bench_pool_new(c: &mut Criterion) {
    let layout = SlotLayout::new(16, 128, 2);
    c.bench_function("block_pool_new_64", |b| {
        b.iter(|| BlockPool::new(black_box(64), black_box(layout)))
    });
    c.bench_function("block_pool_new_1024", |b| {
        b.iter(|| BlockPool::new(black_box(1024), black_box(layout)))
    });
}

fn bench_pool_allocate(c: &mut Criterion) {
    let counts = [1, 2, 4, 8, 16, 32];
    let layout = SlotLayout::new(16, 128, 2);
    for &count in &counts {
        c.bench_with_input(
            BenchmarkId::new("block_pool_allocate", count),
            &count,
            |b, &count| {
                b.iter_batched(
                    || BlockPool::new(64, layout).unwrap(),
                    |mut pool| pool.allocate(black_box(count)),
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

fn bench_pool_free(c: &mut Criterion) {
    let layout = SlotLayout::new(16, 128, 2);
    c.bench_function("block_pool_free_one", |b| {
        b.iter_batched(
            || {
                let mut pool = BlockPool::new(64, layout).unwrap();
                let blocks = pool.allocate(1).unwrap();
                (pool, blocks)
            },
            |(mut pool, blocks)| {
                for block in blocks {
                    pool.free(block).unwrap();
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_pool_free_many(c: &mut Criterion) {
    let layout = SlotLayout::new(16, 128, 2);
    c.bench_function("block_pool_free_many_16", |b| {
        b.iter_batched(
            || {
                let mut pool = BlockPool::new(64, layout).unwrap();
                let blocks = pool.allocate(16).unwrap();
                (pool, blocks)
            },
            |(mut pool, blocks)| pool.free_many(black_box(blocks)),
            BatchSize::SmallInput,
        )
    });
}

fn bench_pool_stats(c: &mut Criterion) {
    let layout = SlotLayout::new(16, 128, 2);
    let pool = BlockPool::new(64, layout).unwrap();
    c.bench_function("block_pool_stats", |b| {
        b.iter(|| black_box(&pool).stats())
    });
}

fn bench_pool_has_leaks(c: &mut Criterion) {
    let layout = SlotLayout::new(16, 128, 2);
    let pool = BlockPool::new(64, layout).unwrap();
    c.bench_function("block_pool_has_leaks", |b| {
        b.iter(|| black_box(&pool).has_leaks())
    });
}

fn bench_pool_block(c: &mut Criterion) {
    let layout = SlotLayout::new(16, 128, 2);
    let pool = BlockPool::new(64, layout).unwrap();
    let id = cache::BlockId::new(0);
    c.bench_function("block_pool_block", |b| {
        b.iter(|| black_box(&pool).block(black_box(id)))
    });
}

fn bench_pool_alloc_free_cycle(c: &mut Criterion) {
    let layout = SlotLayout::new(16, 128, 2);
    let cycle_sizes = [1, 4, 8, 16];
    for &size in &cycle_sizes {
        c.bench_with_input(
            BenchmarkId::new("block_pool_alloc_free_cycle", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || BlockPool::new(64, layout).unwrap(),
                    |mut pool| {
                        let blocks = pool.allocate(size).unwrap();
                        pool.free_many(blocks).unwrap();
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

criterion_group!(
    benches,
    bench_pool_new,
    bench_pool_allocate,
    bench_pool_free,
    bench_pool_free_many,
    bench_pool_stats,
    bench_pool_has_leaks,
    bench_pool_block,
    bench_pool_alloc_free_cycle,
);
criterion_main!(benches);
