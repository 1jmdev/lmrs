use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion, BatchSize, BenchmarkId};
use cache::{BlockPool, CacheManager, SequenceId, SlotLayout};

fn bench_manager_new(c: &mut Criterion) {
    c.bench_function("cache_manager_new", |b| {
        b.iter(|| {
            let pool = BlockPool::new(64, SlotLayout::new(16, 128, 2)).unwrap();
            CacheManager::new(black_box(pool))
        })
    });
}

fn bench_manager_register(c: &mut Criterion) {
    let seq_counts = [1, 8, 32, 128];
    for &count in &seq_counts {
        c.bench_with_input(
            BenchmarkId::new("cache_manager_register", count),
            &count,
            |b, &count| {
                b.iter_batched(
                    || {
                        let pool = BlockPool::new(1024, SlotLayout::new(16, 128, 2)).unwrap();
                        CacheManager::new(pool)
                    },
                    |mut manager| {
                        for i in 0..count {
                            manager.register_sequence(SequenceId::new(i as u64)).unwrap();
                        }
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

fn bench_manager_assign_blocks(c: &mut Criterion) {
    let layout = SlotLayout::new(16, 128, 2);
    c.bench_function("cache_manager_assign_blocks_1", |b| {
        b.iter_batched(
            || {
                let pool = BlockPool::new(256, layout).unwrap();
                let mut manager = CacheManager::new(pool);
                manager.register_sequence(SequenceId::new(1)).unwrap();
                manager
            },
            |mut manager| manager.assign_blocks(black_box(SequenceId::new(1)), black_box(1)),
            BatchSize::SmallInput,
        )
    });
    c.bench_function("cache_manager_assign_blocks_8", |b| {
        b.iter_batched(
            || {
                let pool = BlockPool::new(256, layout).unwrap();
                let mut manager = CacheManager::new(pool);
                manager.register_sequence(SequenceId::new(1)).unwrap();
                manager
            },
            |mut manager| manager.assign_blocks(black_box(SequenceId::new(1)), black_box(8)),
            BatchSize::SmallInput,
        )
    });
}

fn bench_manager_free_sequence(c: &mut Criterion) {
    let layout = SlotLayout::new(16, 128, 2);
    let seq = SequenceId::new(1);
    c.bench_function("cache_manager_free_sequence_4_blocks", |b| {
        b.iter_batched(
            || {
                let pool = BlockPool::new(256, layout).unwrap();
                let mut manager = CacheManager::new(pool);
                manager.register_sequence(seq).unwrap();
                manager.assign_blocks(seq, 4).unwrap();
                manager
            },
            |mut manager| manager.free_sequence(black_box(seq)),
            BatchSize::SmallInput,
        )
    });
}

fn bench_manager_sequence_count(c: &mut Criterion) {
    let pool = BlockPool::new(256, SlotLayout::new(16, 128, 2)).unwrap();
    let mut manager = CacheManager::new(pool);
    for i in 0..32 {
        manager.register_sequence(SequenceId::new(i)).unwrap();
    }
    c.bench_function("cache_manager_sequence_count", |b| {
        b.iter(|| black_box(&manager).sequence_count())
    });
}

fn bench_manager_sequences_iter(c: &mut Criterion) {
    let pool = BlockPool::new(256, SlotLayout::new(16, 128, 2)).unwrap();
    let mut manager = CacheManager::new(pool);
    for i in 0..32 {
        manager.register_sequence(SequenceId::new(i)).unwrap();
    }
    c.bench_function("cache_manager_sequences_iter", |b| {
        b.iter(|| {
            let mut count = 0;
            for _ in black_box(&manager).sequences() {
                count += 1;
            }
            count
        })
    });
}

fn bench_manager_table_lookup(c: &mut Criterion) {
    let pool = BlockPool::new(256, SlotLayout::new(16, 128, 2)).unwrap();
    let mut manager = CacheManager::new(pool);
    let seq = SequenceId::new(7);
    manager.register_sequence(seq).unwrap();
    manager.assign_blocks(seq, 4).unwrap();
    c.bench_function("cache_manager_table_lookup", |b| {
        b.iter(|| black_box(&manager).table(black_box(seq)))
    });
}

criterion_group!(
    benches,
    bench_manager_new,
    bench_manager_register,
    bench_manager_assign_blocks,
    bench_manager_free_sequence,
    bench_manager_sequence_count,
    bench_manager_sequences_iter,
    bench_manager_table_lookup,
);
criterion_main!(benches);
