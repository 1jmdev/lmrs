use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use cache::{BlockId, PrefixHasher, RadixPrefixCache};

fn bench_prefix_hash_tokens(c: &mut Criterion) {
    let lengths = [1, 4, 16, 64, 256, 1024];
    for &len in &lengths {
        let tokens: Vec<u32> = (0..len as u32).collect();
        c.bench_with_input(
            BenchmarkId::new("prefix_hash_tokens", len),
            &len,
            |b, _| {
                b.iter(|| PrefixHasher::hash_tokens(black_box(tokens.iter().copied())))
            },
        );
    }
}

fn bench_radix_insert(c: &mut Criterion) {
    c.bench_function("radix_insert_short", |b| {
        b.iter(|| {
            let mut cache = RadixPrefixCache::new();
            cache.insert(black_box([10, 20, 30]), black_box(vec![BlockId::new(0)]));
        })
    });

    c.bench_function("radix_insert_long", |b| {
        b.iter(|| {
            let mut cache = RadixPrefixCache::new();
            let tokens: Vec<u32> = (0..100).collect();
            cache.insert(black_box(tokens), black_box(vec![BlockId::new(0)]));
        })
    });
}

fn bench_radix_lookup(c: &mut Criterion) {
    c.bench_function("radix_lookup_hit", |b| {
        let mut cache = RadixPrefixCache::new();
        cache.insert([10, 20, 30, 40, 50], vec![BlockId::new(1)]);
        b.iter(|| cache.longest_match(black_box([10, 20, 30, 40, 50])))
    });

    c.bench_function("radix_lookup_miss", |b| {
        let mut cache = RadixPrefixCache::new();
        cache.insert([10, 20, 30], vec![BlockId::new(1)]);
        b.iter(|| cache.longest_match(black_box([99, 98, 97])))
    });

    c.bench_function("radix_lookup_long_hit", |b| {
        let mut cache = RadixPrefixCache::new();
        let tokens: Vec<u32> = (0..128).collect();
        cache.insert(tokens.clone(), vec![BlockId::new(1)]);
        b.iter(|| cache.longest_match(black_box(tokens.iter().copied())))
    });
}

fn bench_radix_insert_many(c: &mut Criterion) {
    c.bench_function("radix_insert_100_unique", |b| {
        b.iter(|| {
            let mut cache = RadixPrefixCache::new();
            for i in 0..100_u32 {
                cache.insert([i, i + 1, i + 2], vec![BlockId::new(i as usize)]);
            }
        })
    });

    c.bench_function("radix_insert_100_shared_prefix", |b| {
        b.iter(|| {
            let mut cache = RadixPrefixCache::new();
            for i in 0..100_u32 {
                cache.insert([1, 2, 3, i, i + 1], vec![BlockId::new(i as usize)]);
            }
        })
    });
}

fn bench_radix_is_empty(c: &mut Criterion) {
    let cache = RadixPrefixCache::new();
    c.bench_function("radix_is_empty", |b| {
        b.iter(|| black_box(&cache).is_empty())
    });
}

criterion_group!(
    benches,
    bench_prefix_hash_tokens,
    bench_radix_insert,
    bench_radix_lookup,
    bench_radix_insert_many,
    bench_radix_is_empty,
);
criterion_main!(benches);
