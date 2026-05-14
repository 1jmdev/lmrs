use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use tokenizer::{ChatMessage, SpecialTokens, TokenizerRegistry};
use tokenizer::wrapper::EncodeOptions;

fn bench_chat_message_new(c: &mut Criterion) {
    c.bench_function("chat_message_new", |b| {
        b.iter(|| ChatMessage::new(black_box("user"), black_box("hello world")))
    });
}

fn bench_chat_message_accessors(c: &mut Criterion) {
    let msg = ChatMessage::new("user", "hello world this is a longer message for testing");
    c.bench_function("chat_message_role", |b| {
        b.iter(|| black_box(&msg).role())
    });
    c.bench_function("chat_message_content", |b| {
        b.iter(|| black_box(&msg).content())
    });
}

fn bench_chat_message_clone(c: &mut Criterion) {
    let msg = ChatMessage::new("user", "hello world");
    c.bench_function("chat_message_clone", |b| {
        b.iter(|| black_box(&msg).clone())
    });
}

fn bench_encode_options_new(c: &mut Criterion) {
    c.bench_function("encode_options_new", |b| {
        b.iter(|| EncodeOptions::new(black_box(true)))
    });
    c.bench_function("encode_options_without_special", |b| {
        b.iter(|| EncodeOptions::without_special_tokens())
    });
    c.bench_function("encode_options_default", |b| {
        b.iter(|| EncodeOptions::default())
    });
}

fn bench_encode_options_add_special(c: &mut Criterion) {
    let opts = EncodeOptions::new(true);
    c.bench_function("encode_options_add_special_tokens", |b| {
        b.iter(|| black_box(&opts).add_special_tokens())
    });
}

fn bench_special_tokens_new(c: &mut Criterion) {
    c.bench_function("special_tokens_new", |b| {
        b.iter(|| SpecialTokens::new(black_box(vec![2, 3, 5, 7])))
    });
}

fn bench_special_tokens_is_eos(c: &mut Criterion) {
    let st = SpecialTokens::new(vec![2, 3, 5, 7]);
    c.bench_function("special_tokens_is_eos_hit", |b| {
        b.iter(|| black_box(&st).is_eos(black_box(3)))
    });
    c.bench_function("special_tokens_is_eos_miss", |b| {
        b.iter(|| black_box(&st).is_eos(black_box(99)))
    });
}

fn bench_special_tokens_eos(c: &mut Criterion) {
    let st = SpecialTokens::new(vec![2, 3, 5, 7]);
    c.bench_function("special_tokens_eos", |b| {
        b.iter(|| black_box(&st).eos())
    });
}

fn bench_tokenizer_registry_new(c: &mut Criterion) {
    c.bench_function("tokenizer_registry_new", |b| {
        b.iter(|| TokenizerRegistry::new())
    });
}

fn bench_tokenizer_registry_empty_ops(c: &mut Criterion) {
    let registry = TokenizerRegistry::new();
    c.bench_function("tokenizer_registry_len", |b| {
        b.iter(|| black_box(&registry).len())
    });
    c.bench_function("tokenizer_registry_is_empty", |b| {
        b.iter(|| black_box(&registry).is_empty())
    });
}

criterion_group!(
    benches,
    bench_chat_message_new,
    bench_chat_message_accessors,
    bench_chat_message_clone,
    bench_encode_options_new,
    bench_encode_options_add_special,
    bench_special_tokens_new,
    bench_special_tokens_is_eos,
    bench_special_tokens_eos,
    bench_tokenizer_registry_new,
    bench_tokenizer_registry_empty_ops,
);
criterion_main!(benches);
