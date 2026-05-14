use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use cache::SequenceId;
use engine::{
    FinishReason, SchedulePolicy, ScheduleResult, Scheduler, SchedulerBudget, Sequence,
    SequenceGroup, SequenceStatus,
};
use sampling::{Greedy, SamplingStrategy};

fn bench_sequence_new(c: &mut Criterion) {
    c.bench_function("sequence_new_short", |b| {
        b.iter(|| {
            Sequence::new(
                black_box(SequenceId::new(1)),
                black_box(vec![1, 2, 3, 4]),
                black_box(1),
                black_box(None),
            )
        })
    });
    c.bench_function("sequence_new_long_prompt", |b| {
        b.iter(|| {
            Sequence::new(
                black_box(SequenceId::new(1)),
                black_box((0..4096).collect::<Vec<_>>()),
                black_box(1),
                black_box(None),
            )
        })
    });
}

fn bench_sequence_accessors(c: &mut Criterion) {
    let seq = Sequence::new(SequenceId::new(1), vec![1, 2, 3, 4], 256, None).unwrap();
    c.bench_function("sequence_id", |b| b.iter(|| black_box(&seq).id()));
    c.bench_function("sequence_status", |b| b.iter(|| black_box(&seq).status()));
    c.bench_function("sequence_finish_reason", |b| {
        b.iter(|| black_box(&seq).finish_reason())
    });
    c.bench_function("sequence_prompt_tokens", |b| {
        b.iter(|| black_box(&seq).prompt_tokens())
    });
    c.bench_function("sequence_generated", |b| {
        b.iter(|| black_box(&seq).generated())
    });
    c.bench_function("sequence_total_len", |b| {
        b.iter(|| black_box(&seq).total_len())
    });
    c.bench_function("sequence_processed_len", |b| {
        b.iter(|| black_box(&seq).processed_len())
    });
    c.bench_function("sequence_pending_tokens", |b| {
        b.iter(|| black_box(&seq).pending_tokens())
    });
    c.bench_function("sequence_has_pending_model_input", |b| {
        b.iter(|| black_box(&seq).has_pending_model_input())
    });
    c.bench_function("sequence_generated_token_ids", |b| {
        b.iter(|| black_box(&seq).generated_token_ids())
    });
}

fn bench_sequence_clone(c: &mut Criterion) {
    let seq = Sequence::new(SequenceId::new(1), (0..128).collect(), 1, None).unwrap();
    c.bench_function("sequence_clone", |b| b.iter(|| black_box(&seq).clone()));
}

fn bench_sequence_mark_processed(c: &mut Criterion) {
    c.bench_function("sequence_mark_processed", |b| {
        b.iter_batched(
            || Sequence::new(SequenceId::new(1), (0..16).collect(), 1024, None).unwrap(),
            |mut seq| seq.mark_processed(black_box(16)),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_sequence_append_sample(c: &mut Criterion) {
    let mut rng = 42_u64;
    let sample = Greedy.sample(&[0.0_f32, 3.0, 1.0], &mut rng).unwrap();
    c.bench_function("sequence_append_sample", |b| {
        b.iter_batched(
            || Sequence::new(SequenceId::new(1), vec![1, 2], 1024, None).unwrap(),
            |mut seq| seq.append_sample(black_box(sample)),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_sequence_status_check(c: &mut Criterion) {
    c.bench_function("sequence_status_can_prefill", |b| {
        b.iter(|| black_box(SequenceStatus::Waiting).can_prefill())
    });
    c.bench_function("sequence_status_is_runnable", |b| {
        b.iter(|| black_box(SequenceStatus::Running).is_runnable())
    });
    c.bench_function("sequence_status_is_terminal", |b| {
        b.iter(|| black_box(SequenceStatus::Finished).is_terminal())
    });
}

fn bench_finish_reason_display(c: &mut Criterion) {
    c.bench_function("finish_reason_eos_display", |b| {
        b.iter(|| black_box(FinishReason::Eos).to_string())
    });
    c.bench_function("finish_reason_length_display", |b| {
        b.iter(|| black_box(FinishReason::Length).to_string())
    });
}

fn bench_sequence_group_new(c: &mut Criterion) {
    c.bench_function("sequence_group_new_single", |b| {
        b.iter(|| {
            let seq = Sequence::new(SequenceId::new(1), vec![1, 2, 3], 1, None).unwrap();
            SequenceGroup::new(black_box(5), black_box(vec![seq]))
        })
    });
}

fn bench_sequence_group_accessors(c: &mut Criterion) {
    let seq = Sequence::new(SequenceId::new(1), vec![1, 2, 3], 1, None).unwrap();
    let group = SequenceGroup::new(5, vec![seq]).unwrap();
    c.bench_function("sequence_group_id", |b| {
        b.iter(|| black_box(&group).id())
    });
    c.bench_function("sequence_group_priority", |b| {
        b.iter(|| black_box(&group).priority())
    });
    c.bench_function("sequence_group_is_finished", |b| {
        b.iter(|| black_box(&group).is_finished())
    });
    c.bench_function("sequence_group_sequences", |b| {
        b.iter(|| black_box(&group).sequences())
    });
}

fn bench_scheduler_new(c: &mut Criterion) {
    c.bench_function("scheduler_new", |b| {
        b.iter(|| Scheduler::new(black_box(SchedulerBudget::new(32, 4096))))
    });
}

fn bench_scheduler_with_policy(c: &mut Criterion) {
    c.bench_function("scheduler_with_policy_fcfs", |b| {
        b.iter(|| {
            Scheduler::new(SchedulerBudget::new(32, 4096)).with_policy(black_box(SchedulePolicy::Fcfs))
        })
    });
    c.bench_function("scheduler_with_policy_priority", |b| {
        b.iter(|| {
            Scheduler::new(SchedulerBudget::new(32, 4096))
                .with_policy(black_box(SchedulePolicy::Priority))
        })
    });
}

fn bench_scheduler_add_group(c: &mut Criterion) {
    let group_counts = [1, 8, 32, 128];
    for &count in &group_counts {
        c.bench_with_input(
            BenchmarkId::new("scheduler_add_group", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let mut scheduler = Scheduler::new(SchedulerBudget::new(256, 65536));
                    for i in 0..count {
                        let seq = Sequence::new(
                            SequenceId::new(i as u64),
                            vec![i as u32; 16],
                            1,
                            None,
                        )
                        .unwrap();
                        let group = SequenceGroup::new(i as u64, vec![seq]).unwrap();
                        scheduler.add_group(group);
                    }
                })
            },
        );
    }
}

fn bench_scheduler_schedule(c: &mut Criterion) {
    c.bench_function("scheduler_schedule_empty", |b| {
        let scheduler = Scheduler::new(SchedulerBudget::new(32, 4096));
        b.iter(|| black_box(&scheduler).schedule())
    });
    c.bench_function("scheduler_schedule_with_groups", |b| {
        b.iter(|| {
            let mut scheduler = Scheduler::new(SchedulerBudget::new(32, 4096));
            for i in 0..8 {
                let seq = Sequence::new(
                    SequenceId::new(i as u64),
                    vec![i as u32; 4],
                    1,
                    None,
                )
                .unwrap();
                let group = SequenceGroup::new(i as u64, vec![seq]).unwrap();
                scheduler.add_group(group);
            }
            scheduler.schedule()
        })
    });
}

fn bench_scheduler_drain_finished(c: &mut Criterion) {
    c.bench_function("scheduler_drain_finished", |b| {
        b.iter(|| {
            let mut scheduler = Scheduler::new(SchedulerBudget::new(32, 4096));
            for i in 0..16 {
                let seq = Sequence::new(SequenceId::new(i as u64), vec![i as u32; 4], 1, None).unwrap();
                let group = SequenceGroup::new(i as u64, vec![seq]).unwrap();
                scheduler.add_group(group);
            }
            scheduler.drain_finished()
        })
    });
}

fn bench_scheduler_budget(c: &mut Criterion) {
    let budget = SchedulerBudget::new(32, 4096);
    c.bench_function("scheduler_budget_can_add", |b| {
        b.iter(|| black_box(budget).can_add(black_box(4), black_box(128)))
    });
    c.bench_function("scheduler_budget_max_sequences", |b| {
        b.iter(|| black_box(budget).max_sequences())
    });
    c.bench_function("scheduler_budget_max_tokens", |b| {
        b.iter(|| black_box(budget).max_tokens())
    });
}

fn bench_scheduler_sequence_mut(c: &mut Criterion) {
    c.bench_function("scheduler_sequence_mut_hit", |b| {
        b.iter_batched(
            || {
                let mut scheduler = Scheduler::new(SchedulerBudget::new(32, 4096));
                let seq = Sequence::new(SequenceId::new(42), vec![1, 2, 3], 1, None).unwrap();
                let group = SequenceGroup::new(42, vec![seq]).unwrap();
                scheduler.add_group(group);
                scheduler
            },
            |mut scheduler| {
                let _ = scheduler.sequence_mut(black_box(SequenceId::new(42)));
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_schedule_result_new(c: &mut Criterion) {
    let entry = engine::BatchEntry::new(
        SequenceId::new(1),
        engine::BatchMode::Decode,
        vec![7],
        0,
    );
    let padded = engine::PaddedBatch::new(vec![vec![7]], vec![1]);
    let varlen = engine::VarLenBatch::new(vec![7], vec![0, 1]);
    let batch = engine::ExecutionBatch::new(vec![entry], padded, varlen);
    c.bench_function("schedule_result_new", |b| {
        b.iter(|| {
            ScheduleResult::new(black_box(batch.clone()), black_box(vec![SequenceId::new(1)]))
        })
    });
}

criterion_group!(
    benches,
    bench_sequence_new,
    bench_sequence_accessors,
    bench_sequence_clone,
    bench_sequence_mark_processed,
    bench_sequence_append_sample,
    bench_sequence_status_check,
    bench_finish_reason_display,
    bench_sequence_group_new,
    bench_sequence_group_accessors,
    bench_scheduler_new,
    bench_scheduler_with_policy,
    bench_scheduler_add_group,
    bench_scheduler_schedule,
    bench_scheduler_drain_finished,
    bench_scheduler_budget,
    bench_scheduler_sequence_mut,
    bench_schedule_result_new,
);
criterion_main!(benches);
