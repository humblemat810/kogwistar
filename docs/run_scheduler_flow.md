# Run Scheduler Flow

This document explains how Kogwistar schedules workflow runs at the AI layer.

## What it does

The scheduler decides which submitted run should start first.
It does not manage CPU scheduling, process priority, or OS-level execution.

## Priority classes

From highest to lowest:

1. `latency-sensitive`
2. `foreground`
3. `background`
4. `batch`

## Admission rules

- `accepted` - run 入 queue, scheduler later start
- `deferred` - run 暫不收，待 later retry / later submit
- `rejected` - run 拒收，通常因 queue_full 且 class too low

## Concurrency rules

- `max_active` bounds total simultaneous runs
- `max_active_by_class` bounds per-class inflight runs
- class limit counts both queued and active work for admission control

## Flow

```text
API request
   │
   v
ChatRunService.submit_workflow_run(...)
   │
   ├─ validates request
   ├─ creates run registry record
   ├─ attaches priority_class
   v
RunScheduler.submit(...)
   │
   ├─ enqueues run in priority heap
   └─ returns queued metadata
   v
Scheduler loop
   │
   ├─ if active runs < max_active
   ├─ and dispatch window has elapsed
   │
   v
Pop highest-priority queued run
   │
   v
Start worker thread for run
   │
   v
_run_workflow(req)
   │
   ├─ emits run.started
   ├─ executes runtime runner
   ├─ emits run.completed / run.failed / run.cancelled
   └─ updates run registry
```

## Policy intent

The scheduler provides AI-layer control over:

- what starts first
- what is deferred
- what is throttled by class

It is intentionally small and lives above backend storage.

## Pause note

Scheduler does not hard-preempt running work.
Current design can only request pause at checkpoint boundary, then resume later.
That keeps existing per-step checkpoint / resume semantics safe.

## Pause modes

- cooperative pause
  - scheduler marks run paused
  - run stops at next checkpoint boundary
  - resume rehydrates `_deps`
- preemptive pause
  - scheduler tries to interrupt running work immediately
  - not implemented yet
  - would need stronger runtime interruption and rollback rules

Note:
- current code implements cooperative pause and pause request only
- it does not implement hard preemption of an already-running step

## Current limits

- concurrency is bounded by scheduler config
- policy is class-based, not tenant-billing aware yet
- fairness is coarse; later slices add budgets and stronger governance
- checkpoint-bound pause is cooperative, not hard preemptive
