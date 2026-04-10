# Benchmarking And Profiling Cheatsheet

## What This Covers

This note is the quick reference for the runtime/conversation write-path profiling work around:

- eager index reconcile cost
- runtime trace fast path
- repeated `model_validate_json(...)` cost inside index job apply
- worker-drain experiments

Main code:

- `kogwistar/runtime/perf_profile.py`
- `tests/runtime/test_in_memory_checkpoint_write_profile.py`
- `kogwistar/runtime/runtime.py`
- `kogwistar/engine_core/indexing.py`

## Quick History

1. We first profiled runtime checkpoint/trace persistence and found the expensive path was not the raw queue insert, but eager reconcile after each write.
2. We added the runtime trace fast path: `WorkflowRuntime._trace_write_mode()` temporarily disables phase-1 index jobs while persisting workflow trace nodes/edges.
3. We then profiled the reconcile loop itself and found repeated `Node.model_validate_json(...)` / `Edge.model_validate_json(...)` in `apply_index_job(...)`.
4. We added a short-lived validation cache per reconcile/worker drain batch in `IndexingSubsystem.reconcile_indexes(...)`.
5. We extended the manual benchmark harness from fake-only to `fake`, `chroma`, and `pg`.
6. We added worker-parallel benchmark cases, but those are informational because backend and threading overhead can dominate.

## What Was Measured

### Runtime checkpoint write

Test:

- `test_manual_profile_in_memory_checkpoint_write`

Purpose:

- compare normal eager trace persistence vs fast-inline trace persistence

Interpretation:

- this is the clearest "user-visible write path" benchmark for workflow trace writes

### Simple resolver workflow

Test:

- `test_manual_profile_simple_resolver_workflow_speedup`

Purpose:

- run a small real workflow with a real `MappingStepResolver`
- retrieve from a preseeded KG
- write to the conversation graph during execution
- compare old baseline vs optimized path

Interpretation:

- this is the best small end-to-end execution benchmark in the suite
- it is end-to-end for workflow execution, but not full application end-to-end

### Total end-to-end speedup

Test:

- `test_manual_profile_in_memory_checkpoint_write_total_speedup`

Purpose:

- compare old baseline:
  - eager trace persistence
  - no validation cache
- against optimized path:
  - fast trace persistence
  - validation cache enabled

Interpretation:

- this is the best "before vs after" benchmark

### Index job breakdown

Test:

- `test_manual_profile_in_memory_index_job_breakdown`

Purpose:

- isolate:
  - `claim_only`
  - `apply_only`
  - `eager_reconcile`
- prove whether repeated validation counts go down

Interpretation:

- this is the best benchmark for the validation-cache change specifically

### Worker parallel benchmark

Test:

- `test_manual_profile_in_memory_index_job_worker_parallel_speedup`

Purpose:

- compare:
  - eager baseline
  - 1 worker drain
  - 4 worker drain

Interpretation:

- useful for experiments
- not a strict "must be faster" benchmark
- very sensitive to backend, local machine, thread overhead, and job batch size

## How To Run

Run one benchmark:

```powershell
.venv\Scripts\python.exe -m pytest tests/runtime/test_in_memory_checkpoint_write_profile.py::test_manual_profile_in_memory_checkpoint_write_total_speedup[fake] -q -s --run-manual
```

Run the whole manual suite:

```powershell
.venv\Scripts\python.exe -m pytest tests/runtime/test_in_memory_checkpoint_write_profile.py -q -s --run-manual
```

Try another backend:

```powershell
.venv\Scripts\python.exe -m pytest tests/runtime/test_in_memory_checkpoint_write_profile.py::test_manual_profile_in_memory_index_job_breakdown[chroma] -q -s --run-manual
```

```powershell
.venv\Scripts\python.exe -m pytest tests/runtime/test_in_memory_checkpoint_write_profile.py::test_manual_profile_in_memory_index_job_breakdown[pg] -q -s --run-manual
```

Notes:

- `pg` may skip if the local pgvector test container is unavailable.
- `chroma` and `pg` are noisier than `fake`.

## Reading The Numbers

Key fields:

- `scenario_total_ms`
  - steady-state timed work for the scenario
- `seed_total_ms`
  - setup/seeding cost
- `wall_total_ms`
  - broader elapsed time, mainly useful in worker benchmarks

For a long-running server interpretation:

- prefer `scenario_total_ms`
- treat `seed_total_ms` as benchmark setup, not request latency

## Profiler Mechanics

Primary profiler:

- `TimingRecorder`

How it works:

- wraps selected methods temporarily
- times them with `time.perf_counter()`
- aggregates count, total, avg, max

Optional profiler:

- `SysMonitoringWallProfiler`

How it works:

- uses Python `sys.monitoring`
- attributes wall time to Python call frames
- only enabled when requested

## Known Wins

### Win 1: runtime trace fast path

Mechanism:

- `WorkflowRuntime._trace_write_mode()` disables eager phase-1 index jobs during runtime trace persistence

Effect:

- much less synchronous reconcile work during:
  - `_persist_workflow_run`
  - `_persist_step_exec`
  - `_persist_checkpoint`

Observed earlier on fake:

- about `3x` faster in the runtime checkpoint benchmark

### Win 2: validation cache inside reconcile/apply

Mechanism:

- `IndexingSubsystem.reconcile_indexes(...)` creates a short-lived cache
- `apply_index_job(...)` reuses one validated parse per exact raw JSON within the drain batch
- callers still receive a deep copy, not the shared cached instance

Effect:

- removes repeated validation of the same node/edge for sibling jobs such as:
  - `node_docs` + `node_refs`
  - `edge_refs` + `edge_endpoints`

Observed earlier on fake:

- about `1.86x` faster for `apply_only`

### Win 3: simple resolver workflow end-to-end execution

Mechanism:

- combines both main improvements:
  - runtime trace fast path
  - validation cache inside reconcile/apply

Effect:

- speeds up a realistic workflow execution that:
  - runs the runtime
  - retrieves from KG
  - writes to the conversation graph
  - persists runtime trace data

Observed:

- `fake`: about `1.79x`
- `chroma`: about `1.62x`
- `pg`: about `1.66x`

Meaning:

- the optimization is not only a fake-backend microbenchmark win
- it shows up across all three backends on a realistic execution path

## What To Expect By Backend

### fake

- best signal for micro-optimizations
- easiest place to see the validation-cache win clearly

### chroma

- backend overhead is larger
- worker mode may be flat or slower on small local runs
- validation-cache win can be drowned out by collection overhead

### pg

- database and container overhead add noise
- good for realism
- not great for tiny one-iteration microbenchmark conclusions

## Interpretation Rules

Good evidence:

- `apply_only` internal validation counts go down
- total speedup is repeatable across several runs
- optimized path stays faster on `fake`

Use caution when:

- the benchmark has `iterations=1`
- comparing worker runs on `chroma` or `pg`
- the difference is small relative to backend startup or I/O variance

## Graph Impact Summary

### Runtime graph

- design writes:
  - mostly unaffected by the fast trace path
  - they write to the workflow engine, not the runtime trace persistence path
- execution writes:
  - directly helped by the fast trace path
  - indirectly helped by the validation cache if/when derived index jobs drain
  - now also confirmed by the simple resolver workflow benchmark across `fake`, `chroma`, and `pg`

### Conversation graph

- normal conversation writes:
  - only partly helped
  - trace nodes/edges written by `WorkflowRuntime` benefit most
- reads:
  - mostly unaffected directly
- tail lookup:
  - mostly unaffected directly
- sequence stamping:
  - unaffected directly
- policy hooks:
  - unaffected directly
- dedupe:
  - logic unchanged
  - still depends on derived endpoint rows being present

## Best Current Story

If you need one short summary:

- the meaningful measured wins are on synchronous runtime trace writes and on repeated validation inside index-job apply
- a small end-to-end resolver workflow also shows real gains across `fake`, `chroma`, and `pg`
- worker parallelism is still experimental and backend-sensitive
