# Live Python/Rust Performance Gate

The live gate is a pytest integration test, not only a validator over checked-in
numbers. It runs the same persistent SQLite workloads through Python and the
current Rust facade, verifies matching state digests, and fails when measured
ratios violate `evaluate_persistent_performance_gate` thresholds.

```powershell
$env:KOGWISTAR_RUN_PERFORMANCE_GATES = "1"
$env:KOGWISTAR_PERFORMANCE_REPORT = ".tmp/rust-persistent-performance.json"
.venv\Scripts\python.exe -m pytest tests/core/test_rust_port_live_performance_gate.py -q -p no:cacheprovider
```

The live gate runs on pull requests, pushes to `main`, and the scheduled
workflow. It can also be selected with `performance` in `workflow_dispatch`.
It is a separate job from correctness tests because fresh-process timing is
slower and host-sensitive. CI preserves the measured JSON report as an artifact.
The live comparison uses 100 items, 20 read cycles, and 10 samples per mode;
it compares Python with the standalone Rust-native executable. Python launches
Rust and verifies its final persisted state only after the timed Rust process
exits. The timed Rust path has no Python/PyO3 seam. The full diagnostic
benchmark still includes Python/Rust facade modes, but those hybrid modes are
not performance-gated.

Current thresholds are intentionally not relaxed: Rust p95 latency may not
exceed Python by more than 10%, throughput must remain at least 95% of Python,
and peak RSS may not exceed Python by more than 10%. A failing host is evidence
to investigate or explicitly scope a platform-specific contract, not permission
to silently widen the threshold.

### Benchmark Version 5 Timing Contract

Version 5 keeps the version-4 per-operation timing contract and changes the
performance gate target from the Python-facing Rust facade to `rust_native`.
The standalone Rust process reports its own Linux peak RSS; platforms without a
supported process-RSS source report `null`, making the performance gate
`not_evaluated` rather than silently passing the memory check. Release-gate
validation accepts only `rust_native` for version-5 performance evidence.

The mixed trace retains separate write, event-read, and projection-read
latency distributions. It is still a synthetic SQLite storage workload, not
an end-to-end HTTP/server benchmark or evidence for PostgreSQL/Chroma.

### Version-5 CI Measurement

GitHub Actions run `44632135094`, Ubuntu CPython 3.13.18: 100 items, 20 read
cycles, 10 fresh samples per mode. Python and standalone Rust-native produced
the same canonical state digest. All three thresholds passed; Rust-native RSS
was 4.2% of Python RSS.

| Workload | Rust-native throughput / Python | Rust-native p95 / Python |
|---|---:|---:|
| Event append | 35.71x | 0.025x |
| Event replay | 20.97x | 0.052x |
| Mixed interleaved | 21.44x | 0.030x |
| Projection list | 9.45x | 0.115x |
| Projection replace | 70.37x | 0.013x |

This is strong evidence that the seam-free Rust SQLite storage slice is
promising. It does not prove whole-stack parity or performance on other
backends. Whole-trace mixed timing had about 50% sample variation, so use the
per-operation distributions and treat whole-trace tail comparisons as noisy.
The Windows CPython 3.13.3 recheck matched the canonical digest and showed the
same direction, but Windows Rust RSS is unavailable; its gate status was
`not_evaluated`, not pass.

### Historical Benchmark Version 4 Timing Contract

The gate now records latency around each logical storage operation and computes
`operation_latency_ms.p50_ms`, `p95_ms`, and `p99_ms` over all operations in all
fresh samples. Append/replace measure one committed write each; replay/list
measure one query call each; mixed mode measures one complete trace step,
including its transaction boundary. Throughput retains the prior workload
counting contract.

`p95_elapsed_ms` remains only as a compatibility field for p95 across whole
workload sample durations. With five samples it is effectively the slowest of
five whole-workload runs; it is not per-operation p95 and is no longer used by
the version-4 performance gate. The release validator independently recomputes
version-4 operation-latency and throughput ratios from mode measurements, then
checks the recorded ratios and expected operation counts from workload size and
sample count. Forged `passed` flags, stale ratios, or incomplete samples cannot
satisfy the gate. Version-1/3 historical reports retain their legacy validation
path and are not operation-tail evidence.

### Version-4 Live Measurement

Windows CPython, 2026-11-19: 100 items, 20 read cycles, 5 fresh samples per
mode. Canonical digests matched. Gate **failed**; thresholds unchanged. Table
shows actual per-operation p95 and throughput ratios, Rust relative to Python.
Read/list counts are 100; append/replace 500; mixed trace steps 1,500 per mode.

| Workload | Python p95 ms | Rust p95 ms | p95 ratio | Throughput ratio |
|---|---:|---:|---:|---:|
| Event append | 33.38 | 32.65 | 0.98x | 1.07x |
| Event replay | 50.12 | 34.51 | 0.69x | 1.58x |
| Projection replace | 32.62 | 37.95 | 1.16x | 0.98x |
| Projection list | 24.67 | 42.03 | 1.70x | 0.67x |
| Mixed interleaved step | 49.95 | 51.07 | 1.02x | 1.01x |

Peak RSS ratio was 0.98x. Projection listing is dominant regression; projection
replace also misses p95 limit despite near-parity throughput. Rust replay and
append improve. Do not infer whole-trace tail from operation p95: per-sample
mixed trace p95 ratio was 1.25x, while pooled mixed-step p95 was 1.02x. Both
measurements remain visible, keeping operation and workload latency distinct.

## Local Diagnostic Result

Windows CPython 3.13.3 diagnostic, 2026-11-19: 100 items, 20 read cycles, 2
samples per mode. All four modes produced the same canonical state digest. The
table shows throughput relative to Python; this two-sample run is diagnostic,
not a gate result.

| Mode | Event append | Event replay | Projection replace | Projection list | Mixed interleaved |
|---|---:|---:|---:|---:|---:|
| Rust facade, per-call | 0.66x | 1.15x | 0.63x | 0.57x | 0.65x |
| Rust facade, cached session | 1.46x | 6.18x | 1.42x | 2.37x | 0.67x |
| Rust-native | 1.85x | 11.36x | 1.78x | 4.55x | 3.80x |

The default uncached facade is slower for write-heavy workloads. Session reuse
improves individual capabilities, but the mixed trace remains slower because
each transaction operation still crosses the Python/PyO3 boundary. Rust-native
results show SQLite implementation itself is not the bottleneck. An experimental
benchmark-only `rust_batched` mode now tests one event+projection batch per
record, preserving that record's atomic commit and the sequential visibility
trace; it does not change product defaults.

### Autocommit Read-Handle Reuse

Windows CPython 3.13.3, 2026-11-19: 100 items, 20 read cycles, 5 samples per
mode, with the Rust store reusing its cached handle for autocommit reads. All
Python/Rust samples produced the same canonical state digest. Rust writes kept
the existing fresh-connection `BEGIN IMMEDIATE` path. Ratios below are Rust
facade relative to Python; the run **did not pass** the current thresholds.

| Workload | Throughput ratio | p95 latency ratio |
|---|---:|---:|
| Event append | 0.63x | 1.62x |
| Event replay | 1.63x | 0.62x |
| Projection replace | 0.61x | 1.62x |
| Projection list | 0.67x | 1.50x |
| Mixed interleaved | 0.99x | 1.02x |

Peak RSS ratio was 0.98x. The prior 0.57x projection-list value used only two
samples, so these measurements are not a controlled before/after comparison.
These version-3 p95 values are whole-workload sample tails, not per-operation
tails. The JSON report remains under ignored `.tmp` benchmark output.

### Cached Write-Transaction Handle

Windows CPython 3.13.3, 2026-11-19: same 100-item, 20-cycle, 5-sample live
gate, after using the cached autocommit handle for `BEGIN IMMEDIATE` writes.
The pre-change numbers above used the same local host and workload. Write
throughput improved materially, while the overall gate remains failed because
projection listing is still below threshold.

| Workload | Before throughput | After throughput | After p95 latency |
|---|---:|---:|---:|
| Event append | 0.63x | 1.068x | 0.942x |
| Event replay | 1.63x | 1.599x | 0.646x |
| Projection replace | 0.61x | 1.070x | 0.937x |
| Projection list | 0.67x | 0.715x | 1.307x |
| Mixed interleaved | 0.99x | 0.999x | 1.002x |

Peak RSS ratio was 0.983x. `kogwistar-store-sqlite` tests passed (72 passed,
1 ignored), including external-commit visibility across cached reads and
writes. This slice removes per-write connection setup; it does not close the
performance gate. JSON report: ignored `.tmp/sqlite-cached-write-performance.json`.

### Experimental Batched Seam

Windows CPython 3.13.3, 2026-11-19: 20 items, 2 read cycles, 2 samples. Python,
regular Rust facade, and `rust_batched` produced identical canonical state
digest (`9388bcd0...95bae9f1ad`). The intentionally small run is directional;
two samples do not establish a stable performance guarantee.

| Mode | Mixed throughput (ops/s) | Relative to Python | p95 (ms) | Relative to Python |
|---|---:|---:|---:|---:|
| Python | 29.12 | 1.00x | 2068.78 | 1.00x |
| Rust facade, per-call | 19.20 | 0.66x | 3134.19 | 1.51x |
| Rust batched, one atomic event+projection call per record | 59.08 | 2.03x | 1043.46 | 0.50x |

This supports the seam-cost hypothesis: grouping related mutations under one
Rust-owned transaction can materially improve the mixed workload while
preserving canonical results. Next evidence needed: repeat at the 100-item gate
size with more samples, then validate concurrency and backend semantics before
considering product adoption. Raw output remains under ignored
`.tmp/rust-persistent-batched-small.json`; the 100-item four-mode run remains
`.tmp/rust-persistent-four-mode-diagnostic.json`.
