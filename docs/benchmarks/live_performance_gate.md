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
The live comparison uses 100 items, 20 read cycles, and 5 samples per mode; it
selects only Python and Rust facade because these are the paths constrained by
this gate. The full four-mode benchmark remains available for broader analysis.

Current thresholds are intentionally not relaxed: Rust p95 latency may not
exceed Python by more than 10%, throughput must remain at least 95% of Python,
and peak RSS may not exceed Python by more than 10%. A failing host is evidence
to investigate or explicitly scope a platform-specific contract, not permission
to silently widen the threshold.

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
This confirms the live gate measures per-workload throughput and tail latency;
read-handle reuse preserves visibility but does not pass the gate. The JSON
report remains under ignored `.tmp` benchmark output.

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
