# Graph Slice Seam Benchmark

This benchmark isolates the typed graph-model slicing seam. It is deliberately
not a storage, MCP, or whole-agent benchmark. The canonical graph envelope and
the pydantic-extension field policy are held constant while the harness
measures Python projection, JSON encoding, the PyO3 Rust projection, and the
public Python bridge.

## Run

From the core repository, build the current native extension using the normal
host-native wheel procedure, then run:

```text
python scripts/graph_slice_seam_benchmark.py --model-count 32 --iterations 30
```

To compare the no-seam Rust projection over the exact same typed envelope
fixture:

```text
python scripts/graph_slice_seam_benchmark.py --model-count 32 --iterations 30 --write-input .tmp-graph-slice.json
cargo run --release --manifest-path rust/Cargo.toml -p kogwistar-contracts --example graph_slice_native_benchmark -- .tmp-graph-slice.json 30
```

The Rust command parses the fixture before timing and keeps typed envelopes in
Rust during timed iterations. Its result therefore excludes Python model
construction, PyO3 dispatch, and JSON serialization. It is a lower-bound
comparison for a future Rust-native service, not an end-to-end application
claim.

The harness verifies three equal projection digests before reporting timing:

- Python `project_with_manifest` output;
- Rust projection output returned by the native extension;
- the public Python bridge output, both one envelope at a time and as a batch.

The batch path is an additive optimization. It sends a JSON array across PyO3
once, while Rust validates and projects every envelope independently. One
invalid envelope fails the whole batch, so batching does not weaken the
contract or create partial model-visible output.

## Interpretation

`python_bridge_end_to_end` includes the existing per-envelope PyO3 and JSON
crossings. `python_bridge_batch_end_to_end` measures the same public contract
with one crossing for the batch. A future Rust-native service can consume the
typed envelope directly and should be compared separately; this benchmark does
not claim that removing PyO3 is automatically faster for every workload.

The result is evidence for seam placement, not a release gate by itself. The
full-stack decision still requires resource-matched MCP, persistence, ACL,
runtime, and recovery benchmarks with equal canonical results.

## Interim Measurement

On the Windows CPython 3.13 checkout environment, with 64 envelopes and 10
iterations, all three projection digests matched. The median timings were:

| Stage | Median |
| --- | ---: |
| Python model projection | 459.98 ms |
| Python JSON encoding | 5.13 ms |
| Rust PyO3 call and output encoding, per envelope | 33.78 ms |
| Rust PyO3 call plus Python decode, per envelope | 40.35 ms |
| Public Python bridge, per envelope | 63.36 ms |
| Public Python bridge, one batch | 75.61 ms |

This sample does **not** demonstrate a batch speedup. The one-batch result is
slower at this payload size, likely because a single large JSON value has a
larger parse/decode working set than many smaller values. The batch API remains
useful as a compatibility and experiment seam, but callers must benchmark their
payload size before selecting it. A Rust-native path that keeps typed values in
Rust is still the candidate for removing the serialization seam; it must be
measured separately before any default change.

The same 64-envelope fixture was then run through the release Rust-native
example for 30 iterations. It produced payload digest
`ea476d0fae660f92166fba3f59dc0fac64b3d99eddb9f6b70b9ef1a988857ce4`, matching
the Python and PyO3 digests, at `0.0619 ms` per typed projection iteration.
That is roughly three orders of magnitude below the public Python bridge in
this small synthetic fixture, but it excludes input construction, transport,
authorization, storage, and output serialization. It should be treated as a
measurement of the projection kernel and seam-removal opportunity only.
