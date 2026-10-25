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

## Conversion Breakdown Recheck

The harness now measures Python input normalization/encoding and Python output
decoding separately. On the Windows CPython 3.13 checkout, 16 envelopes
(44,864 bytes) over 50 iterations produced matching Python, PyO3, one-envelope
bridge, and batch digests. Median measurements were:

| Stage | Median |
| --- | ---: |
| Normalize-copy + encode baseline, 16 envelopes | 7.42 ms |
| Experimental no-copy candidate, 16 envelopes | 7.15 ms |
| Rust PyO3 projection + output JSON encoding, 16 calls | 0.64 ms |
| Python decode of 16 Rust outputs | 0.06 ms |
| Public bridge, 16 individual calls | 7.17 ms |
| Public bridge, one batch call | 6.72 ms |

An experimental no-copy validator/encoder was compared with the existing
normalize-copy-plus-encode implementation in the same 50-iteration run: their
medians were 7.15 ms and 7.42 ms respectively, with overlapping run-to-run
spread. The experiment did not establish a meaningful gain, so the runtime
shortcut was discarded. Input preparation is materially larger than the Rust
kernel in this synthetic slice; output decoding is negligible. A future direct
Python-object/PyO3 conversion should be prototyped and measured against the
same strict JSON fixtures before replacing this compatible string ABI. The
sample is a seam microbenchmark only and does not predict whole-stack latency.
An immediate 20-iteration rerun confirmed the same digest, but end-to-end
medians moved to 9.09 ms (individual calls) and 9.39 ms (batch), while the
input-preparation and Rust-call medians remained 4.72 ms and 0.63 ms. The
spread between runs is a warning that Windows host scheduling/noise dominates
this small workload; do not use these numbers as a release threshold or claim
a stable batch win. A longer controlled run with process isolation is needed
before choosing a new boundary ABI.

### Direct Python-object ABI experiment

A temporary PyO3 variant recursively converted Python dict/list/scalar objects
to `serde_json::Value`, projected the same declared envelope, then returned the
same JSON-string result. On the same 16-envelope fixture, all Python, existing
string-ABI, and object-ABI projection digests matched; invalid non-string keys,
non-finite floats, out-of-range integers, and unsupported objects were rejected.
An alternating-order 12-sample paired run measured median **11.22 ms** for the
existing string ABI and **25.68 ms** for the object ABI (projection plus output
decode). This prototype was about 2.3x slower in this noisy Windows run, so it
was removed rather than kept as an experimental public extension method. The
result is a useful negative finding: avoiding JSON text does not automatically
make a PyO3 seam faster; recursive GIL-bound object traversal and constructing
an intermediate `serde_json::Value` can cost more than Rust's optimized JSON
parser. The native Rust path, which keeps typed data in Rust rather than
round-tripping Python objects, remains the no-language-seam optimization
candidate.

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

## Current Recheck

On the current Windows checkout, an independent 16-envelope fixture was run
through both benchmark paths. The Python, PyO3, batch, and direct Rust outputs
all produced digest
`d8c2e7280f20e6c250dbecd781b2ba3529ce3f9834d19f6006d96fe7cec97eba`.
The direct Rust release example measured `0.01762 ms` per iteration over 10
iterations. The public Python bridge measured `5.842 ms` median over 3
iterations for the same fixture. This confirms the no-seam projection kernel
is materially faster for this workload, while leaving the full-stack transport,
authorization, persistence, and recovery costs outside the claim.
