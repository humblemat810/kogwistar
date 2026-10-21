# Persistent SQLite Facade Benchmark

## Purpose And Method

This benchmark compares the existing Python `EngineSQLite` facade, the current
Rust per-capability-call facade, and an experimental persistent Rust
autocommit session. It measures end-to-end facade cost, not Python versus Rust
language speed in isolation. The Rust path includes PyO3 dispatch, JSON
conversion, SQL execution, and connection/session lifecycle.

Runner: [`scripts/rust_port_persistent_benchmark.py`](../../scripts/rust_port_persistent_benchmark.py).

Run from the Kogwistar repository with its native extension built:

```powershell
python scripts/rust_port_persistent_benchmark.py `
  --output-root .tmp/sqlite-benchmark `
  --items 30 --read-cycles 5 --samples 3
```

The recorded run used CPython 3.13.3 on Windows, 30 event/projection items,
five read cycles, and three fresh child-process samples per mode. Each mode
used a separate temporary SQLite database. The workloads were event append,
event replay, named-projection replacement, and projection listing. All modes
produced the same canonical state digest (`d54d466b...b9d133`). The fixture is
small and local; it is not a cold-cache, contention, or production-scale test.

## Results

| Operation | Python median | Rust per-call median | Rust cached-session median | Cached vs per-call Rust | Cached throughput vs Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| Append 30 events | 486.9 ms | 1,134.3 ms | 720.2 ms | 36.5% faster | 0.68x |
| Replay 150 events | 86.7 ms | 163.1 ms | 37.8 ms | 76.8% faster | 2.30x |
| List 150 projections | 50.6 ms | 154.6 ms | 41.8 ms | 73.0% faster | 1.21x |
| Replace 30 projections | 479.0 ms | 1,089.0 ms | 623.3 ms | 42.8% faster | 0.77x |

The persistent Rust session improves every workload relative to Rust's current
per-call connection lifecycle. Replay and projection listing exceeded Python
throughput, while event append and projection replacement remained slower.
Cached Rust peak RSS was approximately 108.1 MB versus Python's 106.0 MB
(about 2.2% higher). The existing Rust per-call performance gate remains
failed; its RSS gate passed.

## Correctness And Authority Boundary

The experimental cached session is explicitly closed before a fresh Rust
per-call facade reopens the database. A focused regression test verifies
multiple Rust autocommit operations through one cached session, then confirms
their durable results via that fresh Rust connection after close. Existing
tests separately verify ordinary Python/Rust handoffs when no cached Rust
session remains open.

Kogwistar's SQLite authority guard rejects simultaneous Python and Rust
ownership of the same database (`KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT`).
Therefore this prototype is not a transparent optimization for mixed Python/
Rust callers. The benchmark does not bypass that guard to test cross-engine
overlap. Adoption would require a clearly scoped Rust-authoritative lifetime,
with all other SQLite access excluded until the cached session closes. Normal
per-call Rust behavior remains the compatibility path for workflows that hand
authority back and forth.

## Decision

Do not enable persistent Rust sessions globally based on this fixture. A
scoped persistent Rust authority is promising for repeated reads and improves
the writes relative to Rust per-call operation, but event append and
projection replacement remain slower than Python. Keep it opt-in for
Rust-authoritative sessions and evaluate representative concurrency,
longer-lived workloads, shutdown/error cleanup, and larger data before broad
adoption. The benchmark demonstrates a workload-specific optimization, not a
general SQLite backend win for Rust.
