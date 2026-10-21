# Persistent SQLite Facade Benchmark

## Purpose And Method

This benchmark compares the existing Python `EngineSQLite` facade, the current
Rust per-capability-call facade, and an experimental persistent Rust
autocommit session. It measures end-to-end facade cost, not Python versus Rust
language speed in isolation. The Rust path includes PyO3 dispatch, JSON
conversion, SQL execution, and connection/session lifecycle.

Runner: [`scripts/rust_port_persistent_benchmark.py`](../../scripts/rust_port_persistent_benchmark.py).
The direct Rust workload is implemented by the standalone
[`sqlite_native_authority_benchmark`](../../rust/crates/kogwistar-store-sqlite/examples/sqlite_native_authority_benchmark.rs)
example.

Run from the Kogwistar repository with its native extension built:

```powershell
python scripts/rust_port_persistent_benchmark.py `
  --output-root .tmp/sqlite-benchmark `
  --items 30 --read-cycles 5 --samples 3
```

The recorded run used CPython 3.13.3 on Windows, 30 event/projection items,
five read cycles, and three fresh-process samples per mode. Each mode used a
separate temporary SQLite database. The workloads were event append, event
replay, named-projection replacement, and projection listing. All four modes
produced the same canonical state digest (`d54d466b...b9d133`). The fixture is
small and local; it is not a cold-cache, contention, or production-scale test.

`rust_native` calls the production `SqliteStore` directly in a standalone Rust
process. Python only launches the process and checks the output after its
timers stop. Database open/schema initialization is outside the measured
interval. Every event/projection write still uses its own store transaction
and commit; this does not combine independent writes into a batch transaction.
The native process RSS was not collected, so memory comparisons exclude that
mode.

## Results

| Operation | Python median | Rust per-call | Rust cached session | Rust-native | Native throughput vs Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| Append 30 events | 470.5 ms | 1,137.6 ms | 697.7 ms | 590.5 ms | 0.80x |
| Replay 150 events | 90.5 ms | 151.1 ms | 38.2 ms | 24.5 ms | 3.69x |
| List 150 projections | 47.1 ms | 141.5 ms | 38.8 ms | 26.3 ms | 1.79x |
| Replace 30 projections | 471.6 ms | 1,090.3 ms | 596.4 ms | 510.3 ms | 0.92x |

The scoped cached Rust session improves every workload relative to Rust's
per-call connection lifecycle. The native Rust loop further improves on the
cached Python-to-Rust facade in all four workloads, indicating that repeated
bridge and JSON costs are material in this fixture. Even without the bridge,
Python remains faster for the two write workloads: native Rust achieved about
80% of Python's event-append throughput and 92% of its projection-replacement
throughput. Native Rust replay was 3.69x Python throughput; projection listing
was 1.79x. The existing Rust per-call performance gate remains failed; its RSS
gate passed.

## Correctness And Authority Boundary

The experimental cached session is explicitly closed before Python reopens the
database. A focused regression test verifies competing Python access is
rejected while Rust owns the database, then verifies the ownership lease is
released and Python writes are visible to a fresh Rust call after close.
Existing tests separately verify ordinary Python/Rust handoffs when no cached
Rust session remains open.

Kogwistar's SQLite authority guard rejects simultaneous Python and Rust
ownership of the same database (`KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT`).
Therefore this prototype is not a transparent optimization for mixed Python/
Rust callers. The benchmark does not bypass that guard to test cross-engine
overlap. Adoption would require a clearly scoped Rust-authoritative lifetime,
with all other SQLite access excluded until the cached session closes. Normal
per-call Rust behavior remains the compatibility path for workflows that hand
authority back and forth.

## Decision

Do not enable a Rust backend globally based on this fixture. The native result
shows that avoiding Python/PyO3/JSON work pays off for reads, while SQLite
transaction and storage costs still leave Python ahead on these writes. A
future Rust-authoritative read-heavy subsystem may benefit, but evaluate
representative concurrency, longer-lived workloads, shutdown/error cleanup,
larger data, and memory before adoption. This is a workload-specific result,
not a universal Python-versus-Rust ranking.
