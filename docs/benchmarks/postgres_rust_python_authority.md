# Python And Rust Storage Path Benchmark

## Purpose

Compare shipped storage paths at the public Python facade boundary. Do not
interpret these measurements as a comparison of Python and Rust language
throughput in isolation. The Rust path includes PyO3 dispatch, JSON conversion,
transaction/session setup, and the current Rust PostgreSQL connection-pool
lifecycle.

The manual PostgreSQL runner is
[`scripts/postgres_authority_benchmark.py`](../../scripts/postgres_authority_benchmark.py).
It uses precomputed payloads and intentionally excludes embedding inference.
It creates uniquely named `gke_perf_<random>` schemas and drops only those
schemas after each sample. Use a disposable benchmark database.

## Workloads

| Workload | What it measures | Semantic caveat |
| --- | --- | --- |
| Individual event append | One public append call and its transaction per event | Closest to many individually durable writes |
| Grouped event append | Many append calls inside one outer transaction | Changes the commit boundary; sensitivity probe only |
| Atomic conversation record, each | One event plus one named projection per record in one transaction | Preserves per-record atomicity; representative, not a full chat-server benchmark |
| Grouped conversation records | Many event/projection pairs under one outer transaction | Changes durability boundaries; do not adopt without product-level semantic review |
| Event replay | Read a bounded event sequence through each public facade | Uses preloaded records and warm database state |
| Projection listing | List a bounded named-projection set | Uses preloaded records and warm database state |

The Rust transaction case keeps one transaction connection while operations
cross the Python/Rust boundary individually. Thus it estimates the value of
reusing a transaction, not the value of a one-call native batch API. No
conversation turns are combined by the benchmark recommendation.

## Run

Run from the Kogwistar repository with the `pgvector` optional dependencies
(SQLAlchemy) and native Rust extension installed:

```powershell
$env:GKE_PG_DSN = "postgresql+psycopg://..."
python scripts/postgres_authority_benchmark.py --items 100 --samples 5
```

The runner normalizes the DSN for the Rust driver, uses a persistent six-slot
SQLAlchemy pool for Python, and uses the current native Rust facade for Rust.
It reports medians, p95 sample latency, throughput, and Rust/Python median
latency ratios. It does not flush PostgreSQL or OS caches; results describe
steady-state/warm database behavior, not cold-start behavior. Compare like-for-
like runs on the same host and database configuration.

## Evidence To Date

### Persistent SQLite facade, CPython 3.13.3

Re-run on 2026-10-21 using 30 items, five read cycles, and three fresh-process
samples per mode. Both implementations produced the same state digest.

| Operation | Python median | Rust median | Rust/Python latency | Rust/Python throughput |
| --- | ---: | ---: | ---: | ---: |
| Event append, 30 events | 561.8 ms | 1,191.0 ms | 2.15x | 0.47x |
| Event replay, 150 reads | 139.7 ms | 174.5 ms | 1.22x | 0.80x |
| Projection list, 150 reads | 62.3 ms | 173.2 ms | 2.74x | 0.36x |
| Projection replace, 30 writes | 568.4 ms | 1,156.3 ms | 2.03x | 0.49x |

The Rust latency/throughput gate failed; the RSS gate passed. This is the
current SQLite capability facade, not PostgreSQL. It includes Python-to-PyO3
JSON calls and a SQLite connection per Rust capability call. It does not prove
that Rust kernels are inherently slower.

### In-memory native shadow reads, CPython 3.13.3

Ran `scripts/rust_port_phase2_benchmark.py` with 32 dataset items, two warmups,
three fresh-process samples, and ten measured cycles per sample. Each cycle
reads graph records, runs a vector query, replays events, and lists named
projections. Output digests matched; there were no external DB queries or
authoritative writes.

| Mode | p95 cycle latency | Throughput | Peak RSS |
| --- | ---: | ---: | ---: |
| Python | 2.31 ms | 1,800 ops/s | 81.9 MB |
| Rust shadow path | 5.34 ms | 822 ops/s | 82.9 MB |

Rust/Python latency ratio was 2.31x; throughput ratio was 0.46x; peak RSS ratio
was 1.01x. This Rust path is explicitly a read-only safety shadow that rebuilds
an isolated store from a JSON snapshot for each cycle, not a persistent native
request path. The result measures that boundary design, not native Rust graph
query performance.

### PostgreSQL, local disposable Docker instance

Initial baseline on 2026-10-21 used an ephemeral `postgres:16` Docker Desktop
container, 30 items per workload, five samples, and precomputed JSON (no
embedding model). The Rust facade opened its store/pool for individual native
calls. A follow-up run used the same setup and harness after adding an explicit
long-lived Rust store context with a persistent worker, Tokio runtime, and
Postgres pool. Both runs used the same Python path and SQLAlchemy pool. The
database was local and warm, not production-sized. Numbers include bridge,
connection/pool lifecycle, transaction, and SQL costs.

| Workload (30 items) | Python baseline | Rust before | Rust after | Rust improvement | Python after | Rust/Python after latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Individual event append | 472.9 ms | 1,156.1 ms | 612.2 ms | 47.0% faster | 463.1 ms | 1.32x |
| Grouped event append | 286.5 ms | 402.8 ms | 372.6 ms | 7.5% faster | 285.3 ms | 1.31x |
| Per-record atomic event + projection | 757.9 ms | 1,305.7 ms | 728.2 ms | 44.2% faster | 730.8 ms | 1.00x |
| Grouped conversation records | 445.1 ms | 514.6 ms | 477.6 ms | 7.2% faster | 443.4 ms | 1.08x |
| Event replay batch | 8.1 ms | 21.5 ms | 5.0 ms | 76.8% faster | 7.9 ms | 0.63x |
| Projection listing | 8.1 ms | 22.9 ms | 4.5 ms | 80.2% faster | 7.8 ms | 0.58x |

The persistent context substantially narrows or reverses the earlier Rust
penalty. Rust reads are about 1.7x faster than Python in this small local
workload; atomic writes are effectively near parity but remain slightly slower
than Python; ordinary and grouped event appends still favor Python. Do not
interpret the Rust improvement percentages as Python-versus-Rust speedups:
they compare Rust before/after the pool/runtime lifecycle change. The Python
before/after movement is within ordinary small-run variation.

The final follow-up run reported medians, p95 sample latency, and throughput
over five samples. Rust individual append was 612.2 ms median (629.7 ms p95,
49.0 ops/s), grouped append 372.6 ms (385.5 ms p95, 80.5 ops/s), atomic
conversation writes 728.2 ms (740.9 ms p95, 41.2 ops/s), replay 5.0 ms (5.4 ms
p95, 5,995 ops/s), and projection listing 4.5 ms (5.3 ms p95, 6,641 ops/s).
The Python path measured 64.8, 105.2, 41.1, 3,801, and 3,823 ops/s for those
same workloads, respectively.

The benchmark creates unique schemas, checks read counts, and drops only those
schemas. Grouped workloads change commit boundaries and are sensitivity probes,
not recommendations for conversation durability. The prototype uses an
explicit context per `RustPostgresSession`; it is not a global DSN cache. Each
context owns one worker thread/runtime/pool and must be explicitly closed.
Contextless calls retain the original behavior. Active contexts are bounded to
64 per process. The benchmark's grouped path and end-to-end run verified
begin/operations/commit and explicit worker close. No production/network
database or cold-cache run was performed.

The first after-run attempt deadlocked on transaction dispatch because the
worker retained its receiver mutex while re-entering its own receive loop. The
live run showed an idle-open database transaction, so that result was discarded.
The receiver lock lifetime was corrected, and the complete five-sample matrix
then finished. This is why the successful measurements below are the only
valid after results.

An initial attempt failed because the benchmark reused event IDs across
workload namespaces. The metadata store correctly rejected the duplicate; the
runner was corrected to include the namespace in event IDs, then the complete
matrix passed. The runner passed Ruff and compilation checks. Do not generalize
these local results to larger datasets or remote Postgres without another run.

## Implementation Questions And Decision

- Python creates and retains a SQLAlchemy engine/pool for the backend lifetime.
- Contextless native Rust calls still construct `PostgresStore` from the DSN
  for an operation. An explicit `RustPostgresSession` context now reuses one
  Rust store/runtime/pool across calls, while transaction operations remain
  serialized through its worker.
- The Rust facade serializes each operation and result through JSON. Batching
  can amortize this boundary, but must not combine independently durable
  conversation operations.
- Postgres query plans, network latency, transaction commits, indexes, and
  pgvector maintenance may dominate language-level computation.
- Embedding generation is outside this benchmark. A Rust storage port cannot
  accelerate the embedding model or its network request.

## Decision Rule

The prototype is worth retaining for explicit long-lived contexts where read
throughput or repeated atomic transactions matter; it is not evidence to move
all Postgres authority to Rust. Python remains faster for event appends here,
and writing is near parity rather than a decisive win. Next test the same design
on a representative larger/remote database and realistic concurrency. If write
latency is still material, separately prototype a single-call atomic-record
operation, preserving current commit semantics. Do not combine independently
durable conversation operations just to improve throughput.
