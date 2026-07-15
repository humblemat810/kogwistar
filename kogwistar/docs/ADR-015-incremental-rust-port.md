# ADR-015: Incremental Port to Rust

**Status:** Proposed  
**Date:** 2026-07-14  
**Owner:** Maintainers

## Summary

Kogwistar will move its deterministic core, persistence, runtime, and service
implementation from Python to Rust by capability, not by rewriting the repository
in one release.

The migration will use these rules:

- preserve the current Python, REST, MCP, event, and database contracts until an
  explicit versioned change is approved separately;
- keep exactly one authoritative writer for each capability during every phase;
- use read-only shadow execution and differential tests before each cutover;
- cut over and roll back whole capabilities, not individual low-level functions;
- keep Python as a compatibility SDK and as a worker ecosystem for dynamic LLM,
  OCR, LangGraph, and user-defined resolver integrations;
- treat compatibility with `kogwistar-llm-wiki`, `kg-doc-parser`, and
  `kogwistar-obsidian-sink` as a release gate, not an optional downstream task;
- make the final default engine, workflow runtime, and server Rust implementations.

Expected Rust performance is not, by itself, sufficient reason to cut over. Each
phase must demonstrate semantic parity and measured operational benefit or a clear
maintenance benefit.

## Context

The repository is not a small storage wrapper. At the time of this decision it has
approximately 76,000 lines of Python implementation and 63,000 lines of tests. The
largest implementation areas include:

- `engine_core`: approximately 23,000 lines;
- `runtime`: approximately 10,600 lines;
- `server`: approximately 10,800 lines;
- `conversation`: approximately 10,500 lines.

Its behavior spans:

- graph entities, grounding, provenance, ACLs, and deterministic identifiers;
- in-memory, Chroma, SQLite, PostgreSQL, and pgvector behavior;
- synchronous and asynchronous backend contracts;
- unit-of-work transaction boundaries;
- authoritative event streams, replay, disposable projections, queues, leases,
  checkpoints, and server run state;
- workflow scheduling, fan-out, joins, pause/resume, retry, and nested workflows;
- Python package APIs, REST, SSE, MCP, CLI, and optional provider integrations.

The existing backend contract and runtime parity suites are valuable migration
oracles. They cover fake, Chroma, and PostgreSQL paths as well as sync/async runtime
behavior. The current top-level Python API also exposes `GraphKnowledgeEngine`,
`WorkflowRuntime`, conversation services, Pydantic models, and provider contracts.

`engine_postgres_meta.py` illustrates why a file-by-file translation would be
unsafe. One module currently combines sequence allocation, index-job leases, lane
messages, event append/replay, projection compare-and-swap, workflow snapshots and
deltas, and server-run persistence. Those responsibilities share transactions but
do not form one migration unit.

Existing architecture decisions remain binding:

- event history and designated stores remain authoritative where currently defined;
- serving projections remain disposable and rebuildable;
- pgvector remains inside the PostgreSQL authoritative transaction boundary;
- backend parity is a tested behavioral requirement;
- cross-store ACID is not added by this migration.

### Consumer compatibility scope

The port is an implementation replacement for three known consumers. Compilation
and core-only unit parity are insufficient.

| Consumer | Kogwistar surface that must remain usable | Release-blocking outcome |
|---|---|---|
| `kogwistar-llm-wiki` | Models, `GraphKnowledgeEngine`, namespaces and graph spaces, IDs, logical references, runtime, budgets, recovery, maintenance helpers | Ingest, promotion, query, interrupted-run recovery, and vault rebuild preserve workspace and graph-space isolation |
| `kg-doc-parser` | `Node`, `Edge`, `Document`, `Grounding`, `Span`, runtime results, retry helpers, stable IDs, source pointers, fuzzy offsets | Identical validated and grounded parser output without moving parser-owned policy into core |
| `kogwistar-obsidian-sink` | Kogwistar-shaped entities, scoped snapshots, event envelopes, sequence/cursor fields | Incremental projection and full rebuild retain stable notes, relationships, links, deletion behavior, and drift results |

The consumer-owned boundary and event catalogs are normative Phase 0 inputs:

- `kogwistar-llm-wiki/doc/repo_boundary_and_contract_catalog.md`;
- `kogwistar-llm-wiki/doc/inter_repo_api_and_event_catalog.md`;
- `kogwistar-obsidian-sink/docs/obsidian_compatibility_contract.md`.

The compatibility manifest records the exact consumer commit or release used for
each catalog and suite. Their ownership direction remains unchanged: Kogwistar owns
graph truth, events, generic runtime behavior, and reusable mechanisms; product,
parser, and vault rendering policies remain in their owning repositories.

### Reference application and primary acceptance harness

`kogwistar-llm-wiki` is the primary real application for this port, not merely an
example consumer. In the coordinated developer workspace it is a sibling checkout
of this repository (commonly `../kogwistar-llm-wiki`; currently
`C:\Users\chanh\Documents\kogwistar-llm-wiki` on the reference Windows machine).
Automation must accept the application root as configuration and must not encode
that machine-specific absolute path.

The application repository itself contains nested Git repositories:

```text
kogwistar-llm-wiki/
  kogwistar/
  kg-doc-parser/
  kogwistar-obsidian-sink/
  src/kogwistar_llm_wiki/
  tests/
```

Its `pyproject.toml` resolves these as editable local packages through
`tool.uv.sources`. The nested `kogwistar/` checkout is a dependency pin/mirror for
the application; it is not a second source of truth and must not receive an
independent Rust port. The core candidate from this repository is built once and
injected into the coordinated test environment. Updating or syncing the application
pin is a release operation after that candidate passes.

Therefore a green application run is valid port evidence only when setup first
binds the app root and both nested consumer repositories to the exact core candidate
under test and records:

- resolved `kogwistar.__file__` and package version;
- Rust extension build/version and selected implementation mode;
- core, application, parser, and sink commit IDs;
- backend and storage root used by the run.

Tests must fail if they accidentally import the application's nested or stale
Kogwistar checkout instead of the candidate wheel/worktree. The existing import-path
smoke test is part of this gate and must be extended to assert candidate identity,
not merely that an import succeeds.

The application has competing pytest discovery declarations: `pyproject.toml`
names root `tests/`, while root `pytest.ini` currently names root tests plus
`kg-doc-parser/tests` but still omits the sink suite. CI must not depend on config
precedence or implicit recursion. Release rehearsals invoke four explicit suite
layers with one core candidate:

1. this repository's core contract and parity suites;
2. `kogwistar-llm-wiki/kg-doc-parser/tests` in the parser's own environment;
3. `kogwistar-llm-wiki/kogwistar-obsidian-sink/tests` in the sink's own environment;
4. `kogwistar-llm-wiki/tests`, including selected integration and long-run gates.

App-root success cannot substitute for an explicitly identified parser or sink job,
even when parser tests happen to be collected by the current root config. Every
layer records import provenance before executing behavior tests.

Current application source and tests also import below the curated top-level API,
including engine models/backends/jobs, scoped namespaces, runtime models/resolvers,
budgets/pricing/sinks, logical references, provenance, source-pointer utilities,
maintenance/wisdom helpers, and selected server services. Phase 0 must classify
each observed path as runtime, test-only, adapter-backed, or deprecated. For the
first Rust release, an unclassified internal import is a compatibility failure, not
permission to break the application silently.

## Decision Drivers

1. Preserve authority, transaction, replay, and ordering semantics.
2. Allow production rollback without reverse data migration.
3. Keep the public Python API usable while the core changes underneath it.
4. Remove duplicated sync/async orchestration and blocking adapters from the final
   core.
5. Gain Rust memory safety, explicit concurrency, predictable deployment, and
   performance where measurement supports it.
6. Avoid freezing feature development for a repository-wide rewrite.
7. Retain Python ecosystem access where dynamic code is an actual product feature.

## Decision

### 1. Use a strangler migration with capability ownership

Python and Rust may coexist, but each authoritative capability has one active
writer. A capability is a cohesive transaction or state-machine boundary, such as
event append, index-job claiming, graph mutation, or workflow execution. It is not
one method in `StorageBackend`.

Each capability progresses through these modes:

1. `python`: Python is authoritative; Rust is absent or tested offline.
2. `shadow`: Python is authoritative; Rust receives the same immutable input or
   reads the committed result, and its output is compared without authoritative
   writes.
3. `rust`: Rust is authoritative; Python uses the compatibility facade.

There will be no authoritative dual-write mode. Dual writes make rollback and
divergence ambiguous, especially for sequence allocation, leases, event logs, and
projections.

Cutover configuration must be explicit per coarse capability, for example:

```text
KOGWISTAR_IMPL_CONTRACTS=python|shadow|rust
KOGWISTAR_IMPL_META_STORE=python|shadow|rust
KOGWISTAR_IMPL_GRAPH_STORE=python|shadow|rust
KOGWISTAR_IMPL_RUNTIME=python|shadow|rust
KOGWISTAR_IMPL_SERVER=python|rust
```

The exact names may change during implementation, but method-level flags are not
allowed.

### 2. Target architecture

Add a Rust workspace under `rust/` with this intended crate split:

| Crate | Responsibility |
|---|---|
| `kogwistar-contracts` | Versioned DTOs, enums, IDs, canonical JSON, event envelopes, error codes |
| `kogwistar-domain` | Graph, ACL, provenance, workflow invariants, pure folds and reducers |
| `kogwistar-store` | Async storage traits and transaction-scoped capability interfaces |
| `kogwistar-store-memory` | Deterministic in-memory reference backend |
| `kogwistar-store-sqlite` | Local durable metadata and graph behavior where supported |
| `kogwistar-store-postgres` | PostgreSQL metadata, pgvector, queues, event log, and projections |
| `kogwistar-engine` | Graph use cases, indexing orchestration, recovery, and lifecycle |
| `kogwistar-runtime` | Tokio workflow state machine, scheduler, checkpoints, retry, and replay |
| `kogwistar-api` | REST, SSE, auth integration, MCP adapters, and CLI-facing application services |
| `kogwistar-python` | Thin PyO3/maturin compatibility extension used by the Python package |

The split is dependency-directed:

```text
contracts <- domain <- engine <- runtime <- api
                    ^       ^
                    |       |
                 store traits
                    ^
          memory / sqlite / postgres

Python SDK -> coarse PyO3 facade -> Rust application services
Python workers <-> versioned lane/task protocol <-> Rust runtime
```

Crate boundaries may be combined initially to avoid premature abstraction. The
dependency direction and ownership boundaries are mandatory; the exact crate count
is not.

Rust uses one async-first implementation based on Tokio. Sync Python entrypoints
block only in the outer compatibility facade. Rust core code must not maintain
separate sync and async business logic.

### 3. Preserve Python compatibility through a coarse in-process PyO3 facade

During migration, `import kogwistar` and the documented top-level API remain valid.
Pydantic models remain the Python-facing compatibility types until a separately
versioned API change.

The first Rust-backed release uses an in-process PyO3/maturin extension behind the
Python package. Satellite repositories do not import, discover, configure, or
serialize directly for Rust. An out-of-process Rust service is not part of the first
port; introducing one later requires a separate decision covering protocol version,
authentication, configuration, timeouts, cancellation, and offline development.

These existing modules and their used symbols remain import-compatible during the
compatibility window, even when their implementation delegates to Rust:

- `kogwistar.engine_core` and `kogwistar.engine_core.models`;
- `kogwistar.runtime`;
- `kogwistar.id_provider`;
- source-pointer and fuzzy-offset utility modules used by parser consumers.

This ADR does not schedule removal of the Python facade. It remains the supported
consumer API for the first Rust release and thereafter until a separate versioned
decision. The legacy pure-Python core fallback remains available through at least
one full release after Rust becomes default; only that fallback is eligible for
removal under Phase 6.

The native extension will expose coarse use cases rather than every backend method.
Requests and responses cross the boundary as versioned contract values, canonical
JSON bytes, and typed numeric buffers for embeddings. Python object identity,
arbitrary dictionaries, and PyO3 classes must not become a second domain model.

For the first release, model conversion is deliberately compatibility-first:

1. validate Python input through the existing Pydantic model or facade adapter;
2. dump the documented contract shape, including current defaults and view mode;
3. deserialize that shape into Rust contract types;
4. serialize Rust output back to the documented shape;
5. run output through the existing Pydantic `model_validate` path before returning
   it to consumer code.

`Node`, `Edge`, `Document`, `Grounding`, and `Span` provenance fields are mandatory
across this boundary. Source references, mention groundings, character offsets, and
spans must not be dropped, defaulted away, reordered when order is meaningful, or
reinterpreted as Rust implementation metadata.

Rules for the bridge:

- never call arbitrary Python callbacks while holding a database transaction;
- release the GIL for blocking or CPU-heavy Rust work;
- do not hold a Rust transaction open across an unconstrained Python callback;
- map stable Rust error codes to existing Python exception classes and HTTP/MCP
  error shapes;
- inject clock, UUID, and provider dependencies in tests;
- keep one Tokio runtime owner per process instead of creating a runtime per call.

Dynamic Python resolvers and provider integrations will move behind the existing
worker/lane model or a versioned task protocol. The Rust scheduler owns durable
state transitions; Python workers perform explicitly dispatched dynamic work.

### 4. Freeze compatibility contracts before porting behavior

The migration contract includes more than type names. It includes:

- public Python imports, call shapes, defaults, and exception behavior;
- `GraphKnowledgeEngine.read`/`write`, workspace, namespace, graph-space filtering,
  and close/cleanup behavior used by consumers;
- REST, SSE, MCP, and CLI request/response shapes and error status mapping;
- Pydantic JSON Schema and serialized null/default/unknown-field behavior;
- UUIDv5 namespace and input encoding in `stable_id`;
- canonical JSON, SHA-256 digests, float handling, and ordering rules;
- entity-event envelopes, operation names, namespace-local sequence allocation,
  cursor semantics, and replay ordering;
- current consumer CDC envelopes, including the sink's `entity.upsert` input and
  recognized delete/remove/tombstone forms, until a versioned adapter is released;
- append-only replacement and tombstone behavior; a Rust store must not turn a
  logical update or deletion into an unobservable in-place overwrite;
- database tables, column meanings, indexes, transaction boundaries, isolation,
  compare-and-swap, lease, retry, and idempotency behavior;
- vector normalization, distance ordering, tie-breaking, metadata filters, and
  result shapes;
- workflow reducer, fan-out/join, retry, checkpoint, cancellation, resume, and
  terminal-state semantics;
- authoritative versus projection ownership defined in existing ADRs.

Create a machine-readable contract manifest plus committed golden fixtures. Values
that are currently nondeterministic must be made injectable or explicitly
normalized; tests must not silently ignore them.

### 5. Preserve database and event compatibility

Rust reads and writes the existing database and event formats during migration.
There is no bulk data copy merely because implementation language changes.

Before Rust owns a store:

- capture current PostgreSQL and SQLite DDL as a reviewed baseline;
- introduce explicit, forward-only schema migrations and a schema-version table;
- nominate exactly one migration runner for a deployment;
- require additive or backward-readable migrations through the rollback window;
- test opening Python-created databases in Rust and Rust-created databases in
  Python;
- test replay in both directions from the same committed event corpus.

Existing Chroma, SQLite, PostgreSQL, and pgvector stores are production
compatibility surfaces. First Rust-backed startup must not automatically migrate,
re-index, compact, or otherwise mutate them. Any required data migration is an
explicit operator action with preflight validation, backup, resumable progress,
post-validation, and a tested reverse or rollback path.

Schema identifiers, namespace handling, timestamps, JSON encoding, and pgvector
dimensions require explicit tests. Rust must not infer a cleaner schema and change
wire meaning during the port.

For `engine_postgres_meta.py`, port responsibility groups into separate Rust modules
while preserving transactions where a use case spans groups:

```text
sequence.rs
event_log.rs
index_jobs.rs
lane_messages.rs
projections.rs
workflow_design.rs
run_registry.rs
```

The SQLite implementation follows the same contract modules. Shared SQL behavior
must be proven by tests, not assumed from similar function names.

### 6. Treat Chroma and Python integrations as compatibility adapters

The first Rust storage targets are memory, SQLite metadata, and PostgreSQL/pgvector.
The existing Chroma backend stays behind the Python adapter during early phases.
Its future is a separate evidence-based decision: retain the adapter, implement a
Rust client, or deprecate it with a migration path.

Chroma keeps its existing operating and transaction promises. The port must not
claim PostgreSQL-style atomicity for it. Python and Rust must never open the same
single-writer Chroma store concurrently during shadow or canary execution; use
read-only snapshots or isolated per-run roots.

Likewise, LLM SDKs, OCR, LangGraph conversion, and user-authored Python callbacks are
not prerequisites for moving durable engine authority to Rust. They communicate
through contracts and workers rather than being reimplemented merely for language
purity.

## Migration Plan

Phases are gated by evidence, not calendar dates. A later phase may start in a
non-overlapping capability while an earlier one is canarying, but ownership rules
still apply.

### Phase 0: Baseline and contract freeze

Deliverables:

- profile representative graph ingest, retrieval, event replay, queue claiming,
  and workflow fan-out/join workloads;
- record latency, throughput, CPU, peak RSS, database load, and artifact size;
- inventory supported public Python imports, REST/MCP routes, environment variables,
  database schema, and event versions;
- inventory imports, calls, models, view modes, events, and operational assumptions
  from all three consumer repositories and their tests;
- discover and pin the application's nested Git repositories, then run parser and
  sink suites as explicit jobs rather than relying on conflicting root discovery
  settings;
- configure `kogwistar-llm-wiki` as the primary acceptance checkout and make its
  import-path smoke test prove that the exact core candidate is loaded;
- commit JSON Schema, OpenAPI, event, replay, database, and error golden fixtures;
- commit fixtures for grounded parser output, scoped graph snapshots, update and
  tombstone event streams, and deterministic vault projection;
- add deterministic clock/ID hooks where comparison currently depends on time or
  UUIDv4;
- add Rust workspace policy: pinned stable toolchain, formatting, Clippy with
  warnings denied, dependency audit, license policy, and MSRV policy;
- add CI for Linux first and wheel smoke jobs for each platform actually published.

Exit gate:

- existing `ci` and `ci_full` suites pass unchanged;
- golden fixtures reproduce twice without unexplained drift;
- benchmark commands and datasets are repeatable;
- every authoritative surface has an owner and rollback path in the manifest;
- every consumer symbol is classified as preserved, adapter-backed, deprecated, or
  intentionally deferred under an approved semantic-version policy;
- the reference application records core/app/parser/sink identities in every parity
  artifact;
- all four suite layers prove they loaded the same core candidate.

### Phase 1: Contracts and deterministic domain logic

Port first:

- stable IDs and short-ID primitives;
- canonical serialization and provenance hashes;
- event envelopes and error taxonomy;
- metadata-filter evaluation and normalization;
- pure workflow lineage folds, projection reducers, and graph invariants that do not
  perform I/O.

Use Python as the oracle initially. Run table-driven differential tests, property
tests, and fuzz tests against both implementations.

Exit gate:

- byte-identical canonical output for the golden corpus;
- identical validation success/failure and stable error code for invalid inputs;
- property tests cover ordering, idempotency, replay, and round trips;
- the Python package can optionally use the Rust implementation without API changes.

### Phase 2: In-memory backend and read-only engine slices

Implement the Rust in-memory backend to exercise storage traits cheaply. Then port
read-only graph retrieval and projection/replay inspection as complete use cases.

Run Rust in shadow mode against committed Python state. Compare normalized results,
including ordering and vector-distance tie behavior.

Exit gate:

- the backend contract passes against Rust memory storage;
- Python-versus-Rust differential scenarios pass for graph reads and replay;
- no shadow code writes authoritative state;
- benchmark gate is met or a measured exception is recorded.

Phase 2 evidence: `contracts/benchmarks/rust-memory-phase2-windows.json` records
fresh-process Python direct reads against the current Rust isolated-snapshot ABI
for graph list, cosine vector query, replay, and named-projection list. Its gate
is factual: a failed threshold is recorded with measured ratios because this
read-only shadow-safety inspection rebuilds an isolated store and is neither a
cutover nor a default request path. The repeat command is recorded in
`contracts/rust-port-v1.json` under `phase_2_evidence`; projection and benchmark
evidence do not declare the full ADR complete.

### Phase 3: Durable stores and graph mutation

Port in this order:

1. SQLite metadata contract;
2. PostgreSQL sequence and entity-event append/replay;
3. named projections, snapshots, deltas, and run registry;
4. index-job and lane-message claim/lease/retry state machines;
5. graph mutation and pgvector inside the existing unit-of-work boundary;
6. recovery, reindex, and rebuild operations.

Each item is a capability cutover. Queue and event writers require exclusive
ownership; their shadow checks inspect committed rows or execute against isolated
cloned databases.

Exit gate:

- fake/memory, SQLite, and PostgreSQL contract suites pass;
- forced rollback leaves no partial graph, event, vector, or queue state;
- concurrent claim, lease expiry, compare-and-swap, and sequence tests pass under
  stress;
- Python-created and Rust-created stores are mutually readable during the rollback
  window;
- replay and projection rebuild produce the same canonical state;
- update and tombstone replay preserve consumer-visible deletion and replacement
  semantics;
- a canary can revert to Python without data conversion, duplicate events, or
  corrupt projections.

### Phase 4: Workflow runtime and workers

Port the durable execution state machine before porting dynamic resolver code:

- routing, fan-out, joins, nested invocation, retry, and terminal transitions;
- checkpoint, pause/resume, cancellation, replay, and cost/trace events;
- scheduler resource accounting and durable lane dispatch;
- worker request/result envelopes for Python providers and user code.

Shadow execution must use recorded provider/tool results. Live side effects must not
run twice. Only the authoritative runtime may dispatch tools, send messages, charge
budgets, or advance durable checkpoints.

Exit gate:

- existing sync, async, bridge-parity, suspend/resume, and terminal-state suites pass;
- recorded workflows produce identical state, event order, and terminal result;
- crash/restart tests pass at every durable transition;
- cancellation and backpressure behavior meet recorded service objectives;
- Python workers interoperate with the Rust runtime across process restart.

### Phase 5: Server, MCP, auth, and CLI

Move REST and SSE application services to Rust, then MCP and CLI entrypoints. Keep
the Python server as a rollback deployment during the compatibility window.

The Rust server must call the same application capabilities as the Python facade;
it must not create a second set of business rules in route handlers.

Exit gate:

- route, auth, capability, error, SSE resume, and MCP conformance suites pass;
- OpenAPI and MCP tool schemas have no unapproved diff;
- rolling upgrade and mixed-version database compatibility tests pass;
- operational dashboards expose implementation mode, parity mismatches, queue lag,
  replay lag, and schema version.

### Phase 6: Default switch and convergence

Make Rust the default engine, runtime, and server only after canary progression and
one full compatibility release. The Python package becomes a thin SDK, native
extension loader, models layer, and optional worker toolkit.

Delete a Python implementation only when:

- Rust has been default for one full release;
- no unresolved parity mismatch remains;
- rollback no longer depends on that implementation;
- replacement docs, diagnostics, and ownership exist;
- removal is announced as a separate compatibility decision when public imports are
  affected.

Final completion criteria:

- default deployment executes core graph, storage, runtime, and server behavior in
  Rust;
- Python API compatibility tests run against the Rust core;
- no default request path loads the legacy Python core;
- authoritative stores and event history survive upgrade and rollback tests;
- all three consumer suites and the cross-repository end-to-end gate pass against
  the Rust-backed Python facade;
- retained Python code is intentionally classified as SDK, bridge, integration, or
  worker code.

## Verification and Rollout Policy

### Differential testing

Every migrated capability gets a runner that can execute the same scenario against
Python and Rust. Comparison covers output plus durable effects:

- returned values and errors;
- rows, events, projections, and checkpoints;
- ordering and sequence allocation;
- idempotency after retry;
- rollback after injected failure;
- grounded parser artifacts and provenance/offset retention;
- workspace, namespace, and graph-space isolation;
- create, update, tombstone, CDC cursor, and replay behavior;
- incremental Obsidian projection versus deterministic full rebuild.

### Performance gate

Phase 0 records workload-specific objectives. Until more specific objectives are
approved, a cutover candidate must have:

- p95 latency no worse than 110% of the Python baseline;
- throughput at least 95% of the Python baseline;
- peak RSS no worse than 110% of the Python baseline;
- no material increase in database queries, lock time, or write amplification.

A phase may proceed for a strong maintenance or safety benefit despite one failed
performance threshold only through a recorded exception with evidence.

### Production rollout

Use capability-level canaries: internal/test, 1%, 10%, 50%, then 100%. Promotion
requires a defined observation window with zero unexplained correctness mismatch.
Rollback changes only the capability owner flag while schemas remain backward
readable.

Begin with new disposable workspaces. Existing data is shadowed read-only first.
Persistent canaries use explicit roots, capture graph counts and IDs, event
sequences, checkpoint/recovery status, projection manifests, vault hashes, and
error categories, and must exercise rollback before promotion.

Parity mismatch telemetry must avoid sensitive payloads. Record contract version,
capability, stable input digest, mismatch category, and trace correlation ID.

### Consumer release gate

Rust cannot become the default until all of these pass in a clean coordinated
environment:

- existing satellite imports without source changes;
- the reference application's import provenance proves it loaded the Rust-backed
  core candidate rather than a nested or stale editable checkout;
- core, nested parser, nested sink, and app-root suites run as separate jobs against
  that same candidate;
- contract fixtures against both Python and Rust-backed facades;
- `kogwistar-llm-wiki` ingest, workspace isolation, long-run/recovery, promotion,
  lane/queue orchestration, budget accounting, maintenance, and
  projection-consistency suites, including its `integration` and applicable
  `longrun` gates;
- `kg-doc-parser` workflow, grounding, source-pointer, fuzzy-offset, and retry
  suites;
- `kogwistar-obsidian-sink` event-consumer, deterministic projection, round-trip,
  drift, and full-rebuild suites;
- one cross-repository flow: source -> parser -> graph ingest -> event/projection ->
  Obsidian vault -> authoritative replay/full rebuild;
- that flow includes create, update, and tombstone cases;
- an interrupted checkpointed workflow resumes after process restart;
- any changed store format completes explicit migration and rollback rehearsal;
- rollback to Python leaves canary data intact without duplicated events or
  projection corruption.

Release notes must state core version, facade/contract version, supported backends,
storage migration status, and exact consumer versions tested together.

### Test ownership

- Kogwistar Rust-port owner: facade, model conversion, backend/runtime parity,
  storage migration/rollback, and core contract fixtures.
- Each consumer owner: product-specific acceptance fixture and expected output.
- Coordinated release owner: pins compatible commits/releases and blocks default
  promotion unless the cross-repository gate passes.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Python/Pydantic coercion differs from Serde | Golden invalid-input corpus; explicit adapters; stable error codes |
| Provenance or offsets disappear during conversion | Required-field round trips and parser-to-vault end-to-end fixtures |
| Dict/JSON ordering or float behavior changes hashes | Canonical serializer specification and byte-level fixtures |
| SQLx behavior differs from SQLAlchemy/sqlite3 | Cross-language database fixtures, transaction fault injection, testcontainers |
| Queue leases or sequences race differently | Single writer, stress tests, database-native locking, deterministic state-machine tests |
| GIL/Tokio re-entry deadlocks | Coarse facade, one runtime owner, release GIL, no callback inside transaction |
| Python resolver ecosystem blocks runtime migration | Versioned worker protocol and lane dispatch |
| Native wheels complicate installation | Explicit platform matrix, maturin builds, import smoke tests, transition fallback |
| Two implementations slow feature work | Port by capability, shared contract fixtures, delete legacy only after rollback window |
| Rust MCP/LLM libraries lack needed behavior | Keep protocol adapters and Python workers until conformance is demonstrated |
| Big schema cleanup becomes coupled to port | Require separate ADR and migration for semantic/schema changes |
| Core-only parity misses consumer breakage | Pin and run all three consumer suites plus cross-repository create/update/tombstone flow |

## Consequences

### Positive

- Migration is reversible and observable.
- Existing event history and databases remain useful.
- Rust receives clear ownership boundaries instead of inheriting Python module shape.
- Python ecosystem integrations remain available without controlling durable state.
- Contract fixtures become durable architecture documentation.
- Async execution, cancellation, and concurrency have one core implementation.

### Negative

- Python and Rust implementations coexist for multiple releases.
- CI, packaging, and release engineering become more complex before they simplify.
- Compatibility constraints delay desirable schema and API cleanup.
- PyO3 and worker protocols introduce explicit boundary code.
- Differential testing and canary telemetry require substantial initial work.

## Alternatives Considered

### Big-bang rewrite

Rejected. The authority, replay, backend, runtime, and public API surface is too large
to validate only at the end, and rollback would require switching both code and data.

### Translate Python files directly to Rust

Rejected. Current file boundaries, especially the PostgreSQL metadata store and
runtime, combine several capability and transaction concerns. Preserving file shape
would preserve accidental coupling.

### PyO3 extensions only for hot loops

Rejected as the end state. This can help deterministic primitives during migration
but does not remove duplicated orchestration, storage ownership, or runtime
complexity.

### Split every subsystem into a network service first

Rejected. It would add distributed transactions, deployment, latency, and failure
modes before contract parity exists. Process boundaries are reserved for dynamic
workers and independently operated components.

### Keep the core in Python and optimize selectively

Not selected as the target, but Phase 0 may show that some proposed Rust work has no
benefit. Such capabilities may remain Python adapters without blocking migration of
durable core ownership.

## Non-Goals

- Redesign graph, workflow, ACL, or provenance semantics.
- Change which event log or store is authoritative.
- Add cross-store ACID, sharding, or a new consistency model.
- Remove the Python package or Python worker support.
- Rewrite `kogwistar-llm-wiki`, `kg-doc-parser`, or `kogwistar-obsidian-sink` in
  Rust.
- Move product promotion/lane/maintenance policy, parser source-map policy, or
  Obsidian rendering/path policy into Kogwistar core.
- Reimplement every optional provider SDK in Rust.
- Replace Chroma without a separate compatibility and data-migration decision.
- Promise performance gains before benchmarks exist.

## Follow-up Decisions Required

Before the relevant phase, record or amend decisions for:

- exact supported wheel and binary platform matrix;
- contract versioning and deprecation duration;
- production feature-flag and canary control plane;
- Chroma retention, Rust client, or deprecation;
- MCP Rust implementation versus a retained compatibility adapter;
- post-migration schema cleanup after the rollback window closes.

## History

- 2026-07-14: Proposed incremental Rust port strategy.
- 2026-07-14: Added consumer compatibility, cross-repository release gates, and
  explicit first-release facade/storage safeguards.
- 2026-07-14: Recorded the reference application's nested repository topology and
  required separate core, parser, sink, and app-root suites against one candidate.
