# ADR-015 implementation status

Updated: 2026-07-20

ADR-015 is not complete as a production migration. Local implementation and
compatibility work is substantially complete, but authoritative ownership and
production rollout remain intentionally fail-closed.

## Scope-locked completion ledger

This ledger replaces percentage-complete estimates. Percentages mixed local
implementation effort with production rollout gates and therefore overstated
completion. No new capability may be added to this ledger. A defect may enter
the active slice only when an existing item below exposes it; unrelated ideas
are deferred to a later ADR/release.

### Test execution acceleration

- [x] Persist feature/regression/milestone profiles and exclude `slow`, manual,
  real-provider, Ollama, and parser-repository model-selection tests as documented.
- [x] Persist candidate build, identity, bootstrap, Dockerfiles, container worker,
  timing comparison, and report merge scripts; no executable temp script remains.
- [x] Provide opt-in pytest-xdist and measure it without making it the default.
- [x] Provide three ordinary-container LPT shards with private writable roots,
  exact group coverage, one candidate identity, cleanup, and resumable reports.
- [x] Record controlled timings and the decision to prefer container sharding.

The test-harness work is complete. Further test-runner optimization is outside
ADR-015 unless a listed release gate cannot be executed with this harness.

### Capability authority ledger

The authoritative machine-readable source is `contracts/rust-port-v1.json`.

- [x] `deterministic-contracts`: Rust cutover ready.
- [x] `sqlite-meta`: Rust cutover ready, rollback-readable by Python.
- [ ] `postgres-sequence-event-log`: readiness remains false.
- [ ] `projections-snapshots-run-registry`: readiness remains false.
- [ ] `queues-leases-lanes`: readiness remains false.
- [ ] `graph-pgvector`: readiness remains false.
- [ ] `workflow-runtime`: readiness remains false.
- [ ] `server-rest-sse-mcp-cli`: readiness remains false.

`chroma-adapter` is intentionally Python-owned and is not a Rust-port blocker.
LLM SDKs, OCR, LangGraph conversion, and user-authored Python callback bodies
remain integrations/workers as allowed by ADR-015; they are not to be rewritten
for language purity.

### Closed local implementation and evidence backlog

Only these local items remain in scope before production rollout:

- [x] Phase 3: run fake/memory, SQLite, and PostgreSQL contract gates from the
  current candidate; prove fault rollback, lease/CAS/sequence concurrency,
  Python/Rust mutual readability, rebuild equality, and update/tombstone replay.
  `scripts/adr015_phase3_store_gate.py` produced
  `.codex/adr015-phase3-store-gate.json` for source
  `6e2dfbd8c855d83667a1840a73a959604b0946167a2083a8ca8b4419f107fba3`:
  memory 17, SQLite 27, and live pgvector PostgreSQL 36 all passed with zero
  failures, errors, or skips. The gate rejects a skipped PostgreSQL group.
- [ ] Phase 3: promote each of the four pending durable-store capabilities only
  when its capability-specific evidence and rollback evidence pass. Do not flip
  one broad store flag from a narrow test.
- [x] Phase 4: finish the already-bounded recorded runtime cutover. Existing
  reducer, frontier, serving projection, indexed dedup, frozen route validation,
  lane transaction, and Python worker adapter stay within this item. Current
  native evidence is `.codex/adr015-phase4-runtime-gate.json` for source
  `6e2dfbd8c855d83667a1840a73a959604b0946167a2083a8ca8b4419f107fba3`.
- [x] Phase 4: pass existing sync, async, bridge-parity, suspend/resume, and
  terminal suites; recorded state/event/terminal-result parity; crash/restart;
  cancellation/backpressure; and Python-worker restart gates. The persisted
  gate passed durable-worker 55, sync/async bijection 64, bridge parity 17,
  suspend/terminal 23, and live PostgreSQL 10 tests, all with zero failures,
  errors, or skips. Rust async authority remains deliberately fail-closed under
  worker contract v1; this is tested rather than silently delegated.
- [ ] Phase 4: route the public runtime owner through the proven Rust durable
  scheduler while keeping dynamic Python callbacks behind the versioned worker
  boundary. Unsupported nested/sandbox/async/side-effect protocol features stay
  fail-closed; no new runtime-state operation may be invented. Local routing now
  fails closed when `KOGWISTAR_IMPL_RUNTIME=rust` lacks an authority URL (it no
  longer silently falls back to Python), and Rust worker contract v1 explicitly
  rejects async callbacks. The current native boundary subset passed
  durable-worker 57 plus sync/async bijection 64. This line remains unchecked:
  public authority promotion still requires its capability canary and the
  explicit `workflow-runtime` readiness decision.
- [x] Phase 5: explicitly version/defer the 16 currently frozen Python-owned
  server operations listed by `PENDING_SERVER_CUTOVER_ROUTES`. The manifest
  records grouped Python rollback ownership, prerequisite Rust authorities, and
  exit evidence; its exact operation set is bidirectionally checked against the
  Rust fail-closed inventory.
- [x] Phase 5: explicitly version/defer the five conversation/tool syscall
  sub-operations listed by `PENDING_SYSCALL_CUTOVER_OPS`. The same versioned
  manifest entry records Python rollback ownership, the required worker/runtime
  protocol authority, and required exit evidence.
- [ ] Phase 5: pass route/auth/capability/error/SSE-resume/MCP conformance,
  frozen OpenAPI/MCP no-drift, mixed-version/rolling-upgrade, and operational
  implementation/parity/queue/replay/schema telemetry gates.
- [x] Build one fresh candidate after all local code changes and run all four
  compatibility layers (core, parser, sink, application) under one verified
  identity. Report `adr015-milestone-uv-final.json` passed on 2026-07-20 under
  identity `fe22d92e51a9e023a742cd85ec088eed1c74254397ddf7c17c4cb37779e71b1b`.
- [x] Re-run the frozen Phase 3/4 candidate through the same four Linux layers.
  `.codex/adr015-phase34-milestone-container.json` passed on 2026-07-20 with
  wheel SHA-256 `ec08c4484e22fbda209b59945b99e24099e123139815065814293b78d9e4d071`,
  source digest `3710bb6b960df4330f41289695940552119d7dbb21a9c1a20b1028b71af612d1`,
  three ordinary Docker shards, zero xdist workers, exact layer coverage, and
  1010.21 seconds orchestration wall time. PostgreSQL fixture skips inside
  isolated workers remain intentionally non-evidence; the no-skip live
  PostgreSQL Phase 3/4 gates remain the authoritative proof for that backend.
- [x] Re-run the release-readiness report. Current candidate has no
  compatibility or recorded performance blocker; the report correctly remains
  blocked only by six explicitly unready authority capabilities and absent
  production canary evidence.

Fields such as continuation provenance are not independent scope additions.
They are changed only if an existing Phase 4 parity/interoperability gate proves
a concrete mismatch. Performance improvements beyond recorded thresholds and
test-framework changes beyond the completed harness are deferred.

### External rollout and elapsed-release gates

These cannot be satisfied by more local unit-test implementation:

- [ ] Run production `internal/test`, `1%`, `10%`, `50%`, and `100%` capability
  canaries with real observation windows, zero unexplained mismatch, and tested
  rollback. The committed rehearsal proves shape only and cannot promote ownership.
- [ ] Make the ready Rust capabilities default only after production evidence.
- [ ] Complete one full compatibility release with Python rollback retained.
- [ ] Then satisfy Phase 6: default graph/storage/runtime/server behavior in
  Rust, Python API tests against Rust, no default legacy-core request path,
  upgrade/rollback survival, all consumer/E2E gates, and classification of all
  retained Python as SDK, bridge, integration, or worker code.

Current objective facts: 2 of 8 Rust-target capability flags are ready; 6 are
not ready; production canary stages completed are 0 of 5; the required full
compatibility release has not elapsed. These counts, not a floating percentage,
govern ADR completion.

### Current candidate estimate and validation

As of 2026-07-20, the rough effort estimate is **58% complete**. This is a
communication estimate only: the scope-locked ledger and the authority/canary
counts above remain the completion criteria.

The current candidate's fresh Linux wheel was built with the persisted `uv`
builder and passed all three isolated Docker shards. The current milestone
report is `.codex/adr015-milestone-current.json`: candidate identity
`7b4197b594bff5b8787b5d4246f51fb0d3d12b0ba4a3e6aa173c1f90b17a210f`, source
digest `9fddf11185759d56656da9343fed5b14e06fbb1e4c8400af9789b3f8d230cfa6`, and
wheel SHA-256 `b920206ff6c7c292283b4e853e342c4e94a0892d19bb767761aceaf33ae9e75f`.
It passed core, parser, sink, and application with three ordinary containers
and zero pytest-xdist workers. The full compatibility evidence has no code/test
blocker. PostgreSQL live checks remain a separate capability-specific
requirement: the Docker compatibility workers deliberately do not reach the
host Docker daemon, so PostgreSQL fixtures correctly skip there rather than
creating a false environmental pass. Neither result advances a capability flag
until its listed capability-specific evidence and production canary pass.

The later metadata-only lifecycle candidate also passed the full current-source
host selector (`505 passed, 4 skipped, 1182 deselected`) and all four isolated
Linux layers. Its report is `.codex/adr015-lifecycle-milestone-current.json`:
candidate identity `fc17840955c4f2261b9f878890c41815f9d06985fec8df1c42f8219c9706e095`,
source digest `c2d93e57d1cfd0ba31d95e0aaf3abe6c41bb022d19758fa90a4f9edc8f6badce`,
wheel SHA-256 `9dc4bca9cdbc9638d8a1c7978c9978903890032b42039c65492fd1da76215809`.
The milestone used three ordinary containers, zero xdist workers, and completed
in 1075.24 seconds. It proves the lifecycle contract on the candidate; it does
not alter any false readiness flag or substitute for production canaries.

On 2026-07-20, a disposable host `pgvector/pgvector:pg16` instance supplied
the previously unavailable PostgreSQL live evidence. Rust/Python event-log
mutual readability, sequence/idempotency collision, native transaction,
queue/lease, graph rollback, runtime checkpoint/restart/worker reclaim, and
pgvector retrieval suites passed 56 tests total (with one intentional Chroma
rollback skip). This closes the local live-PostgreSQL execution gap; it does
not authorize a broad authority-flag promotion or replace production canaries.

## Completed local gates

- Python-owned Chroma lifecycle projections have an explicit metadata-only
  serving-write contract: tombstone, redirect, and effective-time patches read
  only metadata and call `update(ids=..., metadatas=...)`. They never resend a
  document or omit an embedding while doing so, because Chroma would silently
  invoke its embedding function and replace/recompute a vector. The executable
  fake-and-real-Chroma regression keeps the prior vector and asserts zero
  lifecycle embedding calls. This is adapter correctness/performance hygiene,
  not a new Rust authority capability; full rationale is Case 12 in
  `ADR-015-nondeterministic-test-failures.md`.
- Rust contract, memory, SQLite, PostgreSQL, pgvector, queue/lease/lane,
  run-registry, recorded-runtime, API, Python-worker, and wheel slices exist.
- Runtime state serving uses a disposable full-state projection. Missing,
  stale, or corrupt projections fall back to authoritative event replay.
- Recorded transition idempotency uses indexed `transition_id` lookup in both
  SQLite and PostgreSQL; worker retry uses the same indexed path.
- SQLite additive migration opens older Python queue schemas.
- Runtime event append, current-state/checkpoint/status projections, lane ack,
  and dedup result remain transactionally atomic.
- Cross-repository deterministic acceptance covers source, fake parser, app
  graph promotion, Rust event append/reducer, incremental Obsidian update and
  tombstone, and byte-equivalent full rebuild.
- Python-created stores remain readable after Rust writes. The rollback
  rehearsal deletes Rust current-state projection, recovers from events,
  retries without duplication, and resumes contiguous sequencing.
- Real LLM/OCR provider tests are manual and excluded from ADR CI. Long tests
  with equivalent short semantic coverage are marked slow.
- Clean native wheel import and the four compatibility layers have passed.
- `KOGWISTAR_IMPL_META_STORE=rust` now routes the public SQLite meta facade to
  one persistent Rust writer. Native transaction tokens keep nested operations
  in one atomic unit of work; raw Python writer access fails closed. Python
  rollback reads the same database. A clean Linux wheel core feature run passed
  723 tests with 8 environment-dependent skips.
- `scripts/rust_port_container_compat.py` reproduces clean Linux compatibility
  with a patch-pinned Python image, wheel digest, separate core/consumer venvs,
  full dependency fingerprints, resumable reports, and guaranteed native-bridge
  cleanup. This isolates the parser's dependency set from core server pins.
- The final clean-Linux feature report `.codex/rust-port-container-final.json`
  passed core, parser, sink, and application under one candidate identity
  `d6def4abd1bf481ae7c9ff13858cca374b25d17e67315430b71273380e259ea1`.
  Core, parser, and sink each passed as one process; all 13 isolated application
  groups passed. Measured layer times were 751.39s, 1132.43s, 19.74s, and
  2520.79s respectively.
- Fresh three-shard Linux milestone evidence now passes all four layers under
  one wheel/source identity: `.codex/adr015-milestone-uv-final.json`, candidate
  identity `fe22d92e51a9e023a742cd85ec088eed1c74254397ddf7c17c4cb37779e71b1b`,
  source digest `1081722c404271276dff52d9f93037fde6fc8161428fdaa19b802fd1688f6a85`,
  wheel SHA-256 `985350eb63ba80b205b3c0f6f8cd4842aadb5eb65467a67eee756f22d72e617e`.
  The persisted container source stage now includes the root MCP entrypoint;
  MCP E2E validates wire JSON without opening a competing Chroma engine in the
  parent test process. The builder also installs Maturin through pinned `uv`.
  Final orchestration wall time was 1265.21 seconds with xdist still disabled.
- `.codex/rust-port-release-final-current.json` confirms current compatibility
  and recorded performance gates pass. Its sole blockers are the six explicit
  `rust_cutover_ready: false` capabilities and rehearsal-only (not production)
  canary evidence.
- Sync PostgreSQL now has a native cross-call transaction owner. Coordinated
  `KOGWISTAR_IMPL_META_STORE=rust` plus `KOGWISTAR_IMPL_GRAPH_STORE=rust`
  routes node, edge, document, and domain ADD; existing-entity REPLACE;
  lifecycle patches; edge hard-delete; and document rollback delete through
  one native unit of work with event history and derived-index job admission.
  Partial sync PostgreSQL Rust selection and all async PostgreSQL Rust
  selection fail closed instead of silently retaining a Python writer.
- The focused clean-Linux PostgreSQL suite proves commit/rollback, stale-token
  and second-writer rejection, schema isolation, Python mutual readability,
  legacy default-scope claiming, ambiguous custom-scope rejection, full-row
  replacement with embedding preservation, hard-delete idempotency, and public
  write rollback. Pure node/edge writes now use the same native event transaction
  while preserving their no-fanout and non-idempotent-new-ID contracts. Partial
  reference pruning and extraction rollback use native REPLACE/DELETE, preserve
  embeddings, fall back to base authority when derived rows have not converged,
  and propagate native writer failures instead of reporting false success. The
  latest clean-Linux PostgreSQL transaction suite passed all 23 tests.
- Current Rust workspace gates pass on Linux: formatting, all-target check, 82
  executed workspace tests with one explicit manual scale probe ignored, and
  Clippy with warnings denied. Focused Python/Chroma/PostgreSQL writer and rollback
  coverage passed; two Chroma rollback cases first hit an HNSW segment-reader
  transient and then passed unchanged, recorded as Case 10 in the nondeterminism
  log. That rerun is diagnostic only and is not fresh release evidence.
- Rust server now serves `GET /api/workflow/services/{service_id}/events`
  through REST and MCP by folding authoritative workflow entity events rather
  than trusting disposable graph projections. SQLite tests cover replacement,
  deletion, filtering, limits, missing services, and capability denial;
  PostgreSQL live coverage is compiled and runs when
  `KOGWISTAR_TEST_PG_DSN` is available. The API crate currently passes 26
  tests and Clippy with warnings denied on Linux. Focused Python contract,
  frozen OpenAPI/MCP, health, and capability tests pass 88/88.
- Rust server also owns per-service and bulk service-projection repair over
  REST and MCP. Repair folds authoritative service-definition/event nodes and
  atomically replaces disposable `service_registry` projections in SQLite or
  PostgreSQL. Atomic enable/disable writes a new authoritative definition and
  lifecycle event before replacing the projection. Heartbeat updates lock and
  read the current projection in the same unit of work as health/error events
  and projection replacement. Service reads preserve Python's combined
  `service.inspect + project_view` requirement. Service declaration now writes
  definition, initial lifecycle/health events, and registry projection in one
  transaction. Service trigger now validates Python's trigger vocabulary,
  honors disabled/spec cooldown/debounce suppression, freezes the exact graph
  plan, and atomically creates the run/lane plus authoritative lifecycle events
  and projection. Active-child retries are idempotent; PostgreSQL locks the
  service projection against concurrent duplicate spawn. REST and MCP share
  this path. Rust syscall dispatch now owns spawn/terminate process,
  checkpoint, resume, and approval branches over existing authoritative run
  and capability services. Conversation/tool-dependent syscall branches remain
  explicitly fail-closed in `PENDING_SYSCALL_CUTOVER_OPS`. The
  graph-validation route now validates frozen Node/Edge provenance shape and
  preserves Python's metadata-pointer lift without touching storage. Three
  visualization HTML shells are embedded from the existing packaged templates;
  unresolved Jinja drift fails closed. The data-bearing D3 bundle remains
  Python-owned until graph reads cut over. API coverage is 30/30; 16
  frozen server operations remain Python-owned.
- Focused execution exposed and fixed a pre-existing capability regression:
  capability snapshots again require both `project_view` and
  `read_security_scope`. The capability contract file now imports its fixture
  explicitly, so IDE and isolated-file pytest execution work without relying
  on collection order.
- The bounded public sync-runtime authority now freezes the validated workflow,
  submits caller run identity, claims only that run, executes dynamic callbacks
  through the Python worker, and returns Rust-persisted terminal state. Exact
  admission retry after response loss, cancellation propagation, a fresh facade
  continuing durable work after scheduler restart, and process-local dependency
  restoration have focused executable coverage.
- Public sync resume now validates the exact parked token and maps the external
  `RunSuccess`/`RunFailure` through the existing `u`/`a`/`e` reducer and routing
  semantics into a data-only `resume_effect`. SQLite and PostgreSQL enqueue the
  same effect shape; the worker consumes it without re-running the suspended
  callback. The resume contract exposes the run's frozen routes and node ops;
  current-workflow drift and reserved runtime-state injection fail before a
  resume mutates durable state. Current focused authority/worker coverage passes
  29/29, sync and
  async bijection pass 32/32 each, bridge parity passes 17/17, and focused mypy
  reports no issues. Runtime readiness remains false until native candidate and
  full Phase 4 exit evidence pass.
- Current-source native Windows gates now pass after loading the MSVC/Windows SDK
  environment: the Rust workspace executes 96 tests with one explicit manual
  scale probe ignored, the API crate passes 33/33, and workspace Clippy passes
  with warnings denied. These are local current-source results, not the required
  fresh Linux candidate or PostgreSQL-live evidence.
- A Python-generated runtime wire golden now round-trips through strict Rust
  DTOs for submit, claim, SQLite/PostgreSQL claimed work, worker effect, legacy
  transition result, and resume. SQLite and PostgreSQL share request DTOs,
  claimed-work serialization, and one start-transition builder. This gate
  exposed and fixed a missing resume `turn_node_id`, a `None` versus empty-string
  start identity drift, PostgreSQL's continuation sequence default-to-zero, and
  an effect-only PostgreSQL result request shape. The existing committed-golden
  drift test regenerates this fixture from Python and fails on missing, renamed,
  null/default, or unknown wire fields.
- Runtime continuation lanes now use one strict `RuntimeStepExecutePayload`
  builder in SQLite, PostgreSQL, and both API schedulers. The payload carries
  `turn_node_id` from recorded runtime state, including resume lanes. Current
  Python authority/worker/golden coverage passes 31/31; focused native wire,
  lane, and projection-contract tests pass; workspace Clippy remains warning-free.
  Serving projection namespace/cursor/schema metadata also comes from one shared
  store-contract helper, preventing backend-local projection field drift.
- Current-source Phase 4 runtime evidence is green: sync/async bijection
  68/68, bridge parity 17/17, non-slow suspend/terminal coverage 23/23, and
  SQLite/PostgreSQL checkpoint, fault rollback, restart, claimed-worker, and
  handoff coverage 37/37 (the PostgreSQL cases used a live pgvector container).
  A current native extension was rebuilt before the latter gate; an initial
  34-failure run was diagnosed as the stale ignored local `_rust.pyd` rejecting
  the newer `transaction_id` ABI field, not a runtime semantic regression.
  Rebuilt current Rust server then passed the real TCP SQLite authority E2E.
- Host-native focused verification now has a persisted exact-ABI build and smoke
  path. It builds a wheel for the selected Python interpreter, atomically
  refreshes the ignored source-tree extension, records source/wheel/extension
  digests, and requires the current `transaction_id` store ABI before tests.
  This turns stale local native binaries into an immediate provenance failure
  rather than a misleading durable-runtime failure cascade.

## Performance evidence

`contracts/benchmarks/rust-runtime-serving-current-windows.json` verifies the
specific runtime scan fix. Increasing unrelated recorded history from 100 to
10,000 rows (100x) changed p95 exact-retry lookup by 1.21x and warm current-state
read by 1.14x. The scale gate passes.

`contracts/benchmarks/rust-persistent-sqlite-current-linux.json` passes the
generic persistent-store gate after a `SqliteStore` handle began retaining one
serialized connection instead of reopening and tearing down WAL state per
operation. Across event append/replay and projection replace/list, Rust p95
latency is 0.18-0.69x Python, throughput is 1.54-6.04x Python, peak RSS is
1.016x Python, and the canonical state digest is equal. Public routing,
single-writer enforcement, rollback readability, and clean-wheel compatibility
now unlock the `sqlite-meta` capability only.

## Intentionally not ready

- Public PostgreSQL meta/event-log, projection, run-registry, and queue facades
  are implemented behind the coordinated sync PostgreSQL selector. Fresh
  current-candidate four-layer evidence now passes, but the capability readiness
  flags remain false until capability-specific production canaries exist.
  `KOGWISTAR_IMPL_META_STORE=rust`
  independently promotes only the ready `sqlite-meta` capability on SQLite.
- Sync PostgreSQL base graph mutation, replay, repair, and rollback writers are
  now routed through the coordinated native transaction owner. Derived
  node-ref/endpoint tables remain disposable Python adapters driven by native
  index jobs. `graph-pgvector` remains not ready pending capability-specific
  production canaries. Chroma remains a Python single-writer adapter.
- Coordinated sync PostgreSQL now routes exact singleton base node/edge/document reads
  through the native projection ABI. Rust projection reads accept legacy Python
  default-scope rows lacking scope metadata. Single-vector node/edge queries
  with exact-equality metadata filters also use the native projection ABI;
  multi-row gets, complex-filter queries, and derived-index reads remain Python
  adapters pending ordering and filter contract cutover.
- Rust runtime owns recorded durable reducer/scheduler slices, not the full
  public `WorkflowRuntime` resolver and side-effect lifecycle.
  `ADR-015-runtime-semantics-contract.md` now freezes the bounded Python
  routing/failure/join/worker behavior required for that cutover and names its
  executable evidence; implementation and fresh four-layer proof remain pending.
- Rust server binary and application services exist, but server and runtime
  cutover readiness remain false.
- The 16 frozen server routes and five frozen conversation/tool syscalls are
  now versioned Python-roll-back deferrals in `server_operation_deferrals` in
  `contracts/rust-port-v1.json`. Each records a prerequisite authority and
  capability-specific exit evidence; a contract test requires an exact,
  duplicate-free match with Rust's direct fail-closed inventories. This closes
  their classification task only, not their Rust authority or Phase 5
  conformance gates.
- `index_applied_state` has native SQLite schema/API parity; graph and derived
  index authority nevertheless remain Python-owned until graph-store cutover.
- Production internal/test, 1%, 10%, 50%, and 100% canaries have not occurred.

Every capability now requires explicit `rust_cutover_ready: true`; omitted
readiness fails closed. `scripts/rust_port_readiness.py` validates canary
evidence. Rehearsal evidence may validate its shape but can never authorize an
ownership promotion.

## Remaining completion gates

1. Finish Rust runtime/server authority, operational conformance, and release
   packaging; keep Python rollback deployment available.
2. Run capability canaries through internal/test, 1%, 10%, 50%, and 100% with
   observation windows and zero unexplained correctness mismatches.
3. Keep one full compatibility release before Phase 6 default/removal work.

## Test execution acceleration

The next-release test-harness work is implemented. The compatibility bootstrap
uses persisted `pytest.main()` code, a cached immutable candidate image, three
ordinary container shards, deterministic LPT scheduling from optional timing
history, private writable workspaces, resumable per-group reports, fail-closed
identity merge, and exact no-loss/no-dup group checks. Inline image,
worker, and bootstrap code moved to inspectable `.Dockerfile`, `.sh`, and `.py`
files. Current-source wheel construction likewise uses a persisted host builder,
pinned Dockerfile, and shell entry point. Native-extension discovery is a
persisted Python helper rather than an inline `python -c` payload.

Controlled equal-coverage measurements found application container sharding
1.425x faster (251.12s to 176.23s). Core pytest-xdist with two workers was only
0.724x as fast as serial (124.53s to 172.12s), so xdist remains opt-in and is
off by default. Parser is file-isolated because its module-global SQLite logger
and provider state are not xdist-safe. See `ADR-015-test-harness.md` and
`contracts/benchmarks/adr015-test-parallelism-current-windows.json`.
