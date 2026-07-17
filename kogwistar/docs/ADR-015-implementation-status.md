# ADR-015 implementation status

Updated: 2026-07-17

ADR-015 is not complete as a production migration. Local implementation and
compatibility work is substantially complete, but authoritative ownership and
production rollout remain intentionally fail-closed.

## Completed local gates

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
  are implemented behind the coordinated sync PostgreSQL selector, but the
  capability readiness flags remain false until fresh four-layer evidence passes
  and capability-specific production canaries exist. `KOGWISTAR_IMPL_META_STORE=rust`
  independently promotes only the ready `sqlite-meta` capability on SQLite.
- Sync PostgreSQL base graph mutation, replay, repair, and rollback writers are
  now routed through the coordinated native transaction owner. Derived
  node-ref/endpoint tables remain disposable Python adapters driven by native
  index jobs. `graph-pgvector` remains not ready pending fresh four-layer evidence
  and production canaries. Chroma remains a Python single-writer adapter.
- Rust runtime owns recorded durable reducer/scheduler slices, not the full
  public `WorkflowRuntime` resolver and side-effect lifecycle.
- Rust server binary and application services exist, but server and runtime
  cutover readiness remain false.
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

## Next-release maintenance

- Replace the compatibility-runner bootstrap's deprecated
  `pytest.console_main()` call with `pytest.main()`. Reverify subprocess return
  codes, report persistence, and resume behavior before removing the warning.
- Evaluate bounded parallel execution of isolated application unit groups.
  Preserve per-group temporary storage, deterministic report ordering, progress
  persistence, and the existing native-handle isolation before enabling it.
  Prefer a small standard-library orchestrator over pytest-xdist: build one
  immutable candidate image, distribute `_target_groups()` across 2-3 ordinary
  pytest containers, mount source read-only, give each shard distinct temp/cache,
  database namespace, ports, and report, then merge only reports with identical
  candidate/dependency identities. Unit-test no-loss/no-dup grouping, stable
  ordering, failure/resume, identity rejection, and deterministic merge; compare
  serial and sharded tiny-suite results before making sharding a release gate.
  Emit machine-readable per-file/group timing history for either container
  sharding or a later pytest-xdist evaluation: wall time, setup/call/teardown
  where available, outcome, shard/mode, and candidate/harness/dependency plus
  OS/Python/CPU-class identity. Balance with a recent median-based longest-first
  schedule; retain raw runs as CI artifacts and only a stable aggregate baseline
  in source control. Timing history must not affect candidate identity.
Investigate the use of pytest-xdist to speed up test
