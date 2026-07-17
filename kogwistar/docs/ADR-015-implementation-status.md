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
1.016x Python, and the canonical state digest is equal. Passing evidence does
not itself unlock meta-store authority; the public owner flag remains unrouted
and fail-closed.

## Intentionally not ready

- `EngineSQLite` and public PostgreSQL/meta facades do not route authoritative
  writes through `KOGWISTAR_IMPL_META_STORE`; explicit native bridges are
  integration surfaces.
- Public graph facades remain Python-owned. Chroma remains a Python
  single-writer adapter.
- Rust runtime owns recorded durable reducer/scheduler slices, not the full
  public `WorkflowRuntime` resolver and side-effect lifecycle.
- Rust server binary and application services exist, but server and runtime
  cutover readiness remain false.
- `index_applied_state` remains Python-owned with derived-index authority; it is
  deferred until graph/index authority changes.
- Production internal/test, 1%, 10%, 50%, and 100% canaries have not occurred.

Every capability now requires explicit `rust_cutover_ready: true`; omitted
readiness fails closed. `scripts/rust_port_readiness.py` validates canary
evidence. Rehearsal evidence may validate its shape but can never authorize an
ownership promotion.

## Remaining completion gates

1. Route one public persistent capability through its owner flag without a
   second writer, then meet the ADR performance thresholds or approve a
   measured exception.
2. Finish Rust runtime/server authority, operational conformance, and release
   packaging; keep Python rollback deployment available.
3. Run capability canaries through internal/test, 1%, 10%, 50%, and 100% with
   observation windows and zero unexplained correctness mismatches.
4. Keep one full compatibility release before Phase 6 default/removal work.
