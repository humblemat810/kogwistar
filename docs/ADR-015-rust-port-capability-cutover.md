# ADR-015: Rust Port SQLite Query Binding Fix and Capability Cutover Status

## Status

Draft — SQLite query-binding fix and differential verification complete for `queues-leases-lanes` and `projections-snapshots-run-registry`. The six capability test gates pass, but cutover readiness remains controlled by the current manifest, phase evidence, same-candidate Docker UAT, and deferred-operation policy. `chroma-adapter` remains Python-owned and is not a Rust cutover target.

## Context

The Rust->Python port of `kogwistar-python` store operations uses `rusqlite`. Several store functions bound parameters via `bind_params(&mut stmt, &params)` followed by `stmt.query_map((), ...)`. The `raw_bind_parameter` bindings are cleared when `query_map` is invoked with an empty `()` params tuple, producing `Wrong number of parameters passed to query. Got 0, needed 10` (observed in `claim_index_jobs`).

Five call sites were affected: `list_index_jobs`, `claim_index_jobs`, `claim_projected_lane_messages`, `list_projected_lane_messages`, `list_server_runs` in `rust/crates/kogwistar-python/src/lib.rs`.

Additionally, `project_lane_message` in `lib.rs` had an INSERT with 26 values for 25 columns; the trailing `NULL` list had one extra `NULL`.

## Decision

Replace each `bind_params` + `query_map((), ...)` pair with a `params_refs` slice passed directly to `query_map`:

```rust
let params_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref() as &dyn rusqlite::ToSql).collect();
let rows = se(stmt.query_map(params_refs.as_slice(), |r| ...))?;
```

The `bind_params` helper itself is retained (now unused at these sites).

Fix `project_lane_message` INSERT to match column count (remove one trailing `NULL`).

## Verification

- `cargo build -p kogwistar-python` (workdir=`rust`): passes (warnings only).
- `rust\target\debug\kogwistar_python.dll` copied to `kogwistar\kogwistar\_rust.pyd`.
- `_rust.pyd` loads and exposes `store_sqlite_json`.
- Differential suite `tests/core/test_rust_store_index_jobs_differential.py`: **5 passed** (`-p no:cacheprovider`):
  - `test_enqueue_and_list_index_jobs_parity`
  - `test_claim_index_jobs_parity`
  - `test_enqueue_index_job_parity`
  - `test_list_projected_lane_messages_parity`
  - `test_list_server_runs_parity`
- Direct `store_sqlite_json` calls verified working:
  - `raw_append`, `exclusive_raw_replay`, `list_index_jobs`, `claim_index_jobs`, `list_projected_lane_messages`, `list_server_runs` — no `Wrong number of parameters` errors.

## Capability cutover status (`rust-port-v1.json`)

| Capability | rust_cutover_ready | Covered by this fix | Notes |
|---|---|---|---|
| deterministic-contracts | true | no | already ready |
| sqlite-meta | true | no | already ready |
| postgres-sequence-event-log | false | yes | Test gate passes; cutover flag remains false until current readiness evidence authorizes Rust authority |
| projections-snapshots-run-registry | false | yes | Test gate passes; cutover flag remains false until current readiness evidence authorizes Rust authority |
| queues-leases-lanes | false | yes | Test gate passes; cutover flag remains false until current readiness evidence authorizes Rust authority |
| graph-pgvector | false | yes | Test gate passes; cutover flag remains false until current readiness evidence authorizes Rust authority |
| chroma-adapter | false | no | Python adapter retained; not a Rust cutover target |
| workflow-runtime | false | yes | Test gate passes; cutover flag remains false until current readiness evidence authorizes Rust authority |
| server-rest-sse-mcp-cli | false | yes | Test gate passes; cutover flag remains false until current readiness evidence authorizes Rust authority |



## 6 Capability CI Gate - Grounding Evidence

All six capability **test gates** are verified by named ADR-015 oracle selectors. Test-gate PASS is not itself cutover authorization; the release gate also requires current phase evidence, same-candidate Docker library UAT, no deferred required operation, and manifest alignment. Evidence is recorded under kogwistar/.codex/adr015-phase3-capability-gate/ with per-group JUnit XML.

| # | Capability | Gate group | Tests | Result | Why it worked |
|---|---|---|---|---|---|
| 1 | postgres-sequence-event-log | postgres-sequence-event-log | 4 (test_rust_store_postgres_differential.py, -k postgres) | passed, 0 fail/err/skip, returncode 0 | bind_params + query_map((), ...) replaced by params_refs slice passed directly to query_map; raw parameter bindings no longer cleared by empty () tuple. |
| 2 | projections-snapshots-run-registry | projections-snapshots-run-registry | 6 (projection, run-registry, workflow-history, recovery differentials) | passed, 0 fail/err/skip, returncode 0 | Rust PostgreSQL bridge + PyO3 facade expose store_sqlite_json; Python and Rust read/write the same schema atomically. |
| 3 | queues-leases-lanes | queues-leases-lanes | 3 (index-jobs queue differential, concurrency reclaim + terminal retry, lane python/rust + concurrency) | passed, 0 fail/err/skip, returncode 0 | claim_index_jobs, list_index_jobs, enqueue_index_job verified via differential tests; parameter binding fix resolves Wrong number of parameters passed to query. Got 0, needed 10. |
| 4 | graph-pgvector | graph-pgvector | 47 (test_rust_store_graph_mutation_differential.py, test_rust_postgres_transaction_session.py) | passed, 0 fail/err/skip, returncode 0 | graph/vector infrastructure + pgvector differential tests complete; transaction sessions atomic. |
| 5 | workflow-runtime | server suite (tests/server) | included in tests/server run | 247 passed, 25 skipped, 0 failed | workflow runtime differential tests complete; server REST/SSE/MCP paths exercise workflow runtime. |
| 6 | server-rest-sse-mcp-cli | server suite (tests/server) | included in tests/server run | 247 passed, 25 skipped, 0 failed | REST/SSE/MCP server differential tests complete; OIDC /me 401 fixed by verifying with server.JWT_SECRET/server.JWT_ALG (mint secret) instead of hardcoded TEST_JWT_SECRET, clearing auth_jwt_settings + JWT_ISS/JWT_AUD. |

Native extension: kogwistar/kogwistar/_rust.pyd, sha256 fc22e93ff818b12a5aac85e6891be68bdddad76863432273a5818e1fa6c01dde.
Production authority: not-promoted-local-evidence-only.

Capability chroma-adapter remains false: Python adapter retained, not a Rust cutover target; covered by existing Python chroma tests, not by the Rust gate.

### How the last server gate failure was fixed

tests/server/test_oidc_keycloak_e2e.py::test_oidc_keycloak_container_e2e failed assert 401 == 200 on GET /api/auth/me only in the full suite. Root cause: sequence pollution of JWT auth state. The token was minted by server.app.state.auth_service using server.JWT_SECRET (an import-time constant captured from the OS JWT_SECRET env, kogwistar-test-secret), but /me verified with hardcoded TEST_JWT_SECRET (dev-secret) -> signature mismatch -> 401. Standalone passed only because the OIDC module imported first and os.environ.setdefault("JWT_SECRET", dev-secret) won.

Fix: in test_oidc_keycloak_e2e.py, the pre-/me JWT reset now verifies with the same secrets used to mint (server.JWT_SECRET / server.JWT_ALG), clearing auth_jwt_settings + JWT_ISS/JWT_AUD. tests/server -> 247 passed, 25 skipped, 0 failed (-p no:cacheprovider). auth_middleware.py was temporarily instrumented then reverted (no production code change).

### Revalidation of the six capability implementations

The 2026-11-22 revalidation found one implementation-adjacent defect before the store/PostgreSQL gate completed: importing `kogwistar._rust_bridge` directly entered the `engine_core` package and reached `shortids.py` while `_rust_bridge` was still initializing. The bridge therefore moved the `native_sqlite_enter`/`native_sqlite_exit` import into the managed SQLite session branch. This removes the circular import without changing either Python or Rust ABI. The previously failing child-process transaction-token test now passes, and direct `kogwistar._rust_bridge` import succeeds.

The same revalidation exposed two SQLite transaction-ownership defects: a duplicate native `begin_transaction` did not fail closed, and a rejected nested begin released the live session. The context registry now rejects the duplicate and preserves the active session until `commit_transaction`/`rollback_transaction`. Focused verification: SQLite context invariants **37 passed, 2 skipped**; Rust SQLite differential **12 passed**.

The Python bridge now delegates the complete transaction lifecycle to the native extension instead of double-entering the same transaction; the extension owns the native registry entries for `begin_transaction`, `commit_transaction`, and `rollback_transaction`. Rust SQLite authority suite: **21 passed**.

Revalidation results from the `kogwistar` repository root:

- PostgreSQL store/projection/recovery/lane/queue/graph groups: **143 passed, 0 failed**.
- Rust runtime authority, PostgreSQL durability, and worker handoff: **82 passed, 10 skipped, 0 failed**.
- Server REST/SSE/MCP/CLI suite: **247 passed, 25 skipped, 0 failed**.

The first server attempt run from the outer workspace root reported three missing-file errors because tests resolve committed files relative to the `kogwistar` repository root; rerunning with that root passed. `rust_cutover_ready` remains the explicit cutover ledger and stays false where its tests require it; implementation and local CI gate evidence do not alone promote production ownership.

## Next steps

1. Keep six capability test gates executable and current.
2. Produce same-candidate phase/UAT evidence for each capability.
3. Remove or explicitly account for deferred required operations.
4. Set each manifest readiness flag only when the executable release gate reports that capability `ready`.
5. Finalize ADR-015 and push; watch GitHub CI results (per repo policy).
