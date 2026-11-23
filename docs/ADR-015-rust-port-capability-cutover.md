# ADR-015: Rust Port SQLite Query Binding Fix and Capability Cutover Status

## Status

Status — ADR-015's six named capability test groups and its bounded CI/UAT gates passed for the recorded candidate. The manifest declares 7/8 Rust-targeted capability flags true. The executable release gate derives `graph-pgvector` and `workflow-runtime` ready; `server-rest-sse-mcp-cli` remains blocked by four versioned syscall deferrals. The broader public-surface parity goal remains open; neither the capability gate suite nor this status claims universal Python/Rust parity.

The release-gate report separates deferral-manifest validation from cutover effect: a structurally valid deferral list reports `validation_status=passed`, while non-empty pending operations report `cutover_status=blocked` with their names. A green schema check must never be read as cutover approval.

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

The table reports the manifest's per-capability selection flags, not the effective server-wide readiness calculation. The latter also requires empty pending-route and pending-syscall inventories; see `rust/crates/kogwistar-api/src/cutover.rs` and `rust/crates/kogwistar-api/src/tests/server_readiness.rs`.

| Capability | Current manifest `rust_cutover_ready` | Covered by this fix | Notes |
|---|---|---|---|
| deterministic-contracts | true | no | already ready |
| sqlite-meta | true | no | already ready |
| postgres-sequence-event-log | true | yes | Capability gate passed; four deferred syscalls still block server-wide readiness. |
| projections-snapshots-run-registry | true | yes | Capability gate passed; four deferred syscalls still block server-wide readiness. |
| queues-leases-lanes | true | yes | Capability gate passed; four deferred syscalls still block server-wide readiness. |
| graph-pgvector | true | yes | Capability gate passed; four deferred syscalls still block server-wide readiness. |
| chroma-adapter | false | no | Intentionally Python-owned target; Rust Chroma GraphStore interop is verified separately in `kogwistar/docs/chroma_rust_interop_worklist.md`. |
| workflow-runtime | true | yes | Capability gate passed; four deferred syscalls still block server-wide readiness. |
| server-rest-sse-mcp-cli | false | yes | Named REST/SSE/MCP/CLI test gate passed; effective cutover remains blocked by four deferred syscalls. |



## 6 Capability CI Gate - Grounding Evidence

All six capability **test gates** are verified by named ADR-015 oracle selectors. The accepted CI/UAT evidence closes those declared test gates. The contract still marks `send_message`, `receive_message`, `mount_memory`, and `invoke_tool` as intentionally deferred, so the server cutover flag remains false. These server-level deferrals do not block the independently passing graph-pgvector or workflow-runtime readiness gates. Evidence is recorded under `kogwistar/.codex/adr015-phase3-capability-gate/` with per-group JUnit XML; consult current CI records and manifests rather than treating local historical artifacts as current proof.

| # | Capability | Gate group | Tests | Result | Why it worked |
|---|---|---|---|---|---|
| 1 | postgres-sequence-event-log | postgres-sequence-event-log | 4 (test_rust_store_postgres_differential.py, -k postgres) | passed, 0 fail/err/skip, returncode 0 | bind_params + query_map((), ...) replaced by params_refs slice passed directly to query_map; raw parameter bindings no longer cleared by empty () tuple. |
| 2 | projections-snapshots-run-registry | projections-snapshots-run-registry | 6 (projection, run-registry, workflow-history, recovery differentials) | passed, 0 fail/err/skip, returncode 0 | Rust PostgreSQL bridge + PyO3 facade expose store_sqlite_json; Python and Rust read/write the same schema atomically. |
| 3 | queues-leases-lanes | queues-leases-lanes | 3 (index-jobs queue differential, concurrency reclaim + terminal retry, lane python/rust + concurrency) | passed, 0 fail/err/skip, returncode 0 | claim_index_jobs, list_index_jobs, enqueue_index_job verified via differential tests; parameter binding fix resolves Wrong number of parameters passed to query. Got 0, needed 10. |
| 4 | graph-pgvector | graph-pgvector | 47 (test_rust_store_graph_mutation_differential.py, test_rust_postgres_transaction_session.py) | passed, 0 fail/err/skip, returncode 0 | graph/vector infrastructure + pgvector differential tests complete; transaction sessions atomic. |
| 5 | workflow-runtime | server suite (tests/server) | included in tests/server run | 247 passed, 25 skipped, 0 failed | workflow runtime differential tests complete; server REST/SSE/MCP paths exercise workflow runtime. |
| 6 | server-rest-sse-mcp-cli | server suite (tests/server) | included in tests/server run | 247 passed, 25 skipped, 0 failed | REST/SSE/MCP server differential tests complete; OIDC /me 401 fixed by verifying with server.JWT_SECRET/server.JWT_ALG (mint secret) instead of hardcoded TEST_JWT_SECRET, clearing auth_jwt_settings + JWT_ISS/JWT_AUD. |

Historical native extension artifact: `kogwistar/kogwistar/_rust.pyd`, SHA-256 `fc22e93ff818b12a5aac85e6891be68bdddad76863432273a5818e1fa6c01dde` (local Windows artifact; not current release evidence).
Deployment scope: Kogwistar is a library; isolated Docker UAT is the deployment-like verification environment. No customer-production rollout is implied.

`chroma-adapter` remains false by design: Python adapter retained, not a Rust cutover target. This is distinct from Rust Chroma store/service interoperability, which has its own passing verification worklist. Neither status asserts Chroma is a transactional canonical event store.

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

The first server attempt run from the outer workspace root reported three missing-file errors because tests resolve committed files relative to the `kogwistar` repository root; rerunning with that root passed. The historical table and narrative above that described the six capability flags as false predate the current manifest and are superseded by it. Current release-gate evidence keeps server cutover blocked while preserving independently passing graph-pgvector and workflow-runtime readiness. Full public-surface parity remains incomplete. This is a library/UAT capability decision, not customer production rollout.

## Next steps

1. Keep six capability test gates executable and current.
2. Resolve or explicitly retain the four versioned syscall deferrals before claiming server-wide Rust cutover readiness.
3. Keep manifest per-capability readiness aligned with executable gate evidence; do not infer whole-stack parity from those flags.
4. Track full public-surface parity separately in `kogwistar/docs/ADR-022-dual-language-goal-closure-checklist.md`.
