# ADR-015 Remaining Implementation Tasklets

**Progress:** **7/11 PASS (63.6%)**. Zero tasklets are `IN PROGRESS`; four are `NOT STARTED`.
**Purpose:** Track remaining library implementation work, not production rollout.
**Parent:** [ADR-015 implementation status](ADR-015-implementation-status.md).
**Separate gates:** Production canaries and promotion remain exclusively in
[ADR-015 cutover worklist](ADR-015-cutover-worklist.md). Closed parity evidence
remains in the ADR-022 inventory.

## Rules

- Fixed scope: these 11 tasklets are extracted from the currently documented
  runtime, PostgreSQL query, and server-authority gaps. Do not add items or
  change the denominator without an explicit scope decision.
- A tasklet passes only when its acceptance oracle passes on one identified
  source candidate and its evidence is linked here. Nearby tests do not count.
- `IN PROGRESS` means implementation exists but the complete oracle below has
  not passed. `NOT STARTED` includes intentionally fail-closed Rust operations.
- Preserve Python support, Python-written data compatibility, ACL/scope checks,
  transaction ownership, and fail-closed behavior. No silent language fallback.
- Do not mark production readiness here. Local implementation evidence cannot
  satisfy a canary, promotion, or deployment task.

## Runtime Semantics

Parent: ADR-015 `workflow-runtime` and Phase 4 public-runtime-owner gate;
normative behavior: [runtime semantics contract](ADR-015-runtime-semantics-contract.md).

| ID | Status | Tasklet and single acceptance oracle |
| --- | --- | --- |
| RT-01 | PASS | Match Python frozen-plan route selection for explicit aliases, predicates, priority, fan-out, multiplicity, default routing, and failure suppression. Shadow differential test compares Python/Rust route decision fields and asserts IDs/order: `tests/runtime/test_runtime_parity_bridge_contract.py::test_rust_shadow_route_selection_matches_python_contract` (**1 passed**, `KOGWISTAR_IMPL_RUNTIME=shadow`); Rust route contract: `cargo test --manifest-path rust/Cargo.toml -p kogwistar-runtime route_selection_preserves_explicit_predicate_fanout_failure_and_default_rules --lib` (**1 passed**); existing Rust-mode route bridge: `test_runtime_parity_bridge_route_next_shared_semantics` (**1 passed**). Native runtime candidate: source SHA-256 `f548410c70b09cdba877ef80d4ea3b3f60ed899559f228d48a7f6813623e9ccc`, extension SHA-256 `a8986916cbb62f139582c8f4614d7b0d59c2f33c2e990b385071b8fb9143dfff`. Ruff unavailable in this venv; lint not claimed. |
| RT-02 | PASS | Match successful and handled/unhandled failure effects: updates, errors, token consumption, successor admission, and terminal frontier cleanup. Paired tests compare canonical run state and event history. Evidence: `test_python_workflow_success_matches_rust_durable_successor_admission` compares Python and Rust SQLite start→done→complete state/event histories and terminal frontier; `test_python_workflow_failure_matches_rust_durable_state_and_step_event` compares terminal failure effects; `test_python_handled_failure_matches_rust_durable_recovery_admission` compares handled failure→recover→end→complete histories and terminal state. Adapter and checkpoint rollback/retry tests also pass. Focused combined runtime/parity/checkpoint run: **26 passed**; Rust SQLite worker success/failure atomicity and retry tests: **2 passed**. Suspension/join parity remains explicitly covered by RT-03/RT-04, not claimed here. |
| RT-03 | PASS | Match suspension/resume for one exact token, preserving unrelated pending work and join obligations across restart. Paired trace: `test_python_suspend_resume_matches_rust_durable_ordered_history` runs Python on persistent Chroma, rebuilds both engines before resume, and compares Rust SQLite after reopen; exact token identity, wait reason/payload, ordered step sequence/status/update/errors, final user state, terminal status, and empty frontier match. Supporting branch/join evidence: `test_runtime_parity_bridge_workflow_suspend_and_resume_branching_roundtrip` (**1 passed**) proves Python keeps branch `b` and releases join after suspended `a` resumes; Rust `suspend_resume_preserves_unrelated_pending_and_join_obligations` proves only the owned token moves while other pending work and join waiter/count remain intact. `test_suspend_reopen_explicit_resume_preserves_payload_and_parked_token` (**1 passed**); Rust runtime suite (**49 passed**). Focused parity/join/reopen regression: **3 passed**. |
| RT-04 | PASS | Match fan-out/join behavior through interleaved completion, retry, and cancellation. `test_rust_shadow_join_retry_cancel_matches_python_terminal_frontier` runs one fixed-ID trace in Python and Rust `shadow`: branch `b` fails and retries while `a` reaches the join; cancellation then drains the retry and clears the waiter. Both runs end `cancelled` with identical empty frontier, zero join count, and no waiters (**1 passed**). `test_cancelled_fanout_clears_queued_join_obligations` separately proves an undispatched branch is not run and its join obligation is cleared (**1 passed**). `test_rust_shadow_join_accounting_matches_python_runtime` verifies ordinary fan-out releases join once (**1 passed**); Rust `tests::worker_success_effect_holds_join_arrivals_until_barrier_releases_once` verifies staggered arrivals release once (**1 passed**). The trace uses the installed Rust runtime decision extension in shadow mode; it does not claim Rust owns the whole workflow loop. |
| RT-05 | PASS | `tests/runtime/test_rust_runtime_worker_handoff.py` (**7 passed**) proves claimed-result acceptance atomically records event/projection/successor frontier and completes its lane message; exact duplicate is idempotent. Fault-after-write rollback leaves seed state and claim intact; retry then accepts once, duplicate remains idempotent. Rust reopen and Python `EngineSQLite` reopen of the same file agree on event sequence, run state, completed lane message, projection sequence/state, and successor frontier. Ruff and `git diff --check` passed. |

## PostgreSQL Graph Reads

Parent: ADR-015 `graph-pgvector`; current partial native-read boundary is
described in [implementation status](ADR-015-implementation-status.md).

| ID | Status | Tasklet and single acceptance oracle |
| --- | --- | --- |
| PG-01 | PASS | Python-compatible vector metadata filters support flat equality, `$ne`, and `$and` combinations; unsupported forms fail closed before backend fallback. Shared fixture `contracts/golden/postgres-vector-filter-parity-v1.json` produced identical ordered IDs and distances through Python `PgVectorBackend` and Rust PostgreSQL authority against live pgvector: `tests/pg_sql/test_phase3_e2e_engine_pg.py::test_rust_postgres_vector_filters_match_python_backend` (**1 passed**). Python filter grammar plus Rust-selected native routing/no-fallback tests: **2 passed**. Rust SQL builder also verifies scoped equality/exclusion and hostile quoted-value escaping; PostgreSQL store unit suite: **11 passed, 33 live-DB tests ignored in the package-wide run**. |
| PG-02 | PASS | Move the documented graph-derived lookup reads to Rust PostgreSQL authority and prove Python/Rust parity for IDs, namespace/filter scope, provenance, ordering, and hydration without Rust-to-Python fallback. Shared `contracts/golden/postgres-derived-reference-parity-v1.json` covers node/edge reference filtering, entity hydration, insertion-method primary-record fallback, document-to-node/edge ID reads, and source-context extraction against live pgvector. Python and Rust return identical sorted IDs, complete models, and bounded source context; wrong namespace/method rows are excluded. Rust mode fails if any of five Python backend read methods are called. Evidence: `tests/pg_sql/test_phase3_e2e_engine_pg.py::test_rust_postgres_reference_filters_match_python_backend` plus four native-read regressions in `tests/core/test_rust_postgres_session_facade.py` (**5 passed**, live pgvector + mocked boundary); Ruff, checklist mapping test, and `git diff --check` passed. |

## Rust Server Operations

Parent: ADR-015 `server-rest-sse-mcp-cli`; prerequisites and frozen deferrals are
declared in `contracts/rust-port-v1.json`. Each operation remains fail-closed
until its authority, authorization, persistence, and transport oracle passes.

| ID | Status | Tasklet and single acceptance oracle |
| --- | --- | --- |
| SRV-01 | NOT STARTED | Implement `DELETE /admin/doc/{doc_id}` through native graph mutation, queue/index cleanup, rollback, and public REST contract. Authenticated parity test proves ACL enforcement, durable deletion, replay/reopen behavior, and Python readability. |
| SRV-02 | NOT STARTED | Implement conversation `send_message` and `receive_message` syscalls through the versioned durable runtime boundary. Paired REST/MCP tests prove authorization, scope isolation, idempotent retry, restart recovery, and Python/Rust readability. |
| SRV-03 | NOT STARTED | Implement `mount_memory` syscall through the versioned durable runtime boundary. Paired tests prove ACL and namespace enforcement, bounded projection, restart recovery, and no unauthorized memory exposure. |
| SRV-04 | NOT STARTED | Implement `invoke_tool` syscall through the host-authorized tool catalog and durable runtime boundary. Paired tests prove capability checks, scope/ACL enforcement, idempotency, denied-call non-execution, and restart recovery. |

## Completion Boundary

Passing all 11 tasklets closes this implementation list only. It does not change
`rust_cutover_ready`, satisfy any production canary, make Rust the default, or
authorize removal of Python. Those decisions remain governed by ADR-015 and its
cutover worklist.
