# ADR-022 Dual-Language Goal Fine Checklist

**Updated:** 2026-11-24
**Purpose:** Executable closure tracker for GOAL-06/08/09 in [the goal-closure audit](ADR-022-dual-language-goal-closure-checklist.md). This is not the 177/911 evidence inventory and does not change its denominator.
**Progress:** **1/18 tasklets PASS (5.6%); 3 IN PROGRESS; 14 NOT STARTED; 0 FAIL; 0 BLOCKED.** This is tasklet completion only, not an estimate that the overall goal is 5.6% complete.
**Closure:** GOAL-06 and GOAL-08 close only when every applicable operation/backend/behavior cell below has a passing executable oracle or an explicit unsupported + fail-closed oracle. GOAL-09 closes only when all declared performance tasklets pass with scoped claims. Coarse status must be derived from all mapped fine tasklets; no deferrals remain.

## Rules

- Fixed scope: 18 counted tasklets below. Do not add denominator items for implementation details. If one tasklet has independently passable outcomes, split before work, retain its ID as an uncounted parent, and update this count in the same change.
- Matrix source: `contracts/dual-language-surface-inventory-v1.json`, `contracts/dual-language-support-matrix-v1.json`, and `contracts/rust-port-v1.json`. Enumerate every HTTP operation, MCP operation, syscall, declared backend, language path, and applicable case family. No silently omitted cells.
- A listed test name proves selector mapping only. PASS requires the test to assert the tasklet's stated behavior and Python/Rust parity where both are supported.
- Unsupported cells require an executable test proving explicit rejection before side effects; catalog or dispatch presence is not parity.
- Record exact test selectors, command, result, and any intentionally normalized fields in each row. Keep non-deterministic IDs/times normalized only when contract permits.
- PASS rows remain closed unless a reproduced regression, invalid evidence, or approved contract change requires reopening. Do not re-open ADR-015 or the closed 177/911 inventory to inflate progress.
- GOAL-06/08/09 status in the coarse closure audit must be reconciled whenever their mapped tasklets change. All child tasklets passing is necessary, not sufficient, if the criterion's stated contract is still unmet.

## Tasklets

| ID | Coarse | Status | Single acceptance oracle and evidence |
| --- | --- | --- | --- |
| DLFC-01 | GOAL-06 | PASS | Every inventoried HTTP and MCP operation maps to an executable Python selector and an executable Rust selector; deleting either selector makes the check fail. `tests/core/test_rust_port_contract_manifest.py::test_public_http_and_mcp_tasklets_name_executable_tests` (**1 passed**, 2026-11-23). Selector presence is not behavioral parity. |
| DLFC-02 | GOAL-06 | IN PROGRESS | Generate and validate a stable operation/backend/case-family matrix from the three source contracts. Oracle: every inventory operation has exactly one explicit applicability row per backend and required behavior family; no duplicate, missing, or unknown IDs. Record generated artifact and checker selector here. |
| DLFC-03 | GOAL-06 | NOT STARTED | HTTP success-contract cells: Python and Rust use same request/fixture; normalized status, schema, defaults, ordering, IDs match for every applicable HTTP operation. Oracle: parameterized parity suite reports zero missing/unexecuted cells. |
| DLFC-04 | GOAL-06 | NOT STARTED | HTTP invalid-input cells: missing, malformed, wrong-type, unknown, extra, empty, and boundary inputs as applicable; compare rejection contract and prove no durable side effect. |
| DLFC-05 | GOAL-06 | NOT STARTED | HTTP authority cells: authentication, role, capability, namespace/workspace/graph-space, ACL-hidden target as applicable; prove deny-before-protected-access and no leak/state change. |
| DLFC-06 | GOAL-06 | NOT STARTED | HTTP read cells: filtering, limits/pagination, ordering, hydration, tombstone/redirect, provenance, and not-found behavior as applicable; compare same seeded state. |
| DLFC-07 | GOAL-06 | NOT STARTED | HTTP mutation cells: canonical resulting state, emitted events, retry/idempotency, and injected-failure atomicity for each applicable mutator. |
| DLFC-08 | GOAL-06 | NOT STARTED | MCP success/wire cells: every allowlisted operation matches Python/Rust request schema, result/error envelope, defaults, and transport behavior using shared fixtures. |
| DLFC-09 | GOAL-06 | NOT STARTED | MCP authorization cells: every operation with an authority boundary denies the same principal/scope cases before backend reads/writes; assert unchanged state. |
| DLFC-10 | GOAL-06 | NOT STARTED | MCP effect/retry cells: authorized calls reach the same backend effect, durable state, event/trace, and duplicate/replay outcome; no tool execution from mere catalog presence. |
| DLFC-11 | GOAL-06 | IN PROGRESS | Syscall success/error cells: all 11 catalogued operations have Python/Rust result and error-shape parity for supported backend cells; unsupported paths fail closed. Partial evidence: Python `tests/server/test_syscall_api.py::test_syscall_request_approval_branches` and Rust `in_memory_workflow_syscalls_use_native_run_handlers` pass grant/revoke authorization, response snapshots, and exact shared-fixture parity for missing action and missing grant/revoke capability (`contracts/golden/syscall-request-approval-invalid-v1.json`; 1 Python test and 1 Rust test passed, 2026-11-23; approval slice commit `caba7a07`). Checkpoint string `step_seq` normalization uses `contracts/golden/syscall-checkpoint-step-seq-v1.json`; Python `test_syscall_checkpoint_accepts_conversation_namespace_and_numeric_string_step_seq` and Rust `in_memory_workflow_syscalls_use_native_run_handlers` each passed on 2026-11-23, confirming string `"0"` normalizes to integer `0` (commit `3b523603`). Project-view syscall checks shared resource snapshot required and nested fields via `contracts/golden/workflow-resources-http-v1.json`; Python `test_syscall_project_view_matches_resource_snapshot_contract` and Rust `in_memory_project_view_syscall_is_reachable_through_authenticated_http` passed on 2026-11-23. This asserts required shape, not exact dynamic values; Rust HTTP test also verifies missing capability denial (commit `8e3b8ca4`). Invoke-tool missing/blank name and unknown tool have exact shared-fixture HTTP status/body parity (`contracts/golden/syscall-invoke-tool-errors-v1.json`); Python `test_syscall_invoke_tool_list_transcript_returns_messages` and Rust `in_memory_invoke_builtin_tools_match_shared_contract_and_authority` passed (commit `c6ca243c`). Terminate-process result shape uses `contracts/golden/run-cancel-v1.json`; Python `test_syscall_spawn_process_returns_ok_contract` and Rust `in_memory_workflow_syscalls_use_native_run_handlers` passed on 2026-11-23, asserting required run fields and accepted status only (no exact dynamic-value parity; commit `ec781a02`). Send-message blank-text and non-string idempotency-key errors now match exact status/body via `contracts/golden/syscall-send-message-invalid-v1.json`; Python `test_syscall_send_message_invalid_inputs_match_shared_contract` and Rust `in_memory_send_message_admits_turn_and_run_atomically` passed on 2026-11-23; Rust also verifies no run was created. `sqlite_request_approval_mutations_require_scope_and_return_snapshot` covers SQLite authorization/result. Remaining: other syscall operations, PostgreSQL/SQLite+Chroma cells, and MCP path. |
| DLFC-12 | GOAL-06 | IN PROGRESS | Backend applicability cells: each operation/backend pair is explicitly supported or explicitly rejected; prove no hidden fallback. Preserve SQLite metadata + eventual Chroma projection, PostgreSQL, and memory semantics distinctly. Current syscall cell ledger follows; it is subordinate detail, not added to the 18-tasklet completion denominator. SQLite metadata has five fail-closed cells and six passing supported-path oracles; in-memory `mount_memory` and `project_view` have passing supported-path oracles. Other matrix cells remain open. |
| DLFC-13 | GOAL-08 | NOT STARTED | Durable lifecycle cells: close/reopen, read-after-write, replay/recovery, sequence/version, and Python-write→Rust-read plus Rust-write→Python-read for every applicable durable operation/backend. |
| DLFC-14 | GOAL-08 | NOT STARTED | Runtime trace cells: same fixed-ID workflow traces compare routing, deltas, step/event order, retry, cancel, suspend/resume, terminal frontier, trace parentage, and restart/replay. |
| DLFC-15 | GOAL-08 | NOT STARTED | Transaction/authority seam cells: instrument transaction ownership and failure injection at each declared interop seam; prove no cross-stack partial commit, authority weakening, or unsafe retry. Backend-specific guarantees only. |
| DLFC-16 | GOAL-08 | NOT STARTED | Native isolation cells: Rust-native ingress-to-persistence fails if Python worker/adaptor is invoked and records zero Python handoffs; separately prove Python-only mode does not import/call Rust. |
| DLFC-17 | GOAL-08 | NOT STARTED | Transport/CI cells: HTTP, MCP, SSE/notifications, and CLI where inventoried compare protocol/auth/stream ordering; exact selectors run in supported CPython 3.12–3.14 and PyPy 3.11 CI lanes. |
| DLFC-18 | GOAL-09 | NOT STARTED | Representative performance cells: resource-matched Python-native and Rust-native end-to-end cases across selected high-value operations/backends; publish warm-up, throughput, tail latency, RSS, variation, and seam counts. Pass only declared per-workload thresholds; make no universal speed claim. |

### DLFC-12 Syscall / Backend Cell Ledger

This finite 11-operation × 4-backend ledger makes the currently known support
surface visible. It does **not** change the counted 18-tasklet denominator.
`PARTIAL` means a focused test exists for the operation/backend behavior, but
may not exercise the syscall wire path and does not prove complete Python/Rust
parity. `NOT STARTED` means no adequate cell-level oracle has been verified.
`PASS: fail closed` means unsupported behavior and absence of admitted-run
side effects are explicitly tested. `PASS: supported (live)` means a real
backend service reads the shared fixture successfully; it closes only this
DLFC-12 applicability cell, not DLFC-11's complete behavior families.
Dispatch alone never upgrades a cell.

SQLite metadata `checkpoint` is now `PASS: supported (live)`: Rust
`sqlite_metadata_checkpoint_syscall_reads_recorded_state_and_checks_scope`
reads a recorded transition through the syscall, checks `workflow.run.read`,
accepts numeric-string `step_seq`, and returns not-found for a missing step;
Python `test_syscall_checkpoint_reads_persisted_python_checkpoint` exercises the
real syscall router and persisted checkpoint node, returning the same
fixture result (`contracts/golden/syscall-checkpoint-persisted-v1.json`). The separate
`test_syscall_checkpoint_accepts_conversation_namespace_and_numeric_string_step_seq`
still verifies Python wire normalization. This closes backend applicability for
this cell only; it does not close DLFC-11's complete syscall behavior families.

SQLite metadata `resume` is now `PASS: supported (live)`: Rust
`sqlite_metadata_resume_syscall_resumes_persisted_suspended_token` creates a
durable run, records a suspended worker result, resumes through the syscall,
finishes the queued continuation, and checks the public result against
`contracts/golden/workflow-run-resume-result-v1.json`. Python
`test_syscall_resume_returns_completed_result_contract` checks the same
versioned envelope and result shape. The adapter strips path-only `run_id`
before decoding the strict resume body and waits through the existing runtime
route for the completed result. This closes SQLite metadata applicability only;
cross-backend parity and DLFC-11's full syscall error/retry behavior remain open.

PostgreSQL `resume` is now `PASS: supported (live)`: Rust
`postgres_resume_syscall_resumes_persisted_suspended_token` uses an isolated
PostgreSQL schema, persists a suspended worker result, resumes via the syscall,
claims and completes the continuation, then checks the shared
`contracts/golden/workflow-run-resume-result-v1.json` contract. The syscall
waits through the same public completed-result route as SQLite. This closes
PostgreSQL applicability only; it does not establish all syscall behavior
families or Python/PostgreSQL parity.

SQLite metadata `request_approval` is now `PASS: supported (live)`: Rust
`sqlite_request_approval_mutations_require_scope_and_return_snapshot` checks
denied mutations without required authority and authorized grant/revoke results
through the SQLite-backed service. Python
`test_syscall_request_approval_branches` covers grant/revoke, missing
capabilities, blocked/requested outcomes, and invalid-input fixture cases. This
closes applicability only; dynamic approval snapshots are not compared exactly
between implementations, so DLFC-11 remains open.

SQLite metadata `project_view` is `PASS: supported (live)`: Python
`tests/server/test_syscall_api.py::test_syscall_project_view_matches_resource_snapshot_contract`
and Rust `sqlite_runtime/worker_lifecycle.rs::sqlite_syscall_spawn_and_terminate_use_authoritative_run_service`
both passed on 2026-11-24. Both reach the SQLite-backed project-view operation;
the Python test uses the authenticated syscall route and checks the required
resource snapshot fields, while Rust verifies authorization and successful
response shape. Dynamic values are not compared, so DLFC-11 remains open.

SQLite metadata `spawn_process` and `terminate_process` are now `PASS: supported
(live)`: Rust `sqlite_runtime/worker_lifecycle.rs::sqlite_syscall_spawn_and_terminate_use_authoritative_run_service`
and Python `tests/server/test_syscall_api.py::test_syscall_spawn_process_returns_ok_contract`
both passed on 2026-11-24. The Python route starts a run through the authenticated
syscall and terminates it; the Rust test exercises the authoritative SQLite
run service for both operations. This closes only backend applicability. The
tests do not compare identical dynamic syscall payloads or all authorization,
retry, and failure cases, so DLFC-11 remains open.

| Syscall | PostgreSQL | SQLite metadata | SQLite + Chroma | In-memory |
| --- | --- | --- | --- | --- |
| `checkpoint` | NOT STARTED | PASS: supported (live) | NOT STARTED | PARTIAL |
| `create_conversation` | PARTIAL | PASS: fail closed | PASS: supported (live) | PARTIAL |
| `invoke_tool` | NOT STARTED | PASS: fail closed | NOT STARTED | PARTIAL |
| `mount_memory` | NOT STARTED | PASS: fail closed | NOT STARTED | PASS: supported (live) |
| `project_view` | NOT STARTED | PASS: supported (live) | NOT STARTED | PASS: supported (live) |
| `receive_message` | NOT STARTED | PASS: fail closed | PASS: supported (live) | PARTIAL |
| `request_approval` | NOT STARTED | PASS: supported (live) | NOT STARTED | PARTIAL |
| `resume` | PASS: supported (live) | PASS: supported (live) | NOT STARTED | PARTIAL |
| `send_message` | PARTIAL | PASS: fail closed | PARTIAL | PARTIAL |
| `spawn_process` | NOT STARTED | PASS: supported (live) | NOT STARTED | PARTIAL |
| `terminate_process` | NOT STARTED | PASS: supported (live) | NOT STARTED | PARTIAL |

Evidence pointers for current `PARTIAL` cells (focused evidence only; not gate
closure):

- PostgreSQL: `conversation_admission.rs::postgres_send_message_checks_actor_and_request_before_storage`, `conversation_admission.rs::postgres_send_message_admission_is_one_graph_run_event_lane_transaction`, and `conversation_mcp.rs::mcp_conversation_create_uses_native_postgres_syscall`. `workflow_run_routes.rs::postgres_resume_syscall_resumes_persisted_suspended_token` closes PostgreSQL resume applicability using the live DB and shared result fixture.
- SQLite metadata `request_approval`: `sqlite_request_approval_mutations_require_scope_and_return_snapshot` and Python `tests/server/test_syscall_api.py::test_syscall_request_approval_branches` verify approval support; exact dynamic snapshot parity remains open. `project_view`, `spawn_process`, and `terminate_process` applicability are PASS as recorded above. Worker suspend/resume and operator-dashboard tests remain adjacent runtime/HTTP evidence, not syscall parity evidence.
- SQLite + Chroma: Python `tests/server/test_syscall_api.py::test_syscall_receive_message_reads_chroma_graph_and_matches_shared_fixture` (**1 passed**, 2026-11-23) seeds the fixture into the embedded Chroma-backed conversation graph and compares the authenticated syscall response exactly. Rust `sqlite_transcript_reads.rs::sqlite_chroma_receive_message_reads_live_chroma_and_matches_shared_fixture` (**1 passed**, same date) uses unique collections on the configured live Chroma service and verifies the same result through SQLite canonical ACL events; the fake-endpoint companion `::sqlite_chroma_receive_message_returns_scoped_acl_filtered_transcript` covers denied capability/namespace and missing-ID behavior. This closes only the supported-applicability success cell; other error/retry/persistence parity remains open.
- SQLite + Chroma `create_conversation` is `PASS: supported (live)`: Python `tests/server/test_syscall_api.py::test_syscall_create_conversation_uses_authenticated_owner` (**1 passed**, 2026-11-24) exercises the authenticated route using embedded Chroma; Rust `sqlite_conversation_admission.rs::sqlite_chroma_conversation_create_and_turn_admission_are_canonical_and_atomic` (**1 passed**, same date) was rerun with `KOGWISTAR_TEST_CHROMA_URL=http://127.0.0.1:8000` against the running `chromadb/chroma:1.5.9` container. The Rust test verifies canonical admission and Chroma projection. This closes applicability only; it does not compare complete Python/Rust outputs or all failure/retry cases.
- In-memory `mount_memory` and `project_view` are `PASS: supported (live)`. For `mount_memory`, Rust `in_memory_conversations.rs::in_memory_mount_memory_matches_shared_contract_and_authority` and Python `tests/server/test_chat_server_api.py::test_mount_memory_syscall_uses_python_in_memory_backend` (**1 passed each**, 2026-11-24) verify syscall success and authority; Python constructs the actual `build_in_memory_backend`, checks owner success and outsider denial without leaking snapshot data, while Rust reads the shared result fixture. For `project_view`, Python `tests/server/test_chat_server_api.py::test_project_view_syscall_uses_python_in_memory_backend` and Rust `in_memory_run_admission.rs::in_memory_project_view_syscall_is_reachable_through_authenticated_http` (**1 passed each**, same date) verify authenticated route success and required fields from `workflow-resources-http-v1.json`; Rust also verifies missing-capability denial. These close applicability only; exact dynamic cross-language values and remaining error/retry behavior stay under DLFC-11. `in_memory_conversations.rs` also tests `in_memory_create_conversation_is_atomic_and_matches_shared_contract`, `in_memory_send_message_admits_turn_and_run_atomically`, `in_memory_receive_message_matches_shared_python_contract`, and `in_memory_invoke_builtin_tools_match_shared_contract_and_authority`; `in_memory_run_admission.rs` tests `in_memory_workflow_syscalls_use_native_run_handlers` and `in_memory_resume_syscall_preserves_resume_envelope_and_result`.
- SQLite metadata unsupported cells: `sqlite_runtime/worker_lifecycle.rs::sqlite_metadata_conversation_syscalls_fail_closed` covers `create_conversation`, `invoke_tool`, `mount_memory`, `receive_message`, and `send_message`; it asserts HTTP 501, the capability error code/path, and no admitted runs.

The next DLFC-12 implementation batch should convert one backend column at a
time into shared-fixture Python/Rust parity tests, keeping unsupported cells
as explicit fail-closed tests. Do not mark `PARTIAL` as PASS based only on
these existing focused selectors.

## Coarse Mapping

| Coarse criterion | Required fine items | Close condition |
| --- | --- | --- |
| GOAL-06 | DLFC-01 through DLFC-12 | Matrix complete; each applicable operation/backend/contract cell passes, or unsupported cell passes fail-closed test. |
| GOAL-08 | DLFC-13 through DLFC-17, plus GOAL-06 matrix | All applicable behavior families pass; every intentional difference has owner, scope, and compatibility contract. |
| GOAL-09 | DLFC-18 | Performance protocol and thresholds pass for declared representative workloads; claims remain workload-scoped. |

## Progress Update Protocol

1. On each tasklet change, update its status and record exact evidence in this file.
2. Recompute the top summary from the 18 counted rows; do not infer progress from commits, code volume, or passing aggregate CI alone.
3. Update the matching GOAL row in `ADR-022-dual-language-goal-closure-checklist.md` only when evidence or remaining conditions actually change.
4. When all mapped children pass, audit the coarse exit condition and mark it complete only if the condition itself is satisfied. Never add a new tasklet merely because implementation revealed unrelated possible work.
