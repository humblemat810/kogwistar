# ADR-022 Dual-Language Goal Fine Checklist

**Updated:** 2026-11-23
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
| DLFC-11 | GOAL-06 | IN PROGRESS | Syscall success/error cells: all 11 catalogued operations have Python/Rust result and error-shape parity for supported backend cells; unsupported paths fail closed. Partial evidence: Python `tests/server/test_syscall_api.py::test_syscall_request_approval_branches` plus Rust `in_memory_workflow_syscalls_use_native_run_handlers` and `sqlite_request_approval_mutations_require_scope_and_return_snapshot` pass grant/revoke authorization and response snapshots for memory/SQLite. Remaining: other syscall operations, PostgreSQL/SQLite+Chroma cells, MCP path, and full error-envelope parity. |
| DLFC-12 | GOAL-06 | IN PROGRESS | Backend applicability cells: each operation/backend pair is explicitly supported or explicitly rejected; prove no hidden fallback. Preserve SQLite metadata + eventual Chroma projection, PostgreSQL, and memory semantics distinctly. SQLite metadata inventory identifies five conversation syscalls as unavailable; Rust `sqlite_metadata_conversation_syscalls_fail_closed` asserts the 501 capability error for all five and confirms no run was created. `sqlite_request_approval_mutations_require_scope_and_return_snapshot` verifies an applicable SQLite metadata mutation path. Remaining: enumerate and test every other operation/backend pair; these syscall cells do not close applicability globally. |
| DLFC-13 | GOAL-08 | NOT STARTED | Durable lifecycle cells: close/reopen, read-after-write, replay/recovery, sequence/version, and Python-write→Rust-read plus Rust-write→Python-read for every applicable durable operation/backend. |
| DLFC-14 | GOAL-08 | NOT STARTED | Runtime trace cells: same fixed-ID workflow traces compare routing, deltas, step/event order, retry, cancel, suspend/resume, terminal frontier, trace parentage, and restart/replay. |
| DLFC-15 | GOAL-08 | NOT STARTED | Transaction/authority seam cells: instrument transaction ownership and failure injection at each declared interop seam; prove no cross-stack partial commit, authority weakening, or unsafe retry. Backend-specific guarantees only. |
| DLFC-16 | GOAL-08 | NOT STARTED | Native isolation cells: Rust-native ingress-to-persistence fails if Python worker/adaptor is invoked and records zero Python handoffs; separately prove Python-only mode does not import/call Rust. |
| DLFC-17 | GOAL-08 | NOT STARTED | Transport/CI cells: HTTP, MCP, SSE/notifications, and CLI where inventoried compare protocol/auth/stream ordering; exact selectors run in supported CPython 3.12–3.14 and PyPy 3.11 CI lanes. |
| DLFC-18 | GOAL-09 | NOT STARTED | Representative performance cells: resource-matched Python-native and Rust-native end-to-end cases across selected high-value operations/backends; publish warm-up, throughput, tail latency, RSS, variation, and seam counts. Pass only declared per-workload thresholds; make no universal speed claim. |

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
