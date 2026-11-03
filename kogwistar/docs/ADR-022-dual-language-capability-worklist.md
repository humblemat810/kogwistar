# ADR-022 Remaining Worklist

**Updated:** 2026-11-03
**Purpose:** Track bounded work beneath the coarse gates in [ADR-022-dual-language-capability-checklist.md](ADR-022-dual-language-capability-checklist.md).

## Completion Snapshot

- **Coarse checklist:** **154/177 PASS (87.0%)**.
- **Fine worklist:** **3/47 PASS (6.4%)**; **1 IN PROGRESS**, **43 NOT STARTED**.
- The route-protocol regression is now closed as one bounded task, while the live-Chroma route smoke is tracked separately. This is a one-time correction for that distinct deliverable, not newly discovered product scope. A granularity audit of the other rows remains open; do not claim the list is fully leaf-level until that audit is complete. Fine tasks are a breakdown of the 23 open coarse gates, not new coarse items.

## How To Track

- This is the detailed execution list; the linked checklist remains the sole coarse progress and denominator source.
- The baseline is the 23 coarse IDs currently marked `IN PROGRESS`. Every coarse ID must map to fine work below. Fine IDs do not add to the 177-item coarse inventory.
- This is a first-pass decomposition of all open coarse gates, not yet a proven exhaustive leaf-level inventory of every code/test action. Rows marked as broad inventory or matrix work must be expanded into concrete child tasks before that row is started; do not silently treat an umbrella row as a completed implementation plan.
- A fine task reaches `PASS` only when its stated acceptance evidence is recorded here or linked from its referenced test/report. Tests must exercise the named behavior, not merely compile the surrounding code.
- Close a coarse ID only when its original exit criteria and all mapped required fine tasks pass. Do not reopen a passed coarse ID without a reproduced regression or new required contract.
- A countable fine task must have one independently testable outcome, an explicit evidence target, and a bounded stopping point that can ordinarily be completed in one focused work session. Do not count a multi-backend inventory, whole conformance suite, or broad parity matrix as one leaf task. Complete the current granularity audit once; thereafter freeze the denominator unless an approved requirement changes.
- If a task proves unexpectedly broad, preserve its original ID as a non-counted parent and split only the concrete remaining outcomes; do not expand scope speculatively.

**Status vocabulary:** `NOT STARTED`, `IN PROGRESS`, `PASS`, `FAIL`, `BLOCKED`.

| Fine ID | Coarse ID | Status | Bounded task and acceptance evidence |
| --- | --- | --- | --- |
| FW-02-01 | DL-02 | PASS | The SQLite+Chroma authenticated HTTP protocol/authorization route contract is covered in `tests::sqlite_chroma_graph_topology_reads_are_scoped_and_acl_filtered`: successful canonical node resolution, missing-`read_graph` denial, fail-closed foreign-namespace handling, and metadata-only service rejection with `501 KOGWISTAR_API_CAPABILITY_NOT_CUT_OVER` (**1 passed**). Paired Python ACL/fusion/fixture tests: **5 passed**. |
| FW-02-01B | DL-02 | IN PROGRESS | Add a live-Chroma public-route smoke with the same request, scope, and ACL expectations as the fake-endpoint route test. CI pins `chromadb/chroma:1.5.9`; local Docker currently has no Chroma service running. Keep this as one bounded integration test and do not duplicate the already-passing fake-endpoint cases. |
| FW-02-02 | DL-02 | NOT STARTED | Resolve the remaining write-route contract, including `POST /api/search_index_hybrid`/add-index behavior and deferred syscall gaps. Add paired Python/Rust request-response tests before advertising each path. |
| FW-03-01 | DL-03 | NOT STARTED | Inventory public operations claimed by the capability crosswalk and map each to an executable Python/Rust evidence test; fail the drift test when a claimed operation lacks evidence. |
| FW-03-02 | DL-03 | NOT STARTED | Run the required per-operation conformance suite for the advertised routes and record the exact pass/unsupported set; retain `not-verified` for any untested route. |
| FW-05-01 | DL-05 | NOT STARTED | Extend paired Pydantic/Serde fixtures to remaining graph, storage, REST/MCP request and error DTOs, covering valid and invalid inputs with canonical output comparisons. |
| FW-05-02 | DL-05 | NOT STARTED | Verify identifier/span edge cases and schema-version evolution using shared fixtures; document compatibility behavior for additive and rejected changes. |
| FW-06-01 | DL-06 | NOT STARTED | Enumerate durable resources and state-changing entry points across Python, Rust and mixed mode; prove each operation has one transaction owner or an explicitly weaker backend contract. |
| FW-06-02 | DL-06 | NOT STARTED | Add failure-injection and reopen/retry tests for any authority/resource gaps found in FW-06-01; confirm no partial durable effects or duplicate commits. |
| FW-08-01 | DL-08 | NOT STARTED | Add live PostgreSQL graph-neighbor and graph-snapshot tests for tenant/project scope, source evidence and endpoint dependencies, including malformed or conflicting backend records. |
| FW-08-02 | DL-08 | NOT STARTED | Exercise the public read-effect matrix across supported backends, including ACL changes and stale/missing records; prove unauthorized data is absent from every result shape. |
| FW-08C-01 | DL-08C | NOT STARTED | Add shared Python/Rust cases for tombstone/redirect and source-evidence authorization in vector retrieval across PostgreSQL, SQLite+Chroma and in-memory. |
| FW-08C-02 | DL-08C | NOT STARTED | Verify mutation admission and tenant/project scope parity across the declared runtime/backend matrix; report unsupported combinations explicitly. |
| FW-11-01 | DL-11 | NOT STARTED | Complete trace-field parity for remaining runtime transitions, retries, suspend/resume and terminal errors; compare Python and Rust projections against shared fixtures. |
| FW-11-02 | DL-11 | NOT STARTED | Verify trace parentage and event-to-step linkage through authenticated inspection APIs on SQLite and PostgreSQL, including replay/reopen. |
| FW-12-01 | DL-12 | NOT STARTED | Finish explicit Python-only selector/no-native-loader tests for remaining REST and MCP operations; include legacy defaults and configured graph types. |
| FW-12-02 | DL-12 | NOT STARTED | Exercise Python-only two-stage indexing and retrieval through public interfaces on supported backends, verifying durable job state and vector/visualization readback. |
| FW-13-01 | DL-13 | NOT STARTED | Expand Rust-native executor conformance from the current bounded workflow kinds to each operation advertised as native; assert fixture parity and fail-closed unknown kinds. |
| FW-13-02 | DL-13 | NOT STARTED | Run the native executor matrix on each backend/runtime combination claimed as supported, covering durable restart, error routing and final graph/result digest. |
| FW-14-01 | DL-14 | NOT STARTED | Define the advertised mixed-language capability combinations from existing contracts and execute paired operation/scope tests for each combination. |
| FW-14-02 | DL-14 | NOT STARTED | Add recovery tests for mixed Python/Rust operations (crash, retry, reopen) and verify canonical final-state digests and no duplicate effects. |
| FW-14A-01 | DL-14A | NOT STARTED | Verify Rust-owned scheduling with Python sync and async workers across claim, result delivery, worker loss and scheduler restart; assert durable correlation and state continuity. |
| FW-14A-02 | DL-14A | NOT STARTED | Exercise resolver rejection, bounded JSON projection and process-local dependency isolation on both storage authorities; prove rejected results never reach durable state. |
| FW-14C-01 | DL-14C | PASS | Verified exact environment admission, mismatch rejection, and compatible-recipient-first ordering while preserving generic fallback: `tests/runtime/test_workflow_version_admission.py` and focused `test_python_worker_checks_environment_before_numerical_callback`, `test_environment_pinned_resolver_advertises_compatible_claim_recipients`, `test_async_environment_pinned_resolver_advertises_compatible_claim_recipients`; Rust `cargo test -p kogwistar-api sqlite_environment_pinned_work_resumes_on_another_compatible_worker --lib` (1 passed). Combined focused Python run: 42 passed. |
| FW-14C-02 | DL-14C | PASS | Verified wrong-environment work fails before callback/journal mutation, compatible worker B resumes the persisted workflow after service reopen, and Python sync/async worker journals replay prepared effects after restart/retry without re-running callbacks: `test_worker_environment_mismatch_does_not_poison_execution_journal`, Rust `sqlite_environment_pinned_work_resumes_on_another_compatible_worker` (1 passed), `test_worker_journals_callback_before_send_and_replays_after_restart`, and `test_async_worker_journals_awaited_callback_before_response_retry`; combined focused Python run: 42 passed. |
| FW-15-01 | DL-15 | NOT STARTED | Reconcile every explicit backend/runtime selector with the current supported-mode manifest; classify each fallback as intentional compatibility or an error. |
| FW-15-02 | DL-15 | NOT STARTED | For each Rust-selected operation still open in the selector inventory, inject missing-extension/unsupported-backend conditions and assert explicit failure before effects. |
| FW-15C-01 | DL-15C | NOT STARTED | Cover remaining workflow continuation and callback state-changing call sites with owner/selector guards and negative tests for cross-authority writes. |
| FW-15C-02 | DL-15C | NOT STARTED | Verify transaction/atomicity claims for async backends with and without transactional UoW; document and test fail-closed behavior where atomicity is unavailable. |
| FW-16-01 | DL-16 | NOT STARTED | Add derived-family contract and round-trip tests for remaining production node/edge families, including exact type, wire tag, endpoints and lifecycle fields. |
| FW-16-02 | DL-16 | NOT STARTED | Verify shared payload, digest and recovery parity for the derived families on advertised native and mixed paths, including invalid-payload rejection before mutation. |
| FW-16AC-01 | DL-16AC | NOT STARTED | Complete the shared ACL matrix for Rust graph reads: public, owner, private, scope, shared/group, latest-version, tombstone and no-record denial, paired against Python for nodes and edges. |
| FW-16AC-02 | DL-16AC | NOT STARTED | Add mutation and runtime/backend parity cases for graph reads, including same-string node/edge IDs and stale Chroma projection recovery across service restart; distinguish fake-endpoint from live-service evidence. |
| FW-16AM-01 | DL-16AM | NOT STARTED | Inventory remaining public MCP operations and compare Rust request/response, authorization, discovery and error behavior with Python; add paired contract tests for each advertised operation family. |
| FW-16AM-02 | DL-16AM | NOT STARTED | Verify remaining server-initiated event types, ingestion/parsing/indexing paths, backend paths and production transport conformance; mark unsupported combinations explicitly until exercised. |
| FW-16BQ-01 | DL-16BQ | NOT STARTED | Complete the syscall and REST/MCP parity inventory for advertised operations, verifying Python-compatible behavior across SQLite+Chroma, PostgreSQL and in-memory where supported. |
| FW-16BQ-02 | DL-16BQ | NOT STARTED | Exercise cross-backend execution and restart/retry behavior under the supported interpreter matrix; prove shared ACL, event, worker and final graph-state contracts without claiming universal parity beyond tested paths. |
| FW-17-01 | DL-17 | NOT STARTED | Execute required CI cells for all advertised Python and Rust support versions/platforms and record pass evidence; do not promote best-effort cells to required support without evidence. |
| FW-17-02 | DL-17 | NOT STARTED | Run downstream pinned-consumer and feature/regression checks for each advertised implementation path; confirm package metadata and documented support matrix agree. |
| FW-18-01 | DL-18 | NOT STARTED | Extend the shared versioned workload contract to representative mixed read/write traces and verify identical visibility and final digests across each measured implementation. |
| FW-18-02 | DL-18 | NOT STARTED | Validate batch atomicity boundaries and error/replay behavior per backend; retain per-commit sequential trace semantics and do not claim atomicity for unsupported stores. |
| FW-19-01 | DL-19 | NOT STARTED | Run the fixed-resource benchmark on larger representative fixtures for PostgreSQL/indexing/ACL/agent request paths with warm-up, repeated samples and resource limits recorded. |
| FW-19-02 | DL-19 | NOT STARTED | Add replay/recovery and supported-runtime cells to the benchmark report; verify digest parity and report medians, tail latency and sample variation. |
| FW-20-01 | DL-20 | NOT STARTED | Attribute endpoint latency and throughput to transport, authorization, storage, indexing and worker-transfer stages across representative sizes and backends. |
| FW-20-02 | DL-20 | NOT STARTED | Repeat resource-matched endpoint comparisons with correctness digests and confidence/variation; state only conclusions supported by those measurements. |
| FW-20A-01 | DL-20A | NOT STARTED | Join language-crossing, FFI, worker-dispatch, serialized-byte and transaction counters across representative graph/workflow/storage/indexing/MCP traces. |
| FW-20A-02 | DL-20A | NOT STARTED | Add batch-versus-per-item regression fixtures at representative sizes and define thresholds that detect accidental crossings or serialization loops without duplicating state effects. |

## Current Snapshot

The coarse checklist currently has 23 `IN PROGRESS` IDs. This worklist maps all 23 to 47 counted tasks; 3 are `PASS`, 1 is `IN PROGRESS`, and 43 are `NOT STARTED`. DL-02's protocol/authorization route contract is closed; live Chroma HTTP route evidence is separately active. Granularity for the other coarse gates remains under audit; refine broad rows before starting them, then freeze the denominator. Keep the completion snapshot at the top synchronized whenever either list changes, and keep the linked coarse inventory as the authority for coarse status.
