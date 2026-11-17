# ADR-015 Rust Capability Completion Worklist

**Scope:** Fixed tasklets for the six Rust capabilities still marked incomplete
in `contracts/rust-port-v1.json`. This is the active implementation and
verification worklist; it does not track external production rollout.
**Progress:** **14/31 PASS (45.2%)**; **11 IN PROGRESS**; **6 NOT STARTED**.
**Acceptance environment:** Kogwistar is a library. A clean, containerized
Docker UAT using the release candidate is its deployment-grade end-to-end
acceptance. No separate customer traffic fleet or percentage canary is required
to complete this worklist.
**Excluded:** `deterministic-contracts` and `sqlite-meta` (already ready),
`chroma-adapter` (intentionally Python-owned), downstream adopter rollout, and
the closed ADR-022 evidence inventories.

## Rules

- Fixed denominator: **31 leaf tasklets**. Do not add rows, split rows, or change
  the denominator without a newly approved requirement or demonstrated
  independently failing oracle. Do not reopen a PASS without regression or
  invalidated evidence.
- One row = one invariant, one capability/backend cell, one observable oracle,
  and one focused check. Parent capability names do not count as tasklets.
- `PASS` records behavioral evidence. Shared candidate-release checks remain
  separate and must pass before claiming the full library candidate verified.
- A false `rust_cutover_ready` value keeps Rust from becoming an implicit
  default; it is not evidence that a customer production canary is missing.
  Set it only when mapped capability tasklets and shared Docker/CI acceptance
  are verified.
- Explicit language selection must fail closed when its required authority is
  unavailable. Mixed execution is supported only at declared serialization
  seams; native Rust path claims require a measured empty seam list.
- Current-source reports must use one candidate identity. Docker UAT, supported
  interpreter CI, and backward-readable persisted data are release evidence.

## Tasklets

| ID | Capability | Status | Invariant, cell, and acceptance oracle | Evidence or next check |
| --- | --- | --- | --- | --- |
| PGS-01 | `postgres-sequence-event-log` | PASS | Python-created and Rust-created event schemas interoperate; sequence/cursor and replay results agree. | Current candidate Phase-3 gate: `test_rust_store_postgres_differential.py`, 4 passed; `.tmp/adr015-phase3-refresh-capabilities.json`. |
| PGS-02 | `postgres-sequence-event-log` | PASS | Rust owns each transaction; concurrent idempotency collision and injected rollback leave one canonical event stream. | Same 4-test live `pgvector:pg16` group; no split language transaction ownership. |
| PGS-03 | `postgres-sequence-event-log` | PASS | Commit succeeds but caller loses acknowledgement; after process restart, Python and Rust replay returns the committed sequence without duplicate events. | `tests/core/test_rust_store_postgres_differential.py::test_rust_postgres_lost_ack_replays_committed_event_after_reopen` passed against disposable `pgvector:pg16`; Python read and fresh Rust subprocess retry/replay both observed exactly sequence 1. |
| PGS-04 | `postgres-sequence-event-log` | PASS | Schema isolation and unknown-field rejection survive close/reopen and cross-language reads. | Same current candidate Phase-3 gate; `test_rust_postgres_schema_isolation_and_rejects_unknown_fields`. |
| PRJ-01 | `projections-snapshots-run-registry` | PASS | Named projection schema and atomic batch are readable/writable in both language directions. | Current candidate Phase-3 gate: projection differential tests, 2 passed. |
| PRJ-02 | `projections-snapshots-run-registry` | PASS | Run registry and workflow-history schemas created by either language remain readable and atomically writable by the other. | Current candidate Phase-3 gate: run-registry/workflow-history tests, 3 passed. |
| PRJ-03 | `projections-snapshots-run-registry` | PASS | Bounded recovery/rebuild after reopen yields Python/Rust interoperable projection state. | Current candidate Phase-3 gate: `test_postgres_atomic_bounded_recovery_rebuild_and_python_rust_interop`, passed. |
| QLL-01 | `queues-leases-lanes` | PASS | PostgreSQL queue and lane rows preserve the shared Python/Rust wire contract. | Current candidate Phase-3 queue/lane group, 3 passed. |
| QLL-02 | `queues-leases-lanes` | PASS | Concurrent claims, lease reclaim, and terminal retry remain exclusive and idempotent across runtimes. | Current candidate Phase-3 queue/lane group; live PostgreSQL concurrency cases passed. |
| QLL-03 | `queues-leases-lanes` | NOT STARTED | Worker effect commits but ACK is lost; restart/reconciliation accepts exactly one effect and safely completes the durable lane. | Add live-PostgreSQL crash/lost-ACK test spanning queue, effect, and lane completion. |
| QLL-04 | `queues-leases-lanes` | NOT STARTED | Run, step, message, and trace correlation identifiers remain identical across Python/Rust enqueue, claim, result, and ACK. | Add one shared-fixture trace assertion to the PostgreSQL queue/lane differential test. |
| GRF-01 | `graph-pgvector` | PASS | Graph mutation event, node/edge state, and transaction rollback remain mutually readable across Python/Rust. | Current candidate Phase-3 graph/transaction group, 47 passed. |
| GRF-02 | `graph-pgvector` | IN PROGRESS | Node read and mutation enforce the same tenant, namespace, graph-space, and ACL decision in Python and Rust. | Complete a shared valid/denied fixture for both authorities; prove denied requests cause no writes. |
| GRF-03 | `graph-pgvector` | IN PROGRESS | Edge read/hydration and source-evidence visibility preserve endpoint integrity, source spans, and authorization across authorities. | Complete paired fixtures for edge endpoints and evidence spans; assert no hidden-source leakage. |
| GRF-04 | `graph-pgvector` | IN PROGRESS | Vector query filters, metric, ordering, and returned distances match Python without Rust-to-Python fallback. | Complete shared PostgreSQL query fixtures for supported filters, ranking ties, and unsupported-form rejection. |
| GRF-05 | `graph-pgvector` | IN PROGRESS | A graph write's indexing intent and vector projection recover after reopen without violating the canonical transaction boundary. | Finish restart/rebuild parity for index jobs and vectors; assert Python worker seam only when configured. |
| WFR-01 | `workflow-runtime` | PASS | Route selection, transitions, failure handling, fan-out, join, and terminal frontier match frozen Python semantics. | `ADR-015-implementation-tasklets.md` RT-01/02/04; Rust/Python differential tests passed. |
| WFR-02 | `workflow-runtime` | PASS | Suspension/resume preserves exact token, unrelated pending work, and join obligations across reopen. | `ADR-015-implementation-tasklets.md` RT-03; Python/Rust ordered-history and reopen tests passed. |
| WFR-03 | `workflow-runtime` | PASS | Worker claim/result, replay, duplicate delivery, rollback, and persisted checkpoint state interoperate across Python/Rust. | `ADR-015-implementation-tasklets.md` RT-05; handoff suite 7 passed. |
| WFR-04 | `workflow-runtime` | IN PROGRESS | A declared Rust-native workflow completes ingress-to-persistence with no Python worker and trace `seam_crossings=[]`. | Extend native answer smoke to assert authority, durable result, and empty seams in Docker UAT. |
| SRV-01 | `server-rest-sse-mcp-cli` | PASS | Implemented REST/SSE/MCP operations preserve contract, authentication, scope, and error behavior. | Phase-5 server gate recorded 197 passed; current-candidate revalidation is REL-01/REL-02. |
| SRV-02 | `server-rest-sse-mcp-cli` | IN PROGRESS | `send_message` admits a turn in Rust without a Python syscall worker; owner, ACL, idempotency, event/lane writes stay atomic. | Finish native handler plus paired REST/MCP replay and denial tests. |
| SRV-03 | `server-rest-sse-mcp-cli` | IN PROGRESS | `receive_message` serves scoped transcript/result from Rust-native storage with Python-compatible ordering and projection. | Finish native handler plus cross-language reopen fixture and unauthorized-scope test. |
| SRV-04 | `server-rest-sse-mcp-cli` | IN PROGRESS | `mount_memory` resolves snapshot and ACL natively with Python public response contract. | Finish native HTTP/MCP path plus owner/non-owner/namespace tests after reopen. |
| SRV-05 | `server-rest-sse-mcp-cli` | IN PROGRESS | `invoke_tool` dispatches supported Rust-native built-ins through host authorization; model output cannot grant permission. | Complete allow/deny tool fixtures, audit evidence, and explicit Python seam behavior for unsupported tools. |
| SRV-06 | `server-rest-sse-mcp-cli` | NOT STARTED | Installed Rust CLI operation and error responses match documented Python CLI contract. | Add package-level CLI smoke: help, one success, one structured failure. |
| REL-01 | All six | NOT STARTED | One candidate identity passes core, parser, sink, and application compatibility layers with exact coverage and no required skips. | Run persisted four-layer Docker harness on release candidate; retain merged report and identity. |
| REL-02 | All six | NOT STARTED | Clean Docker consumer installs candidate wheel, imports native extension, passes `pip check`, performs Rust→Python→Rust persisted-data handoff, and exercises Python rollback. | Run deterministic consumer UAT against that wheel and durable database. |
| REL-03 | All six | NOT STARTED | Supported release CI passes CPython 3.12–3.14 and PyPy 3.11 for declared package/test lanes. | Verify GitHub checks for exact candidate commit; label best-effort lanes, do not count them as required passes. |
| REL-04 | All six | IN PROGRESS | Python-only path never enters Rust; Rust-native path never starts Python; mixed path crosses only declared serialized seams. | Extend/execute boundary guards for public selectors and Rust HTTP container; assert no silent fallback. |
| REL-05 | All six | IN PROGRESS | Resource-matched Python, mixed, and seam-free Rust end-to-end paths preserve output parity and report throughput/latency/RSS. | Refresh benchmark on release candidate and Docker resource profile; report seam count, not microbenchmark extrapolation. |

## Closure

A capability is complete when all its tasklets pass and shared REL tasklets pass.
This worklist does not require external customer traffic, percentage rollout,
or a production canary. Completing library acceptance does not silently change
the default language: Python and Rust remain selectable; any default-policy
change must be explicit and preserve the Python path and backward-readable data.
