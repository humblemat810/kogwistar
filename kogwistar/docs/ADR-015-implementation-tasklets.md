# ADR-015 Remaining Implementation Tasklets

**Progress:** **1/11 PASS (9.1%)**. Six tasklets are `IN PROGRESS`; four are `NOT STARTED`.
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
| RT-02 | IN PROGRESS | Match successful and handled/unhandled failure effects: updates, errors, token consumption, successor admission, and terminal frontier cleanup. Paired tests compare canonical run state and event history. Partial evidence: `tests/runtime/test_runtime_parity_bridge_contract.py::test_python_workflow_failure_matches_rust_durable_state_and_step_event` compares Python workflow trace with Rust SQLite durable failure event/state; `tests/runtime/test_rust_runtime_checkpoint_slice.py` covers failure rollback/retry and terminal frontier cleanup. Focused run: **23 passed**. Successor/token parity and handled-failure end-to-end comparison remain open. |
| RT-03 | IN PROGRESS | Match suspension and resume of one exact token while preserving unrelated pending work and join obligations. Restart between suspend and resume; Python and Rust produce the same terminal state and ordered events. |
| RT-04 | IN PROGRESS | Match fan-out and join accounting under interleaved completion, retry, and cancellation. Paired deterministic trace ends with identical frontier, join counts, and terminal status. |
| RT-05 | IN PROGRESS | Prove result acceptance atomicity and idempotency across crash/retry boundaries: one accepted effect, event, projection update, acknowledgement, and successor set. Rust and Python reopen the same persisted state and agree. |

## PostgreSQL Graph Reads

Parent: ADR-015 `graph-pgvector`; current partial native-read boundary is
described in [implementation status](ADR-015-implementation-status.md).

| ID | Status | Tasklet and single acceptance oracle |
| --- | --- | --- |
| PG-01 | IN PROGRESS | Implement the documented Python-compatible complex metadata-filter query subset in the Rust PostgreSQL authority. Shared fixtures prove identical ordered IDs and filtering for supported operators; unsupported operators fail closed before Python/backend fallback. |
| PG-02 | IN PROGRESS | Move remaining graph-derived lookup reads explicitly listed as Python adapters onto the Rust PostgreSQL read authority. Paired fixtures prove identical IDs, scope, provenance, and ordering, with instrumentation proving no Python backend call in Rust mode. |

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
