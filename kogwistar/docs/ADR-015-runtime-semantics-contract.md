# ADR-015 runtime semantics contract

Updated: 2026-07-21

This document freezes the Python runtime behavior that the ADR-015 durable Rust
runtime must preserve. It is intentionally narrower than a full description of
`WorkflowRuntime`: it covers the worker/result boundary, routing, failure,
suspension, frontier, and idempotency rules needed for the current authority
cutover. Public Python APIs remain unchanged.

The Python source and parity tests named below are evidence, not independent
implementations to be approximated. A Rust change that cannot cite an executable
gate for one of these rules is incomplete.

## Worker request

| Normative rule | Python evidence | Executable gate | Rust obligation |
| --- | --- | --- | --- |
| Resolve a step by its frozen workflow `op`, never by assuming `node_id == op`. | `WorkflowRuntime.worker` reads `nodes[node_id].op` before calling `step_resolver(op)`. | `tests/runtime/test_rust_runtime_python_worker.py::test_worker_rejects_claim_without_frozen_workflow_op` | Every lane carries non-empty `op`; worker rejects a missing value before callback execution. |
| `StepContext` contains immutable run/workflow/node/token/step identity and a read-only state view. | `kogwistar.runtime.runtime.StepContext`; construction in `WorkflowRuntime.worker`. | Worker adapter tests in `test_rust_runtime_python_worker.py`. | Build context only from claimed lane payload; callback output cannot overwrite scheduler identity or frontier. |
| Callback execution occurs outside the Rust store transaction. | Python resolver execution precedes persistence; ADR-015 worker protocol separates claim/result. | Crash/replay journal tests in `test_rust_runtime_python_worker.py`. | Claim commits before callback. Result is submitted in a later transaction. |
| A callback whose side effects may have run but whose result was not journaled must not be run again automatically. | ADR worker journal contract. | `test_executing_journal_row_fails_closed_after_crash`. | Keep `executing` journal rows ambiguous; require operator reconciliation. |

## Result mapping

| Python result | Worker effect | Durable meaning |
| --- | --- | --- |
| `RunSuccess` | `status=success`, state updates, optional explicit route names and selected successors | Apply updates, consume executing token, plan validated successors, then complete only if frontier is empty. |
| `RunSuspended` | `status=suspended`, state updates, wait reason, resume payload | Move the exact pending token to suspended; preserve all unrelated pending/join work. |
| `RunFailure` with a handled explicit/predicate route | `status=failed`, errors, state updates, selected successors | Record failed step attempt and errors, consume token, continue on graph-valid successors; run remains running. |
| `RunFailure` without a handled route | `status=failed`, errors, state updates, no successors | Apply updates, clear unfinished scheduler frontier, and persist terminal failed status. |

`RunFailure` is therefore not unconditionally terminal. The authoritative Python
logic is the `status == "failure"` branch in `WorkflowRuntime.run`: explicit or
predicate routing handles a failure; otherwise `_enter_failed()` drains pending
and join work before terminal persistence.

Nested `workflow_invocations`, sandboxed resolver execution, lane-message
sends, and direct trace/event emission are not silently degraded. Until each
has a versioned worker protocol representation, the Rust worker adapter must
reject it before executing the callback (when discoverable from the resolver)
or reject the returned effect before persistence.

Async callbacks are represented by `worker_protocol="async-v2"`. This changes
only Python callback execution: the callback is awaited outside the store
transaction, then submits the same restricted durable worker-effect DTO as
`sync-v1`. The local SQLite worker journal is still the side-effect boundary:
a restart finding `executing` fails closed rather than awaiting the callback
again. A `sync-v1` worker rejects `async-v2`, and an async worker rejects any
other protocol. This permits no new runtime-state operation.

## State shape and process-local dependencies

`WorkflowState` is an open `dict[str, Any]`: workflow authors may add arbitrary
user keys. `ConversationWorkflowState` is separately a narrow JSON-friendly
checkpoint DTO. Do not make the generic runtime state a closed DTO merely
because one conversation workflow has known fields.

`_deps` and `dream_deps` are process-local runtime plumbing, never persisted
state. Pydantic treats underscore attributes as private, so an orchestrator
must attach `_deps` only after `WorkflowStateModel.model_dump()` and before
runtime admission. Checkpoint serialization drops it. This preserves live
resolver dependencies without serializing engines, callbacks, or clients.

## Routing

Frozen routes preserve at least:

- stable edge id;
- source and target node ids;
- target aliases (full id, short id, node label, and node op when present);
- join mask;
- predicate name;
- priority;
- default flag;
- multiplicity;
- source fan-out flag.

The normative selection order is implemented by
`kogwistar.runtime.routing.compute_route_next` and mirrored by
`kogwistar_runtime::select_runtime_route`:

1. Resolve explicit `_route_next` names against frozen aliases in caller order.
2. Otherwise evaluate named predicates; select by existing priority and
   fan-out/multiplicity rules.
3. A failure does not use unconditional/base routing. With no selected explicit
   or predicate edge, it is unhandled and terminal.
4. A non-failure may use unconditional/base routing.
5. If still unmatched, use a default edge.

The Python worker may evaluate dynamic Python predicates, but Rust validates every
reported target and join mask against the exact frozen graph. Static routing,
frontier mutation, token identity, and join accounting remain Rust-owned.

## Frontier and terminal state

| Normative rule | Evidence |
| --- | --- |
| A successful token preserves unrelated pending, suspended, and join work. | `frontier_after_worker_success` unit tests and sync/async runtime join tests. |
| Suspension parks one exact token without decrementing its downstream join obligations. | Python `status == "suspended"` branch; `frontier_after_worker_suspend`; suspend/resume parity suites. |
| Terminal completed/failed/cancelled state has an empty frontier. | `reduce_recorded_transition` rejects `TerminalFrontierNotEmpty`. |
| A terminal transition may reuse the last accepted step sequence but non-terminal transitions must advance it. | `validate_step_sequence` and `terminal_transition_may_share_last_accepted_step_only`. |
| Event history remains authoritative; current-state/checkpoint/status projections are disposable. | ADR-015 and runtime projection rebuild tests. |

## Idempotency and transaction boundary

One result transaction performs all of the following or none:

1. lock run and claimed lane;
2. exact `effect_id`/digest retry check;
3. read current-state projection, falling back to event replay only for repair;
4. reduce and append immutable transition event;
5. update current-state/checkpoint/status and server-run projections;
6. persist dedup result;
7. acknowledge claimed lane;
8. enqueue graph-valid successor lanes or persist terminal status.

An exact retry returns the persisted result without repeating telemetry or callback
effects. A changed digest, handoff, graph target, token, lease, or canonical
frontier order is a conflict.

## Exit evidence for this slice

This slice is complete only when all of these pass from one candidate identity:

- Rust runtime/store/API unit tests for success, suspend, handled failure,
  unhandled failure, forged successor, retry, and rollback;
- Python worker adapter tests for context construction and all three result types;
- existing sync/async routing, join, suspend/resume, and terminal parity tests;
- ADR-015 four compatibility layers.
