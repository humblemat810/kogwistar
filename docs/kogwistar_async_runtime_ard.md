# ARD: Dual Runtime Compatibility & Async Runtime Design

Companion checklist: `docs/kogwistar_async_runtime_impl_checklist.md`

## 1. Overview

Kogwistar introduces a new `AsyncWorkflowRuntime` alongside the existing
thread-based `WorkflowRuntime`.

The system must support one unified workflow design model executed by
multiple runtime implementations.

The existing `WorkflowRuntime` is the semantic reference until async parity
is proven by tests.

---

## 2. Goals

- Single workflow language
- Resolver compatibility
- Stronger execution semantics for task-based concurrency
- Backend-aware durability
- No semantic drift between sync, async, replay, and LangGraph semantics mode

---

## 3. Non-Goals

- Replacing the existing sync runtime as the default before parity is proven
- Changing workflow graph schema
- Changing resolver return semantics
- Making Chroma transactional
- Treating LangGraph visual export as semantic parity

---

## 4. Runtime Architecture

Conceptual execution shape:

```text
WorkflowRun -> Token -> Node -> ExecutionAttempt -> Executor -> StepRunResult
```

Definitions:

- `WorkflowRun`: one execution of one workflow design.
- `Token`: one execution path through the workflow graph.
- `Node`: workflow design node selected for execution.
- `ExecutionAttempt`: one try of one node for one token.
- `Executor`: sync or async step function invocation.

The async runtime must preserve the same graph design contract as
`WorkflowRuntime`:

- workflow design nodes live in `workflow_engine`
- execution trace, step execs, checkpoints, and terminal artifacts live in
  `conversation_engine` or the configured trace engine
- workflow edges are ordered deterministically by priority

---

## 5. Resolver Contract

The runtime family supports:

- `SyncStepFn`: normal callable returning `StepRunResult`
- `AsyncStepFn`: awaitable callable returning `StepRunResult`

Resolver outputs must still be one of the existing `StepRunResult` variants,
such as success, failure, or suspension.

Exceptions from resolvers must become `RunFailure`-style results. A task
exception must not bypass runtime bookkeeping.

The async runtime may adapt sync handlers by executing them through a safe
sync bridge. Blocking sync handlers must not stall unrelated async tasks.

Required invariant:

```text
default_sync_ops == default_async_ops
```

This means operation coverage must stay identical even when execution
mechanics differ.

---

## 6. Token, Fanout, and Join Semantics

Async execution must preserve current token semantics:

- fanout continues the first branch on the current token
- additional branches receive child tokens
- child tokens record `parent_token_id`
- convergence is not automatically joined
- a node executes once per arriving token unless the workflow declares an
  explicit join/barrier

Join semantics remain explicit:

- join nodes are detected by `wf_join=True` or equivalent runtime rule
- join bookkeeping is runtime-owned state
- checkpoints must include enough join state to resume without double-running
  released work

Duplicate downstream execution is allowed for workflows without explicit
joins. Idempotent writes reduce damage, but they are not the primary join
mechanism.

---

## 7. State, Checkpoint, and Replay Semantics

Async runtime must share state merge semantics with:

- `WorkflowRuntime.apply_state_update`
- `kogwistar.runtime.replay`
- LangGraph converter `execution="semantics"`

The following must remain true:

- accepted state changes are applied in deterministic scheduler order
- checkpoint schema stays compatible unless explicitly versioned
- `_deps` stays process-local and is not checkpointed
- runtime-owned bookkeeping such as token and join state is checkpointed when
  needed for resume
- wall-clock timestamps are audit facts only
- replay ordering is based on sequence fields such as `seq` or `step_seq`, not
  `created_at_ms`

---

## 8. Graph Side-Effect Parity

For existing workflows, sync and async runtimes must produce the same accepted
graph side effects when given the same workflow design, same initial state, same
resolver outputs, and same backend profile.

This means parity is checked over:

- final accepted workflow state
- persisted workflow run nodes
- persisted step execution nodes
- checkpoint nodes
- runtime trace edges
- terminal status artifacts
- emitted durable runtime events that are part of replay or server progress

Comparison must normalize volatile fields such as:

- wall-clock timestamps
- durations
- worker/thread/task names
- internal scheduling diagnostics
- backend-assigned sequence values when ordering equivalence is already proven

Async-specific lifecycle details may exist, but they must not change accepted
semantic graph output for workflows that do not opt into async-only behavior.

Examples of async-specific details:

- cooperative task cancellation markers
- task lifecycle telemetry
- await/scheduling diagnostics

Those details may appear as observability events, but they must be explicitly
classified as runtime diagnostics, not workflow semantic side effects.

Side-by-side parity tests should run the same workflow through both runtimes in
isolated engines, normalize volatile fields, and compare the resulting graph
node/edge sets plus final state.

---

## 9. Suspend, Resume, and Cancellation

Suspension is a first-class runtime result, not an error.

Async runtime must preserve:

- suspended token identity
- suspended node id
- suspended wait reason
- checkpoint data needed to continue from the next pending node
- client-supplied resume result application semantics

Cancellation must be cooperative and durable:

- no new steps should be scheduled after cancellation is accepted
- in-flight tasks are allowed to finish or be cancelled according to policy
- terminal cancelled artifacts must be persisted once
- task cancellation must not leave corrupted checkpoint or partial trace state
- cooperative task cancellation is an async-specific runtime behavior and is
  excluded from graph side-effect parity with the current thread runtime, except
  for accepted terminal state and durable user-visible artifacts

---

## 10. Nested Workflow Semantics

Nested workflow invocation must remain compatible:

- parent workflow may invoke a child workflow
- child result is applied to parent state through the same merge semantics
- trace/event emitters must not duplicate sink writers accidentally
- cancellation and failure in a child must produce deterministic parent behavior

The async runtime must make parent/child task ownership explicit so a parent
cannot finish while required child work is still pending.

---

## 11. Backend Contract

Backend durability follows existing storage/runtime rules:

```text
Postgres = strong transactional
Chroma = event-first eventual
In-memory = volatile, no crash recovery promise
```

For Postgres-backed engines, runtime step persistence should share a backend
transaction where supported.

For Chroma-backed engines, SQL/event truth must be committed before derived
projection repair. Chroma projection convergence is replayable and idempotent.

For async execution, unit-of-work state must be task-local. Concurrent async
tasks must not accidentally share one transaction unless explicitly designed to
do so. Nested unit-of-work scopes in the same task may join as they do today.

---

## 12. Trace and Observability

Async runtime must preserve trace artifact shape:

- workflow run nodes
- step execution nodes
- checkpoint nodes
- runtime edges
- progress events used by server APIs

Trace persistence failures remain policy-controlled. Best-effort telemetry
must not silently corrupt accepted workflow state.

Task lifecycle should be observable enough to explain:

- scheduled
- started
- completed
- failed
- suspended
- cancelled

---

## 13. LangGraph Alignment

Only LangGraph converter `execution="semantics"` participates in semantic
parity claims.

`execution="visual"` may diverge and must not be used to prove runtime,
checkpoint, replay, or durability correctness.

Shared reducer logic is preferred. Duplicated reducer logic must be covered by
parity tests until it can be removed.

---

## 14. CI and Migration Principle

Async runtime should land behind explicit opt-in selection.

Fast CI should pin import boundaries, operation coverage, resolver behavior,
state merge behavior, and a tiny linear runtime smoke. Slower `ci_full` tests
should cover backend durability and Chroma/Postgres repair behavior.

---

## 15. Final Principle

One workflow design. Multiple runtimes. Same semantics.
