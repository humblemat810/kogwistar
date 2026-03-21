# ARD: WorkflowRuntime token nesting and convergence behavior (Conversation Orchestration)

Date: 2026-02-26  
Status: **Accepted (documenting current behavior)**  
Scope: `kogwistar/runtime/runtime.py` + the current conversation workflow design used by `ConversationOrchestrator.add_conversation_turn_workflow_v2(...)`.

Updated: 2026-03-08

## Context

The conversation system is split into:

- **Workflow design graph** (stored in `workflow_engine`, `kg_graph_type="workflow"`)
- **Execution trace + checkpoints** (stored in `conversation_engine`, `kg_graph_type="conversation"`)

`ConversationOrchestrator.add_conversation_turn_workflow_v2(...)` runs a workflow design (from `ConversationWorkflowDesigner.ensure_add_turn_flow(mode="full")`)
using `WorkflowRuntime.run(...)`.

The current workflow design includes a **fanout** node (`wf_fanout=True`)
for KG retrieval (`op="kg_retrieve"`), optional pin steps, and an
explicit join barrier at `answer` (`wf_join=True`).

## Decision

### 1) Token-based nesting for fanout

`WorkflowRuntime` models *parallel branches* using **token ids**:

- A token represents one “path of execution” through the workflow graph.
- When a step selects multiple next nodes (fanout), the runtime:
  - **continues** the first next node using the same `token_id`
  - **spawns child tokens** (new `token_id`) for the remaining next nodes
  - records `parent_token_id` on child tokens to preserve lineage

This creates a **nesting tree** of tokens:
- root token (from start)
- child tokens for parallel branches
- grandchildren for fanout inside fanout, etc.

This nesting is *purely runtime execution metadata*; it does not change the workflow design graph.

### 2) Convergence is not automatically joined unless a join node exists

The runtime does **not** automatically merge tokens that converge on the same node **unless** the workflow design includes an explicit join/barrier node.

- Join nodes are detected by node metadata `wf_join=True` or by `op == "join"`.
- Without join nodes, if two tokens reach the same downstream node, the node may be executed multiple times (once per token).

### 3) Current conversation workflow uses an explicit join at `answer`

In the current conversation workflow chart:

- `kg_retrieve` is a fanout node and may route to both `memory_pin` and `kg_pin`.
- `kg_retrieve` may also route directly to `answer`.
- `answer` is marked `wf_join=True`, and inbound convergence is intended
  to be barriered there.

So multiple tokens may reach the `answer` join point, but the current
design no longer treats duplicate `answer` execution as the intended
steady-state behavior.

Deterministic IDs and idempotent writes are still important, but they
are now secondary safety mechanisms rather than the primary convergence
strategy for `answer`.

The runtime currently relies on:

- **Deterministic IDs** (stable IDs for conversation nodes/edges/pointers) so repeated attempts collide rather than duplicate.
- **Idempotent engine writes** (add/upsert semantics where applicable).
- explicit join bookkeeping (`_rt_join`, may-reach join bitsets, waiter release)
  in `WorkflowRuntime`.

This allows fanout parallelism while keeping the main downstream answer
path join-governed.

### 4) Checkpoint/resume preserves token nesting

Runtime persistence stores enough information to resume in-flight work:

- `_deps` is injected for dependency resolution and is intentionally **non-checkpointed** (process-local).
- `_rt_join` is runtime-owned bookkeeping to support fanout/barrier capability tracking and checkpoint/resume.

The runtime persists:
- scheduled/pending tokens
- in-flight tokens
- each token’s `(token_id, parent_token_id)`
- the current “outstanding obligations” counters for join nodes (when present)

This allows resuming with correct nesting lineage and correct barrier counters.

## Rationale

- **Observability**: token_id + parent_token_id yields a natural execution tree for tracing and debugging.
- **Parallelism**: fanout allows independent pin steps to run concurrently.
- **Simplicity**: keep join semantics explicit and opt-in at workflow design points rather than auto-merging every convergence.
- **Safety**: deterministic IDs + idempotent writes reduce the risk of divergence when nodes are executed multiple times.

## Consequences

### Positive
- Fanout is supported with clear lineage and checkpointable routing.
- Execution traces can reconstruct a tree of work for a single turn.
- The workflow designer can declare fanout without committing to join semantics.

### Negative / Risks
- Workflows without explicit join nodes can still trigger duplicate execution of downstream steps.
- Duplicate executions can increase cost (extra LLM calls) even if writes are idempotent.
- State updates may race if two tokens update overlapping keys; last-writer-wins semantics can hide data.

### Mitigations in current version
- Use explicit join nodes (`wf_join=True`) where downstream execution must converge once.
- Prefer downstream steps that are **idempotent** and/or guarded by deterministic IDs.
- Keep fanout branches writing into disjoint state keys when possible.
- If a downstream step must be “exactly-once”, introduce an explicit join node, or add a guard inside the step.

## Future Work (not required for current correctness)

1) Define a formal “state merge” strategy for join nodes:
   - merge-by-key, append-only lists, conflict detection, etc.
2) Extend explicit join/barrier policy to any future convergence points beyond `answer`.
3) Add per-step “exactly-once” guards where semantic idempotence is stronger than write idempotence.

## References (files)

- `kogwistar/runtime/runtime.py` (token spawning + parent_token_id, join detection via `wf_join`)
- `kogwistar/conversation/designer.py` (conversation workflow chart and join metadata)
- `kogwistar/conversation/conversation_orchestrator.py` (v2 runner wiring: designer + runtime + deps injection)
