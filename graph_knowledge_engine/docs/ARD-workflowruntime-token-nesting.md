# ARD: WorkflowRuntime token nesting and convergence behavior (Conversation Orchestration)

Date: 2026-02-26  
Status: **Accepted (documenting current behavior)**  
Scope: `workflow/runtime.py` + the current conversation workflow design used by `ConversationOrchestrator.add_conversation_turn_workflow_v2(...)`.

## Context

The conversation system is split into:

- **Workflow design graph** (stored in `workflow_engine`, `kg_graph_type="workflow"`)
- **Execution trace + checkpoints** (stored in `conversation_engine`, `kg_graph_type="conversation"`)

`ConversationOrchestrator.add_conversation_turn_workflow_v2(...)` runs a workflow design (from `ConversationWorkflowDesigner.ensure_add_turn_flow(mode="full")`)
using `WorkflowRuntime.run(...)`.

The current workflow design includes a **fanout** node (`wf_fanout=True`) for KG retrieval (`op="kg_retrieve"`), and routes to optional pin steps
(`memory_pin`, `kg_pin`) that can run in parallel before continuing to `answer`.

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

### 3) Conversation workflow tolerates convergence by idempotent side effects

In the current conversation workflow chart:

- `kg_retrieve` is a fanout node and may route to both `memory_pin` and `kg_pin`.
- Both pin steps then route to `answer`.

This means it is possible for **multiple tokens** to reach `answer`.

Instead of introducing a join barrier today, the current system tolerates this by relying on:

- **Deterministic IDs** (stable IDs for conversation nodes/edges/pointers) so repeated attempts collide rather than duplicate.
- **Idempotent engine writes** (add/upsert semantics where applicable).
- “Best-effort” linking operations that can be attempted more than once without corrupting the chain.

This choice keeps the workflow design simple and avoids early join semantics while still enabling fanout parallelism.

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
- **Simplicity**: avoid introducing join semantics until we have stronger guarantees about state merging rules.
- **Safety**: deterministic IDs + idempotent writes reduce the risk of divergence when nodes are executed multiple times.

## Consequences

### Positive
- Fanout is supported with clear lineage and checkpointable routing.
- Execution traces can reconstruct a tree of work for a single turn.
- The workflow designer can declare fanout without committing to join semantics.

### Negative / Risks
- Without explicit join nodes, convergence can trigger duplicate execution of downstream steps (e.g., `answer`).
- Duplicate executions can increase cost (extra LLM calls) even if writes are idempotent.
- State updates may race if two tokens update overlapping keys; last-writer-wins semantics can hide data.

### Mitigations in current version
- Prefer downstream steps that are **idempotent** and/or guarded by deterministic IDs.
- Keep fanout branches writing into disjoint state keys when possible.
- If a downstream step must be “exactly-once”, introduce an explicit join node, or add a guard inside the step.

## Future Work (not required for current correctness)

1) Add an explicit `join` step before `answer` if we want “answer runs once even if both pins run”.
2) Define a formal “state merge” strategy for join nodes:
   - merge-by-key, append-only lists, conflict detection, etc.
3) Add per-step “exactly-once” guards (e.g., check if `answer.response_node_id` already exists in state).

## References (files)

- `workflow/runtime.py` (token spawning + parent_token_id, join detection via `wf_join`)
- `workflow/design.py` (conversation workflow chart: `kg_retrieve` fanout → pins → answer)
- `conversation_orchestrator.py` (v2 runner wiring: designer + runtime + deps injection)
