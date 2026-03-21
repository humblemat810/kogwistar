# ARD-P2D Appendix A: State, Effects, Retries, Ordering, and Chain Semantics

**Status:** Accepted as workflow/conversation governance appendix  
**Last Updated:** 2026-03-14  

**Purpose:** This appendix pins down the â€œhard rulesâ€ for WorkflowState, branching, retries, ordering, and how the *main conversation chain* remains linear while still allowing fanout/sidecars and multiple in-chain events per user turn.

This appendix is designed to prevent future drift between:
- v1 linear orchestrator execution
- v2 workflow execution (including fanout)
- future transpiled runtimes (e.g., LangGraph)

## Current Implementation Note (2026-03-08)

The runtime and conversation stack already implement parts of this
appendix:

- `kogwistar/runtime/runtime.py` reserves `_deps` and
  `_rt_join` and applies state updates via explicit `'u'`, `'a'`, and
  `'e'` operations.
- `kogwistar/conversation/resolvers.py` uses structured
  state keys such as `memory`, `kg`, `memory_pin`, `kg_pin`, and
  `answer`.
- `kogwistar/conversation/designer.py` marks the current
  conversation workflow with explicit `wf_join` metadata where join
  semantics are intended.

This appendix remains normative. Not every rule here is enforced
generically by the runtime yet.

---

## A1. Vocabulary

### Design Workflow Graph
The static DAG of *step intent* (nodes, edges, predicates, joins, fanout).

### Runtime Execution Graph
The dynamic execution trace (tokens, step execs, checkpoints).

### Conversation Graph (Domain Artifact)
The persisted domain graph: conversation turns, summaries, pins, snapshots, tool artifacts, errors.

> **Reminder:** Design workflow graph â‰  Conversation graph.

---

## A2. WorkflowState Structure (Checkpoint-safe)

WorkflowState must contain **all meaningful intermediate results** and **all intended side-effects** in a replayable form.

### A2.1 Reserved top-level keys (convention)
- `state["turn"]` â€” canonical identifiers and anchoring context for the *current user input epoch*.
- `state["artifacts"]` â€” jsonable outputs (retrievals, answer payloads, pins, snapshots, etc.).
- `state["effects"]` â€” append-only effect descriptors (what was/will be applied to the conversation graph).
- `state["errors"]` â€” structured errors (also append-only).
- `state["_deps"]` â€” dependency injection (non-checkpointed).

### A2.2 Branch-safe namespaces
Fanout branches MUST NOT clobber each other. Use namespaces:
- `state["artifacts"]["sidecars"][<sidecar_name>][...]`
- `state["errors"]["sidecars"][<sidecar_name>][...]`
- `state["effects"]` is global but must be merge-ordered deterministically (see A5).

---

## A3. Main Chain Semantics: â€œEpochâ€ and Multiple In-Chain Events

### A3.1 Epoch definition
An **epoch** is the period after a user input is accepted and before control is yielded back to a new user input (or the run terminates).

Within one epoch, the system may emit multiple **in-chain events**, e.g.:
- assistant answer turn
- summarization turn/node
- â€œthinking/progress so farâ€ events (future)
- policy interrupt / halt event (future)

### A3.2 Linearity rule
The **main conversation chain is always linear** within an epoch:
- there is a single canonical chain anchor for the epoch (see A4)
- in-chain events append in a deterministic order (see A5)
- sidecars may run concurrently but do not break linearity

### A3.3 In-chain vs non-in-chain nodes
- **In-chain**: `in_conversation_chain=True` (and often `in_ui_chain=True`) and must participate in the linear `next_turn` chain.
- **Non-in-chain**: sidecar artifacts, tool details, prefetches, internal notes, etc. They may form their own subgraphs/chains but are **not** part of the main `next_turn` chain.

### A3.4 â€œMergingâ€ sidecars into main chain
Sidecars MUST NOT implicitly become in-chain.
They may influence the main chain only via explicit bridging steps, e.g.:
- `receive_sidecar_message`
- `check_interrupts`
- `apply_policy_interrupt`

---

## A4. Backbone Contract: `turn_backbone` step

`turn_backbone` is a workflow step (not â€œoutside leaked logicâ€). It establishes the canonical anchor for the epoch.

### A4.1 Required writes (write-once per epoch unless tombstoned)
`state["turn"]` MUST include:
- `conversation_id`
- `user_id`
- `epoch_id` (stable identifier for this user input epoch)
- `user_turn_index` (or equivalent)
- `prev_chain_turn_id` (tail in main chain)
- `user_turn_node_id` (deterministic)
- `user_turn_next_edge_id` (deterministic edge id from prev â†’ user turn, if prev exists)

### A4.2 Allowed subsequent in-chain emissions
Other steps MAY append additional in-chain events (assistant, summary, progress) **but must not rewrite** the backbone identifiers above.

### A4.3 Conflict rule
If `turn_backbone` is (re)executed:
- it must compute the same IDs and re-apply idempotently (upsert/no-op), OR
- fail fast if it would create a different backbone for the same epoch.

---

## A5. Deterministic Ordering (Design-defined, Configurable)

Concurrency means â€œfinish orderâ€ is nondeterministic. Therefore all attachments and merges use a **design-defined order key**.

### A5.1 Default order key (recommended)
Order by the tuple:
1. `wf_priority` (int; default 0; lower runs earlier)
2. `topo_index` (deterministic topological position in design DAG)
3. `step_id` (stable string id)
4. `token_path` (stable fanout path, e.g. `[0,1,0]`)

This yields deterministic â€œDFS-likeâ€ behavior even under parallel execution.

### A5.2 Configurability
Conversation compiler may override ordering by setting:
- `wf_priority`
- or explicit â€œattachment policyâ€ metadata

Default runtime/resolvers remain general; conversation design specializes ordering.

---

## A6. Retries and â€œretryâ€ Relationship Artifacts

### A6.1 Retrieval retries are first-class conversation records
If `memory_retrieve` runs multiple times in an epoch:
- it creates **multiple retrieval record nodes**
- each subsequent attempt links to the previous via an edge:
  - relation: `retry`
  - direction: `attempt_n -> attempt_(n-1)` (or `previous_attempt`), choose one and keep consistent

### A6.2 State representation
Store all attempts in state as an append-only list:
- `state["artifacts"]["memory_retrievals"] = [Attempt, Attempt, ...]`
- each Attempt must be checkpoint-safe (jsonable)

Each Attempt includes:
- `attempt_id` (deterministic from order key + step id + token path)
- `status` in `{ "ok", "err", "timeout" }`
- `value` (jsonable payload) if ok
- `error` (structured) if err/timeout
- `node_id` (deterministic) if materialized into conversation graph
- `retry_edge_id` linking to prior attempt (if any)

Same pattern applies to `kg_retrieve`.

---

## A7. Error Propagation and â€œAbort vs Continueâ€ Policies

### A7.1 Structured result model
Steps should communicate policy-bearing outcomes via structured runtime results:
- `RunSuccess`
- `RunFailure`
- `RunSuspended`

Raised exceptions are still allowed, but they should be treated as runtime/infrastructure failures unless the resolver explicitly converts them into a structured result.

### A7.2 Recoverable vs failed execution outcomes
The key distinction is whether the run can continue immediately:

- **Recoverable / user-fixable**
  - model as `RunSuspended`
  - the run pauses and waits for external correction or input
  - examples:
    - sandboxed code references an undefined variable and a user/agent can fix the code
    - a required business field is missing and must be supplied by a user
    - approval is required before continuing

- **Failed execution result**
  - model as `RunFailure`
  - the step has completed with a failed outcome, but that failure is still valid input to workflow routing policy
  - examples:
    - tool invocation failed and should route to fallback/help/abort
    - retrieval failed and should route to retry or degraded-answer logic
    - sandbox infrastructure failed and should route to retry or terminate

### A7.3 Standard recoverable-error suspension payload
For recoverable/user-fixable errors, `RunSuspended.resume_payload` should use this minimum shape:

- `type`: `"recoverable_error"`
- `op`: workflow op name
- `category`: symbolic code such as `sandbox_code_error` or `missing_input`
- `message`: one human-readable summary
- `errors`: list of concrete error strings
- `repair_payload`: opaque client/agent-specific repair context

This payload is transport-oriented: it tells the external fixer what happened and what is needed to resume.

### A7.4 Routing semantics for `RunFailure`
`RunFailure` is routable.

The runtime must:
- apply the failure result state updates before routing
- persist the step execution with failed status
- evaluate normal outgoing workflow policy using predicates and/or `next_step_names`

If a failure result produces a matching route:
- continue through the selected recovery / retry / help branch

If a failure result produces no route:
- terminate the run with `status="failure"`

So `RunFailure` is neither “always hard-stop” nor “always recoverable”. The graph decides, and unmatched failure ends failed.

### A7.5 Resume semantics for recoverable errors
`resume_run(...)` must accept all structured result types from the external fixer:

- `RunSuccess`
  - continue from the suspended node using success semantics

- `RunFailure`
  - continue from the suspended node using failure-routing semantics
  - if no branch matches, end failed

- `RunSuspended`
  - re-park the same token at the same node
  - replace the prior `resume_payload`
  - allow multi-turn repair loops without changing run identity

The resumed result is the `last_result` seen by downstream predicates exactly as if the step had produced it in-process.

### A7.6 Design guidance
Use these patterns:

- Use `RunSuspended` when the system needs outside help before it can decide the next step.
- Use `RunFailure` when the step has a failed outcome that downstream workflow policy should inspect and route on.
- Let unmatched `RunFailure` terminate the run as failure by default.

---

## A8. Chain-visible Mutations and Tombstones

The conversation system is append-first; â€œmutationsâ€ are modeled as explicit tombstone/replace effects.

### A8.1 What is chain-visible?
Chain-visible elements include (non-exhaustive):
- `next_turn` edges
- `in_conversation_chain`, `in_ui_chain` flags on nodes
- tail metadata (if persisted)
- canonical epoch backbone identifiers (`state["turn"]` keys in A4)

### A8.2 Allowed changes
Instead of mutating existing chain-visible artifacts, produce:
- `tombstone(old_id)`
- `add_node(new_id, ...)`
- `add_edge(new_edge_id, ...)`

All tombstones and replacements must be recorded in `state["effects"]`.

### A8.3 Multiple in-chain events are allowed
This rule does NOT mean â€œonly one in-chain node per epochâ€. It means:
- backbone identifiers are stable for the epoch
- additional in-chain events append linearly and deterministically
- replacements use tombstones, never silent mutation

---

## A9. State Reducer Rules (Branch Merge)

When branches merge (join or terminal aggregation), reduce state deterministically.

### A9.1 Key classes
- **Append-only lists**: concatenate then sort by order key (A5).
  - `effects`, `errors`, `*_retrievals`, logs
- **Write-once per epoch**: must match or error.
  - `state["turn"]` backbone keys (A4.1)
- **Namespaced maps**: merge by namespace; conflicts only within same namespace.
  - `state["artifacts"]["sidecars"]`

### A9.2 Recommendation
Make the reducer explicit and test it. Do not rely on Python dict â€œlast write winsâ€ semantics for cross-branch merges.

---

## A10. Design-time â€œConversation Lintâ€ (Non-blocking for general runtime)

Conversation compiler SHOULD validate the compiled design with additional rules:
- exactly one `turn_backbone` per epoch path
- only designated steps may emit `next_turn`
- non-gating branches do not reach gating joins
- gating joins are reachable by all gated branches
- state key discipline (no cross-branch clobbering)
- ordering defined for any attachments that depend on sequence

These rules do NOT restrict default runtime behavior; they only validate conversation workflows.

---

**End of Appendix A**

