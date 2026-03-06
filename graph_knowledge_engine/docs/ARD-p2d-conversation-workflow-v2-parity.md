# ARD-P2D: Conversation Workflow v2 Parity, State-First Execution, and Resolver Governance

**Status:** Draft (Living Document)\
**Scope:** Conversation orchestrator v1 vs workflow-driven v2; workflow
runtime; resolver packs; state/effects recording; fanout/side-flows\
**Last Updated:** 2026-02-25

------------------------------------------------------------------------

## 1. Background

This system implements:

-   A **Conversation Graph** (user/assistant/tool/memory/summary nodes
    and edges).
-   A **Workflow Runtime** capable of executing a **Workflow Design
    Graph** (DAG of steps with predicates, joins, fanout).

Key architectural truth:

> **Workflow Design Graph â‰  Conversation Graph.**\
> The design graph expresses intent. The conversation graph is the
> domain artifact produced by executing that intent.

### Historical Context

-   **v1**: Linear orchestrator directly mutates conversation graph.
-   **v2**: Workflow-driven execution using resolvers and state
    transitions.

After Phase 1 & 2: - Conversation invariants evolved (deterministic IDs,
idempotency, pin semantics, summary policy). - Default resolvers
diverged from orchestrator behavior. - v2 must now become canonical
while preserving parity with v1 for linear runs.

------------------------------------------------------------------------

## 2. Problem Statement

We currently have overlapping semantic loci:

-   Orchestrator primitives (v1 path)
-   Default resolvers (generic steps)
-   Conversation-specific resolvers

This leads to semantic drift, inconsistent state contracts, and
reconciliation complexity.

We must ensure:

1.  v2 can run purely via workflow primitives and resolvers.
2.  All meaningful intermediate state is recorded in `WorkflowState`.
3.  Fanout and side-flows remain supported.
4.  v1 and v2 can be reconciled by comparing **domain artifacts**, not
    workflow trace.

------------------------------------------------------------------------

## 3. Core Architectural Goals

### 3.1 Parity

For linear runs:

-   Same conversation backbone (turn nodes + `next_turn` edges)
-   Same tool artifacts
-   Same pin artifacts
-   Same summary behavior
-   Same `AddTurnResult`

### 3.2 State-First Execution

All meaningful execution must be represented in `WorkflowState`:

-   Deterministic IDs
-   Pin inputs (checkpoint-safe)
-   Artifacts
-   Effects log
-   Errors

Nothing meaningful should exist only in transient local variables.

### 3.3 Fanout Safety

WorkflowRuntime supports branching and joins.

Design must allow:

-   Prefetching
-   Precondition checks
-   Background tidying
-   Future agentic side-flows

Side branches must not accidentally gate the main chain unless
explicitly joined.

------------------------------------------------------------------------

## 4. Core Invariants

### 4.1 Deterministic IDs

All conversation graph artifacts must have deterministic IDs:

-   Turn node IDs
-   `next_turn` edge IDs
-   Tool call artifacts
-   Pin pointer IDs
-   Summary node IDs

### 4.2 Idempotent Re-execution

Steps must be safe under retry/resume:

-   Deterministic ID + upsert semantics
-   Or no-op detection

### 4.3 Conversation Backbone Uniqueness

-   One valid `next_turn` edge per turn (unless explicitly modeled
    otherwise)
-   Deterministic tail selection

### 4.4 Design Graph vs Domain Graph Separation

Parity comparisons must ignore workflow trace nodes.

------------------------------------------------------------------------

## 5. Canonical Execution Model

### 5.1 WorkflowState Sections (Convention)

-   `state["turn"]` --- backbone metadata
-   `state["artifacts"]` --- retrievals, pins, answer metadata
-   `state["effects"]` --- append-only effect log
-   `state["errors"]` --- structured error state
-   `state["_deps"]` --- injected dependencies (non-checkpointed)

### 5.2 Effect Log Principle

Each step:

1.  Computes deterministic IDs
2.  Appends effect descriptor to `state["effects"]`
3.  Applies effect idempotently

Example:

``` json
{
  "effect_type": "conversation.add_edge",
  "edge_id": "E_xxx",
  "src": "N_prev",
  "dst": "N_turn",
  "edge_type": "next_turn"
}
```

------------------------------------------------------------------------

## 6. Resolver Packs

### 6.1 Default Resolver Pack

-   Generic step primitives
-   Generic validation
-   Error handling patterns
-   Reusable outside conversation domain

### 6.2 Conversation Resolver Pack

-   Backbone creation
-   Chain invariants
-   Pin pointer modeling
-   Summary semantics

### 6.3 Shared Ops Layer

All domain meaning must live here:

-   Summary decision policy
-   Pin selection logic
-   ID generation policy
-   Invariant enforcement

Resolvers are adapters, not semantic engines.

------------------------------------------------------------------------

## 7. Managing New Conversation Features

When adding a feature:

### Step 1: Classify

**General reusable?** â†’ Default resolver pack.

**Conversation-specific?** â†’ Conversation resolver pack.

**Changes meaning or invariants?** â†’ Shared ops layer.

### Step 2: Maintain Step Semantics Stability

A step name must retain stable meaning across:

-   v1 execution
-   v2 workflow execution
-   Future transpiled runtimes (e.g., LangGraph)

### Step 3: Add Dual Tests

1.  Domain parity tests (v1 vs v2).
2.  Workflow behavior tests (fanout correctness).

------------------------------------------------------------------------

## 8. Parity Definition

Must match:

-   Conversation backbone
-   Tool artifacts
-   Pin artifacts
-   Summary behavior
-   AddTurnResult

May differ:

-   Workflow trace artifacts
-   Operational metadata
-   Non-joined side-flow artifacts

------------------------------------------------------------------------

## 9. Branching Design Rules

-   Fanout allowed by default.
-   Non-gating branches must not reach join barriers.
-   Gating branches must explicitly join.
-   Main chain equivalence must remain deterministic.

------------------------------------------------------------------------

## 10. Decisions

-   v2 is workflow-first execution.
-   State-first recording is mandatory.
-   Fanout remains allowed.
-   Step semantics are stable and shared.
-   Parity is defined over conversation graph artifacts.

------------------------------------------------------------------------

## 11. Open Questions

-   Canonical effect descriptor schema?
-   Step versioning strategy?
-   Side-flow result integration policy?

------------------------------------------------------------------------

**This ARD is the reference guardrail for Phase 2D and future workflow
evolution.**

------------------------------------------------------------------------

## Appendix A (State, Retries, Ordering, Chain Semantics)

This appendix is maintained as a separate file:

- **ARD-p2d-appendix-a-state-retries-ordering.md**

(See downloadable artifact.)

------------------------------------------------------------------------

# ARD-P2D Appendix A: State, Effects, Retries, Ordering, and Chain Semantics

**Purpose:** This appendix pins down the â€œhard rulesâ€ for WorkflowState, branching, retries, ordering, and how the *main conversation chain* remains linear while still allowing fanout/sidecars and multiple in-chain events per user turn.

This appendix is designed to prevent future drift between:
- v1 linear orchestrator execution
- v2 workflow execution (including fanout)
- future transpiled runtimes (e.g., LangGraph)

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

### A7.1 Result model (Go/Rust-style)
Steps should record outcomes as:
- `Ok(value)` or
- `Err(error)`
in state, not only via raised exceptions.

### A7.2 Runtime default
Current runtime behavior is typically â€œraise = abort runâ€.

### A7.3 Configurable policy
Make abort/continue behavior configurable per step and/or per workflow:
- `wf_on_error = "abort" | "continue" | "continue_with_flag"`
- When continuing, downstream decision steps can read prior errors and decide:
  - degrade gracefully
  - request user intervention
  - enforce timeout
  - halt main chain via explicit interrupt

This preserves general runtime behavior while enabling conversation-specific â€œstuckable unless timeoutâ€ semantics.

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

## Similarity Contract for v1 vs v2

v2 is allowed to introduce:
- fanout and sidecar branches
- retries and additional diagnostic artifacts
- workflow/runtime trace artifacts

Therefore, strict graph equality between v1 and v2 is not a universal requirement.
Instead, we define **tiered parity** between executions.

### Tier 0: Hard invariants (must match exactly)
These invariants define â€œconversation correctnessâ€ and must hold across v1 and v2:
- Main chain remains **linear** within an epoch.
- At most one outgoing `next_turn` per main-chain turn.
- The main-chain turn sequence (or equivalently the `next_turn` chain) is consistent and deterministic for linear runs.
- Domain relations that define user-visible meaning must exist with correct endpoints:
  - `next_turn` (main chain)
  - `summarizes` (summary linkage, when summary exists)
  - `references` (knowledge pin/reference edges, when present)
- Chain membership flags are consistent (`in_conversation_chain`, `in_ui_chain`) for main-chain turns.

### Tier 1: Canonical equality (compare after normalization)
Two executions are considered equivalent if their **canonicalized** domain views match.
Canonicalization may:
- ignore run-specific stamps (`run_id`, `run_step_seq`, `attempt_seq`, timestamps, worker ids)
- ignore workflow/runtime trace subgraphs
- treat unordered sets as sets (sort deterministically)
- ignore allowed additive side artifacts that are not joined into the main chain

### Tier 2: Superset / monotonicity (v2 may add extra artifacts)
v2 may produce additional artifacts beyond v1, such as:
- extra retrieval attempts linked by `retry`
- sidecar subgraphs
- progress/thinking events
- policy interrupt trace artifacts
These additions are allowed provided Tier 0 invariants hold and Tier 1 canonical equality holds for the main-chain view.

### Testing policy
- Default parity tests should assert **Tier 0 + Tier 1**.
- Dedicated tests may assert Tier 2 behaviors (e.g., retries produce `retry` edges).
- Linear sample parity should run with `max_workers=1` and a conversation design that avoids gating side branches.


