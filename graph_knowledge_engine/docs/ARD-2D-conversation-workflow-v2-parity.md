# ARD-2D: Conversation Workflow v2 Parity, State-First Execution, and Resolver Governance

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

> **Workflow Design Graph ≠ Conversation Graph.**\
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

**General reusable?** → Default resolver pack.

**Conversation-specific?** → Conversation resolver pack.

**Changes meaning or invariants?** → Shared ops layer.

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
