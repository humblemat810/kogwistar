# Conversation Engine Architecture Record (ARD)

**Status:** Active Refactor  
**Scope:** Conversation chain, summary gating, context building, replay determinism  
**Last Updated:** 2026-02  

---

# 1. Background

This system implements a **hypergraph-based, append-only, event-sourced conversational engine**.

It supports:

- Canonical conversation turn sequencing
- Tool calls (memory retrieval, KG retrieval, etc.)
- Knowledge pinning
- Summarization
- Context window construction
- LLM invocation

Long-term goal:

> Deterministic, replayable, audit-safe conversational execution with full provenance.

Historically, summary gating and context building evolved incrementally and now require formalization to prevent semantic drift.

---

# 2. Architectural Context

## 2.1 Canonical Conversation Chain

The conversation has a **single canonical linear chain** defined by:

### Nodes
- `conversation_start`
- `conversation_turn`
- `assistant_turn`
- `conversation_summary`

### Edges
- `next_turn`

`turn_index` increments **only** when appending to this chain.

This chain represents **UI-visible temporal ordering**.

---

## 2.2 Sidecar Graph (Tool / Memory / KG)

Sidecar nodes represent auxiliary operations:

### Nodes
- `tool_call`
- `tool_result`
- `memory_context`
- `kg_pointer`
- etc.

### Edges
- `tool_call_entry_point`
- `run_result`
- `has_memory_context`
- `summarizes`
- etc.

Sidecar nodes:

- May carry `turn_index = parent_turn`
- Must not increment `turn_index`

The sidecar graph forms a DAG.

---

## 2.3 Summary Trigger Logic (Current)

Summary is conditionally invoked based on:

- Turn count since last summary
- Char count since last summary
- Optional LLM decision

Distance fields currently exist on both chain and side nodes, creating ambiguity.

---

## 2.4 Context Building (Current)

Context is built dynamically in memory:

- Node selection
- Token counting
- Budget packing

No context snapshot is persisted.

This is architecturally inconsistent with event-sourcing principles.

---

# 3. Current Invariants (Observed)

1. `next_turn` defines canonical chain ordering.
2. `turn_index` increments only when appending a turn node.
3. Tool nodes reuse parent `turn_index`.
4. Tool binding is represented via explicit edges.
5. Summary nodes are UI-visible turns.
6. Context building is runtime-only (not persisted).
7. Conversation graph is append-only.

---

# 4. Target Invariants (To Enforce)

## 4.1 Canonical Chain Integrity

- Only canonical turn nodes increment `turn_index`.
- Sidecar nodes must never allocate a new turn index.

## 4.2 Pinned Knowledge Immutability

- First pin of a pointer node is immutable.
- Re-pin is a no-op.
- Usage is represented via edges only.

## 4.3 Chain-Only Summary Counters

Persist summary counters **only on `next_turn` edges**:

- `turns_since_summary_at_dst`
- `chars_since_summary_at_dst`

Summary gating must rely exclusively on canonical chain metrics.

Side DAG topology must not influence summary decisions.

## 4.4 Token Budgeting Separation

Two independent systems:

### A. Stable Chain Metrics
Used for summary triggering.

### B. Dynamic Context Metrics
Used for model invocation budgeting.

These must not be conflated.

## 4.5 Token Counter Abstraction

Introduce `TokenCounter` interface:

- Allows model-specific tokenization
- Fallback to approximate char-based estimate
- Persist `token_counter_name` in context snapshot

## 4.6 Context Snapshot (Event-Sourced)

Every LLM invocation must produce a snapshot:

### Node
`entity_type = "context_snapshot"`

### Edges
- `turn --built_context--> context_snapshot`
- `context_snapshot --includes--> node`

### Metadata
- conversation_id
- turn_node_id
- run_step_seq
- model_id
- token_counter_name
- budget_tokens
- total_tokens
- chain_tokens
- side_tokens

### Invariants
- Append-only
- `total_tokens <= budget`
- Exactly one snapshot per LLM call

## 4.7 Internal Ordering

Introduce `run_step_seq`:

- Monotonic per conversation
- Provided by runtime if available
- Else allocated by engine

Stamped on:
- tool_call
- tool_result
- context_snapshot

## 4.8 Append-Only Discipline

No node metadata mutation for:

- sealing
- counter resets
- state transitions

All evolution must be:
- new node
- new edge
- or meta-log entry

---

# 5. Phased Refactor Plan

---

## Phase 1 — Lock Current Contracts

Files:
- `conversation_orchestrator.py`
- `models.py`
- `engine.py`
- `contract.py`

Tasks:
- Document invariants
- Add invariant tests
- Prevent tool nodes from incrementing turn_index

---

## Phase 2 — Normalize Summary Counters

Files:
- `conversation_orchestrator.py`
- `engine.py`

Tasks:
- Persist counters only on `next_turn`
- Remove reliance on side-node distance fields
- Reset counters only on summary append

---

## Phase 3 — Token Counter Abstraction

Files:
- `conversation_context.py`

Tasks:
- Add `TokenCounter`
- Add override mechanism
- Add char-based fallback
- Return true + approx counts

---

## Phase 4 — Persist Context Snapshot

Files:
- `conversation_orchestrator.py`
- `conversation_context.py`
- `engine.py`

Tasks:
- Introduce `context_snapshot`
- Persist before every model call
- Add `includes` edges
- Enforce budget invariant
- Add replay tests

---

## Phase 5 — Internal Sequencing

Files:
- `runtime.py`
- `engine.py`
- `tool_runner.py`

Tasks:
- Add engine-level `next_seq(conversation_id)`
- Stamp seq on sidecar nodes + snapshots

---

## Phase 6 — UI Projection Cleanup

Tasks:
- Compute `ui_turn_index` from visible nodes
- Do not rely directly on raw `turn_index`

---

## Phase 7 — Deterministic Replay

Tasks:
- Rebuild context from snapshot
- Validate token totals
- Ensure generation determinism across runs

---

# 6. Design Principles

1. Canonical chain remains linear.
2. Sidecar graph remains DAG.
3. No mutation of historical data.
4. Context building becomes a first-class event.
5. Model/tokenizer changes must not corrupt invariants.
6. Replay and auditability are first-class concerns.

---

# 7. Architectural Direction

We are moving from:

> “Graph + LLM calls”

to:

> “Fully event-sourced conversational execution engine.”

This ensures:

- Deterministic replay
- Context budget correctness
- Clean separation of chain vs sidecar
- Long-term architectural stability