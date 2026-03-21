# ARD-0006: Conversation Engine Architecture Record

**Status:** Active Refactor (core conversation invariants implemented; snapshot/replay coverage still expanding)  
**Scope:** Conversation chain, summary gating, context building, replay determinism  
**Last Updated:** 2026-03-08  

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

## 1.1 Current Implementation Note (2026-03-08)

The current repo already implements a substantial portion of this ARD:

- `kogwistar/conversation/designer.py` defines the v2 add-turn workflow.
- `kogwistar/conversation/resolvers.py` implements backbone and sidecar workflow ops such as `add_user_turn`, `link_prev_turn`, `link_assistant_turn`, `memory_retrieve`, `kg_retrieve`, `memory_pin`, `kg_pin`, `answer`, and optional `context_snapshot`.
- The current pointer node type is `reference_pointer`, not `kg_pointer`.
- `kogwistar/conversation/service.py` implements `persist_context_snapshot(...)`, and `kogwistar/conversation/agentic_answering.py` persists snapshots around agentic answering calls.


Legacy conversation path is not guaranteed to snapshot
every model invocation automatically.

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
- `reference_pointer`
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

Context is still built dynamically in memory first:

- Node selection
- Token counting
- Budget packing

Persisted context snapshots now exist for explicit snapshot paths and
agentic answering calls, but coverage is not yet universal across all
legacy answer paths.

This remains an area where implementation is ahead of the original
document, but not yet fully complete.

---

# 3. Current Invariants (Observed)

1. `next_turn` defines canonical chain ordering.
2. `turn_index` increments only when appending a turn node.
3. Tool nodes reuse parent `turn_index`.
4. Tool binding is represented via explicit edges.
5. Summary nodes are UI-visible turns.
6. Prompt context is runtime-generated; persisted snapshots exist for
   explicit snapshot and agentic-answering paths, but snapshot coverage
   is not yet universal.
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

## 4.9 Hyperedge Reification Invariant

Hyperedge semantics are defined as:

- `hyperedge = bipartite attachment from full source set to one edge node to full target set`.

Operationally in reified form:

- Let `E` be the relation edge-node.
- For every `s` in `source_ids(E) union source_edge_ids(E)`, attach `s -> E`.
- For every `t` in `target_ids(E) union target_edge_ids(E)`, attach `E -> t`.
- The relation meaning is on `E` itself; endpoint attachments are structural, append-only facts.

This invariant applies to knowledge, workflow, and conversation graphs whenever edge-to-edge endpoints are used.

---

# 5. Phased Refactor Plan

---

## Phase 1 â€” Lock Current Contracts

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

## Phase 2 â€” Normalize Summary Counters

Files:
- `conversation_orchestrator.py`
- `engine.py`

Tasks:
- Persist counters only on `next_turn`
- Remove reliance on side-node distance fields
- Reset counters only on summary append

---

## Phase 3 â€” Token Counter Abstraction

Files:
- `conversation_context.py`

Tasks:
- Add `TokenCounter`
- Add override mechanism
- Add char-based fallback
- Return true + approx counts

---

## Phase 4 â€” Expand Context Snapshot Coverage

Files:
- `conversation_orchestrator.py`
- `conversation_context.py`
- `engine.py`

Tasks:
- Keep `context_snapshot` as the canonical persisted artifact
- Extend snapshot coverage toward every model call
- Add `includes` edges
- Enforce budget invariant
- Add replay tests

---

## Phase 5 â€” Internal Sequencing

Files:
- `runtime.py`
- `engine.py`
- `tool_runner.py`

Tasks:
- Add engine-level `next_seq(conversation_id)`
- Stamp seq on sidecar nodes + snapshots

---

## Phase 6 â€” UI Projection Cleanup

Tasks:
- Compute `ui_turn_index` from visible nodes
- Do not rely directly on raw `turn_index`

---

## Phase 7 â€” Deterministic Replay

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

> â€œGraph + LLM callsâ€

to:

> â€œFully event-sourced conversational execution engine.â€

This ensures:

- Deterministic replay
- Context budget correctness
- Clean separation of chain vs sidecar
- Long-term architectural stability
