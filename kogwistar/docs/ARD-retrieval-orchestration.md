# ARD — Agentic Retrieval Orchestration
## Memory + Knowledge Graph Exploration with Budgeted Iterative Control

**Status:** Draft (one-round retrieval and pinning implemented; multi-round deep controller still draft)  
**Scope:** Retrieval orchestration + retrieval agents + projection contract  
**Out of scope:** Full answering/synthesis, UI, full temporal KG, tool-specific integrations beyond “graph query tools”.

---

## 0. Current Implementation Note (2026-03-08)

The current repo already implements a narrower version of this design:

- `kogwistar/conversation/retrieval_orchestrator.py`
  coordinates one memory retrieval pass and one KG retrieval pass for a
  turn.
- `kogwistar/conversation/memory_retriever.py` retrieves
  cross-conversation memory, extracts KG seeds from selected
  `reference_pointer` nodes, and materializes `memory_context`.
- `kogwistar/conversation/knowledge_retriever.py`
  combines shallow retrieval with seeded graph expansion and pins
  deterministic `reference_pointer` nodes plus `references` edges into
  the conversation canvas.

The following parts of this ARD remain target-state rather than fully
landed behavior:

- generic frontier-state objects
- explicit multi-round iterative control
- first-class budget objects beyond current ad hoc limits
- a standalone projector/pinner abstraction separate from retrievers

---

## 1. Purpose

Define the architecture for **deep retrieval** as an **agentic exploration loop** over:

- **Conversation long-memory** (history graph artifacts)
- **Knowledge graph** (KG nodes/edges)

The subsystem produces **bounded candidate references** and a **plan for further exploration**, under explicit budgets, with deterministic, traceable behavior.

---

## 2. Mental Model

### 2.1 Retrieval is “pinning references”
Retrieval does not copy large content into the canvas. It:
1. Finds **references** to potentially useful entities (memory artifacts, KG entities)
2. Selects a **bounded used boundary**
3. **Pins** selected references into the current conversation turn as pointer nodes + edges

### 2.2 “Deep” retrieval is iterative exploration, not blind k-hop
A deep mode is an **orchestrated loop**:
- each round: propose → expand → score → select → update frontier → decide stop

k-hop expansion remains a **low-level tool**, not the controller.

---

## 3. Components

### 3.1 Retrieval Orchestrator (central controller)
**Responsibility**
- Owns the iterative loop
- Runs Memory retrieval and KG retrieval rounds
- Maintains a frontier, dedupes visited refs
- Enforces budgets and termination conditions
- Produces a final `RetrievalOutcome`

**Does not**
- Persist answer text
- Materialize projections directly (delegates to projector/pinner)

### 3.2 Memory Retrieval Agent
**Responsibility**
- One-round LLM-guided selection of relevant prior conversation artifacts
- Produces memory candidate refs + memory seeds for KG

**Inputs**
- `conversation_id`
- `user_query`
- optional: `active_working_set_ptrs`

**Outputs**
- `selected_memory_refs` (bounded)
- `seed_kg_ids` extracted from pinned refs / summaries
- `next_memory_frontier` (optional)

### 3.3 KG Retrieval Agent
**Responsibility**
- One-round LLM-guided exploration step over KG
- Chooses what relations / neighbors to expand next within limits

**Inputs**
- `user_query`
- `seed_kg_ids`
- optional: `prior_candidates`

**Outputs**
- `selected_kg_refs` (bounded)
- `next_kg_frontier` (bounded)
- optional: explanation / confidence signals

### 3.4 Retrievers (non-agentic primitives)
Deterministic, bounded tools used by agents:

- `vector_search(text_or_embedding, top_k, where)`
- `khop_expand(seed_ids, hop=1, per_seed_limit, where)`
- `incident_edges(seed_ids, limit, rel_types)`
- `fetch_summaries(ids, max_chars)`

> Agents propose “what to expand”; tools enforce “how much”.

### 3.5 Projector / Pinner
**Responsibility**
- Idempotently materialize pointer nodes and linking edges into the **conversation canvas**
- Deduplicate projections
- Apply projection budgets

Current implementation note:
- pinning currently lives in `MemoryRetriever.pin_selected(...)` and
  `KnowledgeRetriever.pin_selected(...)` rather than in a separate
  projector service.

---

## 4. Data Products and Schemas

### 4.1 Candidate Reference (CandidateRef)
Normalized “retrievable thing”:

- `ref_namespace`: `"kg" | "memory" | "tool" | "archive"`
- `ref_kind`: `"node" | "edge" | "run" | "summary" | "pointer" | "artifact"`
- `ref_id`: stable identifier within namespace
- `score`: float (optional)
- `why`: short rationale (optional)
- `seed_of`: list[str] (optional) — provenance to seeds/frontier

### 4.2 Frontier State
- `kg_frontier_ids: list[str]`
- `memory_frontier_ids: list[str]`
- `visited_refs: set[(namespace, kind, id)]`

### 4.3 Retrieval Round Output
**MemoryRoundOutput**
- `selected_memory_refs: list[CandidateRef]`
- `seed_kg_ids: list[str]`
- `next_memory_frontier: list[str]`
- `signals: dict` (confidence, coverage, etc.)

**KGRoundOutput**
- `selected_kg_refs: list[CandidateRef]`
- `next_kg_frontier: list[str]`
- `signals: dict`

### 4.4 RetrievalOutcome (final)
- `selected_memory_refs`
- `selected_kg_refs`
- `pinned_pointer_node_ids` (created in conversation engine)
- `pinned_edge_ids` (conversation edges created)
- `meta` (budgets used, rounds, termination reason)

---

## 5. Orchestration Loop

### 5.1 Default Execution Plan
1. Initialize frontier:
   - from current turn’s pinned pointers (if any)
   - from active working set
2. **Round 0** (speculative):
   - run MemoryRound and KG shallow retrieval concurrently (optional)
3. Merge candidates and seeds
4. Decide whether to run KG deep round
5. Repeat until termination

### 5.2 Concurrency Strategy
- MemoryRound and KG shallow can run concurrently
- KG deep depends on seed availability and/or confidence signals
- Graph writes are serialized

---

## 6. Budgets and Stop Conditions

### 6.1 Mandatory Budgets
- `max_rounds`
- `max_tool_calls`
- `max_nodes_expanded`
- `max_edges_expanded`
- `max_candidates_total`
- `max_pins_per_turn`
- optional: time budget, token budget

### 6.2 Termination Rules
Terminate if any of:
- budget exhausted
- confidence/coverage above threshold
- frontier empty
- cycle detected (same frontier+strategy fingerprint)
- approach exhaustion (all allowed strategies tried)

---

## 7. Cycle Detection & Approach Exhaustion

### 7.1 State Fingerprint
Fingerprint computed from:
- selected strategy (mode flags)
- sorted frontier IDs (bounded sample)
- round index
- recent selected candidate IDs

Repeated fingerprint ⇒ stop or force strategy change.

### 7.2 Strategy Set
Example strategies:
- memory-heavy + shallow KG
- seed-guided KG deep (hop-limited)
- relation-type constrained expansion
- fallback: widen top_k, widen hop depth, then stop

---

## 8. Projection (Pinning) Contract

### 8.1 When to pin
Pin only:
- candidates that cross the **used boundary**
- or the “eligible working set” if you explicitly support a UI-visible shortlist

### 8.2 What to pin
Pin as pointer nodes in the conversation engine, not copies of KG.
- pointer node stores: `(ref_namespace, ref_kind, ref_id)` and minimal summaries
- edges link:
  - `TurnAnchor --references--> PointerNode` (KG)
  - `RunAnchor --uses_prior_context--> PointerNode` (memory)

### 8.3 Idempotency
Pointer identity is deterministic:
- `ptr|scope:{conversation_id}|ns:{ref_namespace}|kind:{ref_kind}|id:{ref_id}`

Edges also deduped by canonical triple:
- `(src, relation, dst)` or deterministic edge id

---

## 9. Provenance and Trace

### 9.1 Trace Graph
Each retrieval run should be traceable:
- `RetrievalRun` node
- `RetrievalRound` nodes
- `ToolCallArtifact` nodes
- edges: `next`, `produced`, `consumed`, `selected`

### 9.2 Snapshot-on-use
If KG is mutable, any KG entity that becomes “used” should be snapshotted to an archive store for provenance stability.

---

## 10. Integration Points

### 10.1 With Conversation Turn Flow
`add_conversation_turn()` delegates retrieval policy to orchestrator:
- produces pinned refs (conversation pointers)
- then answering agent uses pinned refs + selected candidates for synthesis

### 10.2 With Wisdom Layer
Wisdom stores:
- workflow signature (retrieval strategy)
- outcomes (quality, cost, user eval)
- recommended strategy priors per query archetype

---

## 11. Non-goals
- Unbounded traversal across combined graphs
- Copying entire historical conversation content into canvas
- Storing LLM chain-of-thought
- Full temporal KG replay

---

## 12. Invariants (Checklist)
- Retrieval outputs are bounded and budgeted
- Agents propose, tools enforce limits
- Orchestrator controls loops and termination
- Projection is deterministic and idempotent
- Used boundary is explicit
- Traceability is first-class
