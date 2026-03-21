# Architecture Requirements Document (ARD)
## Transactional storage + vector backends (Chroma and Postgres/pgvector)

**Status:** Accepted (storage/runtime boundary implemented; upper-layer backend neutrality still partial)  
**Scope:** `GraphKnowledgeEngine` storage layer + `WorkflowRuntime` persistence boundary + LangGraph conversion compatibility.

## Document History
- 2026-03-05: Status changed from Draft to Accepted (Implemented; code-derived from runtime/backend behavior and tests).
- 2026-03-05: Added explicit concurrent UoW semantics (thread/task isolation, rollback scope, backend-specific consequences).
- 2026-03-08: Clarified that accepted status applies to storage/runtime boundaries and backend adapters, not to every upper-layer module in the repo.

---
## 1. Context

The system has multiple â€œenginesâ€ backed by local persistence:

- **workflow_engine** stores workflow design nodes/edges and can persist the workflow graph.
- **conversation_engine** stores run-time execution artifacts (run anchor, step exec nodes, checkpoint nodes, edges).
- **kg_engine** stores knowledge graph nodes/edges and document-derived artifacts.

A **WorkflowRuntime** executes a workflow graph with fan-out/join semantics. Workers compute results and return `RunResult(state_update=...)`. The scheduler loop applies state updates and persists execution artifacts into `conversation_engine`. The system may also convert a workflow into a LangGraph runnable graph via a `langgraph_converter` (not included here), which must preserve the same state merge semantics as native runtime.

ChromaDB mode uses multiple collections (e.g., `nodes`, `edges`, `documents`, etc.) and already supports `where` filtering and storing documents/metadata (e.g., node/edge JSON as documents).

## 1A. Scope Clarification (2026-03-08)

This ARD is accepted for the storage/runtime persistence boundary only.
It does not claim that every upper-layer module is already backend
neutral.

In particular, the following areas still expose Chroma-shaped collection
surfaces and remain outside the "implemented" claim for this document:

- `kogwistar/strategies/*`
- `kogwistar/visualization/*`
- `kogwistar/graph_query.py`
- engine-facing typing protocols that still expose raw collections

---

## 2. Goals

1. **Support Postgres + pgvector** as an alternative backend with strong transactional semantics.
2. **Preserve existing Chroma mode** (non-transactional vector store) without pretending it can participate in SQL transactions.
3. Ensure **all local SQL writes that must be consistent** (conversation, workflow, KG metadata) can be grouped into a single transaction *when the backend allows*.
4. Provide a **clear, deterministic persistence boundary** in `WorkflowRuntime` (per-step â€œcommit pointâ€).
5. Keep **step resolver contract unchanged**:
   - resolvers read/modify `state` and return `RunResult(state_update=...)`
   - no resolver code should care whether the backend is Chroma or pgvector.
6. Keep **LangGraph converter correctness**:
   - identical state merge semantics (append/update/extend) to native runtime
   - identical observable behavior in tests (parallel fanout merge, joins).

---

## 3. Non-goals

- Making ChromaDB transactional (not possible).
- Implementing distributed transactions across multiple databases.
- Rewriting workflow execution semantics or step resolver APIs.

---

## 4. Current architecture highlights (as-is)

### 4.1 Chroma collections
Chroma mode already has separable collections and `where` filtering, and stores JSON documents + metadata for nodes/edges (e.g., helper functions constructing document+metadata for Chroma inserts).

### 4.2 Runtime write pattern
`WorkflowRuntime.run()`:
- workers execute step resolvers
- scheduler processes completed tasks:
  - merges `state_update` into shared state
  - persists a `WorkflowStepExecNode` + edges into `conversation_engine`
  - optionally persists checkpoint
  - routes tokens and manages join frontier

This scheduler step is the natural persistence/transaction boundary.

### 4.3 LangGraph conversion
There is a converter path used by integration tests that asserts parallel fanout merges are correct. This implies the merge/reducer logic must remain stable and shared between runtime and converter.

---

## 5. Target architecture

### 5.1 Split responsibilities into explicit â€œstoresâ€
Introduce three conceptual stores behind `GraphKnowledgeEngine`:

1. **MetaStore (relational truth)**  
   Nodes/edges, redirects/tombstones, workflow run artifacts, checkpoints, oplog records.

2. **VectorIndex (search index)**  
   - **Chroma implementation:** non-transactional, collection-based.
   - **pgvector implementation:** Postgres table(s) with vector columns.

3. **Outbox (transactional bridge for projections)**  
   Stored in MetaStore database. Represents â€œside effects to applyâ€ (vector updates, external CDC publish, etc.).

### 5.2 Unit of Work (UoW)
Add `engine.uow()` as the only supported way to group writes. The UoW decides whether vector writes are performed:

- **Inline, transactional** (Postgres+pgvector; same DB transaction)
- **Deferred via outbox** (Chroma; SQL commit then apply to Chroma)

**Invariant:** resolvers do not see the UoW; only engine/runtime does.

---

## 6. Backend profiles and transaction model

### 6.1 Profile A â€” Postgres + pgvector (transactional)
- MetaStore + VectorIndex in the same Postgres database.
- `engine.uow()` uses a Postgres transaction:
  - write nodes/edges/exec artifacts
  - write embeddings (`INSERT ... ON CONFLICT DO UPDATE`)
  - write outbox events for external sinks if needed
  - COMMIT / ROLLBACK provides true atomicity.

**Recommended schema (example):**
- `nodes`, `edges`
- `node_embeddings`, `edge_embeddings`, `document_embeddings` (or one table with `collection_key`)
- `outbox`

### 6.2 Profile B â€” SQLite + Chroma (eventual consistency with replay)
- MetaStore in SQLite remains transactional.
- VectorIndex is Chroma collections (non-transactional).
- `engine.uow()` uses SQLite transaction:
  - write nodes/edges/exec artifacts
  - write outbox rows describing vector ops + publish ops
  - COMMIT
- After commit:
  - best-effort drain outbox to Chroma
  - failures remain pending for later replay.

**Invariant:** SQL is the source of truth; Chroma is a projection.

### 6.2.1 Structural ingest contract
The eventual-consistency model in Profile B applies to **derived projections** and replayable side effects. It does **not**
mean that structurally invalid base writes are accepted and repaired later.

Current contract:
- Base node/edge records remain the source of truth.
- Derived projections (`node_docs`, `node_refs`, `edge_refs`, `edge_endpoints`) are rebuilt via durable jobs and replay.
- `add_edge(...)` requires all referenced endpoints to already exist at ingest time; missing endpoints are rejected as an
  invalid base write rather than staged for later repair.

Operational consequence:
- Retry/backoff covers **derived index convergence after a base entity already exists**.
- Retry/backoff does **not** turn out-of-order base ingest into a safe eventual-consistency flow.
- External producers must therefore provide dependency-safe ordering, either:
  - ordered single events (`node` before dependent `edge`)
  - topologically sorted batches
  - or an explicit staging/inbox layer that buffers unresolved edges until endpoints exist

Non-goal of current Profile B:
- accepting unordered concurrent node/edge events as committed base truth and resolving structural dependencies later

If a producer delivers a dependent `edge` before its endpoint `node`, the edge write may fail and must be retried by the
producer or absorbed by a staging layer. This is distinct from replay/repair of already-committed base rows.

### 6.3 Concurrent UoW semantics (threads/tasks)
`engine.uow()` context is isolated per execution context (thread/task). Concurrent callers do not share the same UoW unless they explicitly reuse the same active context.

Behavioral contract:
- Concurrent `with engine.uow()` calls are independent transaction scopes.
- If one scope raises/cancels, only that scope is rolled back.
- A rollback in one thread must not roll back another thread's committed UoW.
- Nested `uow()` in the same thread/context joins the outer UoW; only outermost scope commits or rolls back.

Backend-specific consequences:
- Postgres + pgvector: rollback is atomic for entity rows, vectors, event log rows, and index jobs in that UoW.
- SQLite + Chroma: SQLite MetaStore portion rolls back atomically; Chroma writes are non-transactional and may already be visible if failure occurs after projection write.
- SQLite writer contention: `BEGIN IMMEDIATE` serializes writers; other writers wait, then continue after commit/rollback.

---

## 7. Collections / â€œwhereâ€ support across node/edge/doc types

Chroma mode already uses separable collections and `where` filtering. To ensure all â€œcollectionsâ€ are catered for (including `node_doc`, `edge_doc`-like semantics), the architecture standardizes a **collection key**:

- `collection_key` is a logical namespace (e.g. `node`, `edge`, `document`, `node_index`, `edge_endpoints`).
- Backends map `collection_key` to:
  - Chroma collection name, or
  - Postgres table/partition, or
  - â€œnot indexedâ€ (if disabled by config).

**Outbox vector operations MUST include `collection_key`** so replay does not lose routing.

> Note: `node_doc` / `edge_doc` content can remain as Chroma â€œdocumentsâ€ stored in the same `nodes`/`edges` collections, as currently implemented. The key requirement is that all vector/index updates for every logical collection are routable and replayable.

---

## 8. Runtime persistence boundary

### 8.1 Per-step commit point
In `WorkflowRuntime.run()` scheduler loop, for each completed step:
- apply `state_update`
- persist `WorkflowStepExecNode` + edges
- mutate routing/join frontier
- optionally persist checkpoint

**Requirement:** these operations should be committed as one unit under `conversation_engine.uow()`.

### 8.2 Checkpoint correctness
Checkpoints must reflect a coherent join frontier / pending tokens state. Therefore checkpoint persistence must remain ordered after state and routing updates within the same UoW.

---

## 9. Shared state merge semantics (Runtime + LangGraph)

Define a single shared â€œstate mergeâ€ module used by:
- `WorkflowRuntime.apply_state_update(...)`
- `langgraph_converter` reducers

This module must implement the same update modes (append/update/extend) and pass the existing parallel fanout merge tests unchanged.

---

## 10. Outbox design

### 10.1 Outbox operations
`op_type` examples:
- `UPSERT_VECTOR`
- `DELETE_VECTOR`
- `PUBLISH_CHANGE` (CDC/oplog)

Fields:
- `id` (monotonic)
- `op_type`
- `collection_key`
- `entity_id`
- `payload_json` (document+metadata or embedding+metadata)
- `status` (`pending`, `processing`, `done`, `failed`)
- `attempts`, `last_error`, `next_retry_at`
- `idempotency_key` (unique constraint)

### 10.2 Idempotency
All vector operations must be safe to repeat:
- Chroma: deterministic IDs and upsert semantics
- pgvector: `ON CONFLICT DO UPDATE`
- delete: delete-if-exists

### 10.3 Reconciliation scope
Reconciliation is intended to be near-real-time queue draining, not a once-per-day batch snapshot pass.

Typical trigger paths:
- drain-on-write (`enqueue` + immediate best-effort `reconcile`)
- long-running worker loop over pending jobs
- replay/repair commands for explicit backfill or corruption recovery

Replay/repair is a secondary convergence mechanism. It is not the primary contract for ordinary unordered structural ingest.

---

## 11. Migration plan (incremental)

1. Add `uow()` to `GraphKnowledgeEngine` (SQLite first), keep existing `add_node/add_edge` by auto-wrapping in a UoW for backwards compatibility.
2. Add outbox tables + `flush_outbox()` for Chroma mode (vector/index ops + optional publish ops).
3. Refactor `WorkflowRuntime` persistence helpers to use `with conversation_engine.uow(): ...` per completed step.
4. Extract merge semantics into a shared module, update runtime and converter to use it.
5. Add Postgres MetaStore implementation + migrations; keep API identical.
6. Add pgvector implementation and enable â€œinline vector writes inside transactionâ€ profile.
7. Add backfill/rebuild commands:
   - `rebuild_vector_index(collection_key=...)`
   - `reconcile_outbox()` / `repair_failed_ops()`

---

## 12. Test requirements

### 12.1 Backend-contract tests (parametrized)
1. **SQLite+Chroma crash window**
   - commit meta + outbox
   - simulate failure applying to Chroma
   - assert outbox pending
   - drain outbox later; assert index updated

2. **Postgres+pgvector atomicity**
   - inject error mid-persist in UoW
   - assert rollback (no partial step exec/checkpoint)
   - rerun success; assert all artifacts present

### 12.2 Preserve existing tests
- conversation flow integration should remain stable
- langgraph parallel merge integration must keep passing
- workflow design persist test must keep passing

---

## 13. Risks / trade-offs

- **Chroma mode is eventual consistency:** mitigated with outbox + replay + idempotency.
- **Operational complexity:** outbox drain loop must be reliable; consider a lightweight â€œdrain-on-writeâ€ plus periodic reconciliation.
- **Structural ingest is stricter than projection repair:** callers may incorrectly assume edge-before-node delivery is safe because
  derived index jobs retry. It is not safe without producer ordering, batching, or a staging layer.
- **Schema divergence:** keep logical `collection_key` mapping consistent across backends to avoid migration pain.

---

## 14. Acceptance criteria

- A single code path for runtime + resolver contract works across:
  - SQLite+Chroma mode
  - Postgres+pgvector mode
- In Postgres+pgvector mode:
  - per-step exec persistence + optional embeddings are fully transactional
- In SQLite+Chroma mode:
  - all vector/index updates are replayable and do not silently diverge
- LangGraph conversion:
  - state merge semantics identical to runtime and tests demonstrate correctness

## 9A. LangGraph converter alignment (refined after review)

The `langgraph_converter` introduces **explicit execution modes** and **state-reducer behavior**
that must be reflected in architectural guarantees.

### 9A.1 Execution modes
The converter supports:
- `execution="semantics"`: intended to preserve native runtime semantics (token routing, join gating, fanout behavior).
- `execution="visual"`: best-effort, diagram-oriented export; semantic equivalence is **not guaranteed**.

**Requirement:**
- Any claim of behavioral equivalence with `WorkflowRuntime` MUST be validated using
  `execution="semantics"`.
- `execution="visual"` is explicitly excluded from semantic or transactional correctness claims.

### 9A.2 Reducer / merge semantics
The converter implements `_apply_state_update` and blob reducers that intentionally
mirror `WorkflowRuntime.apply_state_update`.

**Requirement:**
- Merge / reducer logic MUST live in a single shared module.
- Both `WorkflowRuntime` and `langgraph_converter` MUST import and use this module.
- Duplication of reducer logic is disallowed to prevent semantic drift.

### 9A.3 Reserved runtime keys
In semantics mode, the converter uses reserved blob keys such as:
- `__token_id__`
- `__next_step_names__`
- join-related keys (`__join_arrivals__:*`, `__join_done__:*`)

**Requirement:**
- Checkpoint persistence MUST capture the full state blob, including all reserved keys.
- Reserved keys MUST remain namespaced (e.g. `__*__`) and must not collide with user state.

---

## 11A. Updated acceptance criteria (LangGraph-aware)

In addition to Section 14, the following must hold:

1. **Semantic parity**
   - For workflows executed natively and via LangGraph,
     runs executed with `LGConverterOptions(execution="semantics", mode="blob_state")`
     MUST produce equivalent step execution order, join behavior, and final state.

2. **Visual mode isolation**
   - `execution="visual"` outputs may diverge and MUST NOT be used for replay,
     persistence correctness, or transactional guarantees.

3. **Checkpoint completeness**
   - A checkpoint restored into the native runtime MUST contain sufficient information
     to resume execution equivalently to a LangGraph semantics-mode run at the same boundary.




