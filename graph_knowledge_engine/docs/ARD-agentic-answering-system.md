# Architectural Requirements Document (ARD)
## Agentic Answering System with Provenance, Orchestration, and Wisdom

**Status:** Accepted as target architecture; core execution and provenance pieces are implemented, broader wisdom/control-plane split remains partial
**Date:** 2026-03-08
**Owner:** Maintainers

---

## 1. Purpose & Scope

This document specifies the architecture and requirements for an **agentic answering system** built on top of an existing graph-based knowledge and conversation engine.

The system must:
- Answer user queries using structured agent orchestration
- Track *exactly what knowledge and data were used*
- Persist orchestration plans and execution traces
- Project used evidence into the conversation graph
- Support learning across runs via a wisdom layer
- Remain backend-agnostic (Chroma today, SQL/other later)

This document **does not define UI**, nor does it prescribe a single execution strategy. It defines invariants and interfaces.

### 1.1 Current Implementation Note (2026-03-08)

The repo now implements substantial portions of this document, but the
full control-plane / trace-plane / wisdom-layer split is still only
partially realized.

- `graph_knowledge_engine/conversation/agentic_answering.py` already
  routes answer generation and citation repair through `LLMTaskSet`
  rather than through concrete provider SDK classes.
- `graph_knowledge_engine/conversation/designer.py` now defines
  `AgenticAnsweringWorkflowDesigner`, and
  `graph_knowledge_engine/conversation/resolvers.py` implements
  workflow ops such as `aa_prepare`, `aa_get_view_and_question`,
  `aa_retrieve_candidates`, `aa_select_used_evidence`,
  `aa_materialize_evidence_pack`, `aa_generate_answer_with_citations`,
  `aa_validate_or_repair_citations`, `aa_evaluate_answer`,
  `aa_project_pointers`, `aa_maybe_iterate`, and `aa_persist_response`.
- Projection and pointer materialization already exist in
  `graph_knowledge_engine/conversation/knowledge_retriever.py`,
  `graph_knowledge_engine/conversation/memory_retriever.py`,
  `graph_knowledge_engine/conversation/retrieval_orchestrator.py`,
  `graph_knowledge_engine/conversation/conversation_orchestrator.py`,
  and `graph_knowledge_engine/conversation/resolvers.py`.
- Current projected nodes are deterministic `reference_pointer`
  conversation nodes linked with `references` edges.
- The implementation still carries some backend hints in pointer
  properties such as `refers_to_collection`, so the logical-reference
  model in this document remains the intended boundary rather than a
  fully completed repo-wide fact.
- The current implementation still stores orchestration semantics inside
  the existing workflow engine / conversation engine surfaces rather
  than a separately materialized agent control-plane graph family.
- See `ARD-0013-core-vendor-neutrality-and-sidecar-optionality.md` for
  the current vendor-neutrality and backend-surface status.

---

## 2. High-Level Architecture

The system consists of **three primary graph engines**, plus an internal split within orchestration:

### 2.1 Conversation Graph Engine (Canvas)
**Purpose:**  
Human-facing, per-conversation provenance and memory.

**Owns:**
- User turn nodes
- Assistant response nodes
- Agent run anchor nodes
- Projected evidence pointers
- Summary nodes

**Invariant:**  
Append-only provenance for a single conversation thread.

---

### 2.2 Orchestration Graph Engine (Control Plane + Trace)

Split into **two conceptual graphs**:

#### A. Orchestration Control Graph (Templates / Plan Space)
**Purpose:**  
Defines *how the agent can act*.

- Workflow templates
- Step templates
- Branching, retries, loops
- Budget and stop conditions

This graph **can be cyclic**.

#### B. Orchestration Trace Graph (Execution History)
**Purpose:**  
Records *what actually happened in one run*.

- Runs
- Step invocations
- Decisions
- Tool artifacts
- Temporal ordering

This graph is **append-only per run**.

---

### 2.3 Wisdom Graph Engine (Meta-Learning)
**Purpose:**  
Cross-conversation learning and policy priors.

**Owns:**
- Query archetypes
- Workflow signatures
- Aggregated performance metrics
- User evaluations

**Invariant:**  
Stores generalizable patterns, not raw conversation history.

---

## 3. Core Concepts

---

## 4. Agent Run Anchor (Caller Graph Bridge)

### 4.1 Definition
An **AgentRunAnchor** is a node created in the *caller graph* (typically the conversation graph) that represents a single agent execution.

It is the **bridge** between:
- Conversation graph
- Orchestration trace
- Orchestration control template
- Projected evidence

### 4.2 Required Properties
- `run_id` (trace graph)
- `workflow_signature_hash`
- `orchestration_engine_ref`
- `trace_engine_ref`
- `status`
- timestamps (optional)
- model/tool metadata (optional)

### 4.3 Required Edges
- `generated` → AssistantResponse
- `used_knowledge` → Projected KG nodes
- `used_tool_result` → Projected tool results
- optional: `generated_by_step` → specific StepInvocation

---

## 5. Orchestration Control Graph Requirements

### 5.1 Entities
- `WorkflowTemplate`
- `StepTemplate`
- `Gate` / `Decision`
- `Loop`

### 5.2 Edges
- `depends_on`
- `branch_true` / `branch_false`
- `on_success` / `on_failure`
- `loop_back`

### 5.3 Properties
- Stable `workflow_signature_hash`
- Versioning
- Default budgets

### 5.4 Indexability
Workflow templates must be:
- Identifiable by signature hash
- Queryable by metadata and embeddings (optional)

---

## 6. Orchestration Trace Graph Requirements

### 6.1 Entities
- `Run`
- `StepInvocation`
- `Decision`
- `ArtifactRef` (tool outputs, intermediate data)

### 6.2 Required Edges
- `next` (temporal order)
- `instance_of` (StepInvocation → StepTemplate)
- `produced` / `consumed`
- `caused` (Decision → StepInvocation)

### 6.3 Invariants
- Append-only per run
- No mutation of past steps
- Each run references exactly one caller anchor

---

## 7. Cyclic Orchestration & Safety

The orchestration engine **may be cyclic**, but must be gated.

### 7.1 Mandatory Stop Conditions
- Max iterations
- Max tool calls
- Max hop depth
- Max nodes/edges expanded
- Budget exhaustion (tokens, time)

### 7.2 State Fingerprint Cycle Detection
- Compute a compact fingerprint from:
  - approach signature
  - selected candidate IDs
  - hop depth
  - iteration count
- Repetition triggers termination or forced strategy change

### 7.3 Approach Exhaustion
Track attempted strategies (retrieval depth, tools, models).  
Stop when all allowed approaches are tried.

---

## 8. Evidence Projection System

### 8.1 Problem Statement
Knowledge and tool results used by the agent must be:
- Explicitly recorded
- Visible from the conversation graph
- Non-duplicated
- Backend-agnostic

---

### 8.2 Logical Reference Model (Storage-Agnostic)

All projections are based on **logical references**, not backend collections.

#### LogicalRef fields
- `namespace` (e.g. `kg`, `trace`, `tool`)
- `kind` (`node`, `edge`, `artifact`)
- `id` (stable within namespace)

---

### 8.3 Projection Pointer Nodes

Projection creates **pointer nodes** in the caller graph.

#### Pointer Node Properties
- `target_namespace`
- `target_kind`
- `target_id`
- optional backend hints (non-identity)

---

### 8.4 Deterministic Pointer Identity

Pointer node identity is defined by:

```
(scope, pointer_kind, target_namespace, target_kind, target_id)
```

- Scope is usually `conversation_id`
- ID may be deterministic string (not UUID)
- Hashing is optional; structured IDs are allowed

---

### 8.5 Collision Safety
- Deterministic IDs preferred
- If hashing is used:
  - ≥256-bit crypto hash
  - Verify properties on reuse
  - Fail loudly on mismatch

---

### 8.6 Edge Deduplication
Edges are also idempotent.

Canonical identity:
```
(src_id, edge_type, dst_id)
```

---

## 9. Generic Projection API (Conceptual)

A reusable mechanism must exist to:

- Project KG nodes
- Project KG edges
- Project tool artifacts

Responsibilities:
1. Deduplicate pointer nodes
2. Deduplicate edges
3. Enforce projection budgets
4. Rank/select what is considered “used”

---

## 10. Evidence Semantics

### 10.1 “Used” Boundary
A node is considered *used* if it:
- Appears in synthesis context
- Justifies a claim
- Is part of the final reasoning subgraph

Retrieval candidates alone are not automatically “used”.

---

## 11. Cross-Graph Linking Contract

### 11.1 From Conversation → Orchestration
- AgentRunAnchor stores refs to:
  - trace run
  - workflow template

### 11.2 From Orchestration → Caller
- Run stores:
  - `caller_ref` (engine, collection, id)
  - caller kind

This allows **any graph** to invoke orchestration, not only conversations.

---

## 12. Wisdom Layer Integration

### 12.1 Required Join Keys
- `workflow_signature_hash`
- query embedding / archetype

### 12.2 Inputs from Orchestration
- Run summaries
- Cost/latency stats
- Success/failure signals

### 12.3 User Evaluation
Two levels:
- Conversation-level (answer quality)
- Wisdom-level (approach effectiveness)

---

## 13. Concurrency & Execution Model

### 13.1 Within an Attempt
- Execution steps form a DAG
- Independent retrieval steps may run concurrently
- Writes to graphs must be serialized

### 13.2 Across Attempts
- Orchestration loop controls retries/fallbacks
- Each attempt produces its own trace segment

---

## 14. Non-Goals (Explicit)

- No requirement for LLM chain-of-thought storage
- No UI specification
- No hard dependency on a specific graph backend
- No assumption of single-agent future

---

## 15. Summary of Invariants

- Provenance is explicit, not inferred
- Projection is idempotent
- Control ≠ Trace
- Logical identity ≠ backend storage
- Cycles are gated
- Wisdom never mutates live policy mid-run
