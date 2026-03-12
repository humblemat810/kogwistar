# ARD-0012: Repository Restructuring --- Core, Workflow, Conversation (and Future Wisdom)

**Status:** Accepted (conceptual layer split adopted; physical package layout remains compatibility-first)\
**Date:** 2026-03-04\
**Owner:** Maintainers

## Document History

- 2026-03-08: Clarified that `core`, `workflow`, and `conversation` are architectural layer names currently mapped to `graph_knowledge_engine/engine_core/`, `graph_knowledge_engine/runtime/`, and `graph_knowledge_engine/conversation/`. Updated stale path examples and migration wording.
- 2026-03-12: Re-evaluated against the current codebase. Updated the `GraphKnowledgeEngine` wording to reflect that the most serious god-object pressure has been reduced by namespaced subsystems and compatibility shims; the remaining issue is oversized facade/assembly surface, not a single class owning all core behavior.

------------------------------------------------------------------------

# 1. Summary

This document defines the architectural restructuring of the repository
into layered conceptual modules:

-   `core` (currently `graph_knowledge_engine/engine_core/`) --- graph
    substrate, storage, UoW, stable IDs
-   `workflow` (currently `graph_knowledge_engine/runtime/`) ---
    generic workflow runtime and resolver mechanism
-   `conversation` (currently `graph_knowledge_engine/conversation/`)
    --- conversation domain workflows and step implementations
-   `wisdom` (future layer; no dedicated package yet) --- future
    learning/evaluation loops

The goal is to **separate infrastructure from domain logic** while
keeping the workflow runtime reusable for multiple domains.

The layer names in this document are architectural names, not a
requirement that the repository contain literal top-level `core/` or
`workflow/` directories.

------------------------------------------------------------------------

# 2. Problem Statement

Historically the repository mixed:

-   workflow runtime mechanics
-   conversation domain operations
-   graph storage primitives

This led to:

-   circular imports
-   serious "god object" pressure in `GraphKnowledgeEngine`
-   difficulty reusing workflow runtime for non-conversation jobs
-   domain logic appearing inside infrastructure modules

Current status:

-   the runtime/conversation split is now materially better enforced
-   `GraphKnowledgeEngine` remains large, but much of its public API is now a facade over namespaced subsystems such as `read`, `write`, `extract`, `persist`, `rollback`, `adjudicate`, `ingest`, and `embed`
-   the main residual risk is compatibility and assembly bloat in `engine.py`, not the earlier state where one class was the primary owner of most core behavior

The repository must continue enforcing clean architectural
boundaries.

------------------------------------------------------------------------

# 3. Architectural Decision

The repository will follow a **layered architecture**.

Dependency direction:

wisdom â†’ conversation â†’ workflow â†’ core

Rules:

-   `core` imports nothing from `workflow` or `conversation`
-   `workflow` may import `core` but never `conversation`
-   `conversation` may import both `workflow` and `core`
-   `wisdom` may import all lower layers

------------------------------------------------------------------------

# 4. Responsibilities by Layer

## 4.1 Core Layer

Current implementation location:

`graph_knowledge_engine/engine_core/`

Responsibilities:

-   graph storage engine
-   node/edge primitives
-   UoW / transactions
-   lifecycle operations
-   stable ID generation
-   storage backend integrations
-   minimal infrastructure ports

Examples:

graph_knowledge_engine/engine_core/engine.py\
graph_knowledge_engine/engine_core/models.py\
graph_knowledge_engine/engine_core/storage_backend.py\
graph_knowledge_engine/engine_core/chroma_backend.py\
graph_knowledge_engine/engine_core/postgres_backend.py

Core must **not contain domain logic**.

Conversation-specific APIs should eventually be moved out of core.

`GraphKnowledgeEngine` is still the compatibility entrypoint and composition root,
but the namespaced subsystem APIs are now the intended source-of-truth surface for
most core behaviors.

------------------------------------------------------------------------

## 4.2 Workflow Layer

Current implementation location:

`graph_knowledge_engine/runtime/`

Responsibilities:

-   workflow runtime
-   execution scheduling
-   join/fanout semantics
-   resolver dispatch mechanism
-   workflow graph design primitives
-   step context definitions
-   state merge policies

Examples:

graph_knowledge_engine/runtime/runtime.py\
graph_knowledge_engine/runtime/contract.py\
graph_knowledge_engine/runtime/design.py\
graph_knowledge_engine/runtime/resolvers.py\
graph_knowledge_engine/runtime/langgraph_converter.py

Workflow runtime must remain **domain agnostic**.

It must **never import conversation modules**.

------------------------------------------------------------------------

## 4.3 Conversation Layer

Current implementation location:

`graph_knowledge_engine/conversation/`

Responsibilities:

-   conversation graph semantics
-   conversation workflow orchestration
-   domain step implementations
-   agentic answering workflow
-   conversation state models
-   conversation API facade

Examples:

graph_knowledge_engine/conversation/conversation_orchestrator.py\
graph_knowledge_engine/conversation/service.py\
graph_knowledge_engine/conversation/designer.py\
graph_knowledge_engine/conversation/resolvers.py\
graph_knowledge_engine/conversation/agentic_answering.py

Conversation registers its operations into the workflow runtime.

------------------------------------------------------------------------

## 4.4 Wisdom Layer (Future)

Current implementation location:

No dedicated package yet.

Responsibilities:

-   evaluation of conversation outputs
-   improvement loops
-   learning policies
-   system-level analytics

This layer depends on conversation/workflow/core but does not modify
lower layers.

------------------------------------------------------------------------

# 5. Domain Operation Placement

Conversation domain operations must live under `conversation/`.

### Conversation turn flow ops

add_user_turn\
link_prev_turn\
link_assistant_turn\
context_snapshot\
memory_retrieve\
kg_retrieve\
memory_pin\
kg_pin\
answer\
decide_summarize\
summarize

### Agentic answer operations

aa_prepare\
aa_get_view_and_question\
aa_retrieve_candidates\
aa_select_used_evidence\
aa_materialize_evidence_pack\
aa_generate_answer_with_citations\
aa_validate_or_repair_citations\
aa_evaluate_answer\
aa_project_pointers\
aa_maybe_iterate\
aa_persist_response

These operations must **not exist inside workflow modules**.

Today they are implemented primarily through
`graph_knowledge_engine/conversation/resolvers.py`,
`graph_knowledge_engine/conversation/designer.py`, and related
conversation modules. The invariant is logical placement in the
conversation layer, not a required literal `conversation/steps/`
directory.

------------------------------------------------------------------------

# 6. Special Case: start / end Operations

If `start` and `end` are purely structural markers, they belong in
`workflow`.

If they perform conversation-specific side effects, they must be renamed
and moved to conversation:

conv_start\
conv_end

------------------------------------------------------------------------

# 7. Step Pack Registration

Conversation registers its operations into the workflow runtime.

Example pattern:

``` python
def register_conversation_ops(registry):
    registry.register("add_user_turn", add_user_turn)
    registry.register("memory_retrieve", memory_retrieve)
    registry.register("answer", answer)
```

The workflow runtime only knows about **operation names and handlers**,
not conversation semantics.

------------------------------------------------------------------------

# 8. Workflow State Design

Workflow runtime treats state as a **generic dictionary**.

Example:

State = Dict\[str, Any\]

Conversation may provide typed builders for convenience but runtime must
remain domain-neutral.

Example builder:

WorkflowStateModel(...).model_dump()

------------------------------------------------------------------------

# 9. Dependency Injection

Steps may access dependencies through a dependency bundle.

Preferred approach:

ctx.deps: Deps

Alternative transitional approach:

state\["\_deps"\]

------------------------------------------------------------------------

# 10. Core Engine Policy

The core engine must contain only:

-   graph primitives
-   persistence
-   transactions
-   integrity enforcement
-   stable ID helpers

Conversation APIs must eventually move into a conversation facade.

Example:

graph_knowledge_engine/conversation/service.py

Core may temporarily keep **compatibility shims** that forward to
conversation APIs.

------------------------------------------------------------------------

# 11. Migration Plan

Phase 1 --- Establish and preserve the conceptual split in the current
package layout:

graph_knowledge_engine/engine_core/\
graph_knowledge_engine/runtime/\
graph_knowledge_engine/conversation/

Phase 2 --- Remove workflow imports of conversation modules.

Phase 3 --- Split resolver file into mechanism and domain step packs.

Phase 4 --- Inject workflow runtime into orchestrators rather than
constructing it inside methods.

Phase 5 --- Introduce ConversationAPI facade.

Phase 6 --- Enable additional job packs (ingestion, repair, indexing,
wisdom).

------------------------------------------------------------------------

# 12. Guardrails for Code Editing Agents

1.  Do not import `conversation.*` inside `workflow.*`.
2.  Do not implement conversation operations in workflow modules.
3.  Do not subclass `WorkflowRuntime` for conversation logic.
4.  Do not add new conversation-only methods into `core.engine`.
5.  Do not duplicate core node/edge models inside workflow or
    conversation modules.
6.  Rename `start`/`end` if they contain domain behavior.

------------------------------------------------------------------------

# 13. Expected Benefits

-   reusable workflow runtime
-   cleaner architectural boundaries
-   reduced complexity in core engine
-   easier testing and dependency injection
-   safe expansion to wisdom layer

------------------------------------------------------------------------

# 14. Test Design Note

Tests can act as **micro composition roots**, creating:

GraphKnowledgeEngine\
WorkflowRuntime\
ConversationOrchestrator\
FakeLLM

Then executing workflows against them.

------------------------------------------------------------------------

# End of ARD-0012

