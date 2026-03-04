# ARD-0001: Repository Restructuring --- Core, Workflow, Conversation (and Future Wisdom)

**Status:** Proposed (intended for incremental adoption)\
**Date:** 2026-03-04\
**Owner:** Maintainers

------------------------------------------------------------------------

# 1. Summary

This document defines the architectural restructuring of the repository
into layered modules:

-   `core/` --- graph substrate, storage, UoW, stable IDs
-   `workflow/` --- generic workflow runtime and resolver mechanism
-   `conversation/` --- conversation domain workflows and step
    implementations
-   `wisdom/` --- future learning/evaluation loops (not implemented yet)

The goal is to **separate infrastructure from domain logic** while
keeping the workflow runtime reusable for multiple domains.

------------------------------------------------------------------------

# 2. Problem Statement

Currently the repository mixes:

-   workflow runtime mechanics
-   conversation domain operations
-   graph storage primitives

This leads to:

-   circular imports
-   "god object" pressure in `GraphKnowledgeEngine`
-   difficulty reusing workflow runtime for non-conversation jobs
-   domain logic appearing inside infrastructure modules

The repository must be reorganized to enforce clean architectural
boundaries.

------------------------------------------------------------------------

# 3. Architectural Decision

The repository will follow a **layered architecture**.

Dependency direction:

wisdom → conversation → workflow → core

Rules:

-   `core` imports nothing from `workflow` or `conversation`
-   `workflow` may import `core` but never `conversation`
-   `conversation` may import both `workflow` and `core`
-   `wisdom` may import all lower layers

------------------------------------------------------------------------

# 4. Responsibilities by Layer

## 4.1 Core Layer

Location:

core/

Responsibilities:

-   graph storage engine
-   node/edge primitives
-   UoW / transactions
-   lifecycle operations
-   stable ID generation
-   storage backend integrations
-   minimal infrastructure ports

Examples:

core/engine.py\
core/models.py\
core/ids.py\
core/ports.py\
core/storage/

Core must **not contain domain logic**.

Conversation-specific APIs should eventually be moved out of core.

------------------------------------------------------------------------

## 4.2 Workflow Layer

Location:

workflow/

Responsibilities:

-   workflow runtime
-   execution scheduling
-   join/fanout semantics
-   resolver dispatch mechanism
-   workflow graph design primitives
-   step context definitions
-   state merge policies

Examples:

workflow/runtime.py\
workflow/contract.py\
workflow/design.py\
workflow/registry.py\
workflow/state_merge.py

Workflow runtime must remain **domain agnostic**.

It must **never import conversation modules**.

------------------------------------------------------------------------

## 4.3 Conversation Layer

Location:

conversation/

Responsibilities:

-   conversation graph semantics
-   conversation workflow orchestration
-   domain step implementations
-   agentic answering workflow
-   conversation state models
-   conversation API facade

Examples:

conversation/orchestrator.py\
conversation/api.py\
conversation/schema.py

conversation/steps/\
conversation/steps/agentic/

Conversation registers its operations into the workflow runtime.

------------------------------------------------------------------------

## 4.4 Wisdom Layer (Future)

Location:

wisdom/

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

They are implemented in `conversation/steps`.

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

conversation/api.py

Core may temporarily keep **compatibility shims** that forward to
conversation APIs.

------------------------------------------------------------------------

# 11. Migration Plan

Phase 1 --- Create module directories:

core/\
workflow/\
conversation/

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

# End of ARD-0001
