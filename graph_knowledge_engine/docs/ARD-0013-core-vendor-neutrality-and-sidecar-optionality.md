# ARD-0013: Core Vendor Neutrality and Sidecar Vendor Optionality

**Status:** Proposed (aligned to an in-progress refactor)\
**Date:** 2026-03-07\
**Owner:** Maintainers

------------------------------------------------------------------------

# 1. Summary

This document defines the repository direction that is already partially
implemented:

- core packages must depend on vendor-neutral contracts, not concrete LLM
  SDKs
- concrete LLM providers must be optional and isolated behind explicit
  task/provider adapters
- storage backends must be optional, with backend-specific dependencies
  loaded only when selected
- sidecar and service entrypoints may remain vendor-aware, but they must
  not define the core architectural contract

The current codebase is already close to this shape. The main remaining
work is import-boundary cleanup and removal of a small number of
import-time side effects.

------------------------------------------------------------------------

# 2. Problem Statement

Historically, the repository mixed three concerns in the same import
surface:

- graph engine and storage operations
- vendor-specific LLM provider wiring
- sidecar and service entrypoint behavior

That caused three concrete problems:

- importing core modules could require Chroma, Gemini, OpenAI, or
  PostgreSQL-related packages even when the caller only wanted contracts
  or runtime helpers
- `GraphKnowledgeEngine` became the place where vendor defaults were
  chosen and instantiated
- service entrypoints such as the MCP server executed heavy engine setup
  at module import time rather than at application startup

The repository is now being refactored so that the core remains usable
without committing to a specific model vendor, while provider-specific
behavior moves behind optional extras and sidecar-facing adapters.

------------------------------------------------------------------------

# 3. Current Observed State

The following behaviors are already present in the repository as of
2026-03-07.

## 3.1 Task abstraction now exists

`graph_knowledge_engine/llm_tasks/contracts.py` defines a vendor-neutral
task surface:

- `LLMTaskSet`
- `ExtractGraphTaskRequest` / `ExtractGraphTaskResult`
- `AdjudicatePairTaskRequest` / `AdjudicatePairTaskResult`
- `AdjudicateBatchTaskRequest` / `AdjudicateBatchTaskResult`
- `FilterCandidatesTaskRequest` / `FilterCandidatesTaskResult`
- `SummarizeContextTaskRequest` / `SummarizeContextTaskResult`
- `AnswerWithCitationsTaskRequest` / `AnswerWithCitationsTaskResult`
- `RepairCitationsTaskRequest` / `RepairCitationsTaskResult`
- `LLMTaskProviderHints`

This is the main vendor-neutral LLM seam for core and conversation code.

## 3.2 Default provider wiring has moved behind an optional adapter

`graph_knowledge_engine/llm_tasks/default_provider.py` lazily imports
`langchain_google_genai` and `langchain_openai` only when a concrete
provider runner is requested.

If the dependency is missing, it returns a failing runner with explicit
guidance such as:

- `pip install 'kogwistar[gemini]'`
- `pip install 'kogwistar[openai]'`

This file is allowed to remain vendor-aware because it is the provider
adapter, not the core contract.

## 3.3 Core extraction paths now consume task contracts

`graph_knowledge_engine/engine_core/subsystems/extract.py` and
`graph_knowledge_engine/engine_core/subsystems/ingest.py` call
`self._e.llm_tasks.extract_graph(...)` rather than importing a concrete
provider SDK directly.

`extract.py` also resolves extraction mode from
`self._e.llm_tasks.provider_hints.extract_graph_provider` rather than a
Gemini class check.

## 3.4 Adjudication and conversation paths already use the same task seam

`graph_knowledge_engine/strategies/adjudicators.py` calls
`self.e.llm_tasks.adjudicate_pair(...)` and
`self.e.llm_tasks.adjudicate_batch(...)`.

Conversation modules such as:

- `graph_knowledge_engine/conversation/agentic_answering.py`
- `graph_knowledge_engine/conversation/filtering.py`
- `graph_knowledge_engine/conversation/knowledge_retriever.py`
- `graph_knowledge_engine/conversation/memory_retriever.py`
- `graph_knowledge_engine/conversation/conversation_orchestrator.py`

already accept or propagate `LLMTaskSet` rather than requiring a vendor
SDK type.

## 3.5 Tests already validate injection of custom task sets

Multiple tests build deterministic or fake `LLMTaskSet` instances rather
than depending on live vendor SDKs, including:

- `tests/core/test_extraction_schema_modes.py`
- `tests/workflow/test_answer_workflow_v2.py`
- `tests/primitives/test_adjudication_merge.py`
- `tests/primitives/test_adjudication_merge_positive.py`
- `tests/kg_conversation/test_conversation_flow_v2_param_e2e.py`

This confirms that task injection is not only theoretical; it is already
part of the exercised design.

## 3.6 Storage backends are now modeled as optional extras

`pyproject.toml` now declares backend and provider extras, including:

- `chroma`
- `pgvector`
- `openai`
- `gemini`
- `langgraph`
- `ingestion-openai`
- `ingestion-gemini`
- `mcp-adapter`
- `viz`
- `full`

This is directionally correct and matches the intended packaging model.

## 3.7 Backend imports are partially lazy

`graph_knowledge_engine/engine_core/engine.py` contains helper functions
that defer backend-specific imports:

- `_import_chroma_client()`
- `_is_pgvector_backend_instance(...)`
- `_build_postgres_uow_if_needed(...)`

These raise explicit optional-dependency errors instead of assuming the
backend package is always installed.

## 3.8 Core is not fully vendor-neutral yet

`graph_knowledge_engine/engine_core/engine.py` still:

- imports `ChromaBackend` at module scope
- defaults to Chroma when `backend is None`
- builds a default task provider via `build_default_llm_tasks(...)`
  inside `GraphKnowledgeEngine.__init__`
- still tries to enable Azure OpenAI embeddings opportunistically during
  engine construction

This means the core is in transition. The new seam exists, but the main
engine constructor still carries vendor-selection behavior.

## 3.9 Import safety is improved, but not complete

`graph_knowledge_engine/engine_core/__init__.py` now lazy-loads pgvector
symbols via `__getattr__`, which is an improvement.

However:

- it still imports `GraphKnowledgeEngine` eagerly
- `graph_knowledge_engine/runtime/__init__.py` still imports
  `graph_knowledge_engine/runtime/design.py` eagerly
- `graph_knowledge_engine/runtime/design.py` still imports
  `GraphKnowledgeEngine`

So runtime-oriented imports can still pull engine code into otherwise
lightweight paths.

## 3.10 Some non-core layers still expose Chroma-shaped assumptions

`graph_knowledge_engine/typing_interfaces.py` now safely wraps Chroma
type imports in a `try/except`, which improves optional dependency
safety.

But its `EngineLike` protocol still exposes:

- `node_collection`
- `edge_collection`
- `edge_endpoints_collection`
- `document_collection`
- `node_docs_collection`

and strategy/visualization code still depends on those raw collection
attributes. Examples:

- `graph_knowledge_engine/strategies/proposer.py`
- `graph_knowledge_engine/strategies/types.py`
- `graph_knowledge_engine/visualization/basic_visualization.py`

This is a backend-shape leak, not a task-provider leak.

## 3.11 Sidecar packaging is not fully clean yet

`graph_knowledge_engine/server_mcp_with_admin.py` still creates multiple
`GraphKnowledgeEngine` instances at module import time:

- `engine = GraphKnowledgeEngine(...)`
- `conversation_engine = GraphKnowledgeEngine(...)`
- `wisdom_engine = GraphKnowledgeEngine(...)`

This is acceptable as sidecar-specific behavior conceptually, but it is
not yet safe packaging behavior because import and initialization are
still coupled.

------------------------------------------------------------------------

# 4. Architectural Decision

The repository will standardize on the following boundary:

## 4.1 Core packages are vendor-neutral

Core packages may define:

- graph models
- storage interfaces
- workflow/runtime contracts
- task contracts
- provider hints
- engine composition points

Core packages must not require concrete LLM SDK imports in order to
import their contracts or operate on injected abstractions.

## 4.2 Concrete LLM providers are adapters, not core dependencies

Concrete provider wiring belongs in optional provider modules such as:

- `graph_knowledge_engine/llm_tasks/default_provider.py`

Equivalent future provider adapters are allowed, but they must all obey
the `LLMTaskSet` contract.

## 4.3 Sidecars own concrete defaults

MCP servers, OCR helpers, diagnostic tools, and navigation agents may
remain concrete and vendor-aware. These modules are sidecars or service
surfaces, not the core package contract.

They must use optional extras and delayed initialization rather than
forcing all concrete vendors into the base import path.

## 4.4 Backend choice is explicit and optional

Backends such as Chroma and pgvector remain supported, but backend
dependencies must be loaded only when that backend is actually selected.

Core code may depend on storage interfaces and backend adapters, but not
on unconditional installation of all backend libraries.

------------------------------------------------------------------------

# 5. Boundary Rules

## 5.1 Allowed in core

The following are valid core responsibilities:

- `graph_knowledge_engine/engine_core/storage_backend.py`
- `graph_knowledge_engine/engine_core/chroma_backend.py`
- `graph_knowledge_engine/engine_core/postgres_backend.py`
- `graph_knowledge_engine/engine_core/subsystems/*`
- `graph_knowledge_engine/runtime/*`
- `graph_knowledge_engine/conversation/*`
- `graph_knowledge_engine/llm_tasks/contracts.py`

These modules may reference core contracts and injected task sets.

## 5.2 Allowed as optional sidecar or adapter code

The following may remain vendor-aware:

- `graph_knowledge_engine/llm_tasks/default_provider.py`
- `graph_knowledge_engine/ocr.py`
- `graph_knowledge_engine/server_mcp_with_admin.py`
- `graph_knowledge_engine/graph_navigation_agent/*`
- `graph_knowledge_engine/diagnostic/*`

These must be treated as optional surfaces, not as baseline import
requirements for the entire package.

## 5.3 Transitional areas

The following areas are still architecturally transitional:

- `graph_knowledge_engine/engine_core/engine.py`
- `graph_knowledge_engine/engine_core/__init__.py`
- `graph_knowledge_engine/runtime/__init__.py`
- `graph_knowledge_engine/runtime/design.py`
- `graph_knowledge_engine/typing_interfaces.py`
- `graph_knowledge_engine/strategies/*`
- `graph_knowledge_engine/visualization/*`

They must move toward the rules in this document, but they should not be
rewritten wholesale in the same pass.

------------------------------------------------------------------------

# 6. Requirements

## 6.1 Import safety requirements

The following imports must not require Gemini/OpenAI/Chroma/pgvector
packages merely to load the module:

- `graph_knowledge_engine.llm_tasks`
- `graph_knowledge_engine.typing_interfaces`
- `graph_knowledge_engine.runtime.contract`
- `graph_knowledge_engine.runtime.runtime`
- `graph_knowledge_engine.conversation.*` contract-level modules

`graph_knowledge_engine.engine_core` should move in this direction as
far as practical, but the current implementation is still transitional.

## 6.2 Task contract requirements

All LLM-mediated core behavior must be expressible through `LLMTaskSet`.

Core call sites must depend on task requests and task results rather
than vendor SDK classes.

## 6.3 Missing dependency behavior

When a concrete provider or backend is selected without the required
dependency installed, the failure must:

- happen at provider/backend selection or first use
- explain which extra is required
- not masquerade as a generic import failure deep in unrelated modules

## 6.4 Sidecar behavior requirements

Sidecar modules may construct engines and concrete providers, but they
must do so at startup or explicit factory time, not as unavoidable
module import side effects.

## 6.5 Testing requirements

Tests for workflow, conversation, extraction, and adjudication behavior
should prefer injected fake or deterministic `LLMTaskSet` instances over
live vendor SDK calls.

------------------------------------------------------------------------

# 7. Non-Goals for This Pass

This refactor pass should not attempt to:

- remove Chroma as the default engine backend
- redesign the workflow runtime
- rewrite `GraphKnowledgeEngine` from scratch
- fully migrate all strategy code off raw collection attributes
- fully migrate all visualization code off raw collection attributes
- eliminate sidecar-specific vendor integrations such as OCR, MCP, or
  LangGraph-based agent helpers

Those are separable follow-on concerns.

------------------------------------------------------------------------

# 8. Consequences

If this document is followed, the repository gains:

- a stable vendor-neutral task seam for core and conversation logic
- package extras that map to real backend/provider choices
- clearer separation between library behavior and sidecar/service
  behavior
- simpler testing through task injection

The remaining cleanup should focus on:

- removing eager engine imports from package `__init__` files
- moving sidecar engine construction out of module scope
- reducing direct use of raw collection attributes in
  `strategies/` and `visualization/`

------------------------------------------------------------------------

# 9. Decision Record

The repository will treat `LLMTaskSet` as the architectural contract for
LLM-mediated behavior in core and conversation code.

Concrete vendors such as Gemini and OpenAI remain supported, but only as
optional adapters and extras.

Sidecars may stay concrete, but they do not define the core boundary.
