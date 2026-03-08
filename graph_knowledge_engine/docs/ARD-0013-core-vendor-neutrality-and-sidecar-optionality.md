# ARD-0013: Core Vendor Neutrality and Sidecar Vendor Optionality

**Status:** Accepted (implementation largely landed; cleanup ongoing)\
**Date:** 2026-03-08\
**Owner:** Maintainers

------------------------------------------------------------------------

# 1. Summary

The repository is now materially aligned with this boundary:

- core LLM-mediated behavior is expressed through vendor-neutral task
  contracts
- concrete LLM providers and embedding integrations are optional
  adapters
- backend/provider installation is modeled through extras
- sidecars remain allowed to be concrete, but they are not the core
  architectural contract

The main remaining issues are:

- package-root import safety
- import-time side effects in service entrypoints
- Chroma-shaped collection surfaces that still leak into strategies,
  visualization, and some engine-facing protocols

This is no longer primarily an LLM architecture problem. It is mostly an
import-boundary and backend-surface cleanup problem.

------------------------------------------------------------------------

# 2. Problem Statement

Historically, the repository mixed:

- graph engine and storage behavior
- concrete LLM provider setup
- service and sidecar startup behavior

That caused three classes of problems:

- importing core-oriented modules could drag in concrete providers or
  backend-specific code paths
- `GraphKnowledgeEngine` became the place where vendor defaults were
  chosen rather than only where abstractions were consumed
- sidecar entrypoints such as the MCP server performed engine
  initialization at module import time

The repository has now been refactored enough that the desired
architecture is visible in code. This document updates the decision to
match the current state rather than a future aspiration.

------------------------------------------------------------------------

# 3. Current Observed State

The following statements are based on direct inspection of the
repository as of 2026-03-08.

## 3.1 A vendor-neutral LLM task contract now exists

`graph_knowledge_engine/llm_tasks/contracts.py` defines the central LLM
boundary:

- `LLMTaskSet`
- `LLMTaskProviderHints`
- task request/result dataclasses for extraction, adjudication,
  filtering, summarization, answer generation, and citation repair
- `validate_llm_task_set(...)`

This is the actual contract that core and conversation code now consume.

## 3.2 Concrete provider wiring is isolated behind an adapter module

`graph_knowledge_engine/llm_tasks/default_provider.py` is the default
provider adapter.

It:

- imports `langchain_google_genai` only inside the Gemini runner branch
- imports `langchain_openai` only inside the OpenAI runner branch
- raises explicit extra-install guidance when those provider dependencies
  are missing

This is the correct place for vendor-aware provider logic.

`graph_knowledge_engine/llm_tasks/__init__.py` still re-exports
`DefaultTaskProviderConfig` and `build_default_llm_tasks(...)` eagerly,
but that package import does not itself import vendor SDKs at module
scope.

## 3.3 Azure OpenAI embeddings have also been isolated behind an integration helper

`graph_knowledge_engine/integrations/openai_embeddings.py` contains
`build_azure_embedding_fn_from_env()`.

That helper imports `AzureOpenAIEmbeddings` lazily and returns `None`
when dependency or configuration is absent.

This is an improvement over embedding-provider imports living directly in
the engine constructor.

## 3.4 Core LLM call sites have been moved onto the task seam

The following files now use `LLMTaskSet` instead of importing concrete
LLM SDK classes:

- `graph_knowledge_engine/engine_core/subsystems/extract.py`
- `graph_knowledge_engine/engine_core/subsystems/ingest.py`
- `graph_knowledge_engine/strategies/adjudicators.py`
- `graph_knowledge_engine/conversation/filtering.py`
- `graph_knowledge_engine/conversation/knowledge_retriever.py`
- `graph_knowledge_engine/conversation/memory_retriever.py`
- `graph_knowledge_engine/conversation/agentic_answering.py`
- `graph_knowledge_engine/conversation/conversation_orchestrator.py`
- `graph_knowledge_engine/conversation/service.py`

This is the strongest evidence that core LLM behavior is now mostly
vendor-neutral.

## 3.5 Extraction mode selection is now provider-hint based

`graph_knowledge_engine/engine_core/subsystems/extract.py` no longer
uses a Gemini SDK type check to decide schema mode behavior.

It now reads:

- `self._e.llm_tasks.provider_hints.extract_graph_provider`

and resolves `auto` mode from provider hints rather than vendor class
identity.

## 3.6 Tests already exercise injected task sets

Tests now routinely build deterministic or fake `LLMTaskSet` instances,
including:

- `tests/core/test_extraction_schema_modes.py`
- `tests/workflow/test_answer_workflow_v2.py`
- `tests/primitives/test_adjudication_merge.py`
- `tests/primitives/test_adjudication_merge_positive.py`
- `tests/kg_conversation/test_conversation_flow_v2_param_e2e.py`

This confirms that task injection is part of the exercised design, not
just an unused abstraction.

## 3.7 Packaging metadata now reflects optional providers and backends

`pyproject.toml` currently declares:

- base dependencies including `typing-extensions`
- backend extras: `chroma`, `pgvector`
- provider extras: `openai`, `gemini`
- optional surface extras: `langgraph`, `ingestion-openai`,
  `ingestion-gemini`, `mcp-adapter`, `viz`
- aggregate extra: `full`

This is directionally correct and materially better than the previous
state.

The earlier `TypeAlias` portability issue has also been addressed in
code:

- `graph_knowledge_engine/typing_interfaces.py` falls back to
  `typing_extensions.TypeAlias`
- `graph_knowledge_engine/engine_core/engine.py` falls back to
  `typing_extensions.Self` and `TypeAlias`

## 3.8 Import safety is improved, but package roots are still heavier than they should be

There are meaningful improvements:

- `graph_knowledge_engine/conversation/__init__.py` uses lazy
  `__getattr__`
- `graph_knowledge_engine/runtime/contract.py` uses `TYPE_CHECKING` for
  `GraphKnowledgeEngine`
- `graph_knowledge_engine/typing_interfaces.py` wraps Chroma type imports
  in `try/except`

However, several package roots still import more than they should:

- `graph_knowledge_engine/engine_core/__init__.py` eagerly imports
  `GraphKnowledgeEngine`
- `graph_knowledge_engine/runtime/__init__.py` eagerly imports
  `runtime.design`
- `graph_knowledge_engine/runtime/design.py` still imports
  `GraphKnowledgeEngine` directly
- `graph_knowledge_engine/engine_core/subsystems/__init__.py` still
  eagerly imports all subsystem modules

These are import-boundary issues, not vendor-SDK leaks in the old sense.

## 3.9 `GraphKnowledgeEngine` is now mostly LLM-vendor-neutral, but still owns default adapter selection

`graph_knowledge_engine/engine_core/engine.py` no longer imports Gemini
or OpenAI chat SDKs directly at module scope.

It does still:

- import `ChromaBackend` at module scope
- default to Chroma when `backend is None`
- construct a default task provider when `llm_tasks` is not injected
- enable best-effort Azure embedding support through
  `build_azure_embedding_fn_from_env()`

This means the engine is no longer the primary LLM vendor leak, but it
still owns convenience defaults that are stricter than the target
boundary.

## 3.10 True backend neutrality is not complete

The storage adapter seam exists in:

- `graph_knowledge_engine/engine_core/storage_backend.py`
- `graph_knowledge_engine/engine_core/chroma_backend.py`
- `graph_knowledge_engine/engine_core/postgres_backend.py`

But Chroma-shaped surfaces still leak above that seam.

Examples:

- `graph_knowledge_engine/typing_interfaces.py`
- `graph_knowledge_engine/strategies/types.py`
- `graph_knowledge_engine/strategies/proposer.py`
- `graph_knowledge_engine/strategies/merge_policies.py`
- `graph_knowledge_engine/strategies/verifiers.py`
- `graph_knowledge_engine/visualization/basic_visualization.py`
- `graph_knowledge_engine/visualization/graph_viz.py`
- `graph_knowledge_engine/graph_query.py`

These files still depend on raw collection attributes such as:

- `node_collection`
- `edge_collection`
- `edge_endpoints_collection`
- `document_collection`
- `node_docs_collection`

So the repo is now much closer to LLM-vendor neutrality than to full
backend neutrality.

## 3.11 Sidecars are still concrete, and one major sidecar still has import-time setup

Concrete sidecar code remains acceptable in:

- `graph_knowledge_engine/server_mcp_with_admin.py`
- `graph_knowledge_engine/ocr.py`
- `graph_knowledge_engine/graph_navigation_agent/*`
- `graph_knowledge_engine/diagnostic/*`

The main remaining sidecar packaging issue is
`graph_knowledge_engine/server_mcp_with_admin.py`, which still:

- imports `GraphKnowledgeEngine` directly
- creates `engine`, `conversation_engine`, and `wisdom_engine` at module
  import time

That is still too heavy for a clean optional sidecar boundary.

------------------------------------------------------------------------

# 4. Architectural Decision

The repository will standardize on the following interpretation of the
current refactor.

## 4.1 Core LLM behavior is defined by `LLMTaskSet`

Core and conversation code must express LLM-mediated behavior through:

- `LLMTaskSet`
- task request/result objects
- provider hints

Core modules should not depend on concrete provider SDK classes.

## 4.2 Concrete providers remain optional adapters

Concrete provider implementations such as Gemini and OpenAI remain
supported, but only through optional adapter modules such as:

- `graph_knowledge_engine/llm_tasks/default_provider.py`
- `graph_knowledge_engine/integrations/openai_embeddings.py`

These modules are allowed to be vendor-aware. They are not the
architecture boundary for core logic.

## 4.3 Sidecars may remain concrete

Service surfaces and auxiliary tools may remain vendor-aware, including:

- MCP server
- OCR helpers
- navigation/agent sidecars
- diagnostics

They must remain optional and should not define the base package import
contract.

## 4.4 Backend neutrality is a separate concern from LLM neutrality

The storage adapter seam is real and should be preserved.

However, the repository should not pretend that full backend neutrality
is already complete while strategies, visualization, and graph query
code still rely on Chroma-shaped collection attributes.

------------------------------------------------------------------------

# 5. Boundary Rules

## 5.1 Core-neutral surfaces

The following are valid neutral surfaces:

- `graph_knowledge_engine/llm_tasks/contracts.py`
- `graph_knowledge_engine/llm_tasks/errors.py`
- `graph_knowledge_engine/engine_core/storage_backend.py`
- `graph_knowledge_engine/engine_core/subsystems/*`
- `graph_knowledge_engine/runtime/contract.py`
- `graph_knowledge_engine/runtime/runtime.py`
- `graph_knowledge_engine/conversation/*`

These modules may depend on task contracts, storage contracts, and
injected dependencies.

## 5.2 Optional adapter surfaces

The following may remain concrete and optional:

- `graph_knowledge_engine/llm_tasks/default_provider.py`
- `graph_knowledge_engine/integrations/openai_embeddings.py`
- `graph_knowledge_engine/ocr.py`
- `graph_knowledge_engine/server_mcp_with_admin.py`
- `graph_knowledge_engine/graph_navigation_agent/*`
- `graph_knowledge_engine/diagnostic/*`

## 5.3 Transitional surfaces

The following remain transitional and should be treated carefully:

- `graph_knowledge_engine/engine_core/engine.py`
- `graph_knowledge_engine/engine_core/__init__.py`
- `graph_knowledge_engine/runtime/__init__.py`
- `graph_knowledge_engine/runtime/design.py`
- `graph_knowledge_engine/typing_interfaces.py`
- `graph_knowledge_engine/strategies/*`
- `graph_knowledge_engine/visualization/*`
- `graph_knowledge_engine/graph_query.py`

These should move toward cleaner boundaries incrementally, not through a
wholesale rewrite.

------------------------------------------------------------------------

# 6. Requirements

## 6.1 Import safety requirements

The following imports must not require concrete provider SDKs merely to
load the module:

- `graph_knowledge_engine.llm_tasks.contracts`
- `graph_knowledge_engine.typing_interfaces`
- `graph_knowledge_engine.runtime.contract`
- `graph_knowledge_engine.runtime.runtime`
- `graph_knowledge_engine.conversation.__init__`

`graph_knowledge_engine.llm_tasks` may continue to re-export the default
provider builder, but vendor SDK imports inside that package must remain
lazy.

## 6.2 Core task requirements

All new LLM-mediated behavior in core and conversation code must be
expressible through `LLMTaskSet`.

New code must not add direct Gemini/OpenAI SDK dependencies to core
call sites when the same behavior can be expressed through the existing
task contract.

## 6.3 Default adapter requirements

If `GraphKnowledgeEngine` continues to provide default task-provider or
embedding behavior, that behavior must remain:

- lazy
- optional
- replaceable by explicit injection

The injected abstraction remains the real contract.

## 6.4 Backend-surface requirements

No new code should depend directly on raw collection attributes such as
`node_collection` or `edge_collection` when equivalent behavior can be
expressed through `engine.backend`, `engine.read`, `engine.write`, or
other backend-neutral surfaces.

Existing Chroma-shaped callers may remain temporarily, but they should
not expand the leak.

## 6.5 Sidecar startup requirements

Sidecar modules may construct engines and concrete providers, but they
should do so at startup or explicit factory time rather than as
unavoidable module import side effects.

------------------------------------------------------------------------

# 7. Non-Goals for This Pass

This pass does not require:

- removing Chroma as the default backend
- removing default task-provider construction from
  `GraphKnowledgeEngine`
- rewriting the workflow runtime
- fully migrating all strategy code off raw collection attributes
- fully migrating all visualization code off raw collection attributes
- eliminating vendor-specific OCR, MCP, or agent-sidecar code

Those are follow-on concerns and should not be conflated with the now
mostly-complete core LLM neutrality work.

------------------------------------------------------------------------

# 8. Consequences

The current architecture should now be described this way:

- core LLM behavior is mostly vendor-neutral already
- provider integrations are optional adapters
- sidecars are allowed to stay concrete
- the remaining architectural debt is mostly import-root heaviness and
  Chroma-shaped backend surfaces

Future cleanup should focus on:

- making package `__init__` files lighter
- removing module-scope engine construction from the MCP server
- reducing direct collection access in strategies, visualization, and
  graph query helpers

------------------------------------------------------------------------

# 9. Decision Record

The repository will treat `LLMTaskSet` as the architectural contract for
LLM-mediated behavior in core and conversation code.

Concrete providers and embedding integrations remain supported only as
optional adapters.

Sidecars remain concrete by exception, not by default.

The repo should stop describing the current problem as generic vendor
coupling in core. The more accurate description is:

- core LLM neutrality is mostly in place
- package-root import safety is still uneven
- backend-shape neutrality is still incomplete
