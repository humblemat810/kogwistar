# Tutorials Index

Use [docs/LEARNING_PATH.md](../LEARNING_PATH.md) as the main entrypoint. This page is the execution-oriented index: numbered learning-path docs, script-backed ladders, VS Code section-run companions, and legacy ladders that intentionally follow a different depth/pathway.

- Environment variables: [docs/environment-variables.md](../environment-variables.md)

## Learning Path Docs

- [01 Hello Graph Engine](./01_hello_graph_engine.md)
- [02 Core Data Model](./02_core_data_model.md)
- [03 Build a Small Knowledge Graph](./03_build_a_small_knowledge_graph.md)
- [04 Conversation Graph Basics](./04_conversation_graph_basics.md)
- [05 Context Snapshot and Replay](./05_context_snapshot_and_replay.md)
- [06 First Workflow](./06_first_workflow.md)
- [07 Branch Join Workflows](./07_branch_join_workflows.md)
- [08 Storage Backends and Parity](./08_storage_backends_and_parity.md)
- [09 Indexing Pipeline](./09_indexing_pipeline.md)
- [10 Event Log Replay and CDC](./10_event_log_replay_and_cdc.md)
- [11 Build a Mini GraphRAG App](./11_build_a_mini_graphrag_app.md)
- [12 Designer API Integration](./12_designer_api_integration.md)
- [13 How to Test This Repo](./13_how_to_test_this_repo.md)
- [14 Architecture Deep Dive](./14_architecture_deep_dive.md)
- [15 Historical Search With Tombstone and Redirect](./15_historical_search_tombstone_redirect.md)
- [16 Leakage Prevention with Model Slicing](./16_leakage_prevention_with_model_slicing.md)
- [17 Custom LLM Provider (Registry Style)](./17_custom_llm_provider.md)

## VS Code Companion Files

Open these in VS Code and use Run Cell / Run Above / Run Below for notebook-like execution without `.ipynb` files.

- `scripts/tutorial_sections/01_hello_graph_engine.py`
- `scripts/tutorial_sections/03_build_a_small_knowledge_graph.py`
- `scripts/tutorial_sections/04_conversation_graph_basics.py`
- `scripts/tutorial_sections/05_context_snapshot_and_replay.py`
- `scripts/tutorial_sections/06_first_workflow.py`
- `scripts/tutorial_sections/07_branch_join_workflows.py`
- `scripts/tutorial_sections/10_event_log_replay_and_cdc.py`
- `scripts/tutorial_sections/11_build_a_mini_graphrag_app.py`
- `scripts/tutorial_sections/15_historical_search_tombstone_redirect.py`

## RAG Ladder: Simple RAG -> Reinforced RAG

This ladder remains the fastest execution proof for retrieval behavior.

| Level | Focus | Main Script(s) | Runtime Cost | Expected Outcome |
|---|---|---|---|---|
| Level 0 | Simple RAG baseline | `scripts/rag_tutorial_ladder.py level0` | ~10-30s | Deterministic answer with non-empty evidence context |
| Level 1 | Reinforced retrieval orchestration | `scripts/rag_tutorial_ladder.py level1` | ~20-40s | Seeded retrieval changes candidate set under `max_retrieval_level` |
| Level 2 | Reinforced provenance and pinning | `scripts/rag_tutorial_ladder.py level2` | ~20-50s | `reference_pointer` nodes + `references` edges are materialized |
| Level 2b | Equivalent v2 conversation path | `scripts/rag_tutorial_ladder.py level2b` | ~20-50s | `add_turn_workflow_v2(...)` produces the same inspectable evidence pointers through the workflow/runtime registry path, with optional live `gemini` / `openai` / `ollama` answering |
| Level 3 | Reinforced event-sourced loop control | `scripts/claw_runtime_loop.py` | ~1-3m | Inbox/outbox transitions, TTL guardrails, optional CDC stream |

## Runtime Ladder: Resolver -> Pause/Resume -> CDC -> LangGraph

This ladder keeps one canonical workflow all the way through:

`start -> fork -> branch_a(suspends) + branch_b(completes) -> join -> end`

| Level | Focus | Main Script(s) | Runtime Cost | Expected Outcome |
|---|---|---|---|---|
| Level 0 | WorkflowRuntime basics | `scripts/runtime_tutorial_ladder.py level0` | ~5-15s | Suspended run with persisted step exec + checkpoint artifacts |
| Level 1 | Custom resolvers and `_deps` | `scripts/runtime_tutorial_ladder.py level1` | ~5-15s | Resolver registry, injected dependency echo, custom trace events |
| Level 2 | Pause and continue | `scripts/runtime_tutorial_ladder.py level2` | ~5-20s | Suspended token recovered from checkpoint and resumed to completion |
| Level 3 | CDC viewer and LangGraph interop | `scripts/runtime_tutorial_ladder.py level3` | ~5-20s | Trace event inventory, viewer asset path, optional LangGraph export |
| Level 4 | Sandboxed ops and Docker safety | `scripts/runtime_tutorial_ladder.py level4` | ~5-20s | Untrusted generated code is executed in Docker via `SandboxRequest`, not on the host |

## Prerequisites

- Python `3.13`
- Repo dependencies installed, recommended: `pip install -e ".[chroma]"`
- Run from repo root

## One-Time Setup

### RAG Ladder

```powershell
python scripts/rag_tutorial_ladder.py reset --data-dir .gke-data/tutorial-ladder
python scripts/rag_tutorial_ladder.py seed --data-dir .gke-data/tutorial-ladder
```

### Runtime Ladder

```powershell
python scripts/runtime_tutorial_ladder.py reset --data-dir .gke-data/runtime-tutorial-ladder
```

## Legacy Level Docs

These remain supported on purpose. They are not obsolete copies of the numbered learning path; they preserve an alternate depth and pathway for the older ladder-style progression.

- [Runtime Ladder Overview](./runtime-ladder-overview.md)
- [RAG Level 0 - Simple RAG](./level-0-simple-rag.md)
- [RAG Level 1 - Retrieval Orchestration](./level-1-retrieval-orchestration.md)
- [RAG Level 2 - Provenance and Pinning](./level-2-provenance-pinning.md)
- [RAG Level 3 - Event-Sourced Loop Control](./level-3-event-loop-control.md)
- [Runtime Level 0 - WorkflowRuntime Basics](./runtime-level-0-basics.md)
- [Runtime Level 1 - Custom Resolvers](./runtime-level-1-resolvers.md)
- [Runtime Level 2 - Pause and Resume](./runtime-level-2-pause-resume.md)
- [Runtime Level 3 - CDC Viewer and LangGraph Interop](./runtime-level-3-observability-interop.md)
- [Runtime Level 4 - Sandboxed Ops With Docker](./runtime-level-4-sandboxed-ops.md)

## Pattern Matrix

| Problem | Pattern | Where It Is Implemented |
|---|---|---|
| Need quick confidence that retrieval works | Baseline retrieve-then-answer | `scripts/rag_tutorial_ladder.py` (`level0`) |
| Naive top-k misses connected evidence | Seeded retrieval + bounded depth | `kogwistar/conversation/memory_retriever.py`, `kogwistar/conversation/knowledge_retriever.py` |
| Hard to audit what was actually used | Pointer projection with provenance | `reference_pointer` nodes via `MemoryRetriever.pin_selected` and `KnowledgeRetriever.pin_selected` |
| Agent loops can run forever | Event-sourced inbox/outbox + TTL guardrails | `scripts/claw_runtime_loop.py` |
| Need live observability for graph updates | CDC bridge and websocket stream | `scripts/claw_runtime_loop.py run-cdc-bridge` |

