# RAG Ladder and Runtime Ladder

## RAG Ladder: Simple RAG -> Reinforced RAG

This track introduces RAG capabilities in strict levels. Each level reuses previous setup and adds one major concept.

| Level | Focus | Main Script(s) | Runtime Cost | Expected Outcome |
|---|---|---|---|---|
| 0 | Simple RAG baseline | `scripts/rag_tutorial_ladder.py level0` | ~10-30s | Deterministic answer with non-empty evidence context |
| 1 | Reinforced retrieval orchestration | `scripts/rag_tutorial_ladder.py level1` | ~20-40s | Seeded retrieval changes candidate set under `max_retrieval_level` |
| 2 | Reinforced provenance and pinning | `scripts/rag_tutorial_ladder.py level2` | ~20-50s | `reference_pointer` nodes + `references` edges are materialized |
| 3 | Reinforced event-sourced loop control | `scripts/claw_runtime_loop.py` | ~1-3m | Inbox/outbox transitions, TTL guardrails, optional CDC stream |

## Runtime Ladder: Resolver -> Pause/Resume -> CDC -> LangGraph

This track focuses on the workflow runtime itself. It uses one canonical example workflow all the way through:

`start -> fork -> branch_a(suspends) + branch_b(completes) -> join -> end`

| Level | Focus | Main Script(s) | Runtime Cost | Expected Outcome |
|---|---|---|---|---|
| 0 | WorkflowRuntime basics | `scripts/runtime_tutorial_ladder.py level0` | ~5-15s | Suspended run with persisted step exec + checkpoint artifacts |
| 1 | Custom resolvers and `_deps` | `scripts/runtime_tutorial_ladder.py level1` | ~5-15s | Resolver registry, injected dependency echo, custom trace events |
| 2 | Pause and continue | `scripts/runtime_tutorial_ladder.py level2` | ~5-20s | Suspended token recovered from checkpoint and resumed to completion |
| 3 | CDC viewer and LangGraph interop | `scripts/runtime_tutorial_ladder.py level3` | ~5-20s | Trace event inventory, viewer asset path, optional LangGraph export |

## Prerequisites

- Python `3.10+`
- Repo dependencies installed (recommended: `pip install -e ".[chroma]"`)
- Run from repo root

## One-Time Setup (RAG Ladder)

```powershell
python scripts/rag_tutorial_ladder.py reset --data-dir .gke-data/tutorial-ladder
python scripts/rag_tutorial_ladder.py seed --data-dir .gke-data/tutorial-ladder
```

## Level Docs

- [RAG Level 0 - Simple RAG](./level-0-simple-rag.md)
- [RAG Level 1 - Retrieval Orchestration](./level-1-retrieval-orchestration.md)
- [RAG Level 2 - Provenance and Pinning](./level-2-provenance-pinning.md)
- [RAG Level 3 - Event-Sourced Loop Control](./level-3-event-loop-control.md)
- [Runtime Level 0 - WorkflowRuntime Basics](./runtime-level-0-basics.md)
- [Runtime Level 1 - Custom Resolvers](./runtime-level-1-resolvers.md)
- [Runtime Level 2 - Pause and Resume](./runtime-level-2-pause-resume.md)
- [Runtime Level 3 - CDC Viewer and LangGraph Interop](./runtime-level-3-observability-interop.md)

## Pattern Matrix

| Problem | Pattern | Where It Is Implemented |
|---|---|---|
| Need quick confidence that retrieval works | Baseline retrieve-then-answer | `scripts/rag_tutorial_ladder.py` (`level0`) |
| Naive top-k misses connected evidence | Seeded retrieval + bounded depth | `graph_knowledge_engine/conversation/memory_retriever.py`, `graph_knowledge_engine/conversation/knowledge_retriever.py` |
| Hard to audit what was actually used | Pointer projection with provenance | `reference_pointer` nodes via `MemoryRetriever.pin_selected` and `KnowledgeRetriever.pin_selected` |
| Agent loops can run forever | Event-sourced inbox/outbox + TTL guardrails | `scripts/claw_runtime_loop.py` |
| Need live observability for graph updates | CDC bridge and websocket stream | `scripts/claw_runtime_loop.py run-cdc-bridge` |

