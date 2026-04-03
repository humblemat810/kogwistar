# Learning Path

This repo is a graph-native RAG and workflow runtime that treats knowledge, conversations, execution history, and provenance as inspectable graph artifacts instead of opaque helper state.

If the README uses terms like `substrate`, `harness`, `projection`, or `replay`, this page is the next stop. It explains the order in which the repo should be learned.

## Repo Map

- Knowledge graph engine: nodes, edges, provenance, persistence, retrieval.
- Conversation graph: turns, summaries, references, memory, snapshots.
- Workflow runtime: branching, joins, suspend/resume, checkpoints, traces.
- Storage and indexing: local-first persistence, replayable projections, parity across backends.
- Integration surfaces: designer/runtime APIs, CDC viewers, contributor tests.
- Architecture framing: see [positioning.md](./positioning.md) for the layer model and guarantees.

## Who Should Read What

- Beginner / evaluator: start at Tutorials 01-04 if you want quick proof the repo is real.
- Builder / integrator: continue with Tutorials 05-09 and 11-12 if you need to build on it.
- Advanced / contributor: finish with Tutorials 10, 13, 14, 15, 18, and 19 if you need replay, invariants, lifecycle audit, nested runtime orchestration, or encoded governance boundaries.

## Recommended Order

1. [01 Hello Graph Engine](./tutorials/01_hello_graph_engine.md)
2. [02 Core Data Model](./tutorials/02_core_data_model.md)
3. [03 Build a Small Knowledge Graph](./tutorials/03_build_a_small_knowledge_graph.md)
4. [04 Conversation Graph Basics](./tutorials/04_conversation_graph_basics.md)
5. [05 Context Snapshot and Replay](./tutorials/05_context_snapshot_and_replay.md)
6. [06 First Workflow](./tutorials/06_first_workflow.md)
7. [07 Branch Join Workflows](./tutorials/07_branch_join_workflows.md)
8. [08 Storage Backends and Parity](./tutorials/08_storage_backends_and_parity.md)
9. [09 Indexing Pipeline](./tutorials/09_indexing_pipeline.md)
10. [10 Event Log Replay and CDC](./tutorials/10_event_log_replay_and_cdc.md)
11. [11 Build a Mini GraphRAG App](./tutorials/11_build_a_mini_graphrag_app.md)
12. [12 Designer API Integration](./tutorials/12_designer_api_integration.md)
13. [13 How to Test This Repo](./tutorials/13_how_to_test_this_repo.md)
14. [14 Architecture Deep Dive](./tutorials/14_architecture_deep_dive.md)
15. [15 Historical Search With Tombstone and Redirect](./tutorials/15_historical_search_tombstone_redirect.md)
16. [16 Leakage Prevention with Model Slicing](./tutorials/16_leakage_prevention_with_model_slicing.md)
17. [17 Custom LLM Provider (Registry Style)](./tutorials/17_custom_llm_provider.md)
18. [18 Nested Workflow Invocation](./tutorials/18_nested_workflow_invocation.md)
19. [19 Build Artifact Governance Workflow](./tutorials/19_build_artifact_governance_workflow.md)

## How To Use The Tutorials

- Read the numbered Markdown pages first for the mental model.
- Open the companion `scripts/tutorial_sections/*.py` files in VS Code when you want notebook-like execution with `# %%` cells.
- Use [docs/tutorials/README.md](./tutorials/README.md) when you want the script-backed ladders and runtime checkpoints directly.
- If you want the shortest path to the architecture answer, read [positioning.md](./positioning.md) before the later tutorials.

## Existing Script Ladders

- RAG ladder: baseline retrieval, seeded retrieval, provenance pinning, event-sourced loop control.
- Runtime ladder: resolver basics, injected dependencies, suspend/resume, CDC viewer and LangGraph export.

Those ladders remain the execution anchors. The numbered learning path explains why they exist and when to use them.
