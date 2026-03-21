# 02 Core Data Model

Audience: Beginner / evaluator
Time: 15-20 minutes

## What You Will Build

You will build a mental model of the repo's core stored artifacts: nodes, edges, references, documents, and provenance-bearing metadata.

## Why This Matters

This is where readers stop treating the repo as "just another vector DB wrapper." The project is opinionated about graph identity, provenance, and replayable artifacts.

## Run or Inspect

- Inspect `kogwistar/engine_core/models.py` for `Node`, `Edge`, and provenance-bearing fields.
- Read `kogwistar/docs/ARD-0006-conversation.md` after this page if you want the conversation-specific extension of the same idea.
- Compare this page with [01 Hello Graph Engine](./01_hello_graph_engine.md) so the terminology stays grounded in a runnable example.

## Inspect The Result

- Notice that both nodes and edges carry summaries and mentions.
- Notice that provenance is attached to writes, not bolted on as an afterthought.
- Notice that references are explicit graph artifacts instead of hidden side tables.

## Invariant Demonstrated

Provenance is first-class. Stored graph artifacts are designed to preserve where knowledge came from and what later execution depended on.

## Next Tutorial

Continue to [03 Build a Small Knowledge Graph](./03_build_a_small_knowledge_graph.md).
