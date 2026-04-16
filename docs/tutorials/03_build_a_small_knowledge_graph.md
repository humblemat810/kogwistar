# 03 Build a Small Knowledge Graph

Audience: Beginner / evaluator
Time: 15-20 minutes
Companion: [scripts/tutorial_sections/03_build_a_small_knowledge_graph.py](../../scripts/tutorial_sections/03_build_a_small_knowledge_graph.py)

## What You Will Build

You will seed a small knowledge graph, retrieve a few entities and relationships, and inspect the evidence-friendly structure that later tutorials reuse.

## Why This Matters

This is the first point where the repo feels like a knowledge system instead of a persistence layer. You start seeing linked evidence, levels, and graph-shaped retrieval inputs.

## Run or Inspect

- Run the companion file in VS Code section by section.
- Or use the script-backed ladder:

```bash
python scripts/rag_tutorial_ladder.py reset --data-dir .gke-data/tutorial-ladder
python scripts/rag_tutorial_ladder.py seed --data-dir .gke-data/tutorial-ladder
python scripts/rag_tutorial_ladder.py level0 --data-dir .gke-data/tutorial-ladder --question "How does this repo implement simple RAG?"
```

- Follow up with [RAG Level 0 - Simple RAG](./level-0-simple-rag.md).

## Inspect The Result

- Inspect the `K:*` node ids and `E:*` edge ids created by the seeded tutorial graph.
- Confirm retrieval returns graph summaries rather than anonymous chunks.
- Check that seeded entities carry enough metadata to support later bounded-depth traversal.

## Invariant Demonstrated

Retrieved evidence is structured and named. The repo does not need to flatten everything into untraceable text blobs before you can ask questions over it.

## Next Tutorial

Continue to [04 Conversation Graph Basics](./04_conversation_graph_basics.md).
