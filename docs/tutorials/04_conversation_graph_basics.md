# 04 Conversation Graph Basics

Audience: Beginner / evaluator
Time: 20 minutes
Companion: [scripts/tutorial_sections/04_conversation_graph_basics.py](../../scripts/tutorial_sections/04_conversation_graph_basics.py)

## What You Will Build

You will create a conversation, append a turn, inspect stored conversation nodes, and see how pinned references can live beside turns instead of outside the graph.

## Why This Matters

Modeling conversation as a graph is one of the repo's main differentiators. It allows summaries, memory, references, tool artifacts, and snapshots to be stored as peers instead of squeezed into a flat transcript.

## Run or Inspect

- Run the companion file in VS Code to create a small conversation graph and inspect the resulting nodes.
- Inspect `kogwistar/conversation/service.py` for conversation creation and context assembly surfaces.
- Compare the pinned-reference behavior with [RAG Level 2 - Provenance and Pinning](./level-2-provenance-pinning.md).

## Inspect The Result

- Confirm the conversation has a start node plus at least one turn node.
- Inspect the `conversation_id` and `turn_index` fields on those nodes.
- Observe that non-turn artifacts can belong to the same conversation without pretending to be chat messages.

## Invariant Demonstrated

Conversation order and conversation-side artifacts can coexist without collapsing into a lossy flat history.

## Next Tutorial

Continue to [05 Context Snapshot and Replay](./05_context_snapshot_and_replay.md).
