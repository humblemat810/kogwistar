# RAG Level 2: Reinforced Provenance and Pinning

Goal: materialize inspectable evidence references in the conversation graph.

## What You Will Build

You will create a user turn, select evidence, and project that evidence into the conversation graph as pointer nodes and reference edges.

## Why This Matters

This is where the repo's provenance story becomes visible. Instead of only saying "the answer used these facts," the system writes graph artifacts you can inspect later.

## Run or Inspect

## Quick Run

```powershell
python scripts/rag_tutorial_ladder.py level2 `
  --data-dir .gke-data/tutorial-ladder `
  --question "Show evidence and provenance for retrieval decisions." `
  --max-retrieval-level 2
```

Expected output fields:

- `"pinned_kg_pointer_node_ids"`: non-empty
- `"pinned_kg_edge_ids"`: non-empty
- `"checkpoint_pass": true`

## Inspect The Result

- Inspect the conversation graph for the new `reference_pointer` nodes.
- Confirm the turn is linked to those pointers by `references` edges.
- Compare this with [04 Conversation Graph Basics](./04_conversation_graph_basics.md), [11 Build a Mini GraphRAG App](./11_build_a_mini_graphrag_app.md), and [15 Historical Search With Tombstone and Redirect](./15_historical_search_tombstone_redirect.md).

## Inside The Engine

- Creates a user turn via `ConversationService.add_conversation_turn(..., add_turn_only=True)`.
- Retrieves memory and KG candidates with deterministic filtering.
- Calls:
  - `MemoryRetriever.pin_selected(...)`
  - `KnowledgeRetriever.pin_selected(...)`
- Writes `reference_pointer` nodes and `references` edges for replay and audit.

## Checkpoint

Pass when:

- Pointer nodes are created for selected evidence.
- Reference edges link the turn to those pointers.

## Invariant Demonstrated

Used evidence becomes part of the conversation graph. Provenance is explicit, not reconstructed later from guesswork.

## Troubleshooting

- If pinning is empty, verify Level 1 passes first.
- If conversation artifacts look inconsistent, rerun `reset` + `seed`.
- Keep `--data-dir` identical across levels.

## Next Tutorial

Continue to [RAG Level 3 - Event-Sourced Loop Control](./level-3-event-loop-control.md) or return to [10 Event Log Replay and CDC](./10_event_log_replay_and_cdc.md).
