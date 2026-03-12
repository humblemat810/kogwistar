# RAG Level 2: Reinforced Provenance and Pinning

Goal: materialize inspectable evidence references in the conversation graph.

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

## Inside The Engine

- Creates a user turn via `ConversationService.add_conversation_turn(..., add_turn_only=True)`.
- Retrieves memory and KG candidates with deterministic filtering.
- Calls:
  - `MemoryRetriever.pin_selected(...)`
  - `KnowledgeRetriever.pin_selected(...)`
- Writes `reference_pointer` nodes and `references` edges for replay/audit.

## Checkpoint (Pass/Fail)

Pass when:

- Pointer nodes are created for selected evidence.
- Reference edges link the turn to those pointers.

Fail signals:

- Empty pointer/edge IDs in output.
- `"checkpoint_pass": false`

## Troubleshooting

- If pinning is empty, verify Level 1 passes first (seeded retrieval should be working).
- If conversation artifacts look inconsistent, rerun `reset` + `seed`.
- Keep `--data-dir` identical across levels.

