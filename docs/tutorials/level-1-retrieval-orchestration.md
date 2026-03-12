# RAG Level 1: Reinforced Retrieval Orchestration

Goal: show how memory-derived seeds reinforce KG retrieval beyond naive top-k.

## Quick Run

```powershell
python scripts/rag_tutorial_ladder.py level1 `
  --data-dir .gke-data/tutorial-ladder `
  --question "How does architecture reinforce retrieval?" `
  --max-retrieval-level 2
```

Expected output fields:

- `"seed_kg_node_ids"`: non-empty list
- `"added_by_seed"`: candidate IDs that only appear when seeded expansion is enabled
- `"checkpoint_pass": true`

## Inside The Engine

- Runs `MemoryRetriever.retrieve(...)` to pull prior context and extract KG seeds from `reference_pointer` nodes.
- Runs `KnowledgeRetriever.retrieve(...)` twice:
  - without seeds
  - with seeds
- Compares candidate sets under the same `max_retrieval_level`.

## Checkpoint (Pass/Fail)

Pass when:

- Memory retrieval yields at least one seed KG node.
- Seeded KG retrieval changes the candidate set (`added_by_seed` non-empty).

Fail signals:

- Empty seeds from memory retrieval.
- No candidate delta between seeded and unseeded retrieval.

## Troubleshooting

- If seeds are empty, run `python scripts/rag_tutorial_ladder.py seed --data-dir .gke-data/tutorial-ladder` again.
- If retrieval appears unchanged, confirm `--max-retrieval-level` is `>= 1`.
- If you changed the data dir, keep it consistent between `seed` and `level1`.

