# RAG Level 1: Reinforced Retrieval Orchestration

Goal: show how memory-derived seeds reinforce KG retrieval beyond naive top-k.

## What You Will Build

You will compare naive retrieval against seeded expansion and confirm that prior conversation memory can change which knowledge graph candidates are considered.

## Why This Matters

This is the first concrete example of "graph RAG" behaving differently from flat top-k lookup. The memory graph can push retrieval toward connected evidence that a shallow ranking pass would miss.

## Run or Inspect

## Quick Run

```bash
python scripts/rag_tutorial_ladder.py level1 \
  --data-dir .gke-data/tutorial-ladder \
  --question "How does architecture reinforce retrieval?" \
  --max-retrieval-level 2
```

Expected output fields:

- `"seed_kg_node_ids"`: non-empty list
- `"added_by_seed"`: candidate IDs that only appear when seeded expansion is enabled
- `"checkpoint_pass": true`

As you notice, you can retrieve nearby ideas by graph-neighbourhood and proceed.

## Inspect The Result

- Compare `candidate_count_without_seed` with `candidate_count_with_seed`.
- Inspect the `seed_kg_node_ids` and verify they came from conversation-side memory rather than the query alone.
- Read this alongside [05 Context Snapshot and Replay](./05_context_snapshot_and_replay.md) if you want the broader context-assembly implications.

## Inside The Engine

- Runs `MemoryRetriever.retrieve(...)` to pull prior context and extract KG seeds from `reference_pointer` nodes.
- Runs `KnowledgeRetriever.retrieve(...)` twice:
  - without seeds
  - with seeds
- Compares candidate sets under the same `max_retrieval_level`.

## Checkpoint

Pass when:

- Memory retrieval yields at least one seed KG node.
- Seeded KG retrieval changes the candidate set (`added_by_seed` non-empty).

## Invariant Demonstrated

Conversation-derived seeds can change retrieval frontier shape. Retrieval is not locked to a single top-k pass.

## Troubleshooting

- If seeds are empty, run `python scripts/rag_tutorial_ladder.py seed --data-dir .gke-data/tutorial-ladder` again.
- If retrieval appears unchanged, confirm `--max-retrieval-level` is `>= 1`.
- If you changed the data dir, keep it consistent between `seed` and `level1`.

## Next Tutorial

Continue to [RAG Level 2 - Provenance and Pinning](./level-2-provenance-pinning.md) or return to [05 Context Snapshot and Replay](./05_context_snapshot_and_replay.md).
