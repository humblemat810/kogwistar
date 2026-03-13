# 11 Build a Mini GraphRAG App

Audience: Builder / integrator
Time: 25-30 minutes
Companion: [scripts/tutorial_sections/11_build_a_mini_graphrag_app.py](../../scripts/tutorial_sections/11_build_a_mini_graphrag_app.py)

## What You Will Build

You will run a small end-to-end flow: seed documents, retrieve evidence, pin provenance into the conversation graph, and inspect the answer path.

## Why This Matters

This is the main product-intuition tutorial. It converts the separate ideas from the earlier docs into one flow that feels like a real graph-native RAG application.

## Run or Inspect

- Run the companion file for a notebook-like walkthrough of reset, seed, retrieval, and pinning.
- Or use the script-backed ladder directly:

```powershell
python scripts/rag_tutorial_ladder.py reset --data-dir .gke-data/tutorial-ladder
python scripts/rag_tutorial_ladder.py seed --data-dir .gke-data/tutorial-ladder
python scripts/rag_tutorial_ladder.py level0 --data-dir .gke-data/tutorial-ladder --question "How does this repo implement simple RAG?"
python scripts/rag_tutorial_ladder.py level1 --data-dir .gke-data/tutorial-ladder --question "How does architecture reinforce retrieval?" --max-retrieval-level 2
python scripts/rag_tutorial_ladder.py level2 --data-dir .gke-data/tutorial-ladder --question "Show evidence and provenance for retrieval decisions." --max-retrieval-level 2
```

## Inspect The Result

- Confirm the answer is grounded in retrieved node summaries.
- Confirm seeded retrieval changes the candidate set.
- Confirm pinned pointer nodes and reference edges were materialized for the conversation turn.

## Invariant Demonstrated

The answer path is inspectable end to end. Retrieval, expansion, and provenance projection all leave graph artifacts behind.

## Next Tutorial

Continue to [12 Designer API Integration](./12_designer_api_integration.md).

Advanced extension: if you want lifecycle-aware audit semantics for knowledge that changed over time, continue to [15 Historical Search With Tombstone and Redirect](./15_historical_search_tombstone_redirect.md).
