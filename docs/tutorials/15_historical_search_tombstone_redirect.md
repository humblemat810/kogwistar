# 15 Historical Search With Tombstone and Redirect

Audience: Advanced / contributor
Time: 25-30 minutes
Companion: [scripts/tutorial_sections/15_historical_search_tombstone_redirect.py](../../scripts/tutorial_sections/15_historical_search_tombstone_redirect.py)

## What You Will Build

You will model two historical-knowledge revisions using lifecycle semantics:

- Sugar/fat narrative revision
- Egg/cholesterol narrative revision

Then you will query the graph at two time slices (`then` and `now`) with `search_nodes_as_of(...)`, and persist context snapshots that capture what an LLM context would have seen at each slice.

## Why This Matters

Knowledge changes over time. Tombstoning and redirect are not only cleanup features; they are audit features. They let you preserve historical claims while still exposing the current canonical claim for normal retrieval.

This tutorial is an audit semantics demo, not medical guidance.

## Run or Inspect

- Run the companion in VS Code cell by cell:
  - `scripts/tutorial_sections/15_historical_search_tombstone_redirect.py`
- Or run directly:

```powershell
python scripts/tutorial_sections/15_historical_search_tombstone_redirect.py
```

- Inspect engine lifecycle/read behavior in:
  - `graph_knowledge_engine/engine_core/lifecycle.py`
  - `graph_knowledge_engine/engine_core/subsystems/read.py`

## Inspect The Result

- Verify `then` search returns historical node ids (`*_OLD`) and `now` search returns revised node ids (`*_NEW`).
- Compare `resolve_mode` views on old ids:
  - `active_only`: hidden
  - `redirect`: canonical replacement
  - `include_tombstones`: historical tombstoned record still visible
- Compare context snapshot payloads for `historical_then` vs `historical_now`:
  - snapshot node ids differ
  - evidence node id lists differ

## Invariant Demonstrated

Historical retrieval can be reconstructed without losing canonical behavior. The system can answer:

- what is true now?
- what would the model have seen back then?

from persisted lifecycle + snapshot artifacts.

## Next Tutorial

Return to [14 Architecture Deep Dive](./14_architecture_deep_dive.md) or revisit [11 Build a Mini GraphRAG App](./11_build_a_mini_graphrag_app.md) with this audit pattern in mind.
