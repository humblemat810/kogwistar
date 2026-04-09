# 22 Retrieval Approaches Comparison, KG Semantics Edition

Audience: builder, educator, and anyone who wants the same retrieval lesson expressed with Kogwistar graph objects.
Time: 20-30 minutes
Companion: [scripts/tutorial_sections/22_rag_retrieval_comparison_kg_semantics.py](../../scripts/tutorial_sections/22_rag_retrieval_comparison_kg_semantics.py)

## What You Will Build

This tutorial rewrites the retrieval comparison through Kogwistar knowledge-graph semantics:

- documents are converted into provenance-heavy `Node` and `Edge` objects
- the graph is queried through either a first-class in-memory backend or a Chroma-backed engine
- traversal uses the repo's `GraphQuery` helper
- the end of the run checks that this version and the earlier version reach the same answer for each search case

## Why This Matters

This version is useful when you want the tutorial to look like the rest of Kogwistar:

- `Node`, `Edge`, `Span`, and `Grounding` are first-class
- graph traversal is explicit rather than hand-waved
- provenance stays attached to the entities and relations

## Run It

```powershell
python scripts/rag_retrieval_comparison_kg_semantics.py --backend memory
```

Optional:

```powershell
python scripts/rag_retrieval_comparison_kg_semantics.py --backend chroma --persist-directory .gke-data/tutorials/22_rag_retrieval_comparison_kg_semantics
python scripts/rag_retrieval_comparison_kg_semantics.py --top-k 3
python scripts/rag_retrieval_comparison_kg_semantics.py --json
```

## What Is Different From The Earlier Version

The earlier comparison tutorial uses a custom lightweight retrieval store.
This KG edition keeps the same dataset and the same six query cases, but it expresses the graph through the repo's core models:

- `Node` for entities
- `Edge` for relations
- `Grounding` and `Span` for provenance
- `GraphQuery` for traversals and neighborhood inspection

The vector, keyword, and graph layers all read from the same persisted knowledge graph. The backend is selectable so you can teach or test with the in-memory engine and then switch to Chroma when the storage stack is available.

## Parity Check

The final section prints a parity table showing whether the raw-text tutorial and the KG-semantics tutorial return the same answer for each query under:

- vector retrieval
- page index retrieval
- graph retrieval
- hybrid retrieval

That gives you a simple confidence check that the rewrite preserved behavior while changing the internal graph representation.
The script also prints `All parity checks passed: True` when every query and retrieval method matches.

## Next Step

If you want the lighter, non-KG version first, read [21 Retrieval Approaches Comparison](./21_rag_retrieval_comparison.md).
If you want the Kogwistar-native graph semantics version, use this page.
