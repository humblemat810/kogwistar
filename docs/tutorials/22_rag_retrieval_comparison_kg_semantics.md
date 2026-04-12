# 22 Retrieval Approaches Comparison, KG Semantics Edition

Audience: builder, educator, and anyone who wants the same retrieval lesson expressed with Kogwistar graph objects.
Time: 20-30 minutes
Companion: [scripts/tutorial_sections/22_rag_retrieval_comparison_kg_semantics.py](../../scripts/tutorial_sections/22_rag_retrieval_comparison_kg_semantics.py)

This page is written like a notebook. Each section is a small teaching step:
read the explanation, then run the matching script cell in your head.

## Cell 1. Set The Goal

This tutorial rewrites the retrieval comparison through Kogwistar knowledge-graph semantics.

You will see:

- documents converted into provenance-heavy `Node` and `Edge` objects
- the graph queried through a first-class in-memory backend or a Chroma-backed engine
- traversal through the repo's `GraphQuery` helper
- a parity check against the earlier tutorial so both versions return the same answer for each search case

## Cell 2. Understand "Vectorless" Again

In this KG-semantic version, "vectorless" still means retrieval that does not rely on embedding similarity at query time.

The tutorial keeps the non-vector ideas separate:

- lexical vectorless retrieval: page index lookup over the persisted corpus
- structural vectorless retrieval: graph traversal over `Node` and `Edge`

If you want the more agentic root-to-subsection demo, that lives in the earlier tutorial. This page stays focused on graph-native forms.

## Cell 3. Pick A Backend

The backend is selectable so you can use the same tutorial in two ways:

- `--backend memory` for a fast volatile backend
- `--backend chroma` when you want the local persisted backend

Run it like this:

```bash
python scripts/rag_retrieval_comparison_kg_semantics.py --backend memory
```

Optional:

```bash
python scripts/rag_retrieval_comparison_kg_semantics.py --backend chroma --persist-directory .gke-data/tutorials/22_rag_retrieval_comparison_kg_semantics
python scripts/rag_retrieval_comparison_kg_semantics.py --top-k 3
python scripts/rag_retrieval_comparison_kg_semantics.py --json
```

## Cell 4. Load The Dataset Into Graph Semantics

The same dataset from the earlier tutorial is reused here.

What changes is the representation:

- entities become `Node`
- relations become `Edge`
- provenance becomes `Span` and `Grounding`
- traversal happens through `GraphQuery`

This is the main teaching point of the KG rewrite: same information, more explicit structure.

## Cell 5. Build The Vector Path On The KG State

The KG tutorial still includes a normal vector RAG path for comparison.

It is useful because it shows the standard retrieval baseline over the same persisted state:

- query embedding
- top-k similarity search
- document or chunk ranking

This keeps the tutorial honest: graph-native retrieval is compared against a normal vector layer, not against an artificial shortcut.

## Cell 6. Build The Lexical Vectorless Path

The page-index path is still the deterministic lexical baseline.

It is the easiest vectorless form to explain:

- tokenize the query
- look up overlapping terms
- rank by simple overlap and document frequency

This helps show why keyword retrieval is fast, but still brittle when wording changes.

## Cell 7. Build The Graph Path

Graph retrieval is where the KG tutorial shines.

The graph preserves:

- who works on what
- what depends on what
- which entity connects to which other entity

This lets the tutorial show multi-hop reasoning directly instead of hiding it behind a chunk scorer.

The script prints graph traces so you can see the traversal path, not just the answer.

## Cell 8. Build The Hybrid Path

Hybrid retrieval combines the strengths of the other layers:

1. page index candidate retrieval
2. entity extraction from those candidate docs
3. graph expansion from those entities
4. answer synthesis from both sources

That is the production-like pattern this tutorial wants to make visible.

## Cell 9. Compare The Six Questions

The tutorial uses the same six query cases as the earlier version:

- direct lookup
- keyword-heavy retrieval
- synonym-style query
- multi-hop reasoning
- ambiguous entity query
- complex relationship reasoning

For each query, the script prints:

- vector results
- index results
- graph results
- hybrid results
- the final parity check against the earlier tutorial

## Cell 10. Read The Parity Check

The most important end-of-run output is the parity table.

It shows whether the KG tutorial and the earlier tutorial return the same answer for each query under:

- vector retrieval
- page index retrieval
- graph retrieval
- hybrid retrieval

That is the sanity check that the rewrite preserved behavior while changing the internal model.

The script also prints `All parity checks passed: True` when every comparison matches.

## Cell 11. See The Graph View

The tutorial prints an ASCII graph view so you can inspect the structure quickly.

This is useful because it shows:

- which nodes exist
- which edges connect them
- how ambiguity is represented

In other words, it gives you a visual way to reason about the same data the graph traversal uses.

## Cell 12. What To Notice

- Vector RAG is still the best fit when phrasing varies.
- Page index RAG is still the best fit when exact keywords matter.
- Graph RAG is the strongest fit when relationships and multi-hop reasoning matter.
- Hybrid RAG is the most production-like compromise.
- The KG backend makes the graph structure explicit instead of inferred.

## Cell 13. Finish Up

If you want the lighter, non-KG version first, read [21 Retrieval Approaches Comparison](./21_rag_retrieval_comparison.md).
If you want the Kogwistar-native graph semantics version, use this page.
