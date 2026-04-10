# 21 Retrieval Approaches Comparison

Audience: builder, educator, and anyone trying to understand why retrieval systems behave differently.
Time: 20-30 minutes
Companion: [scripts/tutorial_sections/21_rag_retrieval_comparison.py](../../scripts/tutorial_sections/21_rag_retrieval_comparison.py)

This page is written in a notebook style. Each section acts like a small cell:
it explains the step first, then shows what the tutorial is doing and why it matters.

## Cell 1. Set The Goal

We start with one shared tech-company dataset and compare five retrieval styles over the same questions:

- vector RAG
- page index RAG
- section-aware vectorless retrieval
- graph RAG
- hybrid RAG

The goal is not just to retrieve text. The goal is to see which retrieval layer works best for different kinds of questions.

## Cell 2. Understand The Word "Vectorless"

Here, "vectorless" means retrieval that does not rely on embedding similarity at query time.

This tutorial shows three vectorless styles:

- lexical vectorless retrieval: page index lookup
- structural vectorless retrieval: graph traversal
- agentic structural vectorless retrieval: fake section navigation from a root heading to a subsection

That gives you a practical way to compare:

- semantic retrieval with vectors
- exact lexical retrieval without vectors
- structure-aware retrieval without vectors

## Cell 3. Load The Dataset

The dataset lives in [docs/tutorials/data/tech_company_rag_docs.json](./data/tech_company_rag_docs.json).

Each document includes:

- raw text for vector and page-index retrieval
- structured entities for graph ingestion
- structured relations for graph traversal

The domain is a fictional tech-company ecosystem with people, projects, companies, dependencies, and ambiguity around `Aurora` and `Atlas`.

## Cell 4. Build The Vector Path

Vector RAG chunks each document and embeds each chunk with a deterministic lexical-hash embedder.

Why this is useful:

- it behaves like semantic retrieval
- it is tolerant of paraphrase
- it can still find relevant chunks when the wording is not exact

Why it can fail:

- it can drift toward related but wrong context
- it does not preserve explicit relationships
- it can miss precise structure like "who reports to whom"

## Cell 5. Build The Lexical Vectorless Path

Page index RAG builds an inverted index from token to document IDs and scores documents with simple TF-IDF-lite scoring.

Why this is useful:

- very fast
- deterministic
- excellent for exact word overlap

Why it can fail:

- weak on synonyms
- weak on multi-hop reasoning
- brittle when the user asks in different words

This is the tutorial's lexical vectorless path.

## Cell 6. Build The Section-Aware Vectorless Path

This tutorial also simulates a fake agentic parser that breaks a document into headings like `Overview`, `Ownership`, `Dependencies`, and `Risks & Ambiguity`.

The important part is the navigation step:

1. start from the root heading
2. build a fake prompt
3. let the fake LLM choose a subsection
4. retrieve from that subsection

Why this is useful:

- it shows how structure can help even without embeddings
- it feels closer to agentic retrieval than plain keyword lookup
- it is a good bridge between text search and graph reasoning

What to watch for:

- it depends on the quality of the fake section planner
- it can miss facts outside the chosen subsection
- it still needs heuristics, not just raw text matching

The walkthrough prints the fake prompt, the fake response, and the chosen path so you can see the round trip clearly.

## Cell 7. Build The Graph Path

Graph RAG stores `subject -> predicate -> object` relations in an adjacency list and traverses 1-2 hops.

Why this is useful:

- it preserves structure
- it supports reasoning across linked facts
- it handles relationship-heavy and ambiguity-heavy questions well

Why it can fail:

- it needs structured data
- it only knows what was extracted into the graph
- it can struggle if the query has no obvious starting node

The graph and hybrid walkthroughs print hop-by-hop traces so you can see which edges were followed.

## Cell 8. Build The Hybrid Path

Hybrid RAG does four steps:

1. page index candidate retrieval
2. entity extraction from those candidate docs
3. graph expansion from those entities
4. answer synthesis from both sources

Why this matters:

- it gives you better recall than a pure graph pass
- it gives you better precision than raw keyword-only retrieval
- it looks much closer to a real production retrieval stack

## Cell 9. Run The Comparisons

Run the tutorial with:

```powershell
python scripts/rag_retrieval_comparison_tutorial.py
```

Optional:

```powershell
python scripts/rag_retrieval_comparison_tutorial.py --top-k 3
python scripts/rag_retrieval_comparison_tutorial.py --json
```

The script prints:

- dataset summary
- schemas for vector, index, and graph storage
- ASCII graph visualization
- per-query walkthroughs
- hop-by-hop graph traces
- fake section-navigation traces

## Cell 10. Read The Comparison

The comparison table is the fastest way to see the tradeoffs:

| Method | Best at | Weakness | Tutorial takeaway |
|---|---|---|---|
| Vector RAG | semantic similarity | can blur exact entities and structure | useful when wording varies |
| Page Index RAG | exact keyword recall | misses synonyms and reasoning | useful when precision and speed matter |
| Section-aware vectorless retrieval | root-to-subsection navigation | depends on a good section planner | useful when the document has clear headings |
| Graph RAG | multi-hop reasoning | requires structured data | useful when relationships matter |
| Hybrid RAG | balance of recall and structure | more plumbing | the most production-like pattern |

## Cell 11. Read The Example Outputs

These outputs are deterministic and meant to be read side by side.
The table below shows the answer style, not just the matching document:

| Query | Vector | Index | Section | Graph | Hybrid |
|---|---|---|---|---|---|
| Who is the product lead for Atlas? | `Northstar AI is led by Elena Rossi.` | `Aster Labs Atlas overview` | `Agentic section navigation chose Ownership...` | `Ben Ortiz leads Atlas.` | `Ben Ortiz leads Atlas.` |
| Which document mentions keyword index API and vector database? | `Mehta to audit edge cases.` | `QuasarDB Aurora product note` | `Agentic section navigation chose Cross-links...` | `Aurora Vector Database used by Aster Labs.` | `QuasarDB offers Aurora Vector Database.` |
| Who heads the safety project for Atlas? | `Skyline Cloud hosts the Atlas staging environment and the Aurora retrieval service.` | `Safety Project briefing` | `Agentic section navigation chose Risks & Ambiguity...` | `Alice Chen leads Safety Project. Safety Project reviews Atlas.` | `Alice Chen leads Safety Project. Safety Project reviews Atlas.` |
| Which project depends on the Aurora vector database and who leads it? | `Aurora is ambiguous in the corpus...` | `Northstar AI partner note` | `Agentic section navigation chose Dependencies...` | `Atlas depends on Aurora Vector Database. Ben Ortiz leads Atlas.` | `Atlas depends on Aurora Vector Database. Ben Ortiz leads Atlas.` |
| Tell me about Aurora. | `Aurora is ambiguous in the corpus...` | `Aurora audit module` | `Agentic section navigation chose Overview...` | `Aurora is ambiguous in this dataset...` | `Aurora is ambiguous in this dataset...` |
| Who prefers the graph view, and what does she prefer it over? | `Alice Chen prefers the graph view when the answer requires relationships across teams.` | `Migration strategy note` | `Agentic section navigation chose Cross-links...` | `Alice Chen prefers Graph View over Keyword Index.` | `Alice Chen prefers Graph View over Keyword Index.` |

## Cell 12. What To Notice

- Vector retrieval can land on a related sentence instead of the exact fact.
- Page index retrieval is precise when the right words appear, but it cannot infer meaning.
- Section-aware retrieval shows how a fake planner can walk from a root heading to a useful subsection without embeddings.
- Graph retrieval is strongest when the answer depends on named relationships.
- Hybrid retrieval gives you the best practical compromise: exact candidate generation plus structured expansion.

## Cell 13. Finish Up

Run the script once, read the query-by-query comparison, and then inspect the traces.
The script is intentionally verbose so the retrieval styles are easy to compare.

If you want the Kogwistar-native version with a selectable memory-or-Chroma backend, continue to [22 Retrieval Approaches Comparison, KG Semantics Edition](./22_rag_retrieval_comparison_kg_semantics.md).
