# 21 Retrieval Approaches Comparison

Audience: builder, educator, and anyone trying to understand why retrieval systems behave differently.
Time: 20-30 minutes
Companion: [scripts/tutorial_sections/21_rag_retrieval_comparison.py](../../scripts/tutorial_sections/21_rag_retrieval_comparison.py)

## What You Will Build

You will run one local tutorial over one shared tech-company dataset and compare four retrieval styles:

- vector RAG
- page index RAG
- graph RAG
- hybrid RAG

The point is not just to retrieve text. The point is to see when each method is strong, when it fails, and why production systems often combine them.

## Why This Matters

The same question can look very different depending on the retrieval layer:

- vector search is forgiving about wording
- keyword search is fast and exact, but brittle
- graph search preserves structure and multi-hop relationships
- hybrid search usually gives the best practical balance

## Run It

```powershell
python scripts/rag_retrieval_comparison_tutorial.py
```

Optional:

```powershell
python scripts/rag_retrieval_comparison_tutorial.py --top-k 3
python scripts/rag_retrieval_comparison_tutorial.py --json
```

## Dataset Shape

The dataset lives in [docs/tutorials/data/tech_company_rag_docs.json](./data/tech_company_rag_docs.json).
Each document contains:

- raw text for vector and page-index retrieval
- structured entities for graph ingestion
- structured relations for graph traversal

The domain is a fictional tech-company ecosystem with people, projects, companies, dependencies, and ambiguity around `Aurora` and `Atlas`.

## Ingestion Modes

### Vector RAG

The tutorial chunks each document and embeds each chunk with a deterministic lexical-hash embedder.

Strengths:

- good for semantic similarity
- tolerant of paraphrase
- works well when the question and the document do not share exact wording

Weaknesses:

- can drift toward related but wrong context
- does not preserve explicit relationships
- can miss precise structure like "who reports to whom"

### Page Index RAG

The tutorial builds an inverted index from token to document IDs with simple TF-IDF-lite scoring.

Strengths:

- very fast
- deterministic
- excellent for exact word overlap

Weaknesses:

- weak on synonyms
- weak on multi-hop reasoning
- brittle when the user asks in different words

### Graph RAG

The tutorial stores `subject -> predicate -> object` relations in an adjacency list and traverses 1-2 hops.

Strengths:

- preserves structure
- supports reasoning across linked facts
- good for ambiguity and relationship-heavy questions

Weaknesses:

- needs structured data
- only knows what was extracted into the graph
- can struggle if the query has no obvious starting node

### Hybrid RAG

The hybrid path does:

1. page index candidate retrieval
2. entity extraction from those candidate docs
3. graph expansion from those entities
4. answer synthesis from both sources

Strengths:

- better recall than a pure graph pass
- better precision than raw keyword-only retrieval
- closer to how practical retrieval systems are built

## Graph View

The script prints an ASCII adjacency view. A small excerpt looks like this:

```text
Aster Labs
  -> develops -> Atlas
  -> uses -> QuasarDB Vector Database
Atlas
  -> depends_on -> Aurora Retrieval Service
  <- leads <- Ben Ortiz
Aurora Audit Module
  -> watches -> Atlas
  <- distinct_from <- QuasarDB Vector Database
```

## Comparison Table

| Method | Best at | Weakness | Tutorial takeaway |
|---|---|---|---|
| Vector RAG | semantic similarity | can blur exact entities and structure | useful when wording varies |
| Page Index RAG | exact keyword recall | misses synonyms and reasoning | useful when precision and speed matter |
| Graph RAG | multi-hop reasoning | requires structured data | useful when relationships matter |
| Hybrid RAG | balance of recall and structure | more plumbing | the most production-like pattern |

## Example Outputs

These are representative outputs from the deterministic demo run.

| Query | Vector | Index | Graph | Hybrid |
|---|---|---|---|---|
| Who is the product lead for Atlas? | `Northstar AI is led by Elena Rossi.` | `Aster Labs Atlas overview` | `Ben Ortiz leads Atlas.` | `Ben Ortiz leads Atlas.` |
| Which document mentions keyword index API and vector database? | `Mehta to audit edge cases.` | `QuasarDB Aurora product note` | `Aurora Vector Database used by Aster Labs.` | `QuasarDB offers Aurora Vector Database.` |
| Who heads the safety project for Atlas? | `Skyline Cloud hosts the Atlas staging environment and the Aurora retrieval service.` | `Safety Project briefing` | `Alice Chen leads Safety Project. Safety Project reviews Atlas.` | `Alice Chen leads Safety Project. Safety Project reviews Atlas.` |
| Which project depends on the Aurora vector database and who leads it? | `Aurora is ambiguous in the corpus...` | `Northstar AI partner note` | `Atlas depends on Aurora Vector Database. Ben Ortiz leads Atlas.` | `Atlas depends on Aurora Vector Database. Ben Ortiz leads Atlas.` |
| Tell me about Aurora. | `Aurora is ambiguous in the corpus...` | `Aurora audit module` | `Aurora is ambiguous in this dataset...` | `Aurora is ambiguous in this dataset...` |
| Who prefers the graph view, and what does she prefer it over? | `Alice Chen prefers the graph view when the answer requires relationships across teams.` | `Migration strategy note` | `Alice Chen prefers Graph View over Keyword Index.` | `Alice Chen prefers Graph View over Keyword Index.` |

## What To Notice

- Vector retrieval can land on a related sentence instead of the exact fact.
- Page index retrieval is precise when the right words appear, but it cannot infer meaning.
- Graph retrieval is strongest when the answer depends on named relationships.
- Hybrid retrieval gives you the best practical compromise: exact candidate generation plus structured expansion.

## Next Step

Run the script once, then read the printed query-by-query comparison. The output is intentionally verbose so you can see each retrieval style side by side.

If you want the same comparison rewritten with Kogwistar graph objects and a selectable memory-or-Chroma backend, continue to [22 Retrieval Approaches Comparison, KG Semantics Edition](./22_rag_retrieval_comparison_kg_semantics.md).
