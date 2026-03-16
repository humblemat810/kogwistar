# RAG Level 0: Simple RAG Baseline
##### p.s. This repository is much more than just rag, but RAG is an instant gratification quick start if you are coming for better RAG results.
Goal: prove end-to-end retrieval and answering with the smallest mental model.

## What You Will Build

You will run the smallest script-backed retrieval path: query a tiny graph, collect evidence, and synthesize a deterministic answer from the retrieved summaries.

## Why This Matters

This level removes fear early. It proves the repo can answer a question from persisted graph evidence before any seeded expansion, provenance pinning, or workflow machinery enters the picture.

## Run or Inspect

## Quick Run

```bash
python scripts/rag_tutorial_ladder.py level0   --data-dir .gke-data/tutorial-ladder   --question "How does this repo implement simple RAG?"
```
Note that the retrieval and embedding is real, the llm round is precompiled to save llm token.

Expected output fields:

- `"answer"`: synthesized response text
- `"evidence"`: non-empty retrieved node list
- `"checkpoint_pass": true`

## Inspect The Result

- Confirm the answer is composed from retrieved graph summaries.
- Confirm `evidence` contains stable node ids instead of anonymous text chunks.
- Compare this result with [03 Build a Small Knowledge Graph](./03_build_a_small_knowledge_graph.md) if you want the broader mental model first.

## Inside The Engine

- Uses `GraphKnowledgeEngine.query_nodes(...)` with deterministic lexical embeddings.
- Builds a tutorial answer directly from retrieved node summaries.
- Keeps scope intentionally small: retrieval -> answer assembly.

## Checkpoint

Pass when:

- At least one evidence node is returned.
- Answer text is present and grounded in retrieved summaries.

## Invariant Demonstrated

Non-empty graph evidence is enough to drive a grounded answer path. The repo can already produce inspectable retrieval before any advanced orchestration.

## Troubleshooting

- If Chroma is missing: install with `pip install -e ".[chroma]"`.
- If previous runs look stale: rerun `reset` + `seed`.
- If path errors occur: run commands from repo root.

## Next Tutorial

Continue to [RAG Level 1 - Retrieval Orchestration](./level-1-retrieval-orchestration.md) or return to [04 Conversation Graph Basics](./04_conversation_graph_basics.md).
