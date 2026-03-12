# RAG Level 0: Simple RAG Baseline

Goal: prove end-to-end retrieval and answering with the smallest mental model.

## Quick Run

```powershell
python scripts/rag_tutorial_ladder.py level0 `
  --data-dir .gke-data/tutorial-ladder `
  --question "How does this repo implement simple RAG?"
```

Expected output fields:

- `"answer"`: synthesized response text
- `"evidence"`: non-empty retrieved node list
- `"checkpoint_pass": true`

## Inside The Engine

- Uses `GraphKnowledgeEngine.query_nodes(...)` with deterministic lexical embeddings.
- Builds a tutorial answer directly from retrieved node summaries.
- Keeps scope intentionally small: retrieval -> answer assembly.

## Checkpoint (Pass/Fail)

Pass when:

- At least one evidence node is returned.
- Answer text is present and grounded in retrieved summaries.

Fail signals:

- `"evidence": []`
- `"checkpoint_pass": false`

## Troubleshooting

- If Chroma is missing: install with `pip install -e ".[chroma]"`.
- If previous runs look stale: rerun `reset` + `seed`.
- If path errors occur: run commands from repo root.

