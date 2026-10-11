# Pure Vision RAG Smoke Checklist

This checklist prevents a demo run from being reported as successful when it
only tested vector creation or only tested an answer endpoint. A path is
complete only when it creates embeddings, writes image-grounded graph nodes,
retrieves ranked evidence, and produces an answer from that evidence.

## Smoke Matrix

| ID | Retrieval | Embedding source | Answer source | Required result | Status |
| --- | --- | --- | --- | --- | --- |
| M1 | Mock | Checked-in vectors | Deterministic mock answer | End-to-end offline orchestration | Passed: graph retrieval, multimodal grounding, and deterministic answer; fixture has no image bytes |
| R1 | Real | Local CLIP `/v1/represent` | Local protocol stub | Real image/text embedding, graph retrieval, answer | Passed: 512-D CLIP and answer path |
| R2 | Real | Local CLIP `/v1/represent` | Ollama `bonsai-parser-8k:latest` | Grounded multimodal answer | Passed: real image answer |
| H1 | Real handoff | Local CLIP, then stopped | Separate answer service | Saved evidence answers after embedder unload | Passed: saved UTF-8 artifact and stopped service |
| G1 | Real | Standalone GPU Qwen3-VL `/v1/represent` | Selected answer path | GPU-backed embedding and graph retrieval | Compose/TUI contract tested; requires NVIDIA host and real model startup |
| C1 | Real | Local Qwen/Ovis service | Codex/OpenAI-compatible gateway | External answer from local retrieved evidence | Pending / endpoint and credentials required |
| T1 | Selected by TUI | Local CLIP `/v1/represent` with checked-in image | Retrieval-only selection | Real wizard reaches embedding, graph retrieval, and artifact output | Curated selection and fixture are unit-tested; a previous real `/readyz`-wait run passed; current default is 4 GiB RAM / 2 CPUs |

## Per-Run Assertions

- [x] The embedding provider is identified by model, revision, dimension, and
  profile fingerprint.
- [x] At least one real or precomputed embedding is created for every source
  image and for the query.
- [x] Each indexed item is written through the normal Kogwistar graph write
  path with a non-empty `MultimodalSpan`.
- [x] Retrieval returns ranked graph nodes, not an unrelated list of vectors.
- [x] Retrieved evidence includes the source path/resource ID and immutable
  revision/hash metadata.
- [x] The answer request contains the query and only the retrieved evidence
  images or their authorized references.
- [x] The answer is returned by the selected answer provider and recorded in
  the result artifact.
- [x] The embedding profile is not silently mixed with another model or
  dimension.
- [x] Sequential handoff verifies that answer generation still works after
  the embedding service is stopped.
- [x] A protocol stub is labelled as a protocol smoke, never as model-quality
  evidence.

## Interpretation

`M1` proves the graph and answer orchestration without model dependencies; its
answer is deterministic and is not model-quality evidence.
`R1` proves model-backed embedding and retrieval. `R2` is the quality-relevant
path and requires a genuine vision-capable answer model. `H1` proves the
low-VRAM lifecycle. `G1` is the GPU alternative; its Compose contract and
curated TUI selection are tested, but GPU/model availability is environment
dependent. `C1` proves the sidecar retrieval plus external answer
boundary only when an approved endpoint and credentials are configured; a local
OpenAI-compatible endpoint is not evidence that a hosted Codex provider was
reached. `T1` is complete for the curated selection logic; a fresh real run
still depends on the local model service becoming ready within its resource
budget.

## Recorded Commands

The following commands are the minimum reproducible checks. Use UTF-8 file
writing through `--output`; PowerShell redirection can create UTF-16 artifacts.

```text
python -u scripts/pure_vision_rag_tutorial.py --mode mock --query rabbit --top-k 2
python -m pytest tests/tutorials/test_pure_vision_rag_tutorial.py -q -p no:cacheprovider
```

The recorded real TUI run used the cached CPU service from
`compose.embedding-clip-cpu.yml`, selected local `llm-wiki-v1`, indexed
`sample_diagram.png`, searched with the text `diagram`, selected retrieval-only,
and produced a 512-dimensional CLIP hit with a source hash and path. The model
needed about two minutes to load under the earlier 2 GiB container limit; `/readyz` was
used because `/v1/capabilities` can respond before the model is usable.
The Compose default is now 4 GiB RAM and 2 CPUs because the 2 GiB cap was not
reliable for a fresh cold start. A host with less available Docker memory must
increase Docker Desktop allocation or the real path is not considered runnable.

For a real run, record the embedding service capabilities response, the source
and query embedding requests, the graph hit, and the answer endpoint/model.
For a handoff run, record that the embedding service was stopped before
`--mode answer` was invoked.

The curated TUI behavior is protected by
`test_tui_curated_clip_uses_checked_in_fixture`: selecting CPU CLIP chooses the
repository protocol and checked-in image without asking for arbitrary paths.
