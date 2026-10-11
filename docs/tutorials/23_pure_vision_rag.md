# 23 Pure Vision RAG

This tutorial shows a screenshot-native retrieval path using existing Kogwistar
contracts. The graph stores captured document/webpage images and
image-grounded nodes; no OCR, caption, or text chunk is required for
retrieval.

## What The Core Does

The flow is:

```text
document/webpage screenshot captures
  -> vision embedding provider
  -> profile-scoped vectors
  -> Node + Grounding(multimodal_spans)
  -> Kogwistar vector search
  -> ranked image evidence
  -> optional vision answer model
```

The screenshot itself remains the source of truth. `MultimodalSpan` records its
namespace, immutable resource revision, content hash, modality, and spatial
locator. The vector is used for retrieval only; it is not stored inside the
grounding contract.

This is intentionally different from ordinary text RAG: a user text query is
embedded by the same vision-language embedding model into its shared image/text
space, then matched against screenshot embeddings. If the selected model only
accepts images, capture the query as an image too.

## Capturing The Inputs

The capture step belongs outside the core. For a webpage, use a browser or
headless browser to save a screenshot and treat the resulting file as an
immutable resource revision:

```python
from pathlib import Path
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1200})
    page.goto("https://example.invalid/document", wait_until="networkidle")
    Path("media/document-page-001.png").parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path="media/document-page-001.png", full_page=True)
    browser.close()
```

For a PDF or scanned document, render each page to an image with the document
tooling of your choice. Do not OCR the image for this tutorial. Store the
capture hash and revision ID beside the screenshot, then pass the screenshot
resource to the embedding adapter.

## Offline Path: Precomputed Payload

This path needs no model download, GPU, Chroma, or network service. It is useful
for CI, documentation, and machines without an 8 GB GPU.

From the Kogwistar checkout:

```bash
python scripts/pure_vision_rag_tutorial.py --mode mock --query rabbit
python scripts/pure_vision_rag_tutorial.py --mode mock --query lecture --top-k 3
```

The payload is [examples/vision_rag/mock_payload.json](../../examples/vision_rag/mock_payload.json).
It contains screenshot resource IDs, immutable evidence metadata, normalized
image regions, a four-dimensional precomputed vector, and query vectors. The script
still writes `Node` objects through the normal write path and retrieves them
through `search_nodes_as_of_scored`, so this is not a fake list lookup.

Use this path to verify:

- a `Grounding` may contain only `multimodal_spans`;
- image regions survive node serialization and retrieval;
- profile identity is explicit and includes model revision and dimension;
- the vector backend returns normalized similarity scores and graph nodes.

## Real Path: 8 GB-Class Local Models

Kogwistar intentionally does not install or select a vision model. Keep the
heavy model service in a separate process or container so the core remains
backend- and vendor-neutral.

The tutorial's real adapter expects an embedding HTTP endpoint with this
contract:

```text
POST /embed
{
  "inputs": [
    {
      "resource_id": "media/rabbit.png",
      "image_base64": "...",
      "mime_type": "image/png"
    }
  ]
}

200 OK
{
  "embeddings": [[0.12, -0.03, ...]]
}
```

The endpoint can be backed by an 8 GB-class multimodal embedding model such as
an operator's Qwen-VL or Ovis embedding service. The endpoint, model revision,
preprocessing, dimension, and metric must be recorded in the configured
`EmbeddingProfile`; equal dimensions do not make different models comparable.

For a text query in a shared vision-language space, the same endpoint also
accepts:

```json
{"inputs": [{"resource_id": "query-text", "text": "find the rabbit"}]}
```

Run with real images:

```bash
python scripts/pure_vision_rag_tutorial.py --mode real \
  --embedding-endpoint http://127.0.0.1:8001/embed \
  --image rabbit=./media/rabbit.png \
  --image lecture=./media/lecture.png \
  --query-text "find the rabbit"
```

### Small Local Multimodal Demo

The repository also has a real, toy-scale multimodal service using CPU CLIP
ViT-B/32. It produces 512-dimensional vectors in one shared text/image
space, needs no GPU, and is useful for validating the full graph path without
loading Qwen-VL or Ovis. It requires a reliable CPU/RAM budget: the Compose
default is 4 GiB container RAM and 2 CPUs. This is not VRAM, and Docker Desktop
must be configured with at least that much available memory:

```powershell
docker compose -f compose.embedding-clip-cpu.yml up --build -d
python scripts/pure_vision_rag_tutorial.py --mode real \
  --embedding-api llm-wiki-v1 \
  --embedding-endpoint http://127.0.0.1:8792 \
  --image rabbit=./media/rabbit.png \
  --image lecture=./media/lecture.png \
  --query-text "find the rabbit"
```

The `llm-wiki-v1` adapter reads the service's advertised profile and calls
`/v1/represent`; it does not guess the dimension or model identity. CLIP is a
demo baseline, not a quality replacement for Qwen3-VL or Ovis, and its text
input is limited to the model's short context window.

If the service cannot reach `/readyz` after model loading, the real smoke is
not considered runnable on that Docker resource allocation. Increase Docker
Desktop memory or override the limits explicitly, for example:

```powershell
$env:LLM_WIKI_CLIP_MEMORY_LIMIT = "6g"
$env:LLM_WIKI_CLIP_CPU_LIMIT = "4.0"
docker compose -f compose.embedding-clip-cpu.yml up -d
```

### Answer-Model Handoff

Retrieval and answer generation are separate resource lifecycles. On an
8-GB-class machine, run them sequentially when VRAM is tight:

```powershell
# Retrieve while CLIP/Qwen/Ovis is loaded.
python scripts/pure_vision_rag_tutorial.py --mode real `
  --embedding-api llm-wiki-v1 `
  --embedding-endpoint http://127.0.0.1:8792 `
  --image rabbit=./media/rabbit.png `
  --query-text "find the rabbit" > retrieval.json

# Stop the embedding service before starting a separate local vision LLM.
docker compose -f compose.embedding-clip-cpu.yml stop

# Start the separate answer model, then answer from the saved evidence.
python scripts/pure_vision_rag_tutorial.py --mode answer `
  --retrieval-json retrieval.json `
  --answer-endpoint http://127.0.0.1:8002/v1/chat/completions `
  --answer-model local-vision-model `
  --query-text "find the rabbit"
```

This handoff is manual and intentional: the retrieved evidence and profile
are already in `retrieval.json`, while the model process can be replaced by a
vision-capable local Qwen/Ovis server. It avoids claiming that unloading a
remote model can be performed by the core library.

If memory permits, keep the embedding service resident and use an independent
OpenAI-compatible vision answer endpoint:

```bash
python scripts/pure_vision_rag_tutorial.py --mode real \
  --embedding-api llm-wiki-v1 \
  --embedding-endpoint http://127.0.0.1:8792 \
  --answer-endpoint http://127.0.0.1:8002/v1/chat/completions \
  --answer-model local-vision-model \
  --image rabbit=./media/rabbit.png \
  --image lecture=./media/lecture.png \
  --query-text "find the rabbit"
```

The answer endpoint may be a local server or a gateway to an external
Codex/OpenAI-compatible provider. The embedding profile and graph retrieval
remain local; the answer provider receives only the query and retrieved image
evidence. Provider credentials and policy are deployment concerns, not graph
data.

### Interactive TUI

For a guided run, use the dependency-free terminal menu:

```bash
python scripts/pure_vision_rag_tutorial.py --mode tui
```

The wizard is intentionally curated for instruction rather than production
deployment. It offers:

1. **Offline mock**: checked-in vectors and a deterministic answer. This is the
   fastest way to learn the graph and grounding path, but it is not model-quality
   evidence.
2. **CPU CLIP ViT-B/32**: the known local demo profile. It uses the checked-in
   `sample_diagram.png`, the repository's Compose service, and the advertised
   512-dimensional text/image profile. Start the service with:

   ```powershell
   docker compose -f compose.embedding-clip-cpu.yml up -d
   ```

3. **Qwen3-VL or Ovis**: an advanced option for an already deployed service.
   The tutorial does not install arbitrary weights or Python packages. The
   service must implement the repository `/v1/represent` contract so its model,
   revision, metric, dimension, and fingerprint can be checked.

The TUI also offers a known GPU Qwen3-VL profile. It uses the standalone
`compose.embedding-gpu.yml` service, the checked-in image, and the same
`/v1/represent` contract:

```powershell
docker compose -f compose.embedding-gpu.yml up --build -d
```

This requires NVIDIA Container Toolkit, a compatible GPU, and enough host RAM
for the container's 8 GiB loading budget. It is separate from the CPU CLIP
service and does not make the Ovis vLLM stack look like a compatible
`/v1/represent` service. GPU memory is not host RAM; both limits matter.

For every real profile, the wizard can use the checked-in image and a text
query, or use that same image as the query image when text is omitted. It then
offers retrieval-only, local Ollama/OpenAI-compatible answering, or an
advanced Codex/OpenAI-compatible gateway. Resident and sequential handoff
answer lifecycles are explained before execution. Embedding credentials are
never passed to the answer provider.

This is deliberately not a general model installer. If a user needs a new
embedding backend, they should first deploy and verify it separately, then use
the advanced profile with its explicit endpoint.

The smoke checklist and current evidence are recorded in
[23_pure_vision_rag_smoke_checklist.md](23_pure_vision_rag_smoke_checklist.md).
The real local path has been exercised with CPU CLIP retrieval followed by the
Ollama `bonsai-parser-8k:latest` vision answer model. The external
Codex/OpenAI gateway row remains deployment-dependent and is not claimed as
passed until that endpoint is configured and run.

The optional answer endpoint is OpenAI-compatible and receives the query image
plus the retrieved evidence images. It is only an answer/synthesis step; it
does not decide which graph nodes are authoritative:

```bash
python scripts/pure_vision_rag_tutorial.py --mode real \
  --embedding-endpoint http://127.0.0.1:8001/embed \
  --answer-endpoint http://127.0.0.1:8002/v1/chat/completions \
  --answer-model local-vision-model \
  --image rabbit=./media/rabbit.png \
  --image lecture=./media/lecture.png \
  --query-text "find the rabbit"
```

The answer service may be vLLM, Ollama, llama.cpp, or another compatible
server. Its credentials and network boundary belong to the deployment, not to
the Kogwistar graph model.

## Model And Memory Guidance

For an 8 GB machine, start with one screenshot at a time and a small batch.
Keep the VL embedding service resident if possible to avoid repeated model load
latency. A vision answer model can be a separate service and can be started
only for interactive answers. Retrieval remains useful even when the answer
model is offline.

For answer generation, use a vision-capable Qwen-VL/Ovis-grade model that can
receive the retrieved screenshots. A text-only Bonsai2 deployment can be used
only after a separate vision service has converted the screenshots into
grounded text; that fallback is no longer pure screenshot-native answering.
Do not claim direct image understanding from Bonsai2 unless the deployed model
actually exposes a vision tower and its endpoint accepts image inputs.

Do not mix vectors from different models, preprocessing pipelines, or semantic
spaces in one store. Create a new isolated storage scope and profile for each
such model. The profile fingerprint is the compatibility boundary.

## What This Demo Does Not Claim

- It does not turn image similarity into canonical semantic truth.
- It does not perform OCR or image captioning.
- It does not require a particular vendor, model format, quantization, or GPU.
- It does not place image bytes in graph nodes; resources remain external and
  are referenced by immutable IDs and hashes.
- The precomputed path demonstrates the graph/vector contract, not model quality.
