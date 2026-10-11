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
