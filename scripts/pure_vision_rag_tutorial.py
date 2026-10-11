"""Pure vision RAG tutorial using Kogwistar graph and vector primitives.

The mock path is fully offline and uses checked-in image embeddings.  The real
path keeps heavy model dependencies outside Kogwistar and calls two small HTTP
adapters: one for image embeddings and one optional vision answer model.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kogwistar.engine_core import (
    EmbeddingProfile,
    GraphKnowledgeEngine,
    MultimodalSpan,
    SpatialRegionLocator,
    endpoint_fingerprint,
)
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend
from kogwistar.engine_core.models import Grounding, Node
from kogwistar.llm_tasks.contracts import LLMTaskSet


DEFAULT_PAYLOAD = ROOT / "examples" / "vision_rag" / "mock_payload.json"
DEMO_IMAGE_PATH = ROOT.parent / "tests" / "fixtures" / "ovis_multimodal_media" / "sample_diagram.png"
DEFAULT_CLIP_ENDPOINT = "http://127.0.0.1:8792"
DEFAULT_QWEN_GPU_ENDPOINT = "http://127.0.0.1:8794"
DEFAULT_ANSWER_ENDPOINT = "http://127.0.0.1:11434/v1/chat/completions"
DEFAULT_ANSWER_MODEL = "bonsai-parser-8k:latest"


class VisionEmbeddingProvider(Protocol):
    def embed_inputs(
        self, inputs: list[dict[str, str]], *, operation: str
    ) -> list[list[float]]: ...


class VisionAnswerProvider(Protocol):
    def answer(
        self, *, query: dict[str, str], evidence: list[dict[str, Any]]
    ) -> str: ...


@dataclass(frozen=True)
class VisionItem:
    item_id: str
    label: str
    resource_id: str
    resource_revision_id: str
    content_sha256: str
    embedding: list[float]
    region: tuple[float, float, float, float]
    summary: str
    capture_kind: str = "image"
    source_url: str | None = None
    source_path: str | None = None


class PrecomputedVisionProvider:
    def __init__(self, items: list[VisionItem], queries: dict[str, list[float]]) -> None:
        self._items = {item.resource_id: item.embedding for item in items}
        self.queries = queries

    def embed_inputs(
        self, inputs: list[dict[str, str]], *, operation: str
    ) -> list[list[float]]:
        return [list(self._items[item["resource_id"]]) for item in inputs]

    def embed_query(self, query_name: str) -> list[float]:
        return list(self.queries[query_name])


class JsonVisionEmbeddingProvider:
    """HTTP adapter contract for a local vision embedding service.

    The endpoint accepts ``{"inputs": [{"resource_id", "image_base64",
    "mime_type"}]}`` and returns ``{"embeddings": [[...], ...]}``.
    """

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def embed_inputs(
        self, inputs: list[dict[str, str]], *, operation: str
    ) -> list[list[float]]:
        payload = _post_json(self.endpoint, {"inputs": inputs})
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or not all(isinstance(row, list) for row in embeddings):
            raise ValueError("vision embedding endpoint must return embeddings as a list of vectors")
        if len(embeddings) != len(inputs):
            raise ValueError("vision embedding endpoint returned the wrong number of vectors")
        return [[float(value) for value in row] for row in embeddings]


class LlmWikiEmbeddingServiceProvider:
    """Adapter for the repository's real local CLIP/Qwen embedding service."""

    def __init__(self, service_endpoint: str, *, token: str | None = None) -> None:
        self.service_endpoint = service_endpoint.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        capabilities = _get_json(
            f"{self.service_endpoint}/v1/capabilities", headers=self._headers
        )
        profile = capabilities.get("profile")
        if not isinstance(profile, dict):
            raise ValueError("embedding service capabilities did not return a profile")
        fingerprint = profile.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("embedding service capabilities did not return a fingerprint")
        core_profile = dict(profile)
        core_profile["embedding_kind"] = core_profile.pop("embedding", "dense")
        core_profile["similarity_metric"] = {
            "dot": "ip",
            "cosine": "cosine",
        }.get(str(core_profile.pop("metric", "dot")), "ip")
        self.profile = EmbeddingProfile.from_mapping(core_profile)
        self.profile_fingerprint = fingerprint

    def embed_inputs(
        self, inputs: list[dict[str, str]], *, operation: str
    ) -> list[list[float]]:
        items: list[dict[str, object]] = []
        for raw in inputs:
            item: dict[str, object] = {
                "item_id": raw["resource_id"],
                "modality": "image" if "image_base64" in raw else "text",
            }
            if "text" in raw:
                item["text"] = raw["text"]
            if "image_base64" in raw:
                image_bytes = base64.b64decode(raw["image_base64"])
                item["asset"] = {
                    "data_base64": raw["image_base64"],
                    "content_type": raw["mime_type"],
                    "sha256": hashlib.sha256(image_bytes).hexdigest(),
                }
            items.append(item)
        response = _post_json(
            f"{self.service_endpoint}/v1/represent",
            {
                "contract_version": "v1",
                "operation": operation,
                "profile_fingerprint": self.profile_fingerprint,
                "items": items,
            },
            headers=self._headers,
        )
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(inputs):
            raise ValueError("embedding service returned the wrong number of results")
        vectors: list[list[float]] = []
        for result in results:
            if not isinstance(result, dict) or not isinstance(result.get("vectors"), list):
                raise ValueError("embedding service returned an invalid result")
            rows = result["vectors"]
            if len(rows) != 1 or not isinstance(rows[0], list):
                raise ValueError("tutorial requires one dense vector per input")
            vectors.append([float(value) for value in rows[0]])
        return vectors


class OpenAIVisionAnswerProvider:
    """Optional OpenAI-compatible vision answer adapter.

    The service receives a normal chat-completions request with the query image
    and retrieved evidence images.  It is not used for indexing or authority.
    """

    def __init__(self, endpoint: str, model: str) -> None:
        self.endpoint = endpoint
        self.model = model

    def answer(self, *, query: dict[str, str], evidence: list[dict[str, Any]]) -> str:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Answer using only the retrieved image evidence. Identify which "
                    "evidence is relevant and mention uncertainty."
                ),
            },
        ]
        if "image_base64" in query:
            content.append(_image_part(query))
        else:
            content.append({"type": "text", "text": query["text"]})
        for item in evidence:
            content.append({"type": "text", "text": f"Evidence {item['id']}: {item['summary']}"})
            if "image_base64" in item:
                content.append(_image_part(item))
        payload = _post_json(
            self.endpoint,
            {
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "user", "content": content}],
            },
        )
        try:
            answer = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("vision answer endpoint returned no chat content") from exc
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("vision answer endpoint returned empty chat content")
        return answer


class MockVisionAnswerProvider:
    """Deterministic answer stage for the offline orchestration smoke."""

    def answer(self, *, query: dict[str, str], evidence: list[dict[str, Any]]) -> str:
        if not evidence:
            raise ValueError("mock answer requires at least one retrieved evidence item")
        query_label = query.get("text") or query.get("resource_id", "query")
        labels = ", ".join(str(item["label"]) for item in evidence)
        return f"Mock grounded answer for {query_label}: use evidence {labels}."


def _get_json(endpoint: str, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = Request(endpoint, headers=headers or {}, method="GET")
    with urlopen(request, timeout=30) as response:  # noqa: S310 - explicit user-configured endpoint
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("vision service response must be a JSON object")
    return value


def _wait_for_embedding_service(endpoint: str, *, token: str | None = None) -> None:
    """Wait for the repository adapter's model, not merely its HTTP server."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    ready_endpoint = f"{endpoint.rstrip('/')}/readyz"
    print("Waiting for the embedding model to load (/readyz)...")
    last_error = "service did not become ready"
    for _ in range(120):
        try:
            payload = _get_json(ready_endpoint, headers=headers)
            if payload.get("ready") is True:
                print("Embedding model is ready.")
                return
            last_error = "readyz returned no ready=true response"
        except Exception as exc:  # the endpoint is expected to return 503 while loading
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"embedding service did not become ready within 240 seconds: {last_error}")


def _post_json(
    endpoint: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:  # noqa: S310 - explicit user-configured endpoint
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("vision service response must be a JSON object")
    return value


def _image_input(path: Path, *, resource_id: str) -> dict[str, str]:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"resource_id": resource_id, "image_base64": encoded, "mime_type": mime_type}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_part(image: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{image['mime_type']};base64,{image['image_base64']}"
        },
    }


def _load_payload(path: Path) -> tuple[EmbeddingProfile, list[VisionItem], dict[str, list[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile = EmbeddingProfile.from_mapping(payload["profile"])
    items = [
        VisionItem(
            item_id=str(raw["id"]),
            label=str(raw["label"]),
            resource_id=str(raw["resource_id"]),
            resource_revision_id=str(raw["resource_revision_id"]),
            content_sha256=str(raw["content_sha256"]),
            embedding=[float(value) for value in raw["embedding"]],
            region=tuple(float(value) for value in raw["region"]),
            summary=str(raw["summary"]),
            capture_kind=str(raw.get("capture_kind", "image")),
            source_url=str(raw["source_url"]) if raw.get("source_url") else None,
        )
        for raw in payload["items"]
    ]
    queries = {
        str(name): [float(value) for value in vector]
        for name, vector in payload.get("queries", {}).items()
    }
    return profile, items, queries


def _node_for_item(item: VisionItem, profile: EmbeddingProfile) -> Node:
    x, y, width, height = item.region
    evidence = MultimodalSpan(
        source_namespace="pure-vision-rag-demo",
        resource_id=item.resource_id,
        resource_revision_id=item.resource_revision_id,
        content_sha256=item.content_sha256,
        modality="image",
        locator=SpatialRegionLocator(
            x=x,
            y=y,
            width=width,
            height=height,
            coordinate_system="normalized_0_1",
        ),
    )
    return Node(
        id=item.item_id,
        label=item.label,
        type="entity",
        summary=item.summary,
        mentions=[Grounding(multimodal_spans=[evidence])],
        properties={
            "vision_rag": True,
            "embedding_profile": profile.fingerprint,
            "resource_id": item.resource_id,
            "capture_kind": item.capture_kind,
            "source_url": item.source_url,
        },
        embedding=list(item.embedding),
    )


def _offline_llm_tasks() -> LLMTaskSet:
    def unavailable(_request: object) -> object:
        raise RuntimeError("the pure vision RAG tutorial does not invoke an LLM task")

    return LLMTaskSet(
        extract_graph=unavailable,
        adjudicate_pair=unavailable,
        adjudicate_batch=unavailable,
        filter_candidates=unavailable,
        summarize_context=unavailable,
        answer_with_citations=unavailable,
        repair_citations=unavailable,
    )


def run_mock(query_name: str, *, top_k: int = 2, payload_path: Path = DEFAULT_PAYLOAD) -> dict[str, Any]:
    profile, items, queries = _load_payload(payload_path)
    provider = PrecomputedVisionProvider(items, queries)
    try:
        query_embedding = provider.embed_query(query_name)
    except KeyError as exc:
        raise ValueError(f"unknown mock query {query_name!r}; choose from {sorted(queries)}") from exc
    return _run_graph_search(
        profile,
        items,
        query_embedding,
        top_k=top_k,
        answer_provider=MockVisionAnswerProvider(),
        query={"text": query_name, "resource_id": "mock-query"},
    )


def run_real(
    *,
    image_paths: list[tuple[str, Path]],
    query_image: Path | None,
    embedding_endpoint: str,
    embedding_api: str = "simple",
    embedding_token: str | None = None,
    answer_endpoint: str | None,
    answer_model: str,
    top_k: int,
    query_text: str | None = None,
    embedding_model: str = "operator-configured",
    embedding_revision: str = "operator-configured",
) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("real mode requires at least one --image ID=PATH")
    if embedding_api == "llm-wiki-v1":
        provider: VisionEmbeddingProvider = LlmWikiEmbeddingServiceProvider(
            embedding_endpoint, token=embedding_token
        )
        advertised_profile = getattr(provider, "profile", None)
        if not isinstance(advertised_profile, EmbeddingProfile):
            raise ValueError("local embedding service did not advertise a valid profile")
        profile = advertised_profile
    else:
        provider = JsonVisionEmbeddingProvider(embedding_endpoint)
        profile = None
    inputs = [_image_input(path, resource_id=resource_id) for resource_id, path in image_paths]
    vectors = provider.embed_inputs(inputs, operation="document")
    dimension = len(vectors[0]) if vectors else 0
    if profile is None:
        profile = EmbeddingProfile(
            provider="http-vision-embedding",
            model=embedding_model,
            model_revision=embedding_revision,
            dimension=dimension,
            similarity_metric="cosine",
            embedding_kind="single_vector",
            endpoint_fingerprint=endpoint_fingerprint(embedding_endpoint),
        )
    items = [
        VisionItem(
            item_id=f"image:{index}",
            label=resource_id,
            resource_id=resource_id,
            resource_revision_id=f"sha256:{_file_sha256(path)}",
            content_sha256=_file_sha256(path),
            embedding=vector,
            region=(0.0, 0.0, 1.0, 1.0),
            summary=f"Retrieved image evidence for {resource_id}.",
            source_path=str(path),
        )
        for index, ((resource_id, path), vector) in enumerate(zip(image_paths, vectors, strict=True))
    ]
    query_input = (
        {"resource_id": "query-text", "text": query_text}
        if query_text is not None
        else _image_input(query_image, resource_id="query")
    )
    query_vector = provider.embed_inputs([query_input], operation="query")[0]
    answer_provider = (
        OpenAIVisionAnswerProvider(answer_endpoint, answer_model) if answer_endpoint else None
    )
    return _run_graph_search(
        profile,
        items,
        query_vector,
        top_k=top_k,
        answer_provider=answer_provider,
        query=query_input,
    )


def _run_graph_search(
    profile: EmbeddingProfile,
    items: list[VisionItem],
    query_embedding: list[float],
    *,
    top_k: int,
    answer_provider: VisionAnswerProvider | None,
    query: dict[str, str] | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="kogwistar-vision-rag-") as temp_dir:
        engine = GraphKnowledgeEngine(
            persist_directory=temp_dir,
            backend_factory=build_in_memory_backend,
            embedding_function=lambda values: [
                [1.0] + [0.0] * (profile.dimension - 1) for _ in values
            ],
            embedding_profile=profile,
            llm_tasks=_offline_llm_tasks(),
        )
        try:
            for item in items:
                engine.write.add_node(_node_for_item(item, profile))
            hits = engine.search_nodes_as_of_scored(
                query_embeddings=[query_embedding],
                as_of_ts="2030-01-01T00:00:00+00:00",
                n_results=top_k,
            )
            evidence = [
                {
                    "id": hit.node.id,
                    "label": hit.node.label,
                    "summary": hit.node.summary,
                    "similarity": hit.similarity,
                    "multimodal_spans": [
                        span.model_dump(mode="json")
                        for grounding in hit.node.mentions
                        for span in grounding.multimodal_spans
                    ],
                }
                for hit in hits
            ]
            for item, hit in zip(items, hits, strict=False):
                if item.source_path is not None:
                    next(
                        evidence_item
                        for evidence_item in evidence
                        if evidence_item["id"] == hit.node.id
                    )["source_path"] = item.source_path
            result: dict[str, Any] = {
                "profile": profile.as_dict(),
                "profile_fingerprint": profile.fingerprint,
                "hits": evidence,
                "answer": None,
            }
            if answer_provider is not None and query is not None:
                result["answer"] = answer_provider.answer(
                    query=query, evidence=evidence
                )
            return result
        finally:
            engine.close()


def run_answer_only(
    *,
    retrieval_path: Path,
    answer_endpoint: str,
    answer_model: str,
    query_text: str | None,
    query_image: Path | None,
) -> dict[str, Any]:
    """Answer from a saved retrieval result without loading an embedder."""
    result = json.loads(retrieval_path.read_text(encoding="utf-8"))
    hits = result.get("hits")
    if not isinstance(hits, list):
        raise ValueError("retrieval result does not contain a hits list")
    evidence: list[dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            raise ValueError("retrieval result contains an invalid hit")
        source_path = hit.get("source_path")
        if not isinstance(source_path, str):
            raise ValueError("retrieval hit has no saved source_path")
        evidence_item = dict(hit)
        evidence_item.update(_image_input(Path(source_path), resource_id=str(hit["id"])))
        evidence.append(evidence_item)
    if query_text is None and query_image is None:
        raise ValueError("answer mode requires --query-text or --query-image")
    query = (
        {"resource_id": "query-text", "text": query_text}
        if query_text is not None
        else _image_input(query_image, resource_id="query")
    )
    answer = OpenAIVisionAnswerProvider(answer_endpoint, answer_model).answer(
        query=query, evidence=evidence
    )
    return {"profile": result.get("profile"), "hits": hits, "answer": answer}


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default


def _explain(title: str, body: str) -> None:
    print(f"\n{title}\n  {body}\n")


def _demo_image_inputs() -> list[tuple[str, Path]]:
    if not DEMO_IMAGE_PATH.is_file():
        raise FileNotFoundError(f"checked-in demo image is missing: {DEMO_IMAGE_PATH}")
    return [("sample-diagram", DEMO_IMAGE_PATH)]


def run_tui() -> None:
    """Run the curated instructional terminal wizard."""
    print("Pure Vision RAG demo")
    print("This wizard always follows: embed source/query -> graph write -> ranked retrieval -> optional answer.")
    print("It never treats a similarity hit as knowledge truth; the source image remains the evidence.")
    print("Embedding profiles:")
    print("1) Offline mock: no model or service")
    print("2) CPU CLIP ViT-B/32: known local demo profile, 512-D text/image space")
    print("3) Qwen3-VL GPU: known repository service, NVIDIA required")
    print("4) Qwen3-VL or Ovis: advanced operator-provided service")
    profile_choice = _ask("Embedding profile", "1")
    if profile_choice == "1":
        _explain(
            "Mock mode",
            "Uses checked-in vectors and a deterministic answer. It verifies graph wiring and "
            "grounding, but downloads no model and says nothing about retrieval quality or speed.",
        )
        query = _ask("Mock query (rabbit, lecture, or chip)", "rabbit")
        result = run_mock(query)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if profile_choice == "2":
        _explain(
            "Known CPU CLIP profile",
            "This tutorial has a checked-in image fixture and a Compose definition for the exact "
            "sentence-transformers/clip-ViT-B-32 profile. Start it with:\n"
            "docker compose -f compose.embedding-clip-cpu.yml up -d\n"
            "The default container budget is 4 GiB RAM and 2 CPUs; Docker Desktop must have at least that much available. "
            "The wizard waits for /readyz and records the advertised model, revision, dimension, metric, and fingerprint.",
        )
        api = "llm-wiki-v1"
        endpoint = _ask("CLIP service endpoint", DEFAULT_CLIP_ENDPOINT)
        image_paths = _demo_image_inputs()
        using_demo_image = True
        _explain("Built-in input", f"Using the checked-in demo image: {DEMO_IMAGE_PATH}")
    elif profile_choice == "3":
        _explain(
            "Known GPU Qwen3-VL profile",
            "This uses the standalone repository service and the same /v1/represent contract as CPU CLIP. "
            "It requires NVIDIA Container Toolkit, a compatible GPU, and a first-run model download. Start it with:\n"
            "docker compose -f compose.embedding-gpu.yml up --build -d\n"
            "GPU memory is separate from the 8 GiB host-RAM limit; the wizard waits for /readyz before sending data.",
        )
        api = "llm-wiki-v1"
        endpoint = _ask("Qwen GPU service endpoint", DEFAULT_QWEN_GPU_ENDPOINT)
        image_paths = _demo_image_inputs()
        using_demo_image = True
        _explain("Built-in input", f"Using the checked-in demo image: {DEMO_IMAGE_PATH}")
    else:
        _explain(
            "Advanced Qwen3-VL/Ovis profile",
            "The tutorial does not download arbitrary model weights. Select this only when a known service "
            "is already deployed with the repository /v1/represent contract; its advertised profile is the source of truth.",
        )
        api = "llm-wiki-v1" if _ask("Use repository /v1/represent protocol?", "y").lower() in {"y", "yes"} else "simple"
        endpoint_default = DEFAULT_CLIP_ENDPOINT if api == "llm-wiki-v1" else "http://127.0.0.1:8001/embed"
        endpoint = _ask("Embedding endpoint", endpoint_default)
        if _ask("Use the checked-in demo image?", "y").lower() in {"y", "yes"}:
            image_paths = _demo_image_inputs()
            using_demo_image = True
            _explain("Built-in input", f"Using the checked-in demo image: {DEMO_IMAGE_PATH}")
        else:
            using_demo_image = False
            image_paths = []
            for value in _ask("Images as ID=PATH pairs separated by commas").split(","):
                resource_id, separator, path = value.strip().partition("=")
                if separator and resource_id and path:
                    image_paths.append((resource_id, Path(path)))
            if not image_paths:
                raise ValueError("at least one valid ID=PATH image is required")
    query_text = _ask("Text query (leave blank to use a query image)") or None
    if query_text:
        query_image = None
    elif using_demo_image:
        query_image = image_paths[0][1]
        _explain("Built-in query", "No text was supplied, so the same checked-in image is used as the query image.")
    else:
        query_image = Path(_ask("Query image path"))

    print("Answer choices:")
    print("1) Retrieval only")
    print("2) Local Ollama/OpenAI-compatible vision endpoint")
    print("3) Codex/OpenAI-compatible gateway (advanced)")
    answer_choice = _ask("Answer mode", "1")
    if answer_choice == "1":
        _explain(
            "Retrieval only",
            "Stops after ranked evidence. Use this to inspect profile metadata, source paths, hashes, spans, and scores.",
        )
    elif answer_choice == "2":
        _explain(
            "Local vision answer",
            "The answer endpoint receives the query plus only retrieved evidence. It must implement "
            "OpenAI-compatible /v1/chat/completions and must be vision-capable for image evidence. "
            "The default is the local Ollama Bonsai vision model if it is running.",
        )
    else:
        _explain(
            "Codex/OpenAI-compatible gateway",
            "This is a protocol choice, not automatic Codex access. The endpoint must be reachable and "
            "compatible; credentials and provider billing/privacy remain the operator's responsibility.",
        )
    answer_endpoint = None
    answer_model = "vision-model"
    if answer_choice != "1":
        answer_endpoint = _ask(
            "Answer endpoint", DEFAULT_ANSWER_ENDPOINT
        )
        answer_model = _ask("Answer model", DEFAULT_ANSWER_MODEL)
    lifecycle = "retrieval"
    if answer_endpoint:
        print("Answer lifecycle:")
        print("1) Resident: keep the embedding service loaded and answer now (lower latency, higher peak memory).")
        print("2) Sequential handoff: save retrieval, stop the embedder, then answer with --mode answer (lower peak memory).")
        lifecycle = {"1": "resident", "2": "handoff"}.get(
            _ask("Answer lifecycle", "1"), "resident"
        )
    if lifecycle == "handoff":
        _explain(
            "Sequential handoff",
            "The embedding process is used only for creation and retrieval. Save the evidence artifact, "
            "stop that service yourself, then run the printed answer command. This avoids keeping both models in VRAM.",
        )
        answer_endpoint_for_run = None
    else:
        answer_endpoint_for_run = answer_endpoint
    output_default = "retrieval.json" if lifecycle == "handoff" else ""
    output_path = _ask("Save retrieval/answer JSON path (blank for stdout)", output_default)
    if api == "llm-wiki-v1":
        _wait_for_embedding_service(endpoint)
    result = run_real(
        image_paths=image_paths,
        query_image=query_image,
        embedding_endpoint=endpoint,
        embedding_api=api,
        answer_endpoint=answer_endpoint_for_run,
        answer_model=answer_model,
        top_k=int(_ask("Top-k", "2")),
        query_text=query_text,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if output_path:
        Path(output_path).write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved {output_path}")
        if lifecycle == "handoff":
            query_option = f'--query-text "{query_text}"' if query_text else f'--query-image "{query_image}"'
            print(
                "After stopping the embedding service, answer with:\n"
                f"  python scripts/pure_vision_rag_tutorial.py --mode answer --retrieval-json "
                f'"{output_path}" --answer-endpoint "{answer_endpoint}" --answer-model "{answer_model}" {query_option}'
            )
    else:
        print(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("mock", "real", "answer", "tui"),
        default="mock",
        help="mock=offline end-to-end wiring, real=model-backed retrieval, answer=saved handoff, tui=guided wizard",
    )
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--query", default="rabbit", help="Mock query name")
    parser.add_argument("--image", action="append", default=[], help="Real mode: RESOURCE_ID=PATH")
    parser.add_argument("--query-image", type=Path)
    parser.add_argument("--query-text", help="Real mode: text embedded by the same VL model")
    parser.add_argument("--embedding-endpoint")
    parser.add_argument(
        "--embedding-api",
        choices=("simple", "llm-wiki-v1"),
        default="simple",
        help="Embedding service protocol; llm-wiki-v1 uses the local CLIP/Qwen service.",
    )
    parser.add_argument("--embedding-token")
    parser.add_argument("--answer-endpoint")
    parser.add_argument("--answer-model", default="vision-model")
    parser.add_argument("--retrieval-json", type=Path)
    parser.add_argument("--output", type=Path, help="Write result JSON as UTF-8")
    parser.add_argument("--embedding-model", default="operator-configured")
    parser.add_argument("--embedding-revision", default="operator-configured")
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()
    if args.mode == "tui":
        run_tui()
        return
    if args.mode == "mock":
        result = run_mock(args.query, top_k=args.top_k, payload_path=args.payload)
    elif args.mode == "answer":
        if not args.retrieval_json or not args.answer_endpoint:
            parser.error("answer mode requires --retrieval-json and --answer-endpoint")
        result = run_answer_only(
            retrieval_path=args.retrieval_json,
            answer_endpoint=args.answer_endpoint,
            answer_model=args.answer_model,
            query_text=args.query_text,
            query_image=args.query_image,
        )
    else:
        if not args.embedding_endpoint or (args.query_image is None and args.query_text is None):
            parser.error("real mode requires --embedding-endpoint and one of --query-image/--query-text")
        image_paths: list[tuple[str, Path]] = []
        for value in args.image:
            resource_id, separator, path = value.partition("=")
            if not separator or not resource_id or not path:
                parser.error("--image must use RESOURCE_ID=PATH")
            image_paths.append((resource_id, Path(path)))
        result = run_real(
            image_paths=image_paths,
            query_image=args.query_image,
            embedding_endpoint=args.embedding_endpoint,
            embedding_api=args.embedding_api,
            embedding_token=args.embedding_token,
            answer_endpoint=args.answer_endpoint,
            answer_model=args.answer_model,
            top_k=args.top_k,
            query_text=args.query_text,
            embedding_model=args.embedding_model,
            embedding_revision=args.embedding_revision,
        )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Saved {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
