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


class VisionEmbeddingProvider(Protocol):
    def embed_inputs(self, inputs: list[dict[str, str]]) -> list[list[float]]: ...


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


class PrecomputedVisionProvider:
    def __init__(self, items: list[VisionItem], queries: dict[str, list[float]]) -> None:
        self._items = {item.resource_id: item.embedding for item in items}
        self.queries = queries

    def embed_inputs(self, inputs: list[dict[str, str]]) -> list[list[float]]:
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

    def embed_inputs(self, inputs: list[dict[str, str]]) -> list[list[float]]:
        payload = _post_json(self.endpoint, {"inputs": inputs})
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or not all(isinstance(row, list) for row in embeddings):
            raise ValueError("vision embedding endpoint must return embeddings as a list of vectors")
        if len(embeddings) != len(inputs):
            raise ValueError("vision embedding endpoint returned the wrong number of vectors")
        return [[float(value) for value in row] for row in embeddings]


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
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("vision answer endpoint returned no chat content") from exc


def _post_json(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
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
    return _run_graph_search(profile, items, query_embedding, top_k=top_k, answer_provider=None)


def run_real(
    *,
    image_paths: list[tuple[str, Path]],
    query_image: Path | None,
    embedding_endpoint: str,
    answer_endpoint: str | None,
    answer_model: str,
    top_k: int,
    query_text: str | None = None,
    embedding_model: str = "operator-configured",
    embedding_revision: str = "operator-configured",
) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("real mode requires at least one --image ID=PATH")
    provider = JsonVisionEmbeddingProvider(embedding_endpoint)
    inputs = [_image_input(path, resource_id=resource_id) for resource_id, path in image_paths]
    vectors = provider.embed_inputs(inputs)
    dimension = len(vectors[0]) if vectors else 0
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
        )
        for index, ((resource_id, path), vector) in enumerate(zip(image_paths, vectors, strict=True))
    ]
    query_input = (
        {"resource_id": "query-text", "text": query_text}
        if query_text is not None
        else _image_input(query_image, resource_id="query")
    )
    query_vector = provider.embed_inputs([query_input])[0]
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
            embedding_function=lambda values: [[1.0, 0.0, 0.0, 0.0] for _ in values],
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "real"), default="mock")
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--query", default="rabbit", help="Mock query name")
    parser.add_argument("--image", action="append", default=[], help="Real mode: RESOURCE_ID=PATH")
    parser.add_argument("--query-image", type=Path)
    parser.add_argument("--query-text", help="Real mode: text embedded by the same VL model")
    parser.add_argument("--embedding-endpoint")
    parser.add_argument("--answer-endpoint")
    parser.add_argument("--answer-model", default="vision-model")
    parser.add_argument("--embedding-model", default="operator-configured")
    parser.add_argument("--embedding-revision", default="operator-configured")
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()
    if args.mode == "mock":
        result = run_mock(args.query, top_k=args.top_k, payload_path=args.payload)
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
            answer_endpoint=args.answer_endpoint,
            answer_model=args.answer_model,
            top_k=args.top_k,
            query_text=args.query_text,
            embedding_model=args.embedding_model,
            embedding_revision=args.embedding_revision,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
