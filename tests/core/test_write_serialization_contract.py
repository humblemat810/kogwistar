from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import Mock

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, MentionVerification, Span
from kogwistar.engine_core.storage_backend import StorageBackend
from kogwistar.engine_core.subsystems.write import (
    _AsyncBackendCollectionKey,
    _AsyncBackendMethod,
    _entity_payload,
    _refs_fingerprint,
    _sync_backend_operation,
)
from kogwistar.typing_interfaces import EmbeddingFunctionLike
from tests._helpers.fake_backend import build_fake_backend

pytestmark = [pytest.mark.ci, pytest.mark.core]


def test_entity_payload_uses_backend_json_dump_mode() -> None:
    calls: list[dict[str, object]] = []

    class _ModeAwareModel:
        def model_dump(self, **kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return {"id": "workflow-node", "mode": kwargs["field_mode"]}

    payload = _entity_payload(_ModeAwareModel())

    assert payload == {"id": "workflow-node", "mode": "backend"}
    assert calls == [
        {
            "field_mode": "backend",
            "dump_format": "json",
            "exclude": ["embedding"],
        }
    ]


def test_reference_fingerprint_includes_nested_span_changes() -> None:
    def grounding(excerpt: str) -> Grounding:
        return Grounding(
            spans=[
                Span(
                    collection_page_url="collection/doc",
                    document_page_url="document/doc",
                    doc_id="doc",
                    insertion_method="manual",
                    page_number=1,
                    start_char=0,
                    end_char=5,
                    excerpt=excerpt,
                    context_before="",
                    context_after="",
                    verification=MentionVerification(
                        method="human",
                        is_verified=True,
                        score=1.0,
                        notes=None,
                    ),
                )
            ]
        )

    alpha = grounding("alpha")
    assert _refs_fingerprint([alpha]) == _refs_fingerprint([alpha.model_copy(deep=True)])
    assert _refs_fingerprint([alpha]) != _refs_fingerprint([grounding("bravo")])


def test_async_embedding_accepts_an_async_callable_provider(tmp_path) -> None:
    class _AsyncEmbeddingProvider:
        async def __call__(self, documents: list[str]) -> list[list[float]]:
            return [[float(len(documents[0]))]]

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        embedding_function=cast(EmbeddingFunctionLike, _AsyncEmbeddingProvider()),
    )
    assert asyncio.run(engine.write._embedding_async("abc")) == [3.0]


def test_sync_backend_dispatch_uses_declared_collection_operations() -> None:
    backend = Mock(spec=StorageBackend)
    collections = (
        "node",
        "edge",
        "document",
        "domain",
        "edge_endpoints",
        "edge_refs",
        "node_docs",
        "node_refs",
    )
    methods = ("add", "delete", "get", "update", "upsert")

    for collection in collections:
        for method in methods:
            operation = _sync_backend_operation(
                backend,
                cast(_AsyncBackendCollectionKey, collection),
                cast(_AsyncBackendMethod, method),
            )
            assert operation is getattr(backend, f"{collection}_{method}")


def test_async_backend_call_uses_typed_generic_call_capability(tmp_path) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class _AsyncCallBackend:
        async def async_call(
            self, collection_key: str, method: str, **kwargs: object
        ) -> object:
            calls.append((collection_key, method, kwargs))
            return {"ids": ["node-1"]}

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path), backend_factory=build_fake_backend
    )
    original_backend = engine.backend
    engine.backend = cast(StorageBackend, _AsyncCallBackend())
    try:
        result = asyncio.run(
            engine.write._async_backend_call("node", "get", ids=["node-1"])
        )
    finally:
        engine.backend = original_backend
        engine.close()

    assert result == {"ids": ["node-1"]}
    assert calls == [("node", "get", {"ids": ["node-1"]})]


def test_async_backend_call_uses_typed_collection_facade_capability(tmp_path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class _CollectionFacade:
        def add(self, **kwargs: object) -> object:
            calls.append(("add", kwargs))
            return None

        def delete(self, **kwargs: object) -> object:
            calls.append(("delete", kwargs))
            return None

        def get(self, **kwargs: object) -> object:
            calls.append(("get", kwargs))
            return {"ids": ["edge-1"]}

        def update(self, **kwargs: object) -> object:
            calls.append(("update", kwargs))
            return None

        def upsert(self, **kwargs: object) -> object:
            calls.append(("upsert", kwargs))
            return None

    class _AsyncFacadeBackend:
        _is_async_engine = True

        def __init__(self) -> None:
            facade = _CollectionFacade()
            self._nodes_c = facade
            self._edges_c = facade
            self._documents_c = facade
            self._domains_c = facade
            self._edge_endpoints_c = facade
            self._edge_refs_c = facade
            self._node_docs_c = facade
            self._node_refs_c = facade

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path), backend_factory=build_fake_backend
    )
    original_backend = engine.backend
    engine.backend = cast(StorageBackend, _AsyncFacadeBackend())
    try:
        result = asyncio.run(
            engine.write._async_backend_call(
                "edge_refs", "get", where={"edge_id": "edge-1"}
            )
        )
    finally:
        engine.backend = original_backend
        engine.close()

    assert result == {"ids": ["edge-1"]}
    assert calls == [("get", {"where": {"edge_id": "edge-1"}})]


def test_async_semantic_upsert_uses_typed_entity_operations(tmp_path) -> None:
    calls: list[dict[str, object]] = []

    class _AsyncSemanticBackend:
        async def async_node_upsert(self, **kwargs: object) -> None:
            calls.append({"entity_kind": "node", **kwargs})

        async def async_edge_upsert(self, **kwargs: object) -> None:
            calls.append({"entity_kind": "edge", **kwargs})

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path), backend_factory=build_fake_backend
    )
    original_backend = engine.backend
    engine.backend = cast(StorageBackend, _AsyncSemanticBackend())
    try:
        asyncio.run(
            engine.write._async_semantic_upsert(
                entity_kind="node",
                entity_id="node-1",
                document="node document",
                metadata={"kind": "concept"},
                embedding=[0.1, 0.2],
            )
        )
        asyncio.run(
            engine.write._async_semantic_upsert(
                entity_kind="edge",
                entity_id="edge-1",
                document="edge document",
                metadata={"kind": "relation"},
                embedding=[0.3, 0.4],
            )
        )
    finally:
        engine.backend = original_backend
        engine.close()

    assert calls == [
        {
            "entity_kind": "node",
            "ids": ["node-1"],
            "documents": ["node document"],
            "embeddings": [[0.1, 0.2]],
            "metadatas": [{"kind": "concept"}],
        },
        {
            "entity_kind": "edge",
            "ids": ["edge-1"],
            "documents": ["edge document"],
            "embeddings": [[0.3, 0.4]],
            "metadatas": [{"kind": "relation"}],
        },
    ]
