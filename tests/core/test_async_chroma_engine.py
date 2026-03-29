from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("chromadb")

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Node
from tests._kg_factories import kg_document, kg_grounding
from tests.core._async_chroma_real import (
    make_real_async_chroma_backend,
    make_real_async_chroma_uow,
    real_chroma_server,
)

pytestmark = pytest.mark.ci


class _AsyncEmbeddingFunction:
    @staticmethod
    def name() -> str:
        return "async-test"

    async def __call__(self, documents_or_texts):
        await asyncio.sleep(0)
        vectors: list[list[float]] = []
        for text in documents_or_texts:
            base = float(len(str(text)) % 7 + 1)
            vectors.append([base, base + 1.0, base + 2.0])
        return vectors


@pytest.mark.asyncio
async def test_async_embedding_and_document_write_path(real_chroma_server):
    backend_client, backend, collections = await make_real_async_chroma_backend(
        real_chroma_server, collection_prefix="engine_doc"
    )
    _ = backend_client
    uow = make_real_async_chroma_uow()

    eng = GraphKnowledgeEngine(
        persist_directory=str(real_chroma_server.persist_dir),
        embedding_function=_AsyncEmbeddingFunction(),
        backend_factory=lambda _engine: backend,
    )
    eng._phase1_enable_index_jobs = False
    assert eng.backend is backend

    doc = kg_document(
        doc_id="doc::async-chroma-doc",
        content="alpha beta gamma",
        source="async-chroma-test",
    )
    embedding_fn = _AsyncEmbeddingFunction()
    doc_embedding = (await embedding_fn([doc.content]))[0]

    async with uow.transaction():
        await backend.document_add(
            ids=[doc.id],
            documents=[doc.content],
            metadatas=[{"doc_id": doc.id, "kind": "document"}],
            embeddings=[doc_embedding],
        )

    got = await backend.document_get(
        ids=[doc.id], include=["metadatas", "documents", "embeddings"]
    )
    assert got["ids"] == [doc.id]
    assert got["documents"] == ["alpha beta gamma"]
    assert got["metadatas"][0]["doc_id"] == doc.id
    assert list(got["embeddings"][0]) == list(doc_embedding)

    direct = await collections["document"].get(
        ids=[doc.id], include=["embeddings"]
    )
    assert direct["ids"] == [doc.id]


@pytest.mark.asyncio
async def test_async_embedding_and_node_write_path(real_chroma_server):
    backend_client, backend, collections = await make_real_async_chroma_backend(
        real_chroma_server, collection_prefix="engine_node"
    )
    _ = backend_client
    uow = make_real_async_chroma_uow()

    eng = GraphKnowledgeEngine(
        persist_directory=str(real_chroma_server.persist_dir),
        embedding_function=_AsyncEmbeddingFunction(),
        backend_factory=lambda _engine: backend,
    )
    eng._phase1_enable_index_jobs = False
    assert eng.backend is backend

    doc = kg_document(
        doc_id="doc::async-chroma-node",
        content="chlorophyll absorbs light",
        source="async-chroma-test",
    )
    embedding_fn = _AsyncEmbeddingFunction()
    doc_embedding = (await embedding_fn([doc.content]))[0]
    async with uow.transaction():
        await backend.document_add(
            ids=[doc.id],
            documents=[doc.content],
            metadatas=[{"doc_id": doc.id, "kind": "document"}],
            embeddings=[doc_embedding],
        )

    node = Node(
        label="Chlorophyll",
        type="entity",
        summary="pigment that absorbs light",
        mentions=[
            kg_grounding(
                doc.id,
                start_char=0,
                end_char=11,
                excerpt="Chlorophyll",
                collection_page_url=f"document/{doc.id}",
            )
        ],
        metadata={"entity_type": "concept"},
    )

    node_embedding = (await embedding_fn([node.summary]))[0]

    async with uow.transaction():
        await backend.node_add(
            ids=[node.id],
            documents=[node.model_dump_json(field_mode="backend", exclude=["embedding"])],
            metadatas=[{"doc_id": doc.id, "entity_type": "concept"}],
            embeddings=[node_embedding],
        )
        await backend.node_refs_add(
            ids=[f"{node.id}::ref::0"],
            documents=['{"node_id": "%s", "doc_id": "%s"}' % (node.id, doc.id)],
            metadatas=[{"node_id": node.id, "doc_id": doc.id}],
            embeddings=[node_embedding],
        )

    got = await backend.node_get(
        ids=[node.id], include=["metadatas", "documents", "embeddings"]
    )
    assert got["ids"] == [node.id]
    assert got["metadatas"][0]["doc_id"] == doc.id
    assert got["metadatas"][0]["entity_type"] == "concept"
    assert list(got["embeddings"][0]) == list(node_embedding)

    node_refs = await backend.node_refs_get(
        where={"node_id": node.id},
        include=["metadatas", "documents", "embeddings"],
    )
    assert node_refs["ids"]
    assert node_refs["metadatas"][0]["node_id"] == node.id
    assert collections["node_docs"] is not None
