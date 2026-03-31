from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("chromadb")

from tests.core._async_chroma_real import (
    make_real_async_chroma_backend,
    real_chroma_server,  # noqa: F401
)

pytestmark = pytest.mark.ci


@pytest.mark.asyncio
async def test_async_chroma_backend_crud_round_trip(real_chroma_server):  # noqa: F811
    _, backend, collections = await make_real_async_chroma_backend(
        real_chroma_server, collection_prefix="backend_crud"
    )

    await asyncio.to_thread(
        backend.document_add,
        ids=["doc-1"],
        documents=["hello"],
        metadatas=[{"doc_id": "doc-1", "kind": "document"}],
        embeddings=[[0.1, 0.2, 0.3]],
    )
    got = await asyncio.to_thread(backend.document_get, ids=["doc-1"])
    assert got["ids"] == ["doc-1"]
    assert got["documents"] == ["hello"]
    assert got["metadatas"][0]["doc_id"] == "doc-1"

    await asyncio.to_thread(
        backend.document_update,
        ids=["doc-1"],
        documents=["hello updated"],
        metadatas=[{"doc_id": "doc-1", "kind": "document", "version": 2}],
        embeddings=[[0.3, 0.2, 0.1]],
    )
    got = await asyncio.to_thread(backend.document_get, ids=["doc-1"])
    assert got["documents"] == ["hello updated"]
    assert got["metadatas"][0]["version"] == 2

    await asyncio.to_thread(backend.document_delete, ids=["doc-1"])
    got = await asyncio.to_thread(backend.document_get, ids=["doc-1"])
    assert got["ids"] == []

    await asyncio.to_thread(
        backend.node_add,
        ids=["node-1"],
        documents=["node payload"],
        metadatas=[{"doc_id": "doc-2", "kind": "node"}],
        embeddings=[[1.0, 0.0, 0.0]],
    )
    got = await asyncio.to_thread(backend.node_get, ids=["node-1"])
    assert got["ids"] == ["node-1"]
    assert got["documents"] == ["node payload"]
    assert got["metadatas"][0]["kind"] == "node"

    await asyncio.to_thread(
        backend.node_delete,
        ids=["node-1"],
    )
    got = await asyncio.to_thread(backend.node_get, ids=["node-1"])
    assert got["ids"] == []

    assert collections["document"] is not None
    assert collections["node"] is not None
