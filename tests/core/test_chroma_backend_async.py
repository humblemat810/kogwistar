from __future__ import annotations

import pytest

pytest.importorskip("chromadb")

from tests.core._async_chroma_real import (
    make_real_async_chroma_backend,
    real_chroma_server,  # noqa: F401
)
from kogwistar.engine_core.chroma_backend import ChromaBackend

pytestmark = pytest.mark.ci


def test_chroma_backend_round_trips_metadata_with_optional_nulls() -> None:
    import chromadb

    client = chromadb.EphemeralClient()
    collections = {
        name: client.get_or_create_collection(name=f"null_meta_{name}")
        for name in (
            "node_index",
            "node",
            "edge",
            "edge_endpoints",
            "document",
            "domain",
            "node_docs",
            "node_refs",
            "edge_refs",
        )
    }
    backend = ChromaBackend(
        node_index_collection=collections["node_index"],
        node_collection=collections["node"],
        edge_collection=collections["edge"],
        edge_endpoints_collection=collections["edge_endpoints"],
        document_collection=collections["document"],
        domain_collection=collections["domain"],
        node_docs_collection=collections["node_docs"],
        node_refs_collection=collections["node_refs"],
        edge_refs_collection=collections["edge_refs"],
    )

    backend.node_add(
        ids=["node-with-optional-null"],
        documents=["node payload"],
        metadatas=[
            {
                "wf_predicate": None,
                "wf_priority": 100,
                "implementation_selection": {
                    "schema_version": 1,
                    "effective_selectors": {"runtime": "python"},
                },
            }
        ],
        embeddings=[[0.1, 0.2]],
    )

    result = backend.node_get(ids=["node-with-optional-null"])
    assert result["metadatas"] == [
        {
            "wf_priority": 100,
            "implementation_selection": {
                "schema_version": 1,
                "effective_selectors": {"runtime": "python"},
            },
        }
    ]


@pytest.mark.asyncio
async def test_async_chroma_backend_crud_round_trip(real_chroma_server):  # noqa: F811
    _, backend, collections = await make_real_async_chroma_backend(
        real_chroma_server, collection_prefix="backend_crud"
    )

    await backend.document_add(
        ids=["doc-1"],
        documents=["hello"],
        metadatas=[
            {
                "doc_id": "doc-1",
                "kind": "document",
                "implementation_selection": {
                    "schema_version": 1,
                    "effective_selectors": {"runtime": "python"},
                },
            }
        ],
        embeddings=[[0.1, 0.2, 0.3]],
    )
    got = await backend.document_get(ids=["doc-1"])
    assert got["ids"] == ["doc-1"]
    assert got["documents"] == ["hello"]
    assert got["metadatas"][0]["doc_id"] == "doc-1"
    assert got["metadatas"][0]["implementation_selection"] == {
        "schema_version": 1,
        "effective_selectors": {"runtime": "python"},
    }

    await backend.document_update(
        ids=["doc-1"],
        documents=["hello updated"],
        metadatas=[{"doc_id": "doc-1", "kind": "document", "version": 2}],
        embeddings=[[0.3, 0.2, 0.1]],
    )
    got = await backend.document_get(ids=["doc-1"])
    assert got["documents"] == ["hello updated"]
    assert got["metadatas"][0]["version"] == 2

    await backend.document_delete(ids=["doc-1"])
    got = await backend.document_get(ids=["doc-1"])
    assert got["ids"] == []

    await backend.node_add(
        ids=["node-1"],
        documents=["node payload"],
        metadatas=[{"doc_id": "doc-2", "kind": "node"}],
        embeddings=[[1.0, 0.0, 0.0]],
    )
    got = await backend.node_get(ids=["node-1"])
    assert got["ids"] == ["node-1"]
    assert got["documents"] == ["node payload"]
    assert got["metadatas"][0]["kind"] == "node"

    await backend.node_delete(
        ids=["node-1"],
    )
    got = await backend.node_get(ids=["node-1"])
    assert got["ids"] == []

    assert collections["document"] is not None
    assert collections["node"] is not None
