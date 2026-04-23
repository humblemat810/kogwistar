from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.ci_full
pytest.importorskip("sqlalchemy")
pytest.importorskip("pgvector")
pytest.importorskip("psycopg")

from kogwistar.engine_core import postgres_backend as pg_backend_mod
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.models import Node
from kogwistar.engine_core.utils.refs import node_doc_and_meta
from tests._kg_factories import kg_document, kg_grounding


class _AsyncEmbeddingFunction:
    @staticmethod
    def name() -> str:
        return "async-pg-test"

    async def __call__(self, documents_or_texts):
        await asyncio.sleep(0)
        vectors: list[list[float]] = []
        for text in documents_or_texts:
            base = float(len(str(text)) % 7 + 1)
            vectors.append([base, base + 1.0, base + 2.0])
        return vectors


@pytest.mark.asyncio
async def test_async_pg_backend_real_crud_and_query_round_trip(
    async_pg_backend, async_pg_schema, async_pg_uow
):
    backend = async_pg_backend
    assert async_pg_schema is not None
    embedding_fn = _AsyncEmbeddingFunction()

    doc_emb = (await embedding_fn(["hello"]))[0]
    async with async_pg_uow.transaction():
        await backend.document_add(
            ids=["doc-1"],
            documents=["hello"],
            metadatas=[{"doc_id": "doc-1", "kind": "document"}],
            embeddings=[doc_emb],
        )
    got = await backend.document_get(
        ids=["doc-1"], include=["documents", "metadatas", "embeddings"]
    )
    assert got["ids"] == ["doc-1"]
    assert got["documents"] == ["hello"]
    assert got["metadatas"][0]["doc_id"] == "doc-1"

    doc_updated_emb = (await embedding_fn(["hello updated"]))[0]
    async with async_pg_uow.transaction():
        await backend.document_update(
            ids=["doc-1"],
            documents=["hello updated"],
            metadatas=[{"kind": "document", "version": 2}],
            embeddings=[doc_updated_emb],
        )
    got = await backend.document_get(
        ids=["doc-1"], include=["documents", "metadatas", "embeddings"]
    )
    assert got["documents"] == ["hello updated"]
    assert got["metadatas"][0]["version"] == 2

    got = await backend.document_query(
        query_embeddings=[doc_updated_emb],
        where={"doc_id": "doc-1"},
        n_results=1,
        include=["documents", "metadatas"],
    )
    assert got["ids"][0] == ["doc-1"]

    async with async_pg_uow.transaction():
        await backend.document_delete(ids=["doc-1"])
    got = await backend.document_get(ids=["doc-1"])
    assert got["ids"] == []

    node_emb = (await embedding_fn(["node payload"]))[0]
    async with async_pg_uow.transaction():
        await backend.node_add(
            ids=["node-1"],
            documents=["node payload"],
            metadatas=[{"doc_id": "doc-2", "kind": "node"}],
            embeddings=[node_emb],
        )
    got = await backend.node_get(
        ids=["node-1"], include=["metadatas", "embeddings"]
    )
    assert got["ids"] == ["node-1"]
    assert got["metadatas"][0]["kind"] == "node"
    assert list(got["embeddings"][0]) == list(node_emb)

    node_updated_emb = (await embedding_fn(["node payload updated"]))[0]
    async with async_pg_uow.transaction():
        await backend.node_update(
            ids=["node-1"],
            documents=["node payload updated"],
            metadatas=[{"kind": "node", "version": 2}],
            embeddings=[node_updated_emb],
        )
    got = await backend.node_get(
        ids=["node-1"], include=["documents", "metadatas", "embeddings"]
    )
    assert got["documents"] == ["node payload updated"]
    assert got["metadatas"][0]["version"] == 2

    got = await backend.node_query(
        query_embeddings=[node_updated_emb],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )
    assert got["ids"][0] == ["node-1"]

    async with async_pg_uow.transaction():
        await backend.node_delete(ids=["node-1"])
    got = await backend.node_get(ids=["node-1"])
    assert got["ids"] == []

    edge_emb = (await embedding_fn(["edge payload"]))[0]
    async with async_pg_uow.transaction():
        await backend.edge_add(
            ids=["edge-1"],
            documents=["edge payload"],
            metadatas=[{"doc_id": "doc-3", "kind": "edge"}],
            embeddings=[edge_emb],
        )
    got = await backend.edge_get(
        ids=["edge-1"], include=["metadatas", "embeddings"]
    )
    assert got["ids"] == ["edge-1"]
    assert list(got["embeddings"][0]) == list(edge_emb)

    got = await backend.edge_query(
        query_embeddings=[edge_emb],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )
    assert got["ids"][0] == ["edge-1"]

    async with async_pg_uow.transaction():
        await backend.edge_delete(ids=["edge-1"])
    assert (await backend.edge_get(ids=["edge-1"]))["ids"] == []

    domain_emb = (await embedding_fn(["domain payload"]))[0]
    async with async_pg_uow.transaction():
        await backend.domain_add(
            ids=["domain-1"],
            documents=["domain payload"],
            metadatas=[{"kind": "domain"}],
            embeddings=[domain_emb],
        )
    got = await backend.domain_get(
        ids=["domain-1"], include=["metadatas", "embeddings"]
    )
    assert got["ids"] == ["domain-1"]
    got = await backend.domain_query(
        query_embeddings=[domain_emb],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )
    assert got["ids"][0] == ["domain-1"]
    async with async_pg_uow.transaction():
        await backend.domain_delete(ids=["domain-1"])
    assert (await backend.domain_get(ids=["domain-1"]))["ids"] == []

    await backend.node_docs_add(
        ids=["node-docs-1"],
        documents=['{"node_id": "node-1", "doc_id": "doc-2"}'],
        metadatas=[{"node_id": "node-1", "doc_id": "doc-2"}],
        embeddings=[[0.1, 0.1, 0.1]],
    )
    got = await backend.node_docs_query(
        where={"node_id": "node-1"}, include=["documents"]
    )
    assert got["ids"][0] == ["node-docs-1"]
    await backend.node_docs_delete(ids=["node-docs-1"])

    await backend.node_refs_add(
        ids=["node-refs-1"],
        documents=['{"node_id": "node-1", "doc_id": "doc-2"}'],
        metadatas=[{"node_id": "node-1", "doc_id": "doc-2"}],
        embeddings=[[0.1, 0.1, 0.1]],
    )
    got = await backend.node_refs_query(
        where={"node_id": "node-1"}, include=["documents"]
    )
    assert got["ids"][0] == ["node-refs-1"]
    await backend.node_refs_delete(ids=["node-refs-1"])

    await backend.edge_refs_add(
        ids=["edge-refs-1"],
        documents=['{"edge_id": "edge-1", "doc_id": "doc-3"}'],
        metadatas=[{"edge_id": "edge-1", "doc_id": "doc-3"}],
        embeddings=[[0.1, 0.1, 0.1]],
    )
    got = await backend.edge_refs_query(
        where={"edge_id": "edge-1"}, include=["documents"]
    )
    assert got["ids"][0] == ["edge-refs-1"]
    await backend.edge_refs_delete(ids=["edge-refs-1"])

    await backend.edge_endpoints_add(
        ids=["edge-endpoints-1"],
        documents=['{"edge_id": "edge-1", "node_id": "node-1"}'],
        metadatas=[{"edge_id": "edge-1", "node_id": "node-1"}],
        embeddings=[[0.1, 0.1, 0.1]],
    )
    got = await backend.edge_endpoints_query(
        where={"edge_id": "edge-1"},
        include=["documents"],
    )
    assert got["ids"][0] == ["edge-endpoints-1"]
    await backend.edge_endpoints_delete(ids=["edge-endpoints-1"])


@pytest.mark.asyncio
async def test_async_pg_backend_transaction_rollback(
    async_pg_backend, async_pg_uow
):
    backend = async_pg_backend
    nid = "async_txn_node"

    with pytest.raises(RuntimeError):
        async with async_pg_uow.transaction():
            await backend.node_add(
                ids=[nid],
                documents=["doc"],
                metadatas=[{"name": "txn"}],
                embeddings=[[1.0, 0.0, 0.0]],
            )
            raise RuntimeError("boom")

    got = await backend.node_get(ids=[nid], include=["documents"])
    assert got["ids"] == []


@pytest.mark.asyncio
async def test_async_pg_backend_nested_transaction_joins_outer_scope(
    async_pg_backend, async_pg_uow
):
    backend = async_pg_backend
    outer_id = "async_nested_outer_node"
    inner_id = "async_nested_inner_node"

    with pytest.raises(RuntimeError):
        async with async_pg_uow.transaction():
            await backend.node_add(
                ids=[outer_id],
                documents=["outer-doc"],
                metadatas=[{"name": "outer"}],
                embeddings=[[1.0, 0.0, 0.0]],
            )
            async with async_pg_uow.transaction():
                await backend.node_add(
                    ids=[inner_id],
                    documents=["inner-doc"],
                    metadatas=[{"name": "inner"}],
                    embeddings=[[0.0, 1.0, 0.0]],
                )
            raise RuntimeError("boom")

    assert (await backend.node_get(ids=[outer_id]))["ids"] == []
    assert (await backend.node_get(ids=[inner_id]))["ids"] == []


@pytest.mark.asyncio
async def test_async_pg_backend_concurrent_task_local_uow_isolation(
    async_pg_backend, async_pg_uow
):
    backend = async_pg_backend
    first_ready = asyncio.Event()
    second_done = asyncio.Event()
    first_id = "async_task_local_first"
    second_id = "async_task_local_second"

    async def _first_worker():
        with pytest.raises(RuntimeError):
            async with async_pg_uow.transaction():
                await backend.node_add(
                    ids=[first_id],
                    documents=["first-doc"],
                    metadatas=[{"name": "first"}],
                    embeddings=[[1.0, 0.0, 0.0]],
                )
                first_ready.set()
                await second_done.wait()
                raise RuntimeError("first boom")

    async def _second_worker():
        await first_ready.wait()
        async with async_pg_uow.transaction():
            await backend.node_add(
                ids=[second_id],
                documents=["second-doc"],
                metadatas=[{"name": "second"}],
                embeddings=[[0.0, 1.0, 0.0]],
            )
        second_done.set()

    await asyncio.gather(_first_worker(), _second_worker())

    assert (await backend.node_get(ids=[first_id]))["ids"] == []
    assert (await backend.node_get(ids=[second_id]))["ids"] == [second_id]


@pytest.mark.asyncio
async def test_async_pg_engine_uow_rolls_back_writes_together(
    async_pg_backend, async_pg_uow, tmp_path
):
    backend = async_pg_backend
    eng = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "pg_async_engine_rollback"),
        embedding_function=_AsyncEmbeddingFunction(),
        backend=backend,
    )
    assert isinstance(eng.meta_sqlite, EnginePostgresMetaStore)
    eng._phase1_enable_index_jobs = False

    doc = kg_document(
        doc_id="doc::async-pg-rollback",
        content="delta epsilon zeta",
        source="async-pg-test",
    )
    node = Node(
        label="RollbackNode",
        type="entity",
        summary="rollback node",
        mentions=[
            kg_grounding(
                doc.id,
                start_char=0,
                end_char=7,
                excerpt="delta",
                collection_page_url=f"document/{doc.id}",
            )
        ],
        metadata={"entity_type": "concept"},
    )

    with pytest.raises(RuntimeError):
        async with async_pg_uow.transaction():
            await asyncio.to_thread(eng.write.add_document, doc)
            await asyncio.to_thread(eng.write.add_node, node, doc_id=doc.id)
            raise RuntimeError("boom")

    assert (await backend.document_get(ids=[doc.id]))["ids"] == []
    assert (await backend.node_get(ids=[node.id]))["ids"] == []


@pytest.mark.asyncio
async def test_async_pg_engine_sync_surface_with_async_embedding(
    async_pg_backend, async_pg_uow, tmp_path
):
    backend = async_pg_backend
    eng = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "pg_async_engine"),
        embedding_function=_AsyncEmbeddingFunction(),
        backend=backend,
    )
    assert isinstance(eng.meta_sqlite, EnginePostgresMetaStore)
    eng._phase1_enable_index_jobs = False

    doc = kg_document(
        doc_id="doc::async-pg-doc",
        content="alpha beta gamma",
        source="async-pg-test",
    )
    async with async_pg_uow.transaction():
        await asyncio.to_thread(eng.write.add_document, doc)

    node = Node(
        label="Alpha",
        type="entity",
        summary="alpha node",
        mentions=[
            kg_grounding(
                doc.id,
                start_char=0,
                end_char=5,
                excerpt="alpha",
                collection_page_url=f"document/{doc.id}",
            )
        ],
        metadata={"entity_type": "concept"},
    )
    async with async_pg_uow.transaction():
        await asyncio.to_thread(eng.write.add_node, node, doc_id=doc.id)

    node_doc, _ = node_doc_and_meta(node)
    expected_emb = (await _AsyncEmbeddingFunction()([node_doc]))[0]

    got = await backend.node_get(
        ids=[node.id], include=["metadatas", "embeddings"]
    )
    assert got["ids"] == [node.id]
    assert got["metadatas"][0]["doc_id"] == doc.id
    assert list(got["embeddings"][0]) == list(expected_emb)

    ref_rows = await backend.node_refs_get(
        where={"node_id": node.id}, include=["documents"]
    )
    assert ref_rows["ids"]
