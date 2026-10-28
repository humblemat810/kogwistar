from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.rust_meta_sqlite import RustEngineSQLite
from kogwistar.engine_core.subsystems.read import ReadSubsystem
from kogwistar.engine_core.two_stage_chroma import (
    SQLiteChromaTwoStageProjectionAdapter,
    chroma_two_stage_capability,
)
from kogwistar.graph_query import GraphQuery
from tests._helpers.embeddings import build_test_embedding_function
from tests._helpers.graph_builders import build_entity_node, build_relationship_edge
from tests.core.two_stage_case_catalog import two_stage_case


class _Backend:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.endpoint_rows: dict[str, dict] = {}

    def node_get(self, **kwargs):
        ids = [str(value) for value in kwargs.get("ids") or []]
        rows = [self.rows[value] for value in ids if value in self.rows]
        result = {
            "ids": [row["id"] for row in rows],
            "documents": [row["document"] for row in rows],
            "metadatas": [row["metadata"] for row in rows],
        }
        if "embeddings" in kwargs.get("include", []):
            result["embeddings"] = [row.get("embedding") for row in rows]
        return result

    def node_upsert(self, **kwargs):
        for node_id, document, metadata, embedding in zip(
            kwargs["ids"],
            kwargs["documents"],
            kwargs["metadatas"],
            kwargs["embeddings"],
        ):
            self.rows[str(node_id)] = {
                "id": str(node_id),
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def node_delete(self, **kwargs):
        for node_id in kwargs.get("ids") or []:
            self.rows.pop(str(node_id), None)

    def node_query(self, **kwargs):
        rows = list(self.rows.values())
        count = int(kwargs.get("n_results", len(rows)))
        rows = rows[:count]
        result = {
            "ids": [[row["id"] for row in rows]],
            "documents": [[row["document"] for row in rows]],
            "metadatas": [[row["metadata"] for row in rows]],
        }
        if "embeddings" in kwargs.get("include", []):
            result["embeddings"] = [[row.get("embedding") for row in rows]]
        if "distances" in kwargs.get("include", []):
            result["distances"] = [[float(index) for index, _ in enumerate(rows)]]
        return result

    edge_get = node_get
    edge_upsert = node_upsert
    edge_delete = node_delete
    edge_query = node_query

    def edge_endpoints_get(self, **kwargs):
        ids = [str(value) for value in kwargs.get("ids") or []]
        rows = list(self.endpoint_rows.values())
        if ids:
            rows = [row for row in rows if row["id"] in ids]
        where = kwargs.get("where") or {}
        clauses = where.get("$and", [where]) if isinstance(where, dict) else []
        for clause in clauses:
            if not isinstance(clause, dict):
                continue
            for key, value in clause.items():
                rows = [row for row in rows if row["metadata"].get(key) == value]
        return {
            "ids": [row["id"] for row in rows],
            "documents": [row["document"] for row in rows],
            "metadatas": [row["metadata"] for row in rows],
        }

    def edge_endpoints_upsert(self, **kwargs):
        for endpoint_id, document, metadata in zip(
            kwargs["ids"], kwargs["documents"], kwargs["metadatas"]
        ):
            self.endpoint_rows[str(endpoint_id)] = {
                "id": str(endpoint_id),
                "document": document,
                "metadata": metadata,
            }

    def edge_endpoints_delete(self, **kwargs):
        ids = {str(value) for value in kwargs.get("ids") or []}
        where = kwargs.get("where") or {}
        edge_id = where.get("edge_id")
        for endpoint_id, row in list(self.endpoint_rows.items()):
            if endpoint_id in ids or (
                edge_id is not None and row["metadata"].get("edge_id") == edge_id
            ):
                self.endpoint_rows.pop(endpoint_id, None)


class _Indexing:
    def __init__(self) -> None:
        self.fingerprint = "fp-1"
        self.state = "active"
        self.jobs: list[dict] = []

    def canonical_revision_payload(self, *, entity_kind: str, entity_id: str) -> str:
        return json.dumps(
            {
                "source_fingerprint": self.fingerprint,
                "source_state": self.state,
                "canonical_revision": 1,
            }
        )

    def canonical_revision_payloads(self, *, entity_kind: str, entity_ids):
        return {
            entity_id: self.canonical_revision_payload(
                entity_kind=entity_kind, entity_id=entity_id
            )
            for entity_id in entity_ids
        }

    def canonical_entity_revision(self, *, entity_kind: str, entity_id: str):
        return SimpleNamespace(
            revision=1 if self.fingerprint == "fp-1" else 2,
            state=self.state,
        )

    def enqueue_index_job(self, **kwargs) -> None:
        self.jobs.append(kwargs)


def _engine(tmp_path: Path):
    meta = EngineSQLite(tmp_path, filename="meta.sqlite")
    meta.ensure_initialized()
    backend = _Backend()
    engine = SimpleNamespace(
        namespace="tenant-a",
        kg_graph_type="knowledge",
        meta_sqlite=meta,
        metadata=meta,
        backend=backend,
        indexing=_Indexing(),
        _phase1_enable_index_jobs=True,
        write=SimpleNamespace(
            node_doc_and_meta=lambda node: (
                node.model_dump_json(field_mode="backend", exclude=["embedding"]),
                {
                    "node_id": node.safe_get_id(),
                    "kind": "entity",
                    "_class_name": type(node).__name__,
                },
            ),
            enrich_edge_meta=lambda edge: {
                "edge_id": edge.safe_get_id(),
                "_class_name": type(edge).__name__,
            },
        ),
        embed=SimpleNamespace(
            iterative_defensive_emb=lambda document: [float(len(document)), 1.0]
        ),
        persistence_mode="two_stage",
    )
    engine._resolve_redirect_chain = lambda *, initial_items, **_: initial_items
    engine._filter_items_by_resolve_mode = lambda items, _: items
    adapter = SQLiteChromaTwoStageProjectionAdapter(engine)
    engine.two_stage_projection_adapter = adapter
    engine.read = ReadSubsystem(engine)
    return engine, adapter


@two_stage_case("pending_visibility")
@two_stage_case("promotion_handoff")
def test_chroma_arrangement_hands_off_sqlite_stage1_to_explicit_vector_stage2(tmp_path):
    engine, adapter = _engine(tmp_path)
    node = build_entity_node(node_id="n1", doc_id="d1")

    adapter.add_node(node)
    adapter.enqueue_embedding_job(entity_kind="node", entity_id="n1", op="UPSERT")
    pending = ReadSubsystem(engine)._node_get_raw(
        ids=["n1"],
        where=None,
        limit=10,
        include=["documents", "metadatas", "embeddings"],
    )
    assert pending["ids"] == ["n1"]
    assert pending["embeddings"] == [None]
    assert not engine.backend.rows
    assert engine.indexing.jobs[0]["index_kind"] == "node_embedding"

    adapter.apply_embedding_job(
        entity_kind="node",
        entity_id="n1",
        op="UPSERT",
        payload_json=engine.indexing.canonical_revision_payload(
            entity_kind="node", entity_id="n1"
        ),
    )
    expected_document = node.model_dump_json(
        field_mode="backend", exclude=["embedding"]
    )
    assert engine.backend.rows["n1"]["embedding"] == [len(expected_document), 1.0]
    assert engine.meta_sqlite.get_stage1_node_projection("tenant-a", "n1") is None


@two_stage_case("batch_embedding")
def test_chroma_batch_embedding_uses_one_provider_call_per_batch(tmp_path):
    engine, adapter = _engine(tmp_path)
    calls: list[list[str]] = []

    def batch_embed(documents):
        calls.append(list(documents))
        return [[float(index), 1.0] for index, _ in enumerate(documents, 1)]

    engine._ef = batch_embed
    adapter.add_node(build_entity_node(node_id="n1", doc_id="d1"))
    adapter.enqueue_embedding_job(entity_kind="node", entity_id="n1", op="UPSERT")
    adapter.add_node(build_entity_node(node_id="n2", doc_id="d2"))
    adapter.enqueue_embedding_job(entity_kind="node", entity_id="n2", op="UPSERT")

    jobs = [
        SimpleNamespace(job_id=f"job-{index}", **job)
        for index, job in enumerate(engine.indexing.jobs, 1)
    ]
    outcomes = adapter.apply_embedding_jobs_batch(jobs)

    assert set(outcomes) == {"job-1", "job-2"}
    assert all(error is None for error in outcomes.values())
    documents = [
        build_entity_node(node_id=node_id, doc_id=f"d{index}").model_dump_json(
            field_mode="backend", exclude=["embedding"]
        )
        for index, node_id in enumerate(("n1", "n2"), 1)
    ]
    assert calls == [documents]
    assert engine.backend.rows["n1"]["embedding"] == [1.0, 1.0]
    assert engine.backend.rows["n2"]["embedding"] == [2.0, 1.0]
    assert engine.meta_sqlite.list_stage1_node_projections("tenant-a") == []


def test_chroma_stage1_edge_remains_traversable_by_id_and_neighbors(tmp_path):
    engine, adapter = _engine(tmp_path)
    adapter.add_node(build_entity_node(node_id="n1", doc_id="d1"))
    adapter.add_node(build_entity_node(node_id="n2", doc_id="d2"))
    adapter.add_edge(
        build_relationship_edge(
            edge_id="e1", src="n1", tgt="n2", doc_id="d-edge", embedding=None
        )
    )

    graph = GraphQuery(engine)
    assert graph._is_node("n1") is True
    assert graph._is_edge("e1") is True
    assert graph.neighbors("n1") == {"nodes": {"n2"}, "edges": {"e1"}}
    assert graph.neighbors("e1")["nodes"] == {"n1", "n2"}


def test_graph_query_hides_deleted_stage1_edge_before_reconciliation(
    tmp_path: Path,
) -> None:
    engine, adapter = _engine(tmp_path)
    adapter.add_node(build_entity_node(node_id="n1", doc_id="d1"))
    adapter.add_node(build_entity_node(node_id="n2", doc_id="d2"))
    adapter.add_edge(
        build_relationship_edge(
            edge_id="e1", src="n1", tgt="n2", doc_id="d-edge", embedding=None
        )
    )
    graph = GraphQuery(engine)

    assert graph.neighbors("n1") == {"nodes": {"n2"}, "edges": {"e1"}}
    engine.indexing.state = "deleted"
    assert graph.neighbors("n1") == {"nodes": set(), "edges": set()}


@pytest.mark.parametrize("entity_kind", ["node", "edge"])
def test_stage1_fallback_hides_tombstoned_entity_before_reconciliation(
    tmp_path: Path, entity_kind: str
) -> None:
    engine, adapter = _engine(tmp_path)
    if entity_kind == "node":
        adapter.add_node(build_entity_node(node_id="n1", doc_id="d1"))
        entity_id = "n1"
        read = engine.read.node_exists
    else:
        adapter.add_edge(
            build_relationship_edge(
                edge_id="e1", src="n1", tgt="n2", doc_id="d1", embedding=None
            )
        )
        entity_id = "e1"
        read = engine.read.edge_exists

    assert read(ids=[entity_id]) is True
    engine.indexing.state = "deleted"
    assert read(ids=[entity_id]) is False


def test_stage1_fallback_hides_stale_revision_before_reconciliation(
    tmp_path: Path,
) -> None:
    engine, adapter = _engine(tmp_path)
    adapter.add_node(build_entity_node(node_id="n1", doc_id="d1"))

    assert engine.read.node_exists(ids=["n1"]) is True
    engine.indexing.fingerprint = "fp-2"
    assert engine.read.node_exists(ids=["n1"]) is False


def test_stage1_fallback_requires_canonical_lifecycle_metadata(tmp_path: Path) -> None:
    engine, adapter = _engine(tmp_path)
    adapter.add_node(build_entity_node(node_id="n1", doc_id="d1"))

    def incomplete_revision_payload(**_: object) -> str:
        return json.dumps({"source_fingerprint": "fp-1"})

    engine.indexing.canonical_revision_payload = incomplete_revision_payload

    assert engine.read.node_exists(ids=["n1"]) is False


def test_chroma_endpoint_promotion_reuses_edge_vector_not_provider(tmp_path):
    engine, adapter = _engine(tmp_path)
    calls: list[str] = []

    def provider(document):
        calls.append(str(document))
        return [9.0, 9.0]

    engine.embed.iterative_defensive_emb = provider
    edge = build_relationship_edge(
        edge_id="e1", src="n1", tgt="n2", doc_id="d-edge", embedding=None
    )
    adapter.add_edge(edge)
    adapter.apply_embedding_job(
        entity_kind="edge",
        entity_id="e1",
        op="UPSERT",
        payload_json=engine.indexing.canonical_revision_payload(
            entity_kind="edge", entity_id="e1"
        ),
    )

    assert calls == [edge.model_dump_json(field_mode="backend", exclude=["embedding"])]
    assert engine.backend.endpoint_rows
    assert all(
        row["metadata"] and row["document"]
        for row in engine.backend.endpoint_rows.values()
    )


def test_chroma_endpoint_semantic_query_is_rejected():
    from kogwistar.engine_core.chroma_backend import ChromaBackend

    backend = object.__new__(ChromaBackend)
    with pytest.raises(ValueError, match="structural"):
        backend.edge_endpoints_query(query_embeddings=[[0.1, 0.2]])


@two_stage_case("stale_revision")
def test_chroma_arrangement_rejects_stale_promotion_and_deletes_stage1(tmp_path):
    engine, adapter = _engine(tmp_path)
    node = build_entity_node(node_id="n1", doc_id="d1")
    adapter.add_node(node)

    engine.indexing.fingerprint = "fp-2"
    adapter.apply_embedding_job(
        entity_kind="node",
        entity_id="n1",
        op="UPSERT",
        payload_json=json.dumps({"source_fingerprint": "fp-1"}),
    )
    assert not engine.backend.rows
    assert engine.meta_sqlite.get_stage1_node_projection("tenant-a", "n1") is not None

    engine.indexing.canonical_entity_revision = lambda **kwargs: None
    adapter.apply_embedding_job(
        entity_kind="node",
        entity_id="n1",
        op="DELETE",
        payload_json=json.dumps({"source_fingerprint": "fp-2"}),
    )
    assert engine.meta_sqlite.get_stage1_node_projection("tenant-a", "n1") is None
    assert not engine.backend.rows


def test_chroma_arrangement_rejects_stale_delete_for_newer_active_revision(tmp_path):
    engine, adapter = _engine(tmp_path)
    node = build_entity_node(node_id="n1", doc_id="d1")
    adapter.add_node(node)

    engine.indexing.fingerprint = "fp-2"
    adapter.apply_embedding_job(
        entity_kind="node",
        entity_id="n1",
        op="DELETE",
        payload_json=json.dumps({"source_fingerprint": "fp-1"}),
    )

    assert engine.meta_sqlite.get_stage1_node_projection("tenant-a", "n1") is not None


def test_chroma_new_stage1_revision_removes_old_stage2_visibility(tmp_path):
    engine, adapter = _engine(tmp_path)
    node = build_entity_node(node_id="n1", doc_id="d1")
    adapter.add_node(node)
    adapter.apply_embedding_job(
        entity_kind="node",
        entity_id="n1",
        op="UPSERT",
        payload_json=engine.indexing.canonical_revision_payload(
            entity_kind="node", entity_id="n1"
        ),
    )
    assert "n1" in engine.backend.rows

    engine.indexing.fingerprint = "fp-2"
    adapter.add_node(build_entity_node(node_id="n1", doc_id="d2"))
    assert "n1" not in engine.backend.rows
    assert engine.meta_sqlite.get_stage1_node_projection("tenant-a", "n1") is not None


@pytest.mark.parametrize("batched", [False, True])
def test_chroma_old_promotion_cannot_clear_newer_stage1_revision(tmp_path, batched):
    engine, adapter = _engine(tmp_path)
    first = build_entity_node(node_id="n1", doc_id="d1")
    adapter.add_node(first)
    old_payload = engine.indexing.canonical_revision_payload(
        entity_kind="node", entity_id="n1"
    )
    original_upsert = engine.backend.node_upsert

    def upsert_then_stage_new_revision(**kwargs):
        original_upsert(**kwargs)
        engine.indexing.fingerprint = "fp-2"
        adapter.add_node(build_entity_node(node_id="n1", doc_id="d2"))

    engine.backend.node_upsert = upsert_then_stage_new_revision
    if batched:
        from types import SimpleNamespace

        engine._ef = lambda documents: [[1.0, 0.0] for _ in documents]
        result = adapter.apply_embedding_jobs_batch(
            [
                SimpleNamespace(
                    job_id="old-batch-job",
                    entity_kind="node",
                    entity_id="n1",
                    op="UPSERT",
                    payload_json=old_payload,
                )
            ]
        )
        assert result == {"old-batch-job": None}
    else:
        adapter.apply_embedding_job(
            entity_kind="node",
            entity_id="n1",
            op="UPSERT",
            payload_json=old_payload,
        )

    staged = engine.meta_sqlite.get_stage1_node_projection("tenant-a", "n1")
    assert staged is not None
    assert staged["last_authoritative_seq"] == 2
    assert staged["payload"]["source_fingerprint"] == "fp-2"
    assert "n1" not in engine.backend.rows


@two_stage_case("recovery_reconciliation")
def test_chroma_reconciles_stage1_after_stage2_write_before_cleanup(tmp_path):
    engine, adapter = _engine(tmp_path)
    node = build_entity_node(node_id="n1", doc_id="d1")
    adapter.add_node(node)
    staged = engine.meta_sqlite.get_stage1_node_projection("tenant-a", "n1")
    payload = staged["payload"]
    metadata = dict(payload["metadata"])
    metadata["_kogwistar_stage2_ready"] = True
    metadata["_kogwistar_source_fingerprint"] = payload["source_fingerprint"]
    engine.backend.node_upsert(
        ids=["n1"],
        documents=[payload["document"]],
        metadatas=[metadata],
        embeddings=[[1.0, 0.0]],
    )

    assert engine.meta_sqlite.get_stage1_node_projection("tenant-a", "n1") is not None
    assert adapter.reconcile_projection() == 1
    assert engine.meta_sqlite.get_stage1_node_projection("tenant-a", "n1") is None
    assert engine.backend.rows["n1"]["embedding"] == [1.0, 0.0]


@pytest.mark.parametrize("case", ["inactive", "stale", "promoted"])
@pytest.mark.parametrize("rust_metadata", [False, True])
def test_reconcile_does_not_clear_newer_staged_revision(tmp_path, case, rust_metadata, monkeypatch):
    engine, adapter = _engine(tmp_path)
    if rust_metadata:
        meta = RustEngineSQLite(tmp_path, filename="rust-meta.sqlite")
        meta.ensure_initialized()
        engine.metadata = engine.meta_sqlite = meta
    adapter.add_node(build_entity_node(node_id="n1", doc_id="old"))
    if case == "inactive":
        engine.indexing.state = "tombstoned"
    elif case == "stale":
        engine.indexing.fingerprint = "fp-2"
    else:
        staged = engine.meta_sqlite.get_stage1_node_projection("tenant-a", "n1")
        payload = staged["payload"]
        engine.backend.node_upsert(
            ids=["n1"], documents=[payload["document"]],
            metadatas=[{**payload["metadata"], "_kogwistar_source_fingerprint": "fp-1"}],
            embeddings=[[1.0, 0.0]],
        )

    original_clear = adapter.remove_stage1

    def replace_before_clear(**kwargs):
        engine.indexing.state = "active"
        engine.indexing.fingerprint = "fp-2"
        adapter._stage1_upsert(
            entity_kind="node", entity=build_entity_node(node_id="n1", doc_id="new")
        )
        if case == "inactive":
            engine.backend.node_upsert(
                ids=["n1"], documents=["new"], metadatas=[{}], embeddings=[[1.0, 0.0]]
            )
        return original_clear(**kwargs)

    monkeypatch.setattr(adapter, "remove_stage1", replace_before_clear)
    assert adapter.reconcile_projection() == 0
    staged = engine.meta_sqlite.get_stage1_node_projection("tenant-a", "n1")
    assert staged is not None
    assert staged["last_authoritative_seq"] == 2
    assert staged["payload"]["source_fingerprint"] == "fp-2"
    if case == "inactive":
        assert engine.backend.rows["n1"]["document"] == "new"


@pytest.mark.parametrize("kind", ["node", "edge"])
def test_current_stage1_wins_over_stale_chroma_row_on_get(tmp_path, kind):
    engine, adapter = _engine(tmp_path)
    def entity(doc_id):
        if kind == "node":
            return build_entity_node(node_id="n1", doc_id=doc_id)
        return build_relationship_edge(edge_id="n1", src="a", tgt="b", doc_id=doc_id)

    getattr(adapter, f"add_{kind}")(entity("old"))
    key = "n1" if kind == "node" else "edge:n1"
    old = engine.meta_sqlite.get_stage1_node_projection("tenant-a", key)["payload"]
    engine.indexing.fingerprint = "fp-2"
    getattr(adapter, f"add_{kind}")(entity("new"))
    getattr(engine.backend, f"{kind}_upsert")(
        ids=["n1"], documents=[old["document"]],
        metadatas=[{**old["metadata"], "_kogwistar_source_fingerprint": "fp-1"}],
        embeddings=[[1.0, 0.0]],
    )
    result = getattr(engine.read, f"_{kind}_get_raw")(ids=["n1"])
    assert json.loads(result["documents"][0])["doc_id"] == "new"
    assert result["embeddings"] == [None]


@pytest.mark.parametrize("kind", ["node", "edge"])
def test_vector_query_drops_stale_or_unversioned_chroma_records(tmp_path, kind):
    engine, _ = _engine(tmp_path)
    current = (
        build_entity_node(node_id="current", doc_id="d1")
        if kind == "node"
        else build_relationship_edge(edge_id="current", src="a", tgt="b", doc_id="d1")
    )
    stale = (
        build_entity_node(node_id="stale", doc_id="d1")
        if kind == "node"
        else build_relationship_edge(edge_id="stale", src="a", tgt="b", doc_id="d1")
    )
    unversioned = (
        build_entity_node(node_id="legacy", doc_id="d1")
        if kind == "node"
        else build_relationship_edge(edge_id="legacy", src="a", tgt="b", doc_id="d1")
    )
    upsert = getattr(engine.backend, f"{kind}_upsert")
    upsert(
        ids=[current.safe_get_id(), stale.safe_get_id(), unversioned.safe_get_id()],
        documents=[
            current.model_dump_json(field_mode="backend", exclude=["embedding"]),
            stale.model_dump_json(field_mode="backend", exclude=["embedding"]),
            unversioned.model_dump_json(field_mode="backend", exclude=["embedding"]),
        ],
        metadatas=[
            {"_kogwistar_source_fingerprint": "fp-1"},
            {"_kogwistar_source_fingerprint": "old-fp"},
            {},
        ],
        embeddings=[[1.0, 0.0]] * 3,
    )
    queried = getattr(engine.backend, f"{kind}_query")(
        query_embeddings=[[1.0, 0.0]], include=["documents", "metadatas", "embeddings"]
    )
    assert queried["ids"][0]
    parsed = getattr(engine.read, f"{kind}s_from_query_result")(queried)
    assert parsed and parsed[0], (queried, parsed)
    assert parsed[0][0].metadata.get("_kogwistar_source_fingerprint") == "fp-1"
    query = getattr(engine.read, f"query_{kind}s")
    result = query(
        query_embeddings=[1.0, 0.0],
        include=["documents", "metadatas", "embeddings"],
    )
    assert [[item.safe_get_id() for item in batch] for batch in result] == [["current"]]


@two_stage_case("recovery_reconciliation")
def test_chroma_reconciles_missing_edge_endpoints_before_stage1_cleanup(tmp_path):
    engine, adapter = _engine(tmp_path)
    adapter.add_node(build_entity_node(node_id="n1", doc_id="d1"))
    adapter.add_node(build_entity_node(node_id="n2", doc_id="d2"))
    edge = build_relationship_edge(edge_id="e1", src="n1", tgt="n2", doc_id="d-edge")
    adapter.add_edge(edge)

    staged = engine.meta_sqlite.get_stage1_node_projection("tenant-a", "edge:e1")
    payload = staged["payload"]
    metadata = dict(payload["metadata"])
    metadata["_kogwistar_stage2_ready"] = True
    metadata["_kogwistar_source_fingerprint"] = payload["source_fingerprint"]
    engine.backend.edge_upsert(
        ids=["e1"],
        documents=[payload["document"]],
        metadatas=[metadata],
        embeddings=[[1.0, 0.0]],
    )

    assert not engine.backend.endpoint_rows
    assert adapter.reconcile_projection() == 1
    assert engine.meta_sqlite.get_stage1_node_projection("tenant-a", "edge:e1") is None
    assert engine.backend.endpoint_rows
    assert GraphQuery(engine).neighbors("n1") == {"nodes": {"n2"}, "edges": {"e1"}}


def test_chroma_capability_requires_eventual_reconciliation():
    capability = chroma_two_stage_capability()
    assert capability.atomic_promotion == "eventual_reconcile"
    assert capability.is_complete()


@pytest.mark.integration
@pytest.mark.e2e
def test_real_chroma_engine_uses_sqlite_stage1_then_promotes(tmp_path):
    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    try:
        node = build_entity_node(node_id="n1", doc_id="d1")
        engine.write.add_node(node)
        pending = engine.read._node_get_raw(
            ids=["n1"],
            where=None,
            limit=10,
            include=["documents", "metadatas", "embeddings"],
        )
        assert pending["ids"] == ["n1"]
        assert pending["embeddings"] == [None]
        assert not engine.read.query_nodes(query_embeddings=[[0.1, 0.2]], n_results=10)

        jobs = engine.meta_sqlite.list_index_jobs(namespace=engine.namespace)
        job = next(job for job in jobs if job.index_kind == "node_embedding")
        engine.indexing.apply_index_job(
            job_id=job.job_id,
            entity_kind=job.entity_kind,
            entity_id=job.entity_id,
            index_kind=job.index_kind,
            op=job.op,
            namespace=job.namespace,
            payload_json=job.payload_json,
        )
        ready = engine.read._node_get_raw(
            ids=["n1"],
            where=None,
            limit=10,
            include=["documents", "metadatas", "embeddings"],
        )
        assert ready["ids"] == ["n1"]
        assert ready["embeddings"][0] is not None
        ready_hits = engine.read.query_nodes(
            query_embeddings=[[0.1, 0.2]], n_results=10
        )
        assert [[node.id for node in row] for row in ready_hits] == [["n1"]]
        assert (
            engine.meta_sqlite.get_stage1_node_projection(engine.namespace, "n1")
            is None
        )
    finally:
        engine.close()


@pytest.mark.integration
@pytest.mark.e2e
def test_real_chroma_engine_promotes_through_index_job_worker(tmp_path):
    """Exercise the public worker path, not only direct adapter application."""
    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    try:
        engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1"))
        worker = engine.indexing.make_index_job_worker(
            batch_size=8, max_inflight=2, max_jobs_per_tick=8
        )
        metrics = worker.tick()

        assert metrics.claimed >= 1
        assert metrics.done >= 1
        assert engine.read.query_nodes(query_embeddings=[[0.1, 0.2]], n_results=10)
        assert (
            engine.meta_sqlite.get_stage1_node_projection(engine.namespace, "n1")
            is None
        )
    finally:
        engine.close()


@pytest.mark.integration
@pytest.mark.e2e
def test_real_chroma_stage1_pending_survives_engine_restart_and_worker_recovery(
    tmp_path,
):
    """A restart leaves Stage 1 recoverable; the worker completes promotion."""
    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1"))
    assert engine.meta_sqlite.get_stage1_node_projection(engine.namespace, "n1")
    engine.close()

    restarted = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    try:
        worker = restarted.indexing.make_index_job_worker(
            batch_size=8, max_inflight=2, max_jobs_per_tick=8
        )
        metrics = worker.tick()
        assert metrics.done >= 1
        assert restarted.read.query_nodes(query_embeddings=[[0.1, 0.2]], n_results=10)
        assert (
            restarted.meta_sqlite.get_stage1_node_projection(restarted.namespace, "n1")
            is None
        )
    finally:
        restarted.close()


def test_real_chroma_provider_failure_keeps_stage1_retryable(tmp_path, monkeypatch):
    """A provider failure must not acknowledge or lose the staged projection."""
    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    try:
        engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1"))
        original = engine.embed.iterative_defensive_emb

        def fail_provider(document):
            raise RuntimeError(f"provider unavailable for {document}")

        monkeypatch.setattr(engine.embed, "iterative_defensive_emb", fail_provider)
        worker = engine.indexing.make_index_job_worker(
            batch_size=1, max_inflight=1, max_jobs_per_tick=1
        )
        failed = worker.tick()
        assert failed.claimed == 1
        assert failed.done == 0
        assert failed.retried == 1
        assert (
            engine.meta_sqlite.get_stage1_node_projection(engine.namespace, "n1")
            is not None
        )
        assert any(
            item.status == "PENDING"
            for item in engine.meta_sqlite.list_index_jobs(
                namespace=engine.namespace, entity_kind="node", entity_id="n1"
            )
        )

        monkeypatch.setattr(engine.embed, "iterative_defensive_emb", original)
        time.sleep(1.1)
        recovered = worker.tick()
        assert recovered.done == 1
        assert (
            engine.meta_sqlite.get_stage1_node_projection(engine.namespace, "n1")
            is None
        )
        assert engine.read.query_nodes(query_embeddings=[[0.1, 0.2]], n_results=10)
    finally:
        engine.close()


@pytest.mark.integration
@pytest.mark.e2e
def test_real_chroma_reconciles_after_stage2_write_before_stage1_cleanup(
    tmp_path, monkeypatch
):
    """A promotion crash converges without leaving dual projection visibility."""
    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    try:
        engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1"))
        job = next(
            item
            for item in engine.meta_sqlite.list_index_jobs(namespace=engine.namespace)
            if item.index_kind == "node_embedding"
        )
        original = engine.two_stage_projection_adapter.remove_stage1

        def fail_cleanup(**kwargs):
            raise RuntimeError("simulated crash after Stage-2 write")

        monkeypatch.setattr(
            engine.two_stage_projection_adapter, "remove_stage1", fail_cleanup
        )
        with pytest.raises(RuntimeError, match="after Stage-2"):
            engine.two_stage_projection_adapter.apply_embedding_job(
                entity_kind="node",
                entity_id="n1",
                op="UPSERT",
                payload_json=job.payload_json,
            )
        assert engine.read.query_nodes(query_embeddings=[[0.1, 0.2]], n_results=10)
        assert engine.meta_sqlite.get_stage1_node_projection(engine.namespace, "n1")

        monkeypatch.setattr(
            engine.two_stage_projection_adapter, "remove_stage1", original
        )
        assert engine.two_stage_projection_adapter.reconcile_projection() == 1
        assert (
            engine.meta_sqlite.get_stage1_node_projection(engine.namespace, "n1")
            is None
        )
    finally:
        engine.close()


@pytest.mark.integration
@pytest.mark.e2e
def test_real_chroma_rebuilds_lost_embedding_job_after_restart(tmp_path):
    """Canonical state repairs an enqueue that was lost after admission."""
    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1"))
    claimed = engine.meta_sqlite.claim_index_jobs(
        limit=10, lease_seconds=60, namespace=engine.namespace
    )
    for job in claimed:
        engine.meta_sqlite.mark_index_job_done(job.job_id, claim_token=job.claim_token)
    engine.close()

    restarted = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    try:
        assert restarted.indexing.repair_missing_two_stage_embedding_jobs() == 1
        worker = restarted.indexing.make_index_job_worker(
            batch_size=8, max_inflight=2, max_jobs_per_tick=8
        )
        assert worker.tick().done >= 1
        assert restarted.read.query_nodes(query_embeddings=[[0.1, 0.2]], n_results=10)
        assert (
            restarted.meta_sqlite.get_stage1_node_projection(restarted.namespace, "n1")
            is None
        )
    finally:
        restarted.close()


@pytest.mark.integration
@pytest.mark.e2e
@two_stage_case("delete_race")
def test_real_chroma_stage1_delete_is_repaired_from_canonical_event(tmp_path):
    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    try:
        engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1"))
        assert engine.lifecycle.tombstone_node("n1") is False
        assert engine.indexing.repair_missing_two_stage_embedding_jobs() == 1
        jobs = engine.meta_sqlite.list_index_jobs(namespace=engine.namespace)
        assert any(
            job.index_kind == "node_embedding" and job.op == "DELETE" for job in jobs
        )
    finally:
        engine.close()
