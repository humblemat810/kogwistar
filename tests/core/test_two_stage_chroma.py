from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.subsystems.read import ReadSubsystem
from kogwistar.engine_core.two_stage_chroma import (
    SQLiteChromaTwoStageProjectionAdapter,
    chroma_two_stage_capability,
)
from kogwistar.graph_query import GraphQuery
from tests._helpers.graph_builders import build_entity_node, build_relationship_edge
from tests._helpers.embeddings import build_test_embedding_function
from tests.core.two_stage_case_catalog import two_stage_case


class _Backend:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def node_get(self, **kwargs):
        ids = [str(value) for value in kwargs.get("ids") or []]
        rows = [self.rows[value] for value in ids if value in self.rows]
        return {
            "ids": [row["id"] for row in rows],
            "documents": [row["document"] for row in rows],
            "metadatas": [row["metadata"] for row in rows],
        }

    def node_upsert(self, **kwargs):
        for node_id, document, metadata, embedding in zip(
            kwargs["ids"], kwargs["documents"], kwargs["metadatas"], kwargs["embeddings"]
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

    edge_get = node_get
    edge_upsert = node_upsert
    edge_delete = node_delete


class _Indexing:
    def __init__(self) -> None:
        self.fingerprint = "fp-1"
        self.jobs: list[dict] = []

    def canonical_revision_payload(self, *, entity_kind: str, entity_id: str) -> str:
        return json.dumps({"source_fingerprint": self.fingerprint})

    def canonical_entity_revision(self, *, entity_kind: str, entity_id: str):
        return SimpleNamespace(revision=1, state="active")

    def enqueue_index_job(self, **kwargs) -> None:
        self.jobs.append(kwargs)


def _engine(tmp_path: Path):
    meta = EngineSQLite(tmp_path, filename="meta.sqlite")
    meta.ensure_initialized()
    backend = _Backend()
    engine = SimpleNamespace(
        namespace="tenant-a",
        meta_sqlite=meta,
        backend=backend,
        indexing=_Indexing(),
        write=SimpleNamespace(
            node_doc_and_meta=lambda node: (
                f"document:{node.safe_get_id()}",
                {"node_id": node.safe_get_id(), "kind": "entity"},
            ),
            enrich_edge_meta=lambda edge: {"edge_id": edge.safe_get_id()},
        ),
        embed=SimpleNamespace(
            iterative_defensive_emb=lambda document: [float(len(document)), 1.0]
        ),
        persistence_mode="two_stage",
    )
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
    pending = ReadSubsystem(engine)._node_get_raw(ids=["n1"], where=None, limit=10, include=["documents", "metadatas", "embeddings"])
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
    assert engine.backend.rows["n1"]["embedding"] == [len("document:n1"), 1.0]
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
    adapter.add_node(build_entity_node(node_id="n2", doc_id="d2"))

    jobs = [
        SimpleNamespace(job_id=f"job-{index}", **job)
        for index, job in enumerate(engine.indexing.jobs, 1)
    ]
    outcomes = adapter.apply_embedding_jobs_batch(jobs)

    assert set(outcomes) == {"job-1", "job-2"}
    assert all(error is None for error in outcomes.values())
    assert calls == [["document:n1", "document:n2"]]
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
            ids=["n1"], where=None, limit=10, include=["documents", "metadatas", "embeddings"]
        )
        assert pending["ids"] == ["n1"]
        assert pending["embeddings"] == [None]
        assert not engine.read.query_nodes(
            query_embeddings=[[0.1, 0.2]], n_results=10
        )

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
            ids=["n1"], where=None, limit=10, include=["documents", "metadatas", "embeddings"]
        )
        assert ready["ids"] == ["n1"]
        assert ready["embeddings"][0] is not None
        ready_hits = engine.read.query_nodes(
            query_embeddings=[[0.1, 0.2]], n_results=10
        )
        assert [[node.id for node in row] for row in ready_hits] == [["n1"]]
        assert engine.meta_sqlite.get_stage1_node_projection(engine.namespace, "n1") is None
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
        assert engine.meta_sqlite.get_stage1_node_projection(
            engine.namespace, "n1"
        ) is None
    finally:
        engine.close()


@pytest.mark.integration
@pytest.mark.e2e
def test_real_chroma_stage1_pending_survives_engine_restart_and_worker_recovery(tmp_path):
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
        assert restarted.meta_sqlite.get_stage1_node_projection(
            restarted.namespace, "n1"
        ) is None
    finally:
        restarted.close()


@pytest.mark.integration
@pytest.mark.e2e
def test_real_chroma_reconciles_after_stage2_write_before_stage1_cleanup(tmp_path, monkeypatch):
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

        monkeypatch.setattr(engine.two_stage_projection_adapter, "remove_stage1", fail_cleanup)
        with pytest.raises(RuntimeError, match="after Stage-2"):
            engine.two_stage_projection_adapter.apply_embedding_job(
                entity_kind="node",
                entity_id="n1",
                op="UPSERT",
                payload_json=job.payload_json,
            )
        assert engine.read.query_nodes(query_embeddings=[[0.1, 0.2]], n_results=10)
        assert engine.meta_sqlite.get_stage1_node_projection(engine.namespace, "n1")

        monkeypatch.setattr(engine.two_stage_projection_adapter, "remove_stage1", original)
        assert engine.two_stage_projection_adapter.reconcile_projection() == 1
        assert engine.meta_sqlite.get_stage1_node_projection(engine.namespace, "n1") is None
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
        assert restarted.meta_sqlite.get_stage1_node_projection(
            restarted.namespace, "n1"
        ) is None
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
