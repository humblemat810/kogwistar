from __future__ import annotations

import json

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.postgres_backend import PgVectorBackend
from tests._helpers.embeddings import ConstantEmbeddingFunction
from tests._helpers.graph_builders import build_entity_node, build_relationship_edge

pytestmark = [pytest.mark.ci_full, pytest.mark.integration, pytest.mark.e2e]


def _require_postgres(sa_engine, pg_schema) -> None:
    if sa_engine is None or pg_schema is None:
        pytest.skip("live PostgreSQL/pgvector fixture unavailable")


def _engine(sa_engine, pg_schema) -> GraphKnowledgeEngine:
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=pg_schema)
    return GraphKnowledgeEngine(
        backend=backend,
        persistence_mode="two_stage",
        embedding_function=ConstantEmbeddingFunction(dim=3),
    )


def test_postgres_two_stage_pending_then_same_store_promotion(sa_engine, pg_schema) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    node = build_entity_node(node_id="n1", doc_id="d1", embedding=None)

    engine.write.add_node(node)

    stage1 = engine.backend.stage1_projection_query(
        namespace="default", entity_kind="node", ids=["n1"]
    )
    assert [row["entity_id"] for row in stage1] == ["n1"]
    assert engine.backend.node_get(ids=["n1"], include=["embeddings"])["ids"] == []
    assert engine.backend.node_query(
        query_embeddings=[[1.0, 0.0, 0.0]], n_results=10
    )["ids"] == [[]]

    assert engine.indexing.reconcile_indexes(max_jobs=10) == 1

    promoted = engine.backend.node_get(ids=["n1"], include=["embeddings", "metadatas"])
    assert promoted["ids"] == ["n1"]
    assert promoted["embeddings"][0] is not None
    assert promoted["metadatas"][0]["_kogwistar_stage2_ready"] is True
    assert engine.backend.stage1_projection_get(
        namespace="default", entity_kind="node", entity_id="n1"
    ) is None


def test_postgres_two_stage_old_job_cannot_promote_current_revision(
    sa_engine, pg_schema
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))
    jobs = engine.meta_sqlite.list_index_jobs(
        namespace="default", entity_kind="node", entity_id="n1"
    )
    old_payload = jobs[0].payload_json

    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d2", embedding=None))
    engine.two_stage_projection_adapter.apply_embedding_job(
        entity_kind="node", entity_id="n1", op="UPSERT", payload_json=old_payload
    )

    assert engine.backend.node_get(ids=["n1"], include=["embeddings"])["ids"] == []
    current_payload = engine.indexing.canonical_revision_payload(
        entity_kind="node", entity_id="n1"
    )
    assert json.loads(current_payload)["canonical_revision"] == 2


def test_postgres_two_stage_event_and_stage1_admission_roll_back_together(
    sa_engine, pg_schema, monkeypatch
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    original = engine.two_stage_projection_adapter.add_node

    def fail_after_stage1(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected Stage-1 failure")

    monkeypatch.setattr(engine.two_stage_projection_adapter, "add_node", fail_after_stage1)
    with pytest.raises(RuntimeError, match="injected Stage-1 failure"):
        engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))

    assert engine.backend.stage1_projection_get(
        namespace="default", entity_kind="node", entity_id="n1"
    ) is None
    assert list(engine.meta_sqlite.iter_entity_events(namespace="default", from_seq=1)) == []


def test_postgres_two_stage_delete_before_promotion_cleans_stage1(
    sa_engine, pg_schema
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))

    assert engine.tombstone_node("n1") is True
    assert engine.backend.stage1_projection_get(
        namespace="default", entity_kind="node", entity_id="n1"
    ) is None
    assert engine.backend.node_get(ids=["n1"], include=["embeddings"])["ids"] == []


def test_postgres_two_stage_reconciliation_cleans_stage1_after_stage2_write(
    sa_engine, pg_schema
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))
    row = engine.backend.stage1_projection_get(
        namespace="default", entity_kind="node", entity_id="n1"
    )
    assert row is not None
    metadata = dict(row["metadata"])
    metadata["_kogwistar_stage2_ready"] = True
    metadata["_kogwistar_source_fingerprint"] = row["source_fingerprint"]
    engine.backend.node_upsert(
        ids=["n1"], documents=[row["document"]], metadatas=[metadata], embeddings=[[1, 0, 0]]
    )

    assert engine.two_stage_projection_adapter.reconcile_projection() == 1
    assert engine.backend.stage1_projection_get(
        namespace="default", entity_kind="node", entity_id="n1"
    ) is None


def test_postgres_two_stage_delete_during_embedding_cannot_resurrect(
    sa_engine, pg_schema, monkeypatch
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))
    job = engine.meta_sqlite.list_index_jobs(
        namespace="default", entity_kind="node", entity_id="n1"
    )[0]
    original = engine.embed.iterative_defensive_emb
    called = {"value": False}

    def embed_then_delete(document):
        if not called["value"]:
            called["value"] = True
            assert engine.tombstone_node("n1") is True
        return original(document)

    monkeypatch.setattr(engine.embed, "iterative_defensive_emb", embed_then_delete)
    engine.two_stage_projection_adapter.apply_embedding_job(
        entity_kind="node", entity_id="n1", op="UPSERT", payload_json=job.payload_json
    )
    assert engine.backend.node_get(ids=["n1"], include=["embeddings"])["ids"] == []
    assert engine.backend.stage1_projection_get(
        namespace="default", entity_kind="node", entity_id="n1"
    ) is None


def test_postgres_two_stage_late_v1_cannot_overwrite_v2(
    sa_engine, pg_schema
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="v1", embedding=None))
    old_job = engine.meta_sqlite.list_index_jobs(
        namespace="default", entity_kind="node", entity_id="n1"
    )[0]
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="v2", embedding=None))
    assert engine.indexing.reconcile_indexes(max_jobs=10) == 1
    current = engine.backend.node_get(ids=["n1"], include=["metadatas"])
    current_fp = current["metadatas"][0]["_kogwistar_source_fingerprint"]
    engine.two_stage_projection_adapter.apply_embedding_job(
        entity_kind="node", entity_id="n1", op="UPSERT", payload_json=old_job.payload_json
    )
    assert engine.backend.node_get(ids=["n1"], include=["metadatas"])["metadatas"][0][
        "_kogwistar_source_fingerprint"
    ] == current_fp


def test_postgres_two_stage_scanner_repairs_lost_enqueue(
    sa_engine, pg_schema
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))
    job = engine.meta_sqlite.list_index_jobs(
        namespace="default", entity_kind="node", entity_id="n1"
    )[0]
    claimed = engine.meta_sqlite.claim_index_jobs(limit=1, lease_seconds=60, namespace="default")
    assert len(claimed) == 1
    assert engine.meta_sqlite.mark_index_job_done(
        claimed[0].job_id, claim_token=claimed[0].claim_token
    ) is True
    assert engine.indexing.repair_missing_two_stage_embedding_jobs(max_entities=10) == 1
    repaired = engine.meta_sqlite.list_index_jobs(
        namespace="default", entity_kind="node", entity_id="n1"
    )
    assert any(item.status == "PENDING" and item.index_kind == "node_embedding" for item in repaired)


def test_postgres_two_stage_public_api_and_batch_claim_isolate_failure(
    sa_engine, pg_schema, monkeypatch
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))
    engine.write.add_node(build_entity_node(node_id="n2", doc_id="d2", embedding=None))
    public = engine.get_nodes(ids=["n1"], include=["documents", "metadatas", "embeddings"])
    assert [node.id for node in public] == ["n1"]
    assert public[0].embedding is None
    assert [node.id for node in engine.get_nodes(where={"label": "Node n1"})] == ["n1"]
    assert engine.query_nodes(query_embeddings=[[1, 0, 0]], n_results=10) == []

    original = engine.embed.iterative_defensive_emb
    calls = {"count": 0}
    batch_calls = {"count": 0}

    def batch_provider(documents):
        batch_calls["count"] += 1
        if any("n2" in document for document in documents):
            raise RuntimeError("injected batch provider failure")
        return ConstantEmbeddingFunction(dim=3)(documents)

    monkeypatch.setattr(engine, "_ef", batch_provider)

    def one_bad(document):
        calls["count"] += 1
        if "n2" in document:
            raise RuntimeError("injected provider failure")
        return original(document)

    monkeypatch.setattr(engine.embed, "iterative_defensive_emb", one_bad)
    worker = engine.indexing.make_index_job_worker(batch_size=2, max_inflight=2, max_jobs_per_tick=2)
    metrics = worker.tick()
    assert metrics.claimed == 2
    assert metrics.done == 1
    assert metrics.retried == 1
    assert calls["count"] == 2
    assert batch_calls["count"] >= 1
    semantic = engine.query_nodes(query_embeddings=[[1, 0, 0]], n_results=10)
    assert {node.id for row in semantic for node in row} == {"n1"}


def test_postgres_two_stage_edges_follow_same_promotion_contract(sa_engine, pg_schema) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))
    engine.write.add_node(build_entity_node(node_id="n2", doc_id="d2", embedding=None))
    edge = build_relationship_edge(
        edge_id="e1", src="n1", tgt="n2", doc_id="d-edge", embedding=None
    )

    engine.write.add_edge(edge)

    staged = engine.backend.stage1_projection_get(
        namespace="default", entity_kind="edge", entity_id="e1"
    )
    assert staged is not None
    assert engine.backend.edge_query(
        query_embeddings=[[1.0, 0.0, 0.0]], n_results=10
    )["ids"] == [[]]

    assert engine.indexing.reconcile_indexes(max_jobs=10) == 3
    promoted = engine.backend.edge_get(ids=["e1"], include=["embeddings", "metadatas"])
    assert promoted["ids"] == ["e1"]
    assert promoted["embeddings"][0] is not None
    assert engine.backend.stage1_projection_get(
        namespace="default", entity_kind="edge", entity_id="e1"
    ) is None


def test_postgres_two_stage_provider_batch_promotes_multiple_nodes(
    sa_engine, pg_schema, monkeypatch
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))
    engine.write.add_node(build_entity_node(node_id="n2", doc_id="d2", embedding=None))
    jobs = engine.meta_sqlite.list_index_jobs(namespace="default")
    embedding_calls: list[list[str]] = []

    def batch_embed(documents):
        embedding_calls.append(list(documents))
        return [[1.0, 0.0, 0.0] for _ in documents]

    monkeypatch.setattr(engine, "_ef", batch_embed)
    outcomes = engine.two_stage_projection_adapter.apply_embedding_jobs_batch(
        [job for job in jobs if job.index_kind == "node_embedding"]
    )

    assert len(embedding_calls) == 1
    assert len(embedding_calls[0]) == 2
    assert set(outcomes) == {job.job_id for job in jobs if job.index_kind == "node_embedding"}
    assert all(value is None for value in outcomes.values())
    assert engine.backend.node_get(ids=["n1", "n2"], include=["embeddings"])["ids"] == [
        "n1",
        "n2",
    ]


def test_postgres_two_stage_async_backend_fails_closed(async_pg_backend) -> None:
    with pytest.raises(ValueError, match="requires synchronous PostgreSQL"):
        GraphKnowledgeEngine(
            backend=async_pg_backend,
            persistence_mode="two_stage",
            embedding_function=ConstantEmbeddingFunction(dim=3),
        )


def test_postgres_two_stage_rust_authority_fails_closed(
    sa_engine, pg_schema, monkeypatch
) -> None:
    _require_postgres(sa_engine, pg_schema)
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=pg_schema)
    with pytest.raises(ValueError, match="not available with Rust PostgreSQL authority"):
        GraphKnowledgeEngine(
            backend=backend,
            persistence_mode="two_stage",
            embedding_function=ConstantEmbeddingFunction(dim=3),
        )
