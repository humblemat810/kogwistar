from __future__ import annotations

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.chroma_backend import ChromaBackend
from kogwistar.engine_core.storage_backend import (
    TwoStageProjectionCapability,
    get_two_stage_projection_adapter,
    get_two_stage_projection_capability,
)
from tests._helpers.fake_backend import build_fake_backend
from tests._helpers.graph_builders import build_entity_node


def test_two_stage_defaults_to_single_stage_for_existing_backend(tmp_path) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path), backend_factory=build_fake_backend
    )

    assert engine.persistence_mode == "single_stage"
    assert engine.two_stage_projection_capability.supports_two_stage is True


def test_chroma_backend_fails_closed_until_stage1_arrangement_is_implemented() -> None:
    backend = ChromaBackend(
        node_index_collection=object(),
        node_collection=object(),
        edge_collection=object(),
        edge_endpoints_collection=object(),
        document_collection=object(),
        domain_collection=object(),
        node_docs_collection=object(),
        node_refs_collection=object(),
        edge_refs_collection=object(),
    )

    capability = get_two_stage_projection_capability(backend)

    assert not capability.is_complete()
    assert capability.supports_two_stage is False
    assert get_two_stage_projection_adapter(backend) is None


def test_two_stage_rejects_backend_without_complete_capability(tmp_path) -> None:
    def factory(*args, **kwargs):
        backend = build_fake_backend(*args, **kwargs)
        backend.two_stage_projection_capability = TwoStageProjectionCapability()
        return backend

    with pytest.raises(ValueError, match="persistence_mode='two_stage'.*InMemoryBackend"):
        GraphKnowledgeEngine(
            persist_directory=str(tmp_path),
            backend_factory=factory,
            persistence_mode="two_stage",
        )


def test_two_stage_invalid_mode_fails_before_backend_construction(tmp_path) -> None:
    with pytest.raises(ValueError, match="persistence_mode must be"):
        GraphKnowledgeEngine(
            persist_directory=str(tmp_path),
            backend_factory=build_fake_backend,
            persistence_mode="deferred",
        )


def test_optional_backend_capability_declaration_is_immutable() -> None:
    capability = TwoStageProjectionCapability(
        supports_two_stage=True,
        stage1_strategy="transient_projection",
        atomic_promotion="eventual_reconcile",
        reason="test arrangement",
    )

    class _Backend:
        two_stage_projection_capability = capability

    assert get_two_stage_projection_capability(_Backend()) is capability


def test_two_stage_rejects_partial_capability_declaration(tmp_path) -> None:
    capability = TwoStageProjectionCapability(supports_two_stage=True)

    def partial_factory(*args, **kwargs):
        backend = build_fake_backend(*args, **kwargs)
        backend.two_stage_projection_capability = capability
        return backend

    with pytest.raises(ValueError, match="missing capability contract: canonical_event_replay"):
        GraphKnowledgeEngine(
            persist_directory=str(tmp_path),
            backend_factory=partial_factory,
            persistence_mode="two_stage",
        )


def test_complete_capability_descriptor_has_no_missing_contracts() -> None:
    capability = TwoStageProjectionCapability(
        supports_two_stage=True,
        canonical_event_replay=True,
        canonical_read=True,
        stage1_strategy="transient_projection",
        stage1_metadata_query=True,
        stage1_cleanup=True,
        stage2_semantic_projection=True,
        revision_gated_promotion=True,
        semantic_readiness_gate=True,
        delete_reconciliation=True,
        atomic_promotion="eventual_reconcile",
    )

    assert capability.is_complete()
    assert capability.missing_contracts() == ()


def _complete_capability() -> TwoStageProjectionCapability:
    return TwoStageProjectionCapability(
        supports_two_stage=True,
        canonical_event_replay=True,
        canonical_read=True,
        stage1_strategy="transient_projection",
        stage1_metadata_query=True,
        stage1_cleanup=True,
        stage2_semantic_projection=True,
        revision_gated_promotion=True,
        semantic_readiness_gate=True,
        delete_reconciliation=True,
        atomic_promotion="eventual_reconcile",
    )


def test_two_stage_rejects_descriptor_without_executable_adapter(tmp_path) -> None:
    def factory(*args, **kwargs):
        backend = build_fake_backend(*args, **kwargs)
        backend.two_stage_projection_capability = _complete_capability()
        backend.two_stage_projection_adapter = None
        return backend

    with pytest.raises(ValueError, match="requires an executable two-stage projection adapter"):
        GraphKnowledgeEngine(
            persist_directory=str(tmp_path),
            backend_factory=factory,
            persistence_mode="two_stage",
        )


def test_two_stage_rejects_adapter_without_promotion_handler(tmp_path) -> None:
    class _AdmissionOnlyAdapter:
        def add_node(self, node, *, doc_id=None):
            return None

        def add_edge(self, edge, *, doc_id=None):
            return None

    def factory(*args, **kwargs):
        backend = build_fake_backend(*args, **kwargs)
        backend.two_stage_projection_capability = _complete_capability()
        backend.two_stage_projection_adapter = _AdmissionOnlyAdapter()
        return backend

    with pytest.raises(ValueError, match="requires an executable two-stage projection adapter"):
        GraphKnowledgeEngine(
            persist_directory=str(tmp_path),
            backend_factory=factory,
            persistence_mode="two_stage",
        )


def test_two_stage_rejects_transient_adapter_without_stage1_seams(tmp_path) -> None:
    class _IncompleteTransientAdapter:
        def add_node(self, node, *, doc_id=None):
            return None

        def add_edge(self, edge, *, doc_id=None):
            return None

        def apply_embedding_job(self, *, entity_kind, entity_id, op, payload_json):
            return None

    def factory(*args, **kwargs):
        backend = build_fake_backend(*args, **kwargs)
        backend.two_stage_projection_capability = _complete_capability()
        backend.two_stage_projection_adapter = _IncompleteTransientAdapter()
        return backend

    with pytest.raises(ValueError, match="requires an executable two-stage projection adapter"):
        GraphKnowledgeEngine(
            persist_directory=str(tmp_path),
            backend_factory=factory,
            persistence_mode="two_stage",
        )


def test_two_stage_dispatches_to_executable_adapter_without_sync_fallback(tmp_path) -> None:
    class _Adapter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object, str | None]] = []

        def add_node(self, node, *, doc_id=None):
            self.calls.append(("node", node, doc_id))

        def add_edge(self, edge, *, doc_id=None):
            self.calls.append(("edge", edge, doc_id))

        def apply_embedding_job(self, *, entity_kind, entity_id, op, payload_json):
            return None

        def stage1_query(self, *args, **kwargs):
            return []

        def remove_stage1(self, *args, **kwargs):
            return None

        def promote_stage2(self, *args, **kwargs):
            return None

        def remove_stage2_or_invalidate(self, *args, **kwargs):
            return None

        def reconcile_projection(self, *args, **kwargs):
            return None

    adapter = _Adapter()

    def factory(*args, **kwargs):
        backend = build_fake_backend(*args, **kwargs)
        backend.two_stage_projection_capability = _complete_capability()
        backend.two_stage_projection_adapter = adapter
        return backend

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=factory,
        persistence_mode="two_stage",
    )
    assert get_two_stage_projection_adapter(engine.backend) is adapter

    node = build_entity_node(node_id="n1", doc_id="d1", embedding=None)
    engine.write.add_node(node, doc_id="node-doc")
    assert adapter.calls == [("node", node, "node-doc")]


def test_in_memory_two_stage_promotes_pending_node_via_index_job(tmp_path) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        persistence_mode="two_stage",
    )
    node = build_entity_node(node_id="n1", doc_id="d1", embedding=None)

    engine.write.add_node(node)
    pending = engine.backend.node_get(ids=["n1"], include=["documents"])
    assert pending["ids"] == ["n1"]
    assert pending["embeddings"] == [None]
    assert engine.backend.node_query(
        query_embeddings=[[1.0, 0.0, 0.0]], n_results=10
    )["ids"] == [[]]

    assert engine.indexing.reconcile_indexes(max_jobs=10) == 1
    ready = engine.backend.node_get(ids=["n1"], include=["embeddings"])
    assert ready["embeddings"][0] is not None
    assert engine.backend.node_query(
        query_embeddings=[ready["embeddings"][0]], n_results=10
    )["ids"] == [["n1"]]


def test_in_memory_two_stage_uses_existing_index_job_worker(tmp_path) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        persistence_mode="two_stage",
    )
    node = build_entity_node(node_id="n1", doc_id="d1", embedding=None)
    engine.write.add_node(node)
    worker = engine.indexing.make_index_job_worker(
        batch_size=10, max_inflight=10, max_jobs_per_tick=10
    )

    metrics = worker.tick()

    assert metrics.claimed >= 1
    assert metrics.done >= 1
    assert engine.backend.node_get(ids=["n1"])["embeddings"][0] is not None


def test_two_stage_repair_recreates_missing_embedding_job(tmp_path) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        persistence_mode="two_stage",
    )
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))
    engine.indexing.reconcile_indexes(max_jobs=10)
    engine.backend.node_clear_embeddings(ids=["n1"])

    repaired = engine.indexing.repair_missing_two_stage_embedding_jobs()

    assert repaired == 1
    pending = engine.meta_sqlite.list_index_jobs(
        namespace="default", entity_kind="node", entity_id="n1", status="PENDING"
    )
    assert [job.index_kind for job in pending] == ["node_embedding"]
    assert pending[0].payload_json == engine.indexing.canonical_revision_payload(
        entity_kind="node", entity_id="n1"
    )
    assert engine.indexing.reconcile_indexes(max_jobs=10) == 1
    assert engine.backend.node_get(ids=["n1"], include=["embeddings"])["embeddings"][0] is not None


def test_two_stage_repair_recreates_lost_delete_job(tmp_path) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        persistence_mode="two_stage",
    )
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))
    initial = engine.meta_sqlite.claim_index_jobs(
        limit=10, lease_seconds=60, namespace="default"
    )
    for job in initial:
        engine.meta_sqlite.mark_index_job_done(job.job_id, claim_token=job.claim_token)
    engine.reconcile_indexes = lambda **kwargs: 0

    assert engine.lifecycle.tombstone_node("n1") is True
    delete_jobs = engine.meta_sqlite.claim_index_jobs(
        limit=10, lease_seconds=60, namespace="default"
    )
    for job in delete_jobs:
        engine.meta_sqlite.mark_index_job_done(job.job_id, claim_token=job.claim_token)

    assert engine.indexing.repair_missing_two_stage_embedding_jobs() == 1
    pending = engine.meta_sqlite.list_index_jobs(
        namespace="default", entity_kind="node", entity_id="n1", status="PENDING"
    )
    assert [job.op for job in pending] == ["DELETE"]
    assert engine.indexing.reconcile_indexes(max_jobs=10) == 1
    assert engine.backend.node_get(ids=["n1"])["ids"] == []


def test_in_memory_two_stage_rejects_stale_job_after_update(tmp_path) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        persistence_mode="two_stage",
    )
    node = build_entity_node(node_id="n1", doc_id="d1", embedding=None)
    engine.write.add_node(node)
    job = next(
        job
        for job in engine.meta_sqlite.list_index_jobs(namespace="default")
        if job.index_kind == "node_embedding"
    )

    replacement = build_entity_node(node_id="n1", doc_id="d1", embedding=None)
    replacement.summary = "new revision"
    engine._append_event_for_entity(
        namespace="default",
        entity_kind="node",
        entity_id="n1",
        op="ADD",
        payload=replacement.model_dump(field_mode="backend", exclude=["embedding"]),
        required=True,
    )
    engine.backend.node_update(ids=["n1"], documents=["new revision"])
    engine.indexing.apply_index_job(
        job_id=job.job_id,
        entity_kind=job.entity_kind,
        entity_id=job.entity_id,
        index_kind=job.index_kind,
        op=job.op,
        namespace="default",
        payload_json=job.payload_json,
    )

    assert engine.backend.node_get(ids=["n1"])["embeddings"] == [None]


def test_in_memory_two_stage_update_hands_off_old_vector(tmp_path) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        persistence_mode="two_stage",
    )
    first = build_entity_node(node_id="n1", doc_id="d1", embedding=None)
    engine.write.add_node(first)
    engine.indexing.reconcile_indexes(max_jobs=10)
    old_embedding = engine.backend.node_get(ids=["n1"])["embeddings"][0]
    assert old_embedding is not None

    second = build_entity_node(node_id="n1", doc_id="d1", embedding=None)
    second.summary = "new revision"
    engine.write.add_node(second)
    assert engine.backend.node_get(ids=["n1"])["embeddings"] == [None]
    assert engine.backend.node_query(
        query_embeddings=[old_embedding], n_results=10
    )["ids"] == [[]]


def test_in_memory_two_stage_rejects_old_job_after_delete(tmp_path) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        persistence_mode="two_stage",
    )
    node = build_entity_node(node_id="n1", doc_id="d1", embedding=None)
    engine.write.add_node(node)
    job = next(
        job
        for job in engine.meta_sqlite.list_index_jobs(namespace="default")
        if job.index_kind == "node_embedding"
    )

    assert engine.tombstone_node("n1") is True
    engine.indexing.apply_index_job(
        job_id=job.job_id,
        entity_kind=job.entity_kind,
        entity_id=job.entity_id,
        index_kind=job.index_kind,
        op=job.op,
        namespace="default",
        payload_json=job.payload_json,
    )

    assert engine.backend.node_query(
        query_embeddings=[[1.0, 0.0, 0.0]], n_results=10
    )["ids"] == [[]]


def test_in_memory_two_stage_delete_hides_ready_projection(tmp_path) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        persistence_mode="two_stage",
    )
    node = build_entity_node(node_id="n1", doc_id="d1", embedding=None)
    engine.write.add_node(node)
    engine.indexing.reconcile_indexes(max_jobs=10)
    ready = engine.backend.node_get(ids=["n1"])["embeddings"][0]
    assert ready is not None

    assert engine.tombstone_node("n1") is True
    assert engine.backend.node_get(ids=["n1"])["ids"] == []
    assert engine.backend.node_query(
        query_embeddings=[ready], n_results=10
    )["ids"] == [[]]


def test_two_stage_does_not_dispatch_before_canonical_event(tmp_path) -> None:
    class _Adapter:
        def __init__(self) -> None:
            self.calls = 0

        def add_node(self, node, *, doc_id=None):
            self.calls += 1

        def add_edge(self, edge, *, doc_id=None):
            self.calls += 1

        def apply_embedding_job(self, *, entity_kind, entity_id, op, payload_json):
            return None

        def stage1_query(self, *args, **kwargs):
            return []

        def remove_stage1(self, *args, **kwargs):
            return None

        def promote_stage2(self, *args, **kwargs):
            return None

        def remove_stage2_or_invalidate(self, *args, **kwargs):
            return None

        def reconcile_projection(self, *args, **kwargs):
            return None

    adapter = _Adapter()

    def factory(*args, **kwargs):
        backend = build_fake_backend(*args, **kwargs)
        backend.two_stage_projection_capability = _complete_capability()
        backend.two_stage_projection_adapter = adapter
        return backend

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=factory,
        persistence_mode="two_stage",
    )
    node = build_entity_node(node_id="n1", doc_id="d1", embedding=None)

    def fail_event(**kwargs):
        raise RuntimeError("event store unavailable")

    engine._append_event_for_entity = fail_event
    with pytest.raises(RuntimeError, match="event store unavailable"):
        engine.write.add_node(node)
    assert adapter.calls == 0
