import json
import uuid

import pytest

from graph_knowledge_engine.engine_postgres import PgVectorBackend
from graph_knowledge_engine.engine_postgres_meta import EnginePostgresMetaStore
from graph_knowledge_engine.postgres_backend import PostgresUnitOfWork


@pytest.mark.integration
def test_uow_commit_creates_node_event_vector_and_job(sa_engine, pg_schema):
    be = PgVectorBackend(engine=sa_engine, embedding_dim=2, schema=pg_schema)
    be.ensure_schema()

    meta = EnginePostgresMetaStore(engine=sa_engine, schema=pg_schema)
    meta.ensure_initialized()

    uow = PostgresUnitOfWork(engine=sa_engine)

    nid = f"n_inv_commit_{uuid.uuid4().hex[:8]}"
    be.node_delete(ids=[nid])

    event_id = str(uuid.uuid4())
    payload_json = json.dumps({"id": nid, "hello": "world"}, sort_keys=True, separators=(",", ":"))
    job_id = str(uuid.uuid4())

    with uow.transaction():
        be.node_add(
            ids=[nid],
            documents=["doc"],
            metadatas=[{"lifecycle_status": "active"}],
            embeddings=[[1.0, 0.0]],
        )
        meta.append_entity_event(
            namespace="default",
            event_id=event_id,
            entity_kind="node",
            entity_id=nid,
            op="UPSERT",
            payload_json=payload_json,
        )
        meta.enqueue_index_job(
            job_id=job_id,
            namespace="default",
            entity_kind="node",
            entity_id=nid,
            index_kind="node_docs",
            op="UPSERT",
            payload_json=None,
        )

    # Node exists
    got = be.node_get(ids=[nid], include=["documents", "metadatas", "embeddings"])
    assert got["ids"] == [nid]

    # Vector exists (embedding stored)
    assert got.get("embeddings") and got["embeddings"][0] is not None

    # Event exists
    events = list(meta.iter_entity_events(namespace="default", from_seq=1))
    assert any((row[2] == nid) for row in events), "expected at least one event for node id"

    # Job exists
    jobs = meta.list_index_jobs(namespace="default", entity_kind="node", entity_id=nid, limit=50)
    assert any(j.index_kind == "node_docs" for j in jobs)


@pytest.mark.integration
def test_uow_rollback_leaves_no_node_event_vector_or_job(sa_engine, pg_schema):
    be = PgVectorBackend(engine=sa_engine, embedding_dim=2, schema=pg_schema)
    be.ensure_schema()

    meta = EnginePostgresMetaStore(engine=sa_engine, schema=pg_schema)
    meta.ensure_initialized()

    uow = PostgresUnitOfWork(engine=sa_engine)

    nid = f"n_inv_rollback_{uuid.uuid4().hex[:8]}"
    be.node_delete(ids=[nid])

    event_id = str(uuid.uuid4())
    payload_json = json.dumps({"id": nid}, sort_keys=True, separators=(",", ":"))
    job_id = str(uuid.uuid4())

    start_seq = meta.cursor_get(namespace="default", consumer="__phase4_test__")  # best-effort baseline

    with pytest.raises(RuntimeError):
        with uow.transaction():
            be.node_add(
                ids=[nid],
                documents=["doc"],
                metadatas=[{"lifecycle_status": "active"}],
                embeddings=[[1.0, 0.0]],
            )
            meta.append_entity_event(
                namespace="default",
                event_id=event_id,
                entity_kind="node",
                entity_id=nid,
                op="UPSERT",
                payload_json=payload_json,
            )
            meta.enqueue_index_job(
                job_id=job_id,
                namespace="default",
                entity_kind="node",
                entity_id=nid,
                index_kind="node_docs",
                op="UPSERT",
                payload_json=None,
            )
            raise RuntimeError("boom")

    # Node should not exist
    got = be.node_get(ids=[nid], include=["documents", "embeddings"])
    assert got.get("ids") in ([], None) or len(got.get("ids", [])) == 0

    # Job should not exist
    jobs = meta.list_index_jobs(namespace="default", entity_kind="node", entity_id=nid, limit=50)
    assert len(jobs) == 0

    # Event should not exist (check by entity_id scan)
    events = list(meta.iter_entity_events(namespace="default", from_seq=1))
    assert not any((row[2] == nid) for row in events)
