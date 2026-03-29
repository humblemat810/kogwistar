from __future__ import annotations

import re
import threading
import time
import uuid

import pytest

pytest_plugins = ["tests.core._async_chroma_real"]

pytestmark = pytest.mark.ci_full
pytest.importorskip("sqlalchemy")

import sqlalchemy as sa  # noqa: E402

from kogwistar.engine_core.engine import GraphKnowledgeEngine # noqa: E402
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore# noqa: E402
from tests.conftest import _make_async_engine # noqa: E402
from typing import Any # noqa: E402

def _count_events(eng: GraphKnowledgeEngine, ns: str) -> int:
    return sum(1 for _ in eng.meta_sqlite.iter_entity_events(namespace=ns, from_seq=1))


def _count_jobs(eng: GraphKnowledgeEngine, ns: str) -> int:
    return len(eng.meta_sqlite.list_index_jobs(namespace=ns, limit=1000))


@pytest.fixture(
    params=[
        pytest.param("chroma", id="async-chroma"),
        pytest.param("pg", id="async-pg"),
    ],
    ids=["async-chroma", "async-pg"],
)
def async_e2e_engine(
    request: pytest.FixtureRequest, tmp_path
) -> GraphKnowledgeEngine:
    return _make_async_engine(
        backend_kind=str(request.param),
        request=request,
        tmp_path=tmp_path,
        dim=3,
        graph_kind="knowledge",
    )


def _backend_kind(eng: GraphKnowledgeEngine) -> str:
    backend = eng.backend
    if backend.__class__.__name__.lower().startswith("pgvector"):
        return "pg"
    return "chroma"


def _mk_node(node_id: str, *, doc_id: str):
    from kogwistar.engine_core.models import Grounding, Node, Span

    sp = Span.from_dummy_for_document()
    sp.doc_id = doc_id
    return Node(
        id=node_id,
        label=f"Node {node_id}",
        type="entity",
        summary=f"Summary {node_id}",
        doc_id=doc_id,
        mentions=[Grounding(spans=[sp])],
        metadata={"level_from_root": 0, "entity_type": "kg_entity"},
        embedding=[0.1, 0.1, 0.1],
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )


def _mk_edge(edge_id: str, src: str, tgt: str, doc_id: str):
    from kogwistar.engine_core.models import Grounding, Edge, Span

    sp = Span.from_dummy_for_document()
    sp.doc_id = doc_id
    return Edge(
        id=edge_id,
        label=f"Edge {edge_id}",
        type="relationship",
        summary=f"Summary {edge_id}",
        relation="related_to",
        source_ids=[src],
        target_ids=[tgt],
        source_edge_ids=None,
        target_edge_ids=None,
        doc_id=doc_id,
        mentions=[Grounding(spans=[sp])],
        metadata={"level_from_root": 0, "entity_type": "kg_relation"},
        embedding=[0.1, 0.1, 0.1],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )


def _read_all_events(eng: GraphKnowledgeEngine, namespace: str):
    return list(eng.meta_sqlite.iter_entity_events(namespace=namespace, from_seq=1))


def test_phase2b_event_log_replay_async_backends(async_e2e_engine) -> None:
    eng = async_e2e_engine
    ns = getattr(eng, "namespace", "default")

    eng.add_node(_mk_node("n1", doc_id="d1"))
    eng.add_node(_mk_node("n2", doc_id="d2"))
    eng.add_edge(_mk_edge("e1", "n1", "n2", doc_id="d1"))

    before_nodes = {n.safe_get_id() for n in eng.get_nodes()}
    before_edges = {e.safe_get_id() for e in eng.get_edges()}

    last_seq = eng.replay_namespace(namespace=ns, apply_indexes=True)
    assert last_seq >= 3

    after_nodes = {n.safe_get_id() for n in eng.get_nodes()}
    after_edges = {e.safe_get_id() for e in eng.get_edges()}

    assert before_nodes == after_nodes
    assert before_edges == after_edges


def test_phase2b_event_log_no_duplicate_and_payload_sanity_async(
    async_e2e_engine,
) -> None:
    import json

    eng = async_e2e_engine
    ns = getattr(eng, "namespace", "default")

    eng.add_node(_mk_node("n1", doc_id="d1"))
    eng.add_node(_mk_node("n2", doc_id="d2"))
    eng.add_edge(_mk_edge("e1", "n1", "n2", doc_id="d1"))

    events_before = _read_all_events(eng, ns)
    assert len(events_before) >= 3

    for seq, ek, eid, op, payload_json in events_before:
        if op in ("ADD", "REPLACE"):
            payload = json.loads(payload_json)
            assert isinstance(payload, dict)
            assert "id" in payload
            assert str(payload["id"]) == str(eid)

    last_seq = eng.replay_namespace(namespace=ns, apply_indexes=True)
    assert last_seq >= 3

    events_after = _read_all_events(eng, ns)
    assert len(events_after) == len(events_before), "Replay must not append new events"


def test_phase2b_event_log_tombstone_and_cursor_roundtrip_async(
    async_e2e_engine,
) -> None:
    eng = async_e2e_engine
    ns = getattr(eng, "namespace", "default")

    eng.add_node(_mk_node("n1", doc_id="d1"))
    eng.add_node(_mk_node("n2", doc_id="d2"))
    eng.add_edge(_mk_edge("e1", "n1", "n2", doc_id="d1"))

    tomb = getattr(eng, "tombstone_node", None)
    assert callable(tomb)
    try:
        ok = tomb("n2", reason="phase2b-test")
    except TypeError:
        ok = tomb("n2")
    assert ok is True or ok is None

    events = _read_all_events(eng, ns)
    assert any(
        (ek == "node" and eid == "n2" and op in ("TOMBSTONE", "DELETE"))
        for _, ek, eid, op, _ in events
    )

    before_n = len(events)
    last_seq = eng.replay_namespace(namespace=ns, apply_indexes=False)
    assert last_seq >= 4

    after_events = _read_all_events(eng, ns)
    assert len(after_events) == before_n

    consumer = "phase2b-test-consumer"
    eng.meta_sqlite.cursor_set(namespace=ns, consumer=consumer, last_seq=last_seq)
    got = eng.meta_sqlite.cursor_get(namespace=ns, consumer=consumer)
    assert int(got) == int(last_seq)


def test_phase2_coalescing_sample_usage_async(async_e2e_engine) -> None:
    eng = async_e2e_engine
    assert _backend_kind(eng) in {"pg", "chroma"}

    eng.add_node(_mk_node("n_hot", doc_id="d1"))

    job_ids = [
        eng.enqueue_index_job(
            entity_kind="node", entity_id="n_hot", index_kind="node_docs", op="UPSERT"
        )
        for _ in range(10)
    ]

    assert len(set(job_ids)) == 1
    jid = job_ids[0]

    pending = eng.meta_sqlite.list_index_jobs(
        status="PENDING", entity_kind="node", entity_id="n_hot", index_kind="node_docs"
    )
    assert len(pending) == 1
    assert pending[0].job_id == jid

    eng.reconcile_indexes(max_jobs=10)

    done_ids = {j.job_id for j in eng.meta_sqlite.list_index_jobs(status="DONE")}
    assert jid in done_ids

    got = eng.backend.node_docs_get(where={"node_id": "n_hot"})
    assert len(got.get("ids") or []) >= 1


def test_phase2_enqueue_while_doing_creates_new_pending_async(
    async_e2e_engine,
) -> None:
    eng = async_e2e_engine
    assert _backend_kind(eng) in {"pg", "chroma"}

    eng.add_node(_mk_node("n_busy", doc_id="d1"))

    jid1 = eng.enqueue_index_job(
        entity_kind="node", entity_id="n_busy", index_kind="node_docs", op="UPSERT"
    )

    if hasattr(eng.meta_sqlite, "transaction"):
        with eng.meta_sqlite.transaction() as conn:
            if isinstance(eng.meta_sqlite, EnginePostgresMetaStore):
                schema = eng.meta_sqlite.schema
                table = getattr(eng.meta_sqlite, "index_jobs_table", "index_jobs")
                if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", schema):
                    raise AssertionError(f"invalid schema in test: {schema!r}")
                if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
                    raise AssertionError(f"invalid table in test: {table!r}")
                ij = f"{schema}.{table}"
                conn.execute(
                    sa.text(
                        f"UPDATE {ij} "
                        "SET status='DOING', "
                        "    lease_until=NOW() + (:secs || ' seconds')::interval, "
                        "    updated_at=NOW() "
                        "WHERE job_id=:job_id"
                    ),
                    {"secs": 60, "job_id": jid1},
                )
            else:
                conn.execute(
                    "UPDATE index_jobs SET status='DOING', lease_until=?, updated_at=? WHERE job_id=?",
                    (time.time() + 60.0, time.time(), jid1),
                )

    jid2 = eng.enqueue_index_job(
        entity_kind="node", entity_id="n_busy", index_kind="node_docs", op="UPSERT"
    )
    assert jid2 != jid1

    pending = eng.meta_sqlite.list_index_jobs(
        status="PENDING", entity_kind="node", entity_id="n_busy", index_kind="node_docs"
    )
    doing = eng.meta_sqlite.list_index_jobs(
        status="DOING", entity_kind="node", entity_id="n_busy", index_kind="node_docs"
    )
    assert len(pending) == 1 and pending[0].job_id == jid2
    assert len(doing) == 1 and doing[0].job_id == jid1

    eng.reconcile_indexes(max_jobs=10)

    done_ids = {j.job_id for j in eng.meta_sqlite.list_index_jobs(status="DONE")}
    assert jid2 in done_ids

    got = eng.backend.node_docs_get(where={"node_id": "n_busy"})
    assert len(got.get("ids") or []) >= 1


def test_phase3_meta_atomicity_event_and_job_rollback_async(async_e2e_engine) -> None:
    eng = async_e2e_engine
    ns = f"phase3_atomic_{uuid.uuid4().hex}"

    assert _count_events(eng, ns) == 0
    assert _count_jobs(eng, ns) == 0

    with pytest.raises(RuntimeError):
        with eng.uow():
            eng.meta_sqlite.append_entity_event(
                namespace=ns,
                event_id=f"ev_{uuid.uuid4().hex}",
                entity_kind="node",
                entity_id="n1",
                op="upsert",
                payload_json='{"id":"n1","dummy":true}',
            )
            eng.meta_sqlite.enqueue_index_job(
                namespace=ns,
                job_id=f"job_{uuid.uuid4().hex}",
                entity_kind="node",
                entity_id="n1",
                index_kind="node_index",
                op="upsert",
                payload_json='{"id":"n1"}',
            )
            raise RuntimeError("boom")

    assert _count_events(eng, ns) == 0
    assert _count_jobs(eng, ns) == 0


def test_phase3_nested_uow_inner_does_not_commit_if_outer_rolls_back_async(
    async_e2e_engine,
) -> None:
    eng = async_e2e_engine
    ns = f"phase3_nested_{uuid.uuid4().hex}"

    with pytest.raises(RuntimeError):
        with eng.uow():
            with eng.uow():
                eng.meta_sqlite.append_entity_event(
                    namespace=ns,
                    event_id=f"ev_{uuid.uuid4().hex}",
                    entity_kind="edge",
                    entity_id="e1",
                    op="upsert",
                    payload_json='{"id":"e1","dummy":true}',
                )
                eng.meta_sqlite.enqueue_index_job(
                    namespace=ns,
                    job_id=f"job_{uuid.uuid4().hex}",
                    entity_kind="edge",
                    entity_id="e1",
                    index_kind="edge_index",
                    op="upsert",
                    payload_json='{"id":"e1"}',
                )
            raise RuntimeError("boom")

    assert _count_events(eng, ns) == 0
    assert _count_jobs(eng, ns) == 0


def test_phase3_parallel_uow_one_rollback_one_commit_isolated_async(
    async_e2e_engine, request: pytest.FixtureRequest
) -> None:
    eng = async_e2e_engine
    ns1 = f"phase3_parallel_rb_{uuid.uuid4().hex}"
    ns2 = f"phase3_parallel_ok_{uuid.uuid4().hex}"

    started_1 = threading.Event()
    started_2 = threading.Event()
    release = threading.Event()

    t1_exc: list[BaseException] = []
    t2_exc: list[BaseException] = []

    def _wait_evt(evt: threading.Event, *, timeout: float, msg: str) -> None:
        if not evt.wait(timeout=timeout):
            raise AssertionError(msg)

    arena: dict[int, dict[str, Any]] = {
        1: {
            "namespace": ns1,
            "entity_id": "n_rb",
            "started": started_1,
            "success": False,
            "errors": t1_exc,
        },
        2: {
            "namespace": ns2,
            "entity_id": "n_ok",
            "started": started_2,
            "success": True,
            "errors": t2_exc,
        },
    }

    def worker(worker_id: int, success=True) -> None:
        slot = arena[worker_id]
        other_slot = arena[2 if worker_id == 1 else 1]
        try:
            with eng.uow():
                eng.meta_sqlite.append_entity_event(
                    namespace=slot["namespace"],
                    event_id=f"ev_{uuid.uuid4().hex}",
                    entity_kind="node",
                    entity_id=slot["entity_id"],
                    op="upsert",
                    payload_json=f'{{"id":"{slot["entity_id"]}","dummy":true}}',
                )
                eng.meta_sqlite.enqueue_index_job(
                    namespace=slot["namespace"],
                    job_id=f"job_{uuid.uuid4().hex}",
                    entity_kind="node",
                    entity_id=slot["entity_id"],
                    index_kind="node_index",
                    op="upsert",
                    payload_json=f'{{"id":"{slot["entity_id"]}"}}',
                )

                slot["started"].set()
                _wait_evt(
                    other_slot["started"],
                    timeout=5.0,
                    msg="other thread did not start in time",
                )
                _wait_evt(release, timeout=5.0, msg="release was not set")

                if not success:
                    raise RuntimeError("worker rollback")
        except RuntimeError:
            return
        except BaseException as e:  # pragma: no cover
            slot["errors"].append(e)

    t1 = threading.Thread(target=worker, args=[1, False], name="uow-thread-rollback")
    t2 = threading.Thread(target=worker, args=[2, True], name="uow-thread-commit")

    t1.start()
    t2.start()
    if request.node.name.endswith("[async-chroma]"):
        context_manager = pytest.raises(AssertionError)
    else:
        import contextlib

        context_manager = contextlib.nullcontext()
    with context_manager:
        _wait_evt(arena[1]["started"], timeout=5.0, msg="thread1 never entered uow")
        _wait_evt(arena[2]["started"], timeout=5.0, msg="thread2 never entered uow")
    release.set()

    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    assert not t1.is_alive(), "thread1 did not finish"
    assert not t2.is_alive(), "thread2 did not finish"
    with context_manager:
        assert not t1_exc, f"thread1 unexpected errors: {t1_exc!r}"
        assert not t2_exc, f"thread2 unexpected errors: {t2_exc!r}"

    assert _count_events(eng, ns1) == 0
    assert _count_jobs(eng, ns1) == 0
    if request.node.name.endswith("[async-chroma]"):
        return
    assert _count_events(eng, ns2) == 1
    assert _count_jobs(eng, ns2) == 1
