import uuid
import threading
import pytest
pytestmark = pytest.mark.ci_full
pytest.importorskip("sqlalchemy")

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.postgres_backend import PgVectorBackend
from tests.conftest import FakeEmbeddingFunction

EMBEDDING_DIM = 3
TEST_EMBEDDING = FakeEmbeddingFunction(dim=EMBEDDING_DIM)


@pytest.fixture(params=["chroma", "pg"], ids=["chroma", "pg"])
def e2e_engine(request, tmp_path, sa_engine, pg_schema) -> GraphKnowledgeEngine:
    if request.param == "chroma":
        persist_dir = tmp_path / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        eng = GraphKnowledgeEngine(
            persist_directory=str(persist_dir), embedding_function=TEST_EMBEDDING
        )
    else:
        pytest.importorskip("pgvector")
        backend = PgVectorBackend(
            engine=sa_engine, embedding_dim=EMBEDDING_DIM, schema=pg_schema
        )
        eng = GraphKnowledgeEngine(backend=backend, embedding_function=TEST_EMBEDDING)

    eng._test_backend_kind = request.param  # type: ignore[attr-defined]
    return eng


def _count_events(eng: GraphKnowledgeEngine, ns: str) -> int:
    return sum(1 for _ in eng.meta_sqlite.iter_entity_events(namespace=ns, from_seq=1))


def _count_jobs(eng: GraphKnowledgeEngine, ns: str) -> int:
    return len(eng.meta_sqlite.list_index_jobs(namespace=ns, limit=1000))


def test_phase3_meta_atomicity_event_and_job_rollback(e2e_engine):
    """
    Inside engine.uow():
      - append entity_events
      - enqueue index_jobs
    then raise => BOTH tables must have 0 rows for that namespace.
    """
    eng = e2e_engine
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


def test_phase3_nested_uow_inner_does_not_commit_if_outer_rolls_back(e2e_engine):
    eng = e2e_engine
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
            # Outer fails after inner block returns
            raise RuntimeError("boom")

    assert _count_events(eng, ns) == 0
    assert _count_jobs(eng, ns) == 0


def test_phase3_parallel_uow_one_rollback_one_commit_isolated(e2e_engine, request):
    """
    Two threads enter eng.uow() concurrently.
      - thread1 writes meta rows, then rolls back
      - thread2 writes meta rows, then commits

    Expected: rollback in thread1 does not cancel committed rows from thread2.
    """
    eng = e2e_engine
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

    arena = {
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
                    # fail inside UoW so this worker rolls back
                    raise RuntimeError("worker rollback")
        except RuntimeError:
            # expected rollback path
            return
        except BaseException as e:  # pragma: no cover
            slot["errors"].append(e)

    t1 = threading.Thread(target=worker, args=[1, False], name="uow-thread-rollback")
    t2 = threading.Thread(target=worker, args=[2, True], name="uow-thread-commit")

    t1.start()
    t2.start()
    if request.node.name.endswith("[chroma]"):
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
    if request.node.name.endswith("[chroma]"):
        return
    assert _count_events(eng, ns2) == 1
    assert _count_jobs(eng, ns2) == 1
