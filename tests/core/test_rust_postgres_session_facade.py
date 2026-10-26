from __future__ import annotations

from types import SimpleNamespace

import threading

import pytest
from pydantic_extension.model_slicing import use_mode

from kogwistar.engine_core.rust_postgres_session import (
    RustEnginePostgresMetaStore,
    RustPostgresConnectionUnavailable,
    RustPostgresSession,
    _RustPostgresTransactionToken,
)
from kogwistar.engine_core.subsystems.read import ReadSubsystem
from kogwistar.engine_core.subsystems.write import WriteSubsystem
from kogwistar.engine_core.lifecycle import LifecycleSubsystem
from kogwistar.implementation_selection import (
    capture_implementation_selection,
    callback_execution_scope,
    implementation_selection_scope,
    transaction_authority_scope,
)
from tests._helpers.graph_builders import build_entity_node, build_relationship_edge

pytestmark = [pytest.mark.ci, pytest.mark.core]


@pytest.mark.parametrize(
    "operation",
    [
        "add",
        "pure-node-add",
        "pure-edge-add",
        "edge-delete",
        "rollback-delete",
        "replace",
        "metadata-patch",
        "async-metadata-patch",
        "authority-check",
    ],
)
def test_selected_rust_postgres_mutations_fail_closed_without_native_meta_store(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")
    writer = WriteSubsystem.__new__(WriteSubsystem)
    writer._e = SimpleNamespace(meta_sqlite=object())

    with pytest.raises(RuntimeError, match="refusing Python persistence fallback"):
        if operation == "add":
            writer._rust_postgres_add(
                entity_kind="node",
                entity_id="node-1",
                document="{}",
                metadata={},
                embedding=[0.0],
                payload={},
            )
        elif operation == "pure-node-add":
            from kogwistar.engine_core.models import PureChromaNode

            writer._e = SimpleNamespace(
                meta_sqlite=object(),
                kg_graph_type="knowledge",
                namespace="test",
                backend=SimpleNamespace(
                    node_add=lambda **_values: pytest.fail(
                        "selected Rust operation fell back to Python node persistence"
                    )
                ),
            )
            writer.add_pure_node(
                PureChromaNode(
                    id="pure-node-1",
                    label="pure node",
                    type="entity",
                    summary="pure node summary",
                    embedding=[0.0],
                )
            )
        elif operation == "pure-edge-add":
            from kogwistar.engine_core.models import PureChromaEdge

            writer._e = SimpleNamespace(
                meta_sqlite=object(),
                kg_graph_type="knowledge",
                namespace="test",
                backend=SimpleNamespace(
                    edge_add=lambda **_values: pytest.fail(
                        "selected Rust operation fell back to Python edge persistence"
                    )
                ),
                adjudicate=SimpleNamespace(
                    split_endpoints=lambda source_ids, target_ids: (
                        source_ids,
                        [],
                        target_ids,
                        [],
                    )
                ),
                persist=SimpleNamespace(assert_endpoints_exist=lambda _edge: None),
                pre_add_pure_edge_hooks=[],
            )
            writer.add_pure_edge(
                PureChromaEdge(
                    id="pure-edge-1",
                    label="pure edge",
                    type="relationship",
                    summary="pure edge summary",
                    relation="related_to",
                    source_ids=["source"],
                    target_ids=["target"],
                    source_edge_ids=None,
                    target_edge_ids=None,
                    embedding=[0.0],
                )
            )
        elif operation == "edge-delete":
            writer._rust_postgres_delete_edges(["edge-1"])
        elif operation == "metadata-patch":
            writer.patch_base_projection_metadata(
                entity_kind="node",
                entity_id="node-1",
                metadata_patch={"reference_fingerprint": "abc"},
            )
        elif operation == "async-metadata-patch":
            pending = writer._async_patch_base_projection_metadata(
                entity_kind="node",
                entity_id="node-1",
                metadata_patch={"reference_fingerprint": "abc"},
            )
            pending.send(None)
        elif operation == "authority-check":
            writer.uses_rust_postgres_authority()
        elif operation == "rollback-delete":
            writer.rust_postgres_delete_existing(
                entity_kind="node", entity_ids=["node-1"]
            )
        else:
            writer.rust_postgres_replace_existing(
                entity_kind="node",
                entity_id="node-1",
                document="{}",
                metadata_patch={},
                payload={},
            )


def test_python_postgres_authority_keeps_legacy_native_probe_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", raising=False)
    writer = WriteSubsystem.__new__(WriteSubsystem)
    updates: list[dict[str, object]] = []
    writer._e = SimpleNamespace(
        meta_sqlite=object(),
        namespace="default",
        backend=SimpleNamespace(
            node_update=lambda **kwargs: updates.append(kwargs),
        ),
    )

    assert writer._rust_postgres_add(
        entity_kind="node",
        entity_id="node-1",
        document="{}",
        metadata={},
        embedding=[0.0],
        payload={},
    ) is False
    writer.patch_base_projection_metadata(
        entity_kind="node",
        entity_id="node-1",
        metadata_patch={"reference_fingerprint": "abc"},
    )
    assert updates == [
        {"ids": ["node-1"], "metadatas": [{"reference_fingerprint": "abc"}]}
    ]


@pytest.mark.parametrize(
    "operation",
    [
        "tombstone_node",
        "redirect_node",
        "tombstone_edge",
        "redirect_edge",
        "replay",
    ],
)
def test_selected_rust_lifecycle_fails_closed_without_native_meta_store(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")
    engine = SimpleNamespace(
        meta_sqlite=object(),
        _implementation_selection_snapshot=capture_implementation_selection(),
    )
    lifecycle = LifecycleSubsystem(engine)

    with pytest.raises(RuntimeError, match="refusing Python lifecycle fallback"):
        if operation == "tombstone_node":
            lifecycle.tombstone_node("node-1")
        elif operation == "redirect_node":
            lifecycle.redirect_node("node-1", "node-2")
        elif operation == "tombstone_edge":
            lifecycle.tombstone_edge("edge-1")
        elif operation == "redirect_edge":
            lifecycle.redirect_edge("edge-1", "edge-2")
        else:
            lifecycle.apply_replayed_lifecycle_patch(
                entity_kind="node",
                entity_id="node-1",
                lifecycle_patch={"lifecycle_status": "tombstoned"},
            )


def test_postgres_session_nested_commit_uses_one_native_transaction(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    def fake_store_postgres(*, operation, transaction_id, **kwargs):
        del kwargs
        calls.append((operation["kind"], transaction_id))
        return None

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    session = RustPostgresSession(dsn="postgresql://example/db", schema="test")

    with session.transaction() as outer:
        session.call("raw_append")
        with session.transaction() as inner:
            session.require_token(inner)
            assert inner.value == outer.value
            session.call("replace_named_projection")

    transaction_ids = {value for _, value in calls if value is not None}
    assert len(transaction_ids) == 1
    assert [kind for kind, _ in calls] == [
        "begin_transaction",
        "raw_append",
        "replace_named_projection",
        "commit_transaction",
    ]


def test_postgres_meta_facade_session_context_is_nested_and_reopenable(monkeypatch):
    calls: list[str] = []

    def fake_store_postgres(*, operation, **kwargs):
        del kwargs
        calls.append(operation["kind"])
        return None

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    store = RustEnginePostgresMetaStore(
        dsn="postgresql://example/db", schema="test"
    )

    with store.session() as outer:
        assert outer is store
        with store.session() as inner:
            assert inner is store
        with pytest.raises(
            RustPostgresConnectionUnavailable,
            match="active execution session",
        ):
            store.close()
        assert not store.native_session._closed
    assert store.native_session._closed

    with store.session():
        assert not store.native_session._closed
    assert store.native_session._closed
    assert calls == ["ensure_schema", "close_store", "ensure_schema", "close_store"]


def test_postgres_meta_facade_keeps_store_open_for_overlapping_threads(monkeypatch):
    calls: list[str] = []
    calls_lock = threading.Lock()

    def fake_store_postgres(*, operation, **kwargs):
        del kwargs
        with calls_lock:
            calls.append(operation["kind"])
        return None

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    store = RustEnginePostgresMetaStore(
        dsn="postgresql://example/db", schema="test"
    )
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()

    def hold_session(entered: threading.Event, release: threading.Event) -> None:
        with store.session():
            entered.set()
            assert release.wait(timeout=5)

    first = threading.Thread(
        target=hold_session, args=(first_entered, release_first), daemon=True
    )
    second = threading.Thread(
        target=hold_session, args=(second_entered, release_second), daemon=True
    )
    first.start()
    assert first_entered.wait(timeout=5)
    second.start()
    assert second_entered.wait(timeout=5)

    release_first.set()
    first.join(timeout=5)
    assert not first.is_alive()
    assert "close_store" not in calls

    release_second.set()
    second.join(timeout=5)
    assert not second.is_alive()
    assert calls == ["ensure_schema", "close_store"]


def test_postgres_graph_mutation_batch_uses_one_native_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    store = RustEnginePostgresMetaStore(
        dsn="postgresql://example/db", schema="test"
    )

    def fake_call(kind: str, **values: object) -> object:
        calls.append((kind, values))
        return [
            {"event_seq": 1, "entity_id": "node-1"},
            {"event_seq": 2, "entity_id": "edge-1"},
            "job-1",
        ]

    monkeypatch.setattr(store, "_call", fake_call)
    batch = {
        "contract_version": 1,
        "atomic": True,
        "mutations": [
            {
                "scope": {"namespace": "ns"},
                "table": "gke_nodes",
                "entity_kind": "node",
                "event_id": "node-event",
                "op": "ADD",
                "payload": {"id": "node-1", "entity_type": "node"},
                "record": {"id": "node-1", "document": "node text"},
                "embedding_dim": 0,
            },
            {
                "scope": {"namespace": "ns"},
                "table": "gke_edges",
                "entity_kind": "edge",
                "event_id": "edge-event",
                "op": "ADD",
                "payload": {"id": "edge-1", "entity_type": "edge"},
                "record": {"id": "edge-1", "document": "edge text"},
                "embedding_dim": 0,
            },
        ],
        "index_jobs": [
            {
                "job_id": "job-1",
                "namespace": "ns",
                "entity_kind": "node",
                "entity_id": "node-1",
                "index_kind": "node_docs",
                "op": "UPSERT",
            }
        ],
    }

    result = store.apply_graph_mutations_atomic(batch)

    assert result == [
        {"event_seq": 1, "entity_id": "node-1"},
        {"event_seq": 2, "entity_id": "edge-1"},
    ]
    assert len(calls) == 1
    kind, values = calls[0]
    assert kind == "batch"
    operations = values["operations"]
    assert isinstance(operations, list)
    assert [operation["kind"] for operation in operations] == [
        "graph_mutation",
        "graph_mutation",
        "enqueue_index_job",
    ]
    assert operations[0]["record"]["id"] == "node-1"
    assert operations[1]["record"]["id"] == "edge-1"


@pytest.mark.parametrize(
    "mutation_update",
    [
        {"scope": {"namespace": ""}},
        {"table": "gke_edges"},
        {"payload": {"id": "different-id"}},
        {"record": {"id": "node-1", "embedding": [True]}, "embedding_dim": 1},
        {"record": {"id": "node-1", "embedding": ["1"]}, "embedding_dim": 1},
        {"embedding_dim": False},
    ],
)
def test_postgres_graph_mutation_batch_rejects_invalid_contract_before_native_call(
    monkeypatch: pytest.MonkeyPatch,
    mutation_update: dict[str, object],
) -> None:
    store = RustEnginePostgresMetaStore(
        dsn="postgresql://example/db", schema="test"
    )
    calls: list[str] = []
    monkeypatch.setattr(
        store,
        "_call",
        lambda kind, **values: calls.append(kind),
    )
    mutation: dict[str, object] = {
        "scope": {"namespace": "ns"},
        "table": "gke_nodes",
        "entity_kind": "node",
        "event_id": "node-event",
        "op": "ADD",
        "payload": {"id": "node-1"},
        "record": {"id": "node-1"},
        "embedding_dim": 0,
    }
    mutation.update(mutation_update)
    batch = {"contract_version": 1, "atomic": True, "mutations": [mutation]}

    with pytest.raises(ValueError):
        store.apply_graph_mutations_atomic(batch)

    assert calls == []


def test_native_session_rejects_rust_transaction_inside_python_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "python")
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "python")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "python")
    snapshot = capture_implementation_selection()
    calls: list[str] = []
    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        lambda *, operation, **_kwargs: calls.append(operation["kind"]),
    )
    session = RustPostgresSession(dsn="postgresql://example/db", schema="test")

    with implementation_selection_scope(snapshot):
        with pytest.raises(RuntimeError, match="does not match frozen implementation selection"):
            with session.transaction():
                pytest.fail("a Python-selected run must not open a Rust transaction")

    assert calls == []


def test_native_session_rejects_direct_call_inside_python_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "python")
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "python")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "python")
    snapshot = capture_implementation_selection()
    calls: list[str] = []
    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        lambda *, operation, **_kwargs: calls.append(operation["kind"]),
    )
    session = RustPostgresSession(dsn="postgresql://example/db", schema="test")

    with implementation_selection_scope(snapshot):
        with pytest.raises(
            RuntimeError, match="does not match frozen implementation selection"
        ):
            session.call("graph_projection_records")

    assert calls == []


def test_native_session_rejects_nested_python_transaction_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        lambda *, operation, **_kwargs: calls.append(operation["kind"]),
    )
    session = RustPostgresSession(dsn="postgresql://example/db", schema="test")

    with transaction_authority_scope("python"):
        with pytest.raises(RuntimeError, match="nested transaction uses a different"):
            with session.transaction():
                pytest.fail("a Rust transaction must not join Python ownership")

    assert calls == []


def test_native_session_rejects_direct_call_inside_python_transaction_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        lambda *, operation, **_kwargs: calls.append(operation["kind"]),
    )
    session = RustPostgresSession(dsn="postgresql://example/db", schema="test")

    with transaction_authority_scope("python"):
        with pytest.raises(RuntimeError, match="nested transaction uses a different"):
            session.call("graph_projection_records")

    assert calls == []


def test_native_session_rejects_direct_call_from_worker_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        lambda *, operation, **_kwargs: calls.append(operation["kind"]),
    )
    session = RustPostgresSession(dsn="postgresql://example/db", schema="test")

    with callback_execution_scope():
        with pytest.raises(
            RuntimeError,
            match="transaction authority cannot be opened from a worker callback",
        ):
            session.call("replace_named_projection")

    assert calls == []


def test_postgres_session_exposes_typed_graph_authority_capability(monkeypatch):
    calls: list[str] = []

    def fake_store_postgres(*, operation, **kwargs):
        del kwargs
        calls.append(operation["kind"])
        return {
            "level": "atomic_batch_with_index_jobs",
            "single_mutation": True,
            "atomic_index_jobs": True,
        }

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    session = RustPostgresSession(dsn="postgresql://example/db", schema="test")

    assert session.graph_authority_capability()["level"] == "atomic_batch_with_index_jobs"
    assert calls == ["graph_authority_capability"]


def test_postgres_session_rolls_back_and_rejects_raw_or_stale_writer(monkeypatch):
    calls: list[str] = []

    def fake_store_postgres(*, operation, **kwargs):
        del kwargs
        calls.append(operation["kind"])
        return None

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    session = RustPostgresSession(dsn="postgresql://example/db", schema="test")
    with pytest.raises(RustPostgresConnectionUnavailable):
        session.connect()

    stale = None
    with pytest.raises(RuntimeError, match="stop"):
        with session.transaction() as token:
            stale = token
            raise RuntimeError("stop")
    assert calls == ["begin_transaction", "rollback_transaction"]
    assert stale is not None
    with pytest.raises(RustPostgresConnectionUnavailable, match="stale"):
        session.require_token(stale)


def test_postgres_transaction_token_cannot_cross_session_owner():
    first = RustPostgresSession(dsn="postgresql://example/db", schema="test")
    second = RustPostgresSession(dsn="postgresql://example/db", schema="test")
    token = _RustPostgresTransactionToken("same-transaction", first.context_id)
    active = second._transaction_id.set(token.value)
    try:
        with pytest.raises(RustPostgresConnectionUnavailable, match="stale"):
            second.require_token(token)
    finally:
        second._transaction_id.reset(active)


def test_postgres_session_commit_error_clears_local_owner_without_retry(monkeypatch):
    calls: list[tuple[str, str | None]] = []
    fail_commit = True

    def fake_store_postgres(*, operation, transaction_id, **kwargs):
        nonlocal fail_commit
        del kwargs
        calls.append((operation["kind"], transaction_id))
        if operation["kind"] == "commit_transaction" and fail_commit:
            fail_commit = False
            raise RuntimeError("commit acknowledgement lost")
        return None

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    session = RustPostgresSession(dsn="postgresql://example/db", schema="test")
    stale = None
    with pytest.raises(RuntimeError, match="acknowledgement lost"):
        with session.transaction() as token:
            stale = token
            session.call("raw_append")

    assert not session.transaction_active
    assert stale is not None
    with pytest.raises(RustPostgresConnectionUnavailable, match="stale"):
        session.require_token(stale)
    assert [kind for kind, _ in calls] == [
        "begin_transaction",
        "raw_append",
        "commit_transaction",
    ]
    assert len({transaction_id for _, transaction_id in calls}) == 1

    with session.transaction():
        pass
    next_transaction_id = calls[-1][1]
    assert next_transaction_id is not None
    assert next_transaction_id != calls[0][1]


def test_postgres_session_owns_stable_context_and_closes_worker(monkeypatch):
    calls: list[tuple[str, str, str | None]] = []

    def fake_store_postgres(*, operation, context_id, transaction_id, **kwargs):
        del kwargs
        calls.append((operation["kind"], context_id, transaction_id))
        return None

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    session = RustPostgresSession(dsn="postgresql://example/db", schema="test")
    context_id = session.context_id
    session.call("current_global_seq")
    with pytest.raises(RustPostgresConnectionUnavailable, match="active transaction"):
        with session.transaction():
            session.close()

    session.close()
    session.close()
    assert [kind for kind, _, _ in calls] == [
        "current_global_seq",
        "begin_transaction",
        "rollback_transaction",
        "close_store",
    ]
    assert {item_context for _, item_context, _ in calls} == {context_id}
    with pytest.raises(RustPostgresConnectionUnavailable, match="closed"):
        session.call("current_global_seq")


def test_postgres_meta_facade_routes_nested_sequence_and_queue_calls(monkeypatch):
    calls: list[tuple[str, str | None]] = []
    results = {
        "next_global_seq": 1,
        "next_user_seq": 1,
        "enqueue_index_job": "job",
    }

    def fake_store_postgres(*, operation, transaction_id, **kwargs):
        del kwargs
        calls.append((operation["kind"], transaction_id))
        return results.get(operation["kind"])

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    store = RustEnginePostgresMetaStore(
        dsn="postgresql://example/db", schema="test"
    )
    with store.transaction() as token:
        assert store.next_global_seq_conn(token) == 1
        assert store.next_scoped_seq("scope") == 1
        assert store.enqueue_index_job(
            job_id="job",
            entity_kind="node",
            entity_id="n",
            index_kind="node_docs",
            op="UPSERT",
        ) == "job"
    assert [kind for kind, _ in calls] == [
        "begin_transaction",
        "next_global_seq",
        "next_user_seq",
        "enqueue_index_job",
        "commit_transaction",
    ]
    assert len({value for _, value in calls if value is not None}) == 1


def test_postgres_meta_facade_reuses_the_shared_atomic_batch_seam(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    def fake_store_postgres(*, operation, transaction_id, **kwargs):
        del kwargs
        calls.append((operation["kind"], transaction_id))
        return [1, None]

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    store = RustEnginePostgresMetaStore(
        dsn="postgresql://example/db", schema="test"
    )
    assert store.execute_batch(
        [{"kind": "next_global_seq"}, {"kind": "current_global_seq"}]
    ) == [1, None]
    assert calls == [("batch", None)]


def test_postgres_meta_facade_validates_native_graph_read_shapes(monkeypatch):
    valid_record = {
        "id": "n1",
        "document": '{"id":"n1"}',
        "metadata": {"namespace": "alpha"},
        "embedding": [1.0, 0.0],
    }
    results = {
        "graph_projection_records": [valid_record],
        "graph_projection_vector_query": [
            {"record": valid_record, "distance": 0.0}
        ],
    }

    def fake_store_postgres(*, operation, **kwargs):
        del kwargs
        return results[operation["kind"]]

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    store = RustEnginePostgresMetaStore(
        dsn="postgresql://example/db", schema="test"
    )
    assert store.graph_projection_records(table="gke_nodes") == [valid_record]
    assert store.graph_projection_vector_query(table="gke_nodes")[0][
        "distance"
    ] == 0.0

    results["graph_projection_records"] = {"id": "wrong"}
    with pytest.raises(RuntimeError, match="invalid records"):
        store.graph_projection_records(table="gke_nodes")

    results["graph_projection_records"] = [{**valid_record, "embedding": [float("nan")]}]
    with pytest.raises(RuntimeError, match="invalid records"):
        store.graph_projection_records(table="gke_nodes")

    results["graph_projection_records"] = [
        {**valid_record, "embedding": [10**10000]}
    ]
    with pytest.raises(RuntimeError, match="invalid records"):
        store.graph_projection_records(table="gke_nodes")

    results["graph_projection_vector_query"] = [
        {"record": valid_record, "distance": float("inf")}
    ]
    with pytest.raises(RuntimeError, match="invalid matches"):
        store.graph_projection_vector_query(table="gke_nodes")


def test_postgres_meta_facade_forwards_optional_graph_projection_cursor(monkeypatch):
    operations: list[dict] = []

    def fake_store_postgres(*, operation, **kwargs):
        del kwargs
        operations.append(operation)
        return []

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    store = RustEnginePostgresMetaStore(
        dsn="postgresql://example/db", schema="test"
    )

    assert store.graph_projection_records(
        table="gke_nodes", namespace="alpha", limit=2, after_id="node-1"
    ) == []
    assert store.graph_projection_records(
        table="gke_nodes", namespace="alpha", limit=2
    ) == []

    assert operations == [
        {
            "kind": "graph_projection_records",
            "table": "gke_nodes",
            "namespace": "alpha",
            "limit": 2,
            "after_id": "node-1",
        },
        {
            "kind": "graph_projection_records",
            "table": "gke_nodes",
            "namespace": "alpha",
            "limit": 2,
        },
    ]


def test_read_subsystem_routes_simple_base_reads_to_native_projection(monkeypatch):
    store = RustEnginePostgresMetaStore(
        dsn="postgresql://example/db", schema="test"
    )
    calls: list[dict] = []

    native_node = build_entity_node(
        node_id="n1",
        doc_id="doc-1",
        summary="native",
        metadata={"namespace": "alpha", "team": "red"},
        embedding=[1.0, 0.0, 0.0],
    )
    with use_mode("backend"):
        native_document = native_node.model_dump_json()

    def native_records(**values):
        calls.append(values)
        return [
            {
                "id": "n1",
                "document": native_document,
                "metadata": {"namespace": "alpha", "team": "red"},
                "embedding": [1.0, 0.0, 0.0],
            }
        ]

    monkeypatch.setattr(store, "graph_projection_records", native_records)

    class Backend:
        nodes = SimpleNamespace(name="gke_nodes")
        edges = SimpleNamespace(name="gke_edges")
        documents = SimpleNamespace(name="gke_documents")
        embedding_dim = 3
        distance = "cosine"

        def node_get(self, **_values):
            return {"ids": ["python"]}

        def edge_get(self, **_values):
            return {"ids": ["python-edge"]}

        def node_query(self, **_values):
            return {"ids": [["python-query"]]}

        def edge_query(self, **_values):
            return {"ids": [["python-edge-query"]]}

        def document_get(self, **_values):
            raise AssertionError("exact document read used Python backend")

    engine = SimpleNamespace(
        meta_sqlite=store,
        backend=Backend(),
        namespace="alpha",
        kg_graph_type="knowledge",
        _resolve_redirect_chain=lambda *, initial_items, **_values: initial_items,
        _filter_items_by_resolve_mode=lambda items, _mode: items,
    )
    reader = ReadSubsystem(engine)
    result = reader._node_get_raw(
        ids=["n1"], where={"team": "red"}, limit=5
    )
    assert result == {
        "ids": ["n1"],
        "documents": [native_document],
        "metadatas": [{"namespace": "alpha", "team": "red"}],
        "embeddings": [[1.0, 0.0, 0.0]],
    }
    assert calls == [
        {
            "namespace": "alpha",
            "workspace_id": None,
            "graph_space": None,
            "table": "gke_nodes",
            "ids": ["n1"],
            "metadata": {"team": "red"},
            "limit": 5,
        }
    ]

    no_include = reader._node_get_raw(ids=["n1"], include=[], limit=5)
    assert "documents" in no_include and "metadatas" in no_include
    assert "embeddings" not in no_include

    fallback = reader._node_get_raw(
        where={"team": {"$in": ["red"]}}, include=[], limit=5
    )
    assert fallback == {"ids": ["python"]}
    assert reader._node_get_raw(where={"team": "red"}, include=[], limit=5) == {
        "ids": ["python"]
    }

    def reject_python_node_read(**_values):
        raise AssertionError("Rust-selected singleton read fell back to Python")

    monkeypatch.setattr(engine.backend, "node_get", reject_python_node_read)
    hydrated = reader.get_nodes(ids=["n1"], where={"team": "red"}, limit=5)
    assert len(hydrated) == 1
    assert hydrated[0].summary == "native"
    assert hydrated[0].embedding == [1.0, 0.0, 0.0]

    native_query_calls: list[dict] = []

    def native_vector_query(**values):
        native_query_calls.append(values)
        return [
            {
                "record": {
                    "id": "n1",
                    "document": native_document,
                    "metadata": {"namespace": "alpha", "team": "red"},
                    "embedding": [1.0, 0.0, 0.0],
                },
                "distance": 0.125,
            }
        ]

    monkeypatch.setattr(store, "graph_projection_vector_query", native_vector_query)

    def reject_python_vector_query(**_values):
        raise AssertionError("Rust-selected vector query fell back to Python")

    monkeypatch.setattr(engine.backend, "node_query", reject_python_vector_query)
    native_hits = reader.query_nodes(
        query_embeddings=[1.0, 0.0, 0.0], where={"team": "red"}, n_results=1
    )
    assert len(native_hits) == 1
    assert native_hits[0][0].summary == "native"
    assert native_query_calls == [
        {
            "namespace": "alpha",
            "workspace_id": None,
            "graph_space": None,
            "table": "gke_nodes",
            "embedding": [1.0, 0.0, 0.0],
            "embedding_dim": 3,
            "metadata": {"team": "red"},
            "metric": "cosine",
            "limit": 1,
        }
    ]

    scored_hits = reader.search_nodes_as_of_scored(
        query_embeddings=[1.0, 0.0, 0.0],
        as_of_ts="2025-01-01T00:00:00Z",
        where={"team": "red"},
        n_results=1,
    )
    assert len(scored_hits) == 1
    assert scored_hits[0].node.summary == "native"
    assert scored_hits[0].raw_distance == 0.125
    assert scored_hits[0].similarity == 0.875

    legacy_query_calls: list[dict] = []

    def legacy_vector_query(**values):
        legacy_query_calls.append(values)
        return {
            "ids": [["n1"]],
            "documents": [[native_document]],
            "metadatas": [[{"namespace": "alpha", "team": "red"}]],
            "embeddings": [[[1.0, 0.0, 0.0]]],
        }

    monkeypatch.setattr(engine.backend, "node_query", legacy_vector_query)
    legacy_hits = reader.query_nodes(
        query_embeddings=[1.0, 0.0, 0.0],
        where={"team": {"$in": ["red"]}},
        n_results=1,
    )
    assert legacy_hits[0][0].summary == "native"
    assert len(legacy_query_calls) == 1
    scored_legacy_hits = reader.search_nodes_as_of_scored(
        query_embeddings=[1.0, 0.0, 0.0],
        as_of_ts="2025-01-01T00:00:00Z",
        where={"team": {"$in": ["red"]}},
        n_results=1,
    )
    assert len(scored_legacy_hits) == 1
    assert scored_legacy_hits[0].node.summary == "native"
    assert scored_legacy_hits[0].raw_distance is None
    assert len(legacy_query_calls) == 2

    native_query_count = len(native_query_calls)
    engine.backend.supports_historical_tombstone_query = True
    tombstone_aware_hits = reader.search_nodes_as_of_scored(
        query_embeddings=[1.0, 0.0, 0.0],
        as_of_ts="2025-01-01T00:00:00Z",
        where={"team": "red"},
        n_results=1,
    )
    assert len(tombstone_aware_hits) == 1
    assert legacy_query_calls[-1]["include_tombstoned"] is True
    assert len(native_query_calls) == native_query_count

    native_edge = build_relationship_edge(
        edge_id="e1",
        src="n1",
        tgt="n2",
        doc_id="doc-1",
        summary="native edge",
        metadata={"namespace": "alpha", "team": "red"},
        source_edge_ids=["source-edge"],
        target_edge_ids=["target-edge"],
        embedding=[0.0, 1.0, 0.0],
    )
    with use_mode("backend"):
        native_edge_document = native_edge.model_dump_json()
    monkeypatch.setattr(
        store,
        "graph_projection_records",
        lambda **_values: [
            {
                "id": "e1",
                "document": native_edge_document,
                "metadata": {"namespace": "alpha", "team": "red"},
                "embedding": [0.0, 1.0, 0.0],
            }
        ],
    )

    def reject_python_edge_read(**_values):
        raise AssertionError("Rust-selected edge read fell back to Python")

    monkeypatch.setattr(engine.backend, "edge_get", reject_python_edge_read)
    hydrated_edges = reader.get_edges(ids=["e1"], where={"team": "red"}, limit=5)
    assert len(hydrated_edges) == 1
    assert hydrated_edges[0].summary == "native edge"
    assert hydrated_edges[0].source_edge_ids == ["source-edge"]
    assert hydrated_edges[0].target_edge_ids == ["target-edge"]
    assert hydrated_edges[0].embedding == [0.0, 1.0, 0.0]

    monkeypatch.setattr(
        store,
        "graph_projection_vector_query",
        lambda **_values: [
            {
                "record": {
                    "id": "e1",
                    "document": native_edge_document,
                    "metadata": {"namespace": "alpha", "team": "red"},
                    "embedding": [0.0, 1.0, 0.0],
                },
                "distance": 0.25,
            }
        ],
    )

    def reject_python_edge_vector_query(**_values):
        raise AssertionError("Rust-selected edge vector query fell back to Python")

    monkeypatch.setattr(engine.backend, "edge_query", reject_python_edge_vector_query)
    native_edge_hits = reader.query_edges(
        query_embeddings=[0.0, 1.0, 0.0], where={"team": "red"}, n_results=1
    )
    assert len(native_edge_hits) == 1
    assert native_edge_hits[0][0].summary == "native edge"
    assert native_edge_hits[0][0].source_edge_ids == ["source-edge"]

    native_records_result = {
        "id": "doc-1",
        "document": "content",
        "metadata": {"type": "text", "processed": True},
        "embedding": None,
    }
    monkeypatch.setattr(store, "graph_projection_records", lambda **_values: [native_records_result])
    document = reader.get_document("doc-1")
    assert document.id == "doc-1" and document.content == "content"

    monkeypatch.setattr(
        store,
        "graph_projection_vector_query",
        lambda **_values: [
            {"record": native_records_result, "distance": 0.25}
        ],
    )
    native_query = reader._rust_postgres_projection_query(
        entity_kind="node",
        query_embeddings=[[1.0, 0.0, 0.0]],
        where={"team": "red"},
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    assert native_query == {
        "ids": [["doc-1"]],
        "documents": [["content"]],
        "metadatas": [[{"type": "text", "processed": True}]],
        "distances": [[0.25]],
    }
    assert reader._rust_postgres_projection_query(
        entity_kind="node",
        query_embeddings=[[1.0, 0.0, 0.0]],
        where={"team": {"$in": ["red"]}},
        n_results=3,
        include=[],
    ) is None
