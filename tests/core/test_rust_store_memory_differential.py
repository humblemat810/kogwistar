from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from kogwistar._rust_bridge import (
    RustParityError,
    graph_store_implementation_mode,
    meta_store_implementation_mode,
    store_memory_read,
)
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend


pytestmark = [pytest.mark.ci, pytest.mark.core]


@pytest.fixture(scope="module", autouse=True)
def _native_extension() -> Any:
    return pytest.importorskip("kogwistar._rust")


def _backend() -> Any:
    """Build actual Python in-memory backend; no duplicate graph oracle."""
    return build_in_memory_backend(SimpleNamespace())


def _record_snapshot(backend: Any) -> list[dict[str, Any]]:
    """Test-side adapter: Python backend has collections, not graph namespaces.

    `node` and `edge` are deliberately mapped to isolated snapshot namespaces.
    This is a bounded test adapter, not a backend API change; true collection-to-
    namespace routing remains deferred until an allowed public-backend slice.
    """
    records: list[dict[str, Any]] = []
    for namespace, collection in (("alpha", backend.node), ("beta", backend.edge)):
        committed = collection.get(include=["documents", "metadatas", "embeddings"])
        for row_id, document, metadata, embedding in zip(
            committed["ids"],
            committed["documents"],
            committed["metadatas"],
            committed["embeddings"],
            strict=True,
        ):
            records.append(
                {
                    "namespace": namespace,
                    "id": row_id,
                    "document": document,
                    "metadata": metadata,
                    "embedding": embedding,
                }
            )
    return records


def _meta_snapshot(meta: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Snapshot committed InMemoryMetaStore rows, retaining event identity.

    Public event iteration omits `event_id`; adapter reads only its committed test
    state so idempotency can be verified without fabricating a second oracle.
    """
    events = [
        {
            "namespace": namespace,
            "seq": row.seq,
            "event_id": row.event_id,
            "entity_kind": row.entity_kind,
            "entity_id": row.entity_id,
            "op": row.op,
            "payload": json.loads(row.payload_json),
        }
        for namespace, rows in meta._state.entity_events.items()
        for row in rows
    ]
    cursors = [
        {"namespace": namespace, "consumer": consumer, "last_seq": last_seq}
        for (namespace, consumer), last_seq in meta._state.replay_cursors.items()
    ]
    return events, cursors


def _projection_snapshot(meta: Any, namespaces: tuple[str, ...]) -> list[dict[str, Any]]:
    """Read named projections through public meta APIs; preserve Python row shape."""
    return [
        row
        for namespace in namespaces
        for row in meta.list_named_projections(namespace)
    ]


def _snapshot(
    backend: Any, *, projection_namespaces: tuple[str, ...] = ()
) -> dict[str, Any]:
    events, cursors = _meta_snapshot(backend._engine.meta_sqlite)
    return {
        "records": _record_snapshot(backend),
        "events": events,
        "cursors": cursors,
        "projections": _projection_snapshot(
            backend._engine.meta_sqlite, projection_namespaces
        ),
    }


def _state_hash(backend: Any, *, projection_namespaces: tuple[str, ...] = ()) -> str:
    return hashlib.sha256(
        json.dumps(
            _snapshot(backend, projection_namespaces=projection_namespaces),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _native(
    *,
    snapshot: dict[str, Any],
    operation: dict[str, Any],
    python_value: Any,
    store: str,
) -> Any:
    return store_memory_read(
        snapshot=snapshot,
        operation=operation,
        python_value=python_value,
        store=store,
    )


def _python_records(backend: Any, *, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    got = backend.node.get(where=where, include=["documents", "metadatas", "embeddings"])
    return [
        {
            "id": row_id,
            "document": document,
            "metadata": metadata,
            "embedding": embedding,
        }
        for row_id, document, metadata, embedding in zip(
            got["ids"],
            got["documents"],
            got["metadatas"],
            got["embeddings"],
            strict=True,
        )
    ]


def _python_vector_matches(
    backend: Any,
    *,
    embedding: list[float],
    where: dict[str, Any] | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Convert actual Chroma-shaped memory query output, preserving rank order."""
    got = backend.node.query(
        query_embeddings=[embedding], where=where, n_results=limit,
        include=["documents", "metadatas", "embeddings", "distances"],
    )
    return [
        {
            "record": {
                "id": row_id,
                "document": document,
                "metadata": metadata,
                "embedding": vector,
            },
            "distance": distance,
        }
        for row_id, document, metadata, vector, distance in zip(
            got["ids"][0],
            got["documents"][0],
            got["metadatas"][0],
            got["embeddings"][0],
            got["distances"][0],
            strict=True,
        )
    ]


def _seed() -> Any:
    backend = _backend()
    backend.node.add(
        ids=["z", "a", "missing", "mismatch", "zero"],
        documents=["Z", "A", "M", "D", "0"],
        metadatas=[{"team": "red"}] * 5,
        embeddings=[[1.0, 0.0], [0.0, 1.0], None, [1.0, 0.0, 0.0], [0.0, 0.0]],
    )
    backend.edge.add(
        ids=["a"],
        documents=["beta A"],
        metadatas=[{"team": "red"}],
        embeddings=[[1.0, 1.0]],
    )
    meta = backend._engine.meta_sqlite
    first = meta.append_entity_event(
        namespace="alpha",
        event_id="evt-replace",
        entity_kind="node",
        entity_id="a",
        op="UPSERT",
        payload_json='{"replacement":{"id":"a","name":"new"}}',
    )
    assert first == 1
    assert (
        meta.append_entity_event(
            namespace="alpha",
            event_id="evt-replace",
            entity_kind="node",
            entity_id="changed",
            op="UPSERT",
            payload_json='{"replacement":{"id":"changed"}}',
        )
        == first
    )
    assert meta.append_entity_event(
        namespace="alpha",
        event_id="evt-tombstone",
        entity_kind="node",
        entity_id="z",
        op="TOMBSTONE",
        payload_json='{"tombstone":true,"reason":"deleted"}',
    ) == 2
    assert meta.append_entity_event(
        namespace="beta",
        event_id="evt-beta",
        entity_kind="edge",
        entity_id="a",
        op="UPSERT",
        payload_json='{"replacement":{"id":"a"}}',
    ) == 1
    meta.cursor_set(namespace="alpha", consumer="replay", last_seq=1)
    return backend


@pytest.mark.parametrize("mode", ["python", "shadow", "rust"])
def test_graph_store_modes_namespace_filter_stable_order_and_read_only(monkeypatch, mode: str) -> None:
    backend = _seed()
    snapshot = _snapshot(backend)
    before = _state_hash(backend)
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", mode)

    # Consumer-visible Python collection insertion order is retained end-to-end.
    raw_python = _python_records(backend, where={"team": "red"})
    assert [record["id"] for record in raw_python] == ["z", "a", "missing", "mismatch", "zero"]
    result = _native(
        snapshot=snapshot,
        operation={"kind": "graph_records", "namespace": "alpha", "metadata": {"team": "red"}},
        python_value=raw_python,
        store="graph",
    )
    assert result == raw_python
    assert [record["id"] for record in result] == ["z", "a", "missing", "mismatch", "zero"]

    beta = _native(
        snapshot=snapshot,
        operation={"kind": "graph_record", "namespace": "beta", "id": "a"},
        python_value={
            "id": "a",
            "document": "beta A",
            "metadata": {"team": "red"},
            "embedding": [1.0, 1.0],
        },
        store="graph",
    )
    assert beta["document"] == "beta A"
    assert _native(
        snapshot=snapshot,
        operation={"kind": "graph_record", "namespace": "beta", "id": "z"},
        python_value=None,
        store="graph",
    ) is None
    assert _state_hash(backend) == before


@pytest.mark.parametrize("mode", ["python", "shadow", "rust"])
def test_meta_store_modes_replay_cursor_latest_identity_and_payload(monkeypatch, mode: str) -> None:
    backend = _seed()
    meta = backend._engine.meta_sqlite
    snapshot = _snapshot(backend)
    before = _state_hash(backend)
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", mode)

    # Python from_seq is inclusive; native after_seq is exclusive. Explicit mapping:
    # Python from_seq=2 == Rust after_seq=1, not a hidden normalization.
    python_rows = list(meta.iter_entity_events(namespace="alpha", from_seq=2))
    assert [row[0] for row in python_rows] == [2]
    replay = _native(
        snapshot=snapshot,
        operation={"kind": "replay_events", "namespace": "alpha", "after_seq": 1, "limit": 10},
        python_value=[
            {
                "namespace": "alpha",
                "seq": 2,
                "event_id": "evt-tombstone",
                "entity_kind": "node",
                "entity_id": "z",
                "op": "TOMBSTONE",
                "payload": {"tombstone": True, "reason": "deleted"},
            }
        ],
        store="meta",
    )
    assert replay[0]["payload"] == {"tombstone": True, "reason": "deleted"}

    all_events = _native(
        snapshot=snapshot,
        operation={"kind": "replay_events", "namespace": "alpha", "after_seq": 0, "limit": 10},
        python_value=[
            {
                "namespace": "alpha",
                "seq": 1,
                "event_id": "evt-replace",
                "entity_kind": "node",
                "entity_id": "a",
                "op": "UPSERT",
                "payload": {"replacement": {"id": "a", "name": "new"}},
            },
            {
                "namespace": "alpha",
                "seq": 2,
                "event_id": "evt-tombstone",
                "entity_kind": "node",
                "entity_id": "z",
                "op": "TOMBSTONE",
                "payload": {"tombstone": True, "reason": "deleted"},
            },
        ],
        store="meta",
    )
    assert all_events[0]["entity_id"] == "a"  # duplicate identity kept original payload
    assert _native(
        snapshot=snapshot,
        operation={"kind": "replay_cursor", "namespace": "alpha", "consumer": "replay"},
        python_value={"namespace": "alpha", "consumer": "replay", "last_seq": 1},
        store="meta",
    )["last_seq"] == 1
    assert _native(
        snapshot=snapshot,
        operation={"kind": "latest_event_seq", "namespace": "alpha"},
        python_value=2,
        store="meta",
    ) == 2
    assert _state_hash(backend) == before


@pytest.mark.parametrize("mode", ["python", "shadow", "rust"])
def test_cosine_vector_query_matches_python_memory_order_and_candidates(monkeypatch, mode: str) -> None:
    backend = _seed()
    snapshot = _snapshot(backend)
    before = _state_hash(backend)
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", mode)

    python_ties = _python_vector_matches(
        backend, embedding=[1.0, 1.0], where={"team": "red"}, limit=5
    )
    assert [item["record"]["id"] for item in python_ties] == [
        "z", "a", "missing", "mismatch", "zero"
    ]
    result = _native(
        snapshot=snapshot,
        operation={
            "kind": "vector_query",
            "namespace": "alpha",
            "embedding": [1.0, 1.0],
            "limit": 5,
            "metadata": {"team": "red"},
            "metric": "cosine",
        },
        python_value=python_ties,
        store="graph",
    )
    assert [item["record"]["id"] for item in result] == [
        "z", "a", "missing", "mismatch", "zero"
    ]
    for expected, actual in zip(python_ties, result, strict=True):
        assert expected["record"]["id"] == actual["record"]["id"]
        assert actual["distance"] == pytest.approx(expected["distance"], abs=1e-6)
    assert [item["distance"] for item in result[-3:]] == [2.0, 2.0, 2.0]

    # Dimension mismatch and zero query vector remain normal candidates at 2.0.
    python_short = _python_vector_matches(backend, embedding=[1.0], where=None, limit=5)
    short = _native(
        snapshot=snapshot,
        operation={
            "kind": "vector_query",
            "namespace": "alpha",
            "embedding": [1.0],
            "limit": 5,
            "metric": "cosine",
        },
        python_value=python_short,
        store="graph",
    )
    assert [item["record"]["id"] for item in short] == ["z", "a", "missing", "mismatch", "zero"]
    assert [item["distance"] for item in short] == [2.0] * 5

    python_zero = _python_vector_matches(backend, embedding=[0.0, 0.0], where=None, limit=5)
    zero = _native(
        snapshot=snapshot,
        operation={
            "kind": "vector_query",
            "namespace": "alpha",
            "embedding": [0.0, 0.0],
            "limit": 5,
            "metric": "cosine",
        },
        python_value=python_zero,
        store="graph",
    )
    assert [item["distance"] for item in zero] == [2.0] * 5
    assert _state_hash(backend) == before


@pytest.mark.parametrize("mode", ["python", "shadow", "rust"])
def test_pruned_event_snapshot_prefix_and_high_cursor_round_trip(monkeypatch, mode: str) -> None:
    """Test-side snapshot adapter models Python meta's retained event prefix.

    The public in-memory meta API has no prefix-prune verb. Its committed state
    is adjusted only to form the snapshot scenario, then all Python results are
    read from that real meta store before native comparison.
    """
    backend = _seed()
    meta = backend._engine.meta_sqlite
    with meta._lock:
        meta._state.entity_events["alpha"] = meta._state.entity_events["alpha"][1:]
    meta.cursor_set(namespace="alpha", consumer="replay", last_seq=9)
    meta.cursor_set(namespace="empty", consumer="replay", last_seq=12)
    snapshot = _snapshot(backend)
    before = _state_hash(backend)
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", mode)

    retained = snapshot["events"]
    assert [(event["namespace"], event["seq"]) for event in retained] == [
        ("alpha", 2),
        ("beta", 1),
    ]
    alpha_replay = [event for event in retained if event["namespace"] == "alpha"]
    assert list(meta.iter_entity_events(namespace="alpha", from_seq=1)) == [
        (2, "node", "z", "TOMBSTONE", '{"tombstone":true,"reason":"deleted"}')
    ]
    assert _native(
        snapshot=snapshot,
        operation={"kind": "replay_events", "namespace": "alpha", "after_seq": 1, "limit": 10},
        python_value=alpha_replay,
        store="meta",
    ) == alpha_replay
    assert _native(
        snapshot=snapshot,
        operation={"kind": "replay_cursor", "namespace": "alpha", "consumer": "replay"},
        python_value={"namespace": "alpha", "consumer": "replay", "last_seq": 9},
        store="meta",
    ) == {"namespace": "alpha", "consumer": "replay", "last_seq": 9}
    assert _native(
        snapshot=snapshot,
        operation={"kind": "latest_event_seq", "namespace": "alpha"},
        python_value=meta.get_latest_entity_event_seq(namespace="alpha"),
        store="meta",
    ) == 2
    assert _native(
        snapshot=snapshot,
        operation={"kind": "replay_cursor", "namespace": "empty", "consumer": "replay"},
        python_value={"namespace": "empty", "consumer": "replay", "last_seq": 12},
        store="meta",
    ) == {"namespace": "empty", "consumer": "replay", "last_seq": 12}
    assert _native(
        snapshot=snapshot,
        operation={"kind": "latest_event_seq", "namespace": "empty"},
        python_value=meta.get_latest_entity_event_seq(namespace="empty"),
        store="meta",
    ) == 0
    assert _state_hash(backend) == before


@pytest.mark.parametrize("mode", ["python", "shadow", "rust"])
def test_named_projection_read_snapshot_parity_and_read_only(monkeypatch, mode: str) -> None:
    backend = _seed()
    meta = backend._engine.meta_sqlite
    namespaces = ("bridge_governance", "workflow_design", "isolated")
    meta.replace_named_projection(
        "bridge_governance",
        "z-last",
        {"nested": {"members": ["a", {"id": "b"}]}, "enabled": True},
        last_authoritative_seq=21,
        last_materialized_seq=20,
        projection_schema_version=3,
        materialization_status="rebuilding",
    )
    meta.replace_named_projection(
        "bridge_governance",
        "a-first",
        {"kind": "bridge", "metadata": {"region": "test"}},
        last_authoritative_seq=9,
        last_materialized_seq=9,
        projection_schema_version=2,
        materialization_status="ready",
    )
    meta.replace_workflow_design_projection(
        workflow_id="wf-1",
        head={
            "current_version": 4,
            "active_tip_version": 5,
            "last_authoritative_seq": 30,
            "last_materialized_seq": 29,
            "projection_schema_version": 7,
            "snapshot_schema_version": 2,
            "materialization_status": "ready",
        },
        versions=[{"version": 4, "prev_version": 3, "target_seq": 30, "created_at_ms": 44}],
        dropped_ranges=[{"start_seq": 1, "end_seq": 2, "start_version": 0, "end_version": 1}],
    )
    meta.replace_named_projection(
        "isolated",
        "a-first",
        {"kind": "other"},
        last_authoritative_seq=1,
        last_materialized_seq=1,
        projection_schema_version=1,
        materialization_status="ready",
    )
    snapshot = _snapshot(backend, projection_namespaces=namespaces)
    before = _state_hash(backend, projection_namespaces=namespaces)
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", mode)

    bridge_rows = meta.list_named_projections("bridge_governance")
    assert [row["key"] for row in bridge_rows] == ["a-first", "z-last"]
    bridge_get = meta.get_named_projection("bridge_governance", "z-last")
    assert bridge_get is not None
    assert bridge_get["payload"]["nested"]["members"][1] == {"id": "b"}
    assert isinstance(bridge_get["updated_at_ms"], int)
    assert _native(
        snapshot=snapshot,
        operation={
            "kind": "named_projection",
            "namespace": "bridge_governance",
            "key": "z-last",
        },
        python_value=bridge_get,
        store="meta",
    ) == bridge_get
    assert _native(
        snapshot=snapshot,
        operation={
            "kind": "named_projection",
            "namespace": "bridge_governance",
            "key": "missing",
        },
        python_value=meta.get_named_projection("bridge_governance", "missing"),
        store="meta",
    ) is None
    assert _native(
        snapshot=snapshot,
        operation={"kind": "named_projections", "namespace": "bridge_governance"},
        python_value=bridge_rows,
        store="meta",
    ) == bridge_rows
    workflow_row = meta.get_named_projection("workflow_design", "wf-1")
    assert workflow_row is not None
    assert _native(
        snapshot=snapshot,
        operation={"kind": "named_projection", "namespace": "workflow_design", "key": "wf-1"},
        python_value=workflow_row,
        store="meta",
    ) == workflow_row
    # Rust returns named row only; Python retains workflow-specific transformation.
    assert meta.get_workflow_design_projection(workflow_id="wf-1") is not None
    isolated = meta.get_named_projection("isolated", "a-first")
    assert isolated is not None
    assert _native(
        snapshot=snapshot,
        operation={"kind": "named_projection", "namespace": "isolated", "key": "a-first"},
        python_value=isolated,
        store="meta",
    ) == isolated
    assert _state_hash(backend, projection_namespaces=namespaces) == before


def test_store_selectors_shadow_oracle_requirement_and_error_codes(monkeypatch, _native_extension) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "shadow")
    with pytest.raises(RuntimeError, match="caller-computed Python result"):
        store_memory_read(snapshot={}, operation={"kind": "latest_event_seq", "namespace": "ns"})
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "bad")
    with pytest.raises(ValueError, match="KOGWISTAR_IMPL_GRAPH_STORE"):
        graph_store_implementation_mode()
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "bad")
    with pytest.raises(ValueError, match="KOGWISTAR_IMPL_META_STORE"):
        meta_store_implementation_mode()
    with pytest.raises(Exception) as invalid_json:
        _native_extension.store_memory_read_json("not-json")
    assert getattr(invalid_json.value, "code") == "KOGWISTAR_STORE_INVALID_JSON"
    with pytest.raises(Exception) as invalid_payload:
        _native_extension.store_memory_read_json('{"snapshot":{},"operation":{"kind":"nope"}}')
    assert getattr(invalid_payload.value, "code") == "KOGWISTAR_STORE_INVALID_PAYLOAD"


def test_shadow_compares_caller_value_without_oracle_recursion(monkeypatch) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "shadow")
    with pytest.raises(RustParityError, match="graph_store_memory_read"):
        store_memory_read(
            snapshot={},
            operation={"kind": "latest_event_seq", "namespace": "ns"},
            python_value=99,
        )
