from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kogwistar._rust_bridge import RustParityError, store_postgres, store_sqlite
from kogwistar.document_types import Chunk
from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.models import Document
from kogwistar.engine_core.rust_meta_sqlite import RustEngineSQLite
from kogwistar.engine_core.rust_postgres_session import RustEnginePostgresMetaStore
from kogwistar.runtime import (
    CheckpointedProjectionStore,
    ProjectionCheckpoint,
    ProjectionLoadResult,
    refresh_checkpointed_named_projection,
)
from kogwistar.runtime.models import (
    WorkflowCancelledNode,
    WorkflowCheckpointNode,
    WorkflowCompletedNode,
    WorkflowFailedNode,
)

pytestmark = [pytest.mark.ci, pytest.mark.core]


@pytest.fixture(scope="module", autouse=True)
def _native_extension() -> Any:
    return pytest.importorskip("kogwistar._rust")


def _sqlite(path: Path, operation: dict[str, Any]) -> Any:
    return store_sqlite(path=path, operation=operation)


def _postgres(dsn: str | None, schema: str, operation: dict[str, Any]) -> Any:
    return store_postgres(dsn=dsn, schema=schema, operation=operation)


def _event(*, entity_id: str, revision: int) -> str:
    """Full Node-shaped Python event payload, intentionally not Rust-specific."""
    source_document = Document.from_text("alpha beta", id="source-doc")
    source_document.source_map = {
        "chunk-0": Chunk(
            doc_id="source-doc", start_char=0, end_char=5, text="alpha"
        ),
        "chunk-1": Chunk(
            doc_id="source-doc", start_char=5, end_char=10, text=" beta"
        ),
    }
    source_map = source_document.model_dump(
        field_mode="backend", dump_format="json", exclude={"embeddings"}
    )["source_map"]
    return json.dumps(
        {
            "id": entity_id,
            "document": f"node {entity_id} revision {revision}",
            "metadata": {"revision": revision, "source": "python"},
            "groundings": [],
            "source_map": source_map,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _recover(
    namespace: str = "alpha",
    consumer: str = "sink",
    projection_namespace: str = "entity_projection",
    projection_key: str = "main",
    batch_limit: int = 2,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "kind": "recover_entity_projection",
        "namespace": namespace,
        "consumer": consumer,
        "projection_namespace": projection_namespace,
        "projection_key": projection_key,
        "batch_limit": batch_limit,
        **extra,
    }


def _rebuild(
    namespace: str = "alpha",
    consumer: str = "sink",
    projection_namespace: str = "entity_projection",
    projection_key: str = "main",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "kind": "rebuild_entity_projection",
        "namespace": namespace,
        "consumer": consumer,
        "projection_namespace": projection_namespace,
        "projection_key": projection_key,
        **extra,
    }


def _assert_projection_shape(payload: dict[str, Any]) -> None:
    entities = payload["entities"]
    deleted = entities['["node","n1"]']
    assert deleted["deleted"] is True
    assert deleted["op"] == "DELETE"
    active = entities['["alien_kind","n2"]']
    assert active["deleted"] is False
    assert active["entity"]["id"] == "n2"
    expected_source_map = Document.from_text("alpha beta", id="source-doc")
    expected_source_map.source_map = {
        "chunk-0": Chunk(
            doc_id="source-doc", start_char=0, end_char=5, text="alpha"
        ),
        "chunk-1": Chunk(
            doc_id="source-doc", start_char=5, end_char=10, text=" beta"
        ),
    }
    assert active["entity"]["source_map"] == expected_source_map.model_dump(
        field_mode="backend", dump_format="json", exclude={"embeddings"}
    )["source_map"]


def test_rust_meta_facades_implement_checkpointed_projection_store_protocol(
    tmp_path: Path,
) -> None:
    """Both native metadata facades preserve the shared Python projection seam."""

    sqlite = RustEngineSQLite(tmp_path)
    postgres = RustEnginePostgresMetaStore(
        dsn="postgresql://unconfigured.example.invalid/db",
        schema="contract_only",
    )
    assert isinstance(sqlite, CheckpointedProjectionStore)
    assert isinstance(postgres, CheckpointedProjectionStore)


def _run_recovery_contract(
    *,
    append_python: Any,
    append_rust: Any,
    call: Any,
    projection: Any,
    cursor: Any,
    event_count: Any,
) -> None:
    """Run same recovery invariants over a Python-owned durable corpus."""
    append_python("py-add", "node", "n1", "ADD", _event(entity_id="n1", revision=1))
    append_python("py-replace", "node", "n1", "REPLACE", _event(entity_id="n1", revision=2))
    append_python("py-tombstone", "node", "n1", "TOMBSTONE", '{"reason":"gone"}')
    append_python("py-delete", "node", "n1", "DELETE", '{"reason":"confirmed"}')
    append_rust(
        "rust-add-unknown",
        "alien_kind",
        "n2",
        "ADD",
        _event(entity_id="n2", revision=1),
    )
    append_rust(
        "rust-other-namespace",
        "node",
        "other",
        "ADD",
        _event(entity_id="other", revision=1),
        namespace="beta",
    )
    assert event_count() == 5

    first = call(_recover())
    assert first["processed_count"] == 2
    assert (first["prior_cursor"], first["new_cursor"], first["caught_up"]) == (0, 2, False)
    assert cursor("sink") == 2

    beta = call(
        _recover(
            namespace="beta",
            projection_namespace="beta_entity_projection",
            projection_key="main",
        )
    )
    assert (beta["processed_count"], beta["prior_cursor"], beta["new_cursor"]) == (1, 0, 1)
    assert projection("beta_entity_projection", "main")["payload"]["entities"]

    # Fold and write a real next batch, then deliberately abort. The named
    # projection and consumer cursor must both remain at the first batch.
    before_abort = projection("entity_projection", "main")
    before_cursor = cursor("sink")
    with pytest.raises(RustParityError) as aborted:
        call(_recover(abort_after_projection=True))
    assert aborted.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    assert projection("entity_projection", "main") == before_abort
    assert cursor("sink") == before_cursor

    # Every bridge call opens the durable store again; this is the restart edge.
    second = call(_recover())
    assert second["processed_count"] == 2
    assert (second["prior_cursor"], second["new_cursor"], second["caught_up"]) == (2, 4, False)
    assert cursor("sink") == 4

    third = call(_recover())
    assert third["processed_count"] == 1
    assert (third["prior_cursor"], third["new_cursor"], third["caught_up"]) == (4, 5, True)
    assert cursor("sink") == 5
    baseline = projection("entity_projection", "main")
    assert baseline is not None
    _assert_projection_shape(baseline["payload"])

    # Replaying a caught-up consumer is idempotent, including canonical bytes.
    idempotent = call(_recover())
    assert idempotent["processed_count"] == 0
    assert idempotent["canonical_payload"] == third["canonical_payload"]
    assert idempotent["digest"] == third["digest"]
    assert event_count() == 5

    # Same event namespace, independent consumer and target projection.
    isolated = call(_recover(consumer="audit", projection_key="audit", batch_limit=1))
    assert (isolated["prior_cursor"], isolated["new_cursor"]) == (0, 1)
    assert cursor("audit") == 1
    assert projection("entity_projection", "audit") is not None
    assert projection("entity_projection", "main") == baseline

    # Explicit rebuild changes no event rows and is byte-identical when corpus is unchanged.
    rebuilt = call(_rebuild())
    rebuilt_again = call(_rebuild())
    assert rebuilt["canonical_payload"] == rebuilt_again["canonical_payload"]
    assert rebuilt["digest"] == rebuilt_again["digest"]
    assert rebuilt_again["new_cursor"] == 5
    assert event_count() == 5

    # Invalid authoritative payload must roll back the named projection and cursor.
    append_rust("rust-bad", "node", "bad", "REPLACE", '{"id":"different"}')
    before_invalid = projection("entity_projection", "main")
    before_invalid_cursor = cursor("sink")
    with pytest.raises(RustParityError) as invalid:
        call(_recover())
    assert invalid.value.code == "KOGWISTAR_STORE_INVALID_ENTITY_EVENT"
    assert projection("entity_projection", "main") == before_invalid
    assert cursor("sink") == before_invalid_cursor

    with pytest.raises(RustParityError) as zero_limit:
        call(_recover(batch_limit=0))
    assert zero_limit.value.code == "KOGWISTAR_STORE_INVALID_ENTITY_EVENT"
    with pytest.raises(RustParityError) as unknown_field:
        call({**_recover(), "unexpected": True})
    assert unknown_field.value.code == "KOGWISTAR_STORE_INVALID_PAYLOAD"
    with pytest.raises(RustParityError) as invalid_namespace:
        call(_recover(namespace="bad\x00namespace"))
    assert invalid_namespace.value.code == "KOGWISTAR_STORE_INVALID_ENTITY_EVENT"


def test_sqlite_atomic_bounded_recovery_rebuild_and_python_rust_interop(tmp_path: Path) -> None:
    python = EngineSQLite(tmp_path)
    python.ensure_initialized()
    path = python.db_path

    def append_python(event_id: str, kind: str, entity_id: str, op: str, payload: str) -> None:
        assert (
            python.append_entity_event(
                namespace="alpha",
                event_id=event_id,
                entity_kind=kind,
                entity_id=entity_id,
                op=op,
                payload_json=payload,
            )
            > 0
        )

    def append_rust(
        event_id: str, kind: str, entity_id: str, op: str, payload: str, *, namespace: str = "alpha"
    ) -> None:
        _sqlite(
            path,
            {
                "kind": "raw_append",
                "namespace": namespace,
                "event_id": event_id,
                "entity_kind": kind,
                "entity_id": entity_id,
                "op": op,
                "payload_json": payload,
            },
        )

    def call(operation: dict[str, Any]) -> Any:
        return _sqlite(path, operation)

    def projection(namespace: str, key: str) -> Any:
        return _sqlite(path, {"kind": "get_named_projection", "namespace": namespace, "key": key})

    def cursor(consumer: str) -> int:
        return _sqlite(path, {"kind": "cursor_get", "namespace": "alpha", "consumer": consumer})[
            "last_seq"
        ]

    def event_count() -> int:
        return len(
            _sqlite(
                path,
                {"kind": "exclusive_raw_replay", "namespace": "alpha", "after_seq": 0, "limit": 100},
            )
        )

    _run_recovery_contract(
        append_python=append_python,
        append_rust=append_rust,
        call=call,
        projection=projection,
        cursor=cursor,
        event_count=event_count,
    )
    # Rust-created event remains Python-readable through existing public meta-store.
    assert [row[1] for row in python.iter_entity_events(namespace="alpha", from_seq=5)] == [
        "alien_kind",
        "node",
    ]


def test_sqlite_rebuild_recovers_python_persisted_workflow_run_node(tmp_path: Path) -> None:
    """Python-authored derived node survives Rust event-log rebuild/reopen."""
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "workflow-run-node-model-v1.json"
        ).read_text(encoding="utf-8")
    )
    payload = fixture["payload"]
    python_store = EngineSQLite(tmp_path)
    python_store.ensure_initialized()
    event_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    python_store.append_entity_event(
        namespace="alpha",
        event_id="python-workflow-run-node",
        entity_kind="node",
        entity_id=payload["id"],
        op="ADD",
        payload_json=event_payload,
    )

    operation = _rebuild(
        namespace="alpha",
        consumer="workflow-run-recovery",
        projection_namespace="derived-node-projection",
        projection_key="workflow-run",
    )
    first = _sqlite(python_store.db_path, operation)
    reopened = _sqlite(python_store.db_path, operation)

    assert first["canonical_payload"] == reopened["canonical_payload"]
    assert first["digest"] == reopened["digest"]
    assert (
        hashlib.sha256(first["canonical_payload"].encode("utf-8")).hexdigest()
        == first["digest"]
    )
    restored = json.loads(reopened["canonical_payload"])
    entity_key = json.dumps(["node", payload["id"]], separators=(",", ":"))
    assert restored["entities"][entity_key]["entity"] == payload


def test_sqlite_rebuild_recovers_python_persisted_workflow_checkpoint_node(
    tmp_path: Path,
) -> None:
    """Python checkpoint payload survives Rust SQLite replay and reopen."""
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "workflow-checkpoint-node-model-v1.json"
        ).read_text(encoding="utf-8")
    )
    payload = fixture["payload"]
    python_store = EngineSQLite(tmp_path)
    python_store.ensure_initialized()
    python_store.append_entity_event(
        namespace="alpha",
        event_id="python-workflow-checkpoint-node",
        entity_kind="node",
        entity_id=payload["id"],
        op="ADD",
        payload_json=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )

    operation = _rebuild(
        namespace="alpha",
        consumer="workflow-checkpoint-recovery",
        projection_namespace="derived-node-projection",
        projection_key="workflow-checkpoint",
    )
    first = _sqlite(python_store.db_path, operation)
    reopened = _sqlite(python_store.db_path, operation)

    assert first["canonical_payload"] == reopened["canonical_payload"]
    assert first["digest"] == reopened["digest"]
    assert (
        hashlib.sha256(first["canonical_payload"].encode("utf-8")).hexdigest()
        == first["digest"]
    )
    restored = json.loads(reopened["canonical_payload"])
    entity_key = json.dumps(["node", payload["id"]], separators=(",", ":"))
    assert restored["entities"][entity_key]["entity"] == payload


def test_sqlite_rebuild_recovers_python_persisted_workflow_completed_node(
    tmp_path: Path,
) -> None:
    """Python completion payload survives Rust SQLite replay and reopen."""
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "workflow-completed-node-model-v1.json"
        ).read_text(encoding="utf-8")
    )
    payload = fixture["payload"]
    python_store = EngineSQLite(tmp_path)
    python_store.ensure_initialized()
    python_store.append_entity_event(
        namespace="alpha",
        event_id="python-workflow-completed-node",
        entity_kind="node",
        entity_id=payload["id"],
        op="ADD",
        payload_json=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )

    operation = _rebuild(
        namespace="alpha",
        consumer="workflow-completed-recovery",
        projection_namespace="derived-node-projection",
        projection_key="workflow-completed",
    )
    first = _sqlite(python_store.db_path, operation)
    reopened = _sqlite(python_store.db_path, operation)

    assert first["canonical_payload"] == reopened["canonical_payload"]
    assert first["digest"] == reopened["digest"]
    assert (
        hashlib.sha256(first["canonical_payload"].encode("utf-8")).hexdigest()
        == first["digest"]
    )
    restored = json.loads(reopened["canonical_payload"])
    entity_key = json.dumps(["node", payload["id"]], separators=(",", ":"))
    assert restored["entities"][entity_key]["entity"] == payload


def test_sqlite_rebuild_recovers_python_persisted_workflow_runtime_edge(tmp_path: Path) -> None:
    """Rust recovery preserves Python WorkflowRuntimeEdge endpoint identity."""
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "workflow-runtime-edge-model-v1.json"
        ).read_text(encoding="utf-8")
    )
    payload = fixture["payload"]
    python_store = EngineSQLite(tmp_path)
    python_store.ensure_initialized()
    event_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    python_store.append_entity_event(
        namespace="alpha",
        event_id="python-workflow-runtime-edge",
        entity_kind="edge",
        entity_id=payload["id"],
        op="ADD",
        payload_json=event_payload,
    )

    operation = _rebuild(
        namespace="alpha",
        consumer="workflow-runtime-edge-recovery",
        projection_namespace="derived-edge-projection",
        projection_key="workflow-runtime",
    )
    first = _sqlite(python_store.db_path, operation)
    reopened = _sqlite(python_store.db_path, operation)

    assert first["canonical_payload"] == reopened["canonical_payload"]
    assert first["digest"] == reopened["digest"]
    assert (
        hashlib.sha256(first["canonical_payload"].encode("utf-8")).hexdigest()
        == first["digest"]
    )
    restored = json.loads(reopened["canonical_payload"])
    entity_key = json.dumps(["edge", payload["id"]], separators=(",", ":"))
    recovered_payload = restored["entities"][entity_key]["entity"]
    assert recovered_payload == payload
    assert recovered_payload["source_ids"] == ["parity-node"]
    assert recovered_payload["target_ids"] == ["external:target"]


@pytest.mark.parametrize(
    ("fixture_name", "projection_key"),
    [
        ("workflow-cancelled-node-model-v1.json", "workflow-cancelled"),
        ("workflow-failed-node-model-v1.json", "workflow-failed"),
    ],
)
def test_sqlite_rebuild_recovers_python_persisted_workflow_terminal_node(
    tmp_path: Path,
    fixture_name: str,
    projection_key: str,
) -> None:
    """Rust SQLite replay preserves Python-authored workflow terminal nodes."""
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / fixture_name
        ).read_text(encoding="utf-8")
    )
    payload = fixture["payload"]
    python_store = EngineSQLite(tmp_path)
    python_store.ensure_initialized()
    python_store.append_entity_event(
        namespace="alpha",
        event_id=f"python-{projection_key}-node",
        entity_kind="node",
        entity_id=payload["id"],
        op="ADD",
        payload_json=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )

    operation = _rebuild(
        namespace="alpha",
        consumer=f"{projection_key}-recovery",
        projection_namespace="derived-node-projection",
        projection_key=projection_key,
    )
    first = _sqlite(python_store.db_path, operation)
    reopened = _sqlite(python_store.db_path, operation)

    assert first["canonical_payload"] == reopened["canonical_payload"]
    assert first["digest"] == reopened["digest"]
    assert (
        hashlib.sha256(first["canonical_payload"].encode("utf-8")).hexdigest()
        == first["digest"]
    )
    entity_key = json.dumps(["node", payload["id"]], separators=(",", ":"))
    restored = json.loads(reopened["canonical_payload"])
    assert restored["entities"][entity_key]["entity"] == payload


@pytest.mark.parametrize(
    ("fixture_name", "projection_key"),
    [
        ("llm-node-model-v1.json", "llm-node"),
        ("llm-node-extraction-model-v1.json", "llm-node-extraction"),
        ("llm-node-flattened-model-v1.json", "llm-node-flattened"),
        ("llm-node-assoc-flattened-model-v1.json", "llm-node-assoc-flattened"),
    ],
)
def test_sqlite_rebuild_recovers_python_persisted_llm_derived_node(
    tmp_path: Path,
    fixture_name: str,
    projection_key: str,
) -> None:
    """Rust SQLite replay preserves Python-authored LLM-derived node payloads."""
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / fixture_name
        ).read_text(encoding="utf-8")
    )
    payload = fixture["payload"]
    python_store = EngineSQLite(tmp_path)
    python_store.ensure_initialized()
    python_store.append_entity_event(
        namespace="alpha",
        event_id=f"python-{projection_key}-node",
        entity_kind="node",
        entity_id=payload["id"],
        op="ADD",
        payload_json=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )

    operation = _rebuild(
        namespace="alpha",
        consumer=f"{projection_key}-recovery",
        projection_namespace="derived-node-projection",
        projection_key=projection_key,
    )
    first = _sqlite(python_store.db_path, operation)
    reopened = _sqlite(python_store.db_path, operation)

    assert first["canonical_payload"] == reopened["canonical_payload"]
    assert first["digest"] == reopened["digest"]
    assert (
        hashlib.sha256(first["canonical_payload"].encode("utf-8")).hexdigest()
        == first["digest"]
    )
    entity_key = json.dumps(["node", payload["id"]], separators=(",", ":"))
    restored = json.loads(reopened["canonical_payload"])
    assert restored["entities"][entity_key]["entity"] == payload


@pytest.mark.parametrize(
    ("fixture_name", "projection_key"),
    [
        ("conversation-edge-model-v1.json", "conversation-edge"),
        ("workflow-edge-model-v1.json", "workflow-edge"),
        ("llm-edge-model-v1.json", "llm-edge"),
        ("llm-edge-flattened-model-v1.json", "llm-edge-flattened"),
        ("llm-edge-assoc-flattened-model-v1.json", "llm-edge-assoc-flattened"),
    ],
)
def test_sqlite_rebuild_recovers_python_persisted_derived_edge(
    tmp_path: Path,
    fixture_name: str,
    projection_key: str,
) -> None:
    """Rust SQLite replay preserves Python-authored derived edge payloads."""
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / fixture_name
        ).read_text(encoding="utf-8")
    )
    payload = fixture["payload"]
    python_store = EngineSQLite(tmp_path)
    python_store.ensure_initialized()
    python_store.append_entity_event(
        namespace="alpha",
        event_id=f"python-{projection_key}",
        entity_kind="edge",
        entity_id=payload["id"],
        op="ADD",
        payload_json=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )

    operation = _rebuild(
        namespace="alpha",
        consumer=f"{projection_key}-recovery",
        projection_namespace="derived-edge-projection",
        projection_key=projection_key,
    )
    first = _sqlite(python_store.db_path, operation)
    reopened = _sqlite(python_store.db_path, operation)

    assert first["canonical_payload"] == reopened["canonical_payload"]
    assert first["digest"] == reopened["digest"]
    assert (
        hashlib.sha256(first["canonical_payload"].encode("utf-8")).hexdigest()
        == first["digest"]
    )
    entity_key = json.dumps(["edge", payload["id"]], separators=(",", ":"))
    restored = json.loads(reopened["canonical_payload"])
    assert restored["entities"][entity_key]["entity"] == payload


@pytest.mark.parametrize(
    ("fixture_name", "entity_kind"),
    [
        ("workflow-run-node-model-v1.json", "node"),
        ("workflow-checkpoint-node-model-v1.json", "node"),
        ("workflow-completed-node-model-v1.json", "node"),
        ("workflow-cancelled-node-model-v1.json", "node"),
        ("workflow-failed-node-model-v1.json", "node"),
        ("llm-node-model-v1.json", "node"),
        ("llm-node-extraction-model-v1.json", "node"),
        ("llm-node-flattened-model-v1.json", "node"),
        ("llm-node-assoc-flattened-model-v1.json", "node"),
        ("conversation-edge-model-v1.json", "edge"),
        ("workflow-edge-model-v1.json", "edge"),
        ("llm-edge-model-v1.json", "edge"),
        ("llm-edge-flattened-model-v1.json", "edge"),
        ("llm-edge-assoc-flattened-model-v1.json", "edge"),
        ("workflow-runtime-edge-model-v1.json", "edge"),
    ],
)
def test_postgres_rebuild_recovers_python_persisted_derived_entity(
    sa_engine: Any,
    pg_dsn: str | None,
    pg_schema: str | None,
    fixture_name: str,
    entity_kind: str,
) -> None:
    """PostgreSQL event authority replays Python-derived graph payload in Rust."""
    pytest.importorskip("sqlalchemy")
    if pg_dsn is None or pg_schema is None or sa_engine is None:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore

    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / fixture_name
        ).read_text(encoding="utf-8")
    )
    payload = fixture["payload"]
    namespace = f"derived-recovery-{uuid.uuid4().hex}"
    python_store = EnginePostgresMetaStore(engine=sa_engine, schema=pg_schema)
    python_store.ensure_initialized()
    event_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    python_store.append_entity_event(
        namespace=namespace,
        event_id=f"python-{uuid.uuid4().hex}",
        entity_kind=entity_kind,
        entity_id=payload["id"],
        op="ADD",
        payload_json=event_payload,
    )

    operation = _rebuild(
        namespace=namespace,
        consumer=f"consumer-{uuid.uuid4().hex}",
        projection_namespace=f"projection-{uuid.uuid4().hex}",
        projection_key="derived",
    )
    first = _postgres(pg_dsn, pg_schema, operation)
    reopened = _postgres(pg_dsn, pg_schema, operation)

    assert first["canonical_payload"] == reopened["canonical_payload"]
    assert first["digest"] == reopened["digest"]
    assert (
        hashlib.sha256(first["canonical_payload"].encode("utf-8")).hexdigest()
        == first["digest"]
    )
    restored = json.loads(reopened["canonical_payload"])
    entity_key = json.dumps([entity_kind, payload["id"]], separators=(",", ":"))
    recovered_entity = restored["entities"][entity_key]["entity"]
    assert recovered_entity == payload
    if fixture_name == "workflow-checkpoint-node-model-v1.json":
        python_model = WorkflowCheckpointNode.model_validate(recovered_entity)
        assert type(python_model) is WorkflowCheckpointNode
        assert (
            python_model.model_dump(field_mode="backend", dump_format="json")
            == payload
        )
    elif fixture_name == "workflow-completed-node-model-v1.json":
        python_model = WorkflowCompletedNode.model_validate(recovered_entity)
        assert type(python_model) is WorkflowCompletedNode
        assert (
            python_model.model_dump(field_mode="backend", dump_format="json")
            == payload
        )
    elif fixture_name == "workflow-cancelled-node-model-v1.json":
        python_model = WorkflowCancelledNode.model_validate(recovered_entity)
        assert type(python_model) is WorkflowCancelledNode
        assert (
            python_model.model_dump(field_mode="backend", dump_format="json")
            == payload
        )
    elif fixture_name == "workflow-failed-node-model-v1.json":
        python_model = WorkflowFailedNode.model_validate(recovered_entity)
        assert type(python_model) is WorkflowFailedNode
        assert (
            python_model.model_dump(field_mode="backend", dump_format="json")
            == payload
        )


def _assert_python_projection_recovery_over_rust_owned_store(
    open_store: Callable[[], RustEngineSQLite],
) -> None:
    """Run the shared Python reducer over a Rust-owned durable metadata store."""

    def decode_current(row: dict[str, Any]) -> ProjectionLoadResult[dict[str, int]]:
        payload = dict(row["payload"])
        checkpoint = ProjectionCheckpoint(
            projection_id=str(payload["projection_id"]),
            workspace_id=str(payload["workspace_id"]),
            projection_schema_version=int(payload["projection_schema_version"]),
            source_namespace=str(payload["source_namespace"]),
            source_from_seq=int(payload["source_from_seq"]),
            source_to_seq=int(payload["source_to_seq"]),
            last_authoritative_seq=int(row["last_authoritative_seq"]),
            last_materialized_seq=int(row["last_materialized_seq"]),
            projected_at_ms=int(payload["projected_at_ms"]),
            last_source_event_ts_ms=None,
            snapshot_id=str(payload["snapshot_id"]),
            raw_event_count=int(payload["raw_event_count"]),
            materialization_status=str(row["materialization_status"]),
            rebuild_reason=(
                str(payload["rebuild_reason"])
                if payload.get("rebuild_reason") is not None
                else None
            ),
            processed_event_ids=tuple(str(value) for value in payload.get("processed_event_ids", [])),
        )
        return ProjectionLoadResult(
            state={"total": int(payload["total"])},
            checkpoint=checkpoint,
            payload=payload,
        )

    def decode_event(payload_json: str) -> dict[str, int | str]:
        value = json.loads(payload_json)
        assert isinstance(value, dict)
        return {"event_id": str(value["event_id"]), "delta": int(value["delta"])}

    def build_payload(
        state: dict[str, int],
        checkpoint: ProjectionCheckpoint,
        processed_event_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "projection_id": checkpoint.projection_id,
            "workspace_id": checkpoint.workspace_id,
            "projection_schema_version": checkpoint.projection_schema_version,
            "source_namespace": checkpoint.source_namespace,
            "source_from_seq": checkpoint.source_from_seq,
            "source_to_seq": checkpoint.source_to_seq,
            "projected_at_ms": checkpoint.projected_at_ms,
            "last_source_event_ts_ms": checkpoint.last_source_event_ts_ms,
            "snapshot_id": checkpoint.snapshot_id,
            "raw_event_count": checkpoint.raw_event_count,
            "rebuild_reason": checkpoint.rebuild_reason,
            "processed_event_ids": list(processed_event_ids),
            "total": state["total"],
        }

    def refresh(store: RustEngineSQLite, *, rebuild_from_scratch: bool = False) -> dict[str, Any]:
        return dict(
            refresh_checkpointed_named_projection(
                store,
                namespace="projection",
                key="counter",
                projection_id="counter-v1",
                workspace_id="workspace",
                source_namespace="events",
                projection_schema_version=1,
                decode_current=decode_current,
                create_state=lambda: {"total": 0},
                decode_event=decode_event,
                event_key=lambda event: str(event["event_id"]),
                apply_event=lambda state, event, _seq: state.__setitem__(
                    "total", state["total"] + int(event["delta"])
                ),
                build_payload=build_payload,
                build_snapshot=lambda row: dict(row["payload"]),
                rebuild_from_scratch=rebuild_from_scratch,
            )
        )

    first = open_store()
    first.ensure_initialized()
    for event_id, delta in (("rust-1", 2), ("rust-2", 3)):
        assert (
            first.append_entity_event(
                namespace="events",
                event_id=event_id,
                entity_kind="counter",
                entity_id="total",
                op="ADD",
                payload_json=json.dumps({"event_id": event_id, "delta": delta}),
            )
            > 0
        )
    assert refresh(first, rebuild_from_scratch=True)["total"] == 5
    first.close()

    reopened = open_store()
    reopened.ensure_initialized()
    replayed = refresh(reopened, rebuild_from_scratch=True)
    assert replayed["total"] == 5
    assert replayed["source_to_seq"] == 2

    assert (
        reopened.append_entity_event(
            namespace="events",
            event_id="rust-3",
            entity_kind="counter",
            entity_id="total",
            op="ADD",
            payload_json=json.dumps({"event_id": "rust-3", "delta": 7}),
        )
        > 0
    )
    assert refresh(reopened)["total"] == 12
    assert refresh(reopened)["total"] == 12
    reopened.close()


def test_python_projection_recovery_over_rust_owned_sqlite_store(tmp_path: Path) -> None:
    _assert_python_projection_recovery_over_rust_owned_store(
        lambda: RustEngineSQLite(tmp_path)
    )


def test_python_projection_recovery_over_rust_owned_postgres_store(
    pg_dsn: str | None, pg_schema: str | None
) -> None:
    if pg_dsn is None or pg_schema is None:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    _assert_python_projection_recovery_over_rust_owned_store(
        lambda: RustEnginePostgresMetaStore(dsn=pg_dsn, schema=pg_schema)
    )


def test_postgres_atomic_bounded_recovery_rebuild_and_python_rust_interop(
    sa_engine: Any, pg_dsn: str | None, pg_schema: str | None
) -> None:
    pytest.importorskip("sqlalchemy")
    if pg_dsn is None or pg_schema is None:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore

    python = EnginePostgresMetaStore(engine=sa_engine, schema=pg_schema)
    python.ensure_initialized()
    prefix = uuid.uuid4().hex

    def append_python(event_id: str, kind: str, entity_id: str, op: str, payload: str) -> None:
        assert (
            python.append_entity_event(
                namespace="alpha",
                event_id=f"{prefix}-{event_id}",
                entity_kind=kind,
                entity_id=entity_id,
                op=op,
                payload_json=payload,
            )
            > 0
        )

    def append_rust(
        event_id: str, kind: str, entity_id: str, op: str, payload: str, *, namespace: str = "alpha"
    ) -> None:
        _postgres(
            pg_dsn,
            pg_schema,
            {
                "kind": "raw_append",
                "namespace": namespace,
                "event_id": f"{prefix}-{event_id}",
                "entity_kind": kind,
                "entity_id": entity_id,
                "op": op,
                "payload_json": payload,
            },
        )

    def call(operation: dict[str, Any]) -> Any:
        return _postgres(pg_dsn, pg_schema, operation)

    def projection(namespace: str, key: str) -> Any:
        return _postgres(
            pg_dsn, pg_schema, {"kind": "get_named_projection", "namespace": namespace, "key": key}
        )

    def cursor(consumer: str) -> int:
        return _postgres(
            pg_dsn, pg_schema, {"kind": "cursor_get", "namespace": "alpha", "consumer": consumer}
        )["last_seq"]

    def event_count() -> int:
        return len(
            _postgres(
                pg_dsn,
                pg_schema,
                {"kind": "exclusive_raw_replay", "namespace": "alpha", "after_seq": 0, "limit": 100},
            )
        )

    _run_recovery_contract(
        append_python=append_python,
        append_rust=append_rust,
        call=call,
        projection=projection,
        cursor=cursor,
        event_count=event_count,
    )
    assert [row[1] for row in python.iter_entity_events(namespace="alpha", from_seq=5)] == [
        "alien_kind",
        "node",
    ]
