from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator, cast

import pytest

from kogwistar.json_types import JsonObject, JsonValue, decode_json_object
from kogwistar.engine_core.two_stage_async import (
    AsyncRustPostgresTwoStageProjectionAdapter,
)
from kogwistar.engine_core.two_stage_rust_postgres import (
    RustPostgresTwoStageProjectionAdapter,
)


class _FakeMeta:
    def __init__(self, record: JsonObject) -> None:
        self.record = record
        self.last_read_for_update = False

    def graph_projection_records(self, **values: JsonValue) -> list[JsonObject]:
        self.last_read_for_update = bool(values.get("for_update"))
        return [self.record]

    def upsert_graph_projection(self, **values: JsonValue) -> None:
        record = values.get("record")
        if not isinstance(record, dict):
            raise TypeError("projection record must be a JSON object")
        self.record = cast(JsonObject, record)


class _FakeEngine:
    namespace = "default"
    backend = SimpleNamespace(
        embedding_dim=3,
        nodes=SimpleNamespace(name="gke_nodes"),
        edges=SimpleNamespace(name="gke_edges"),
    )
    _ef: object | None = None

    def __init__(self, *, state: str, fingerprint: str) -> None:
        self.state = state
        self.fingerprint = fingerprint
        self.uow_entries = 0
        self.enqueued: list[JsonObject] = []

    @contextmanager
    def uow(self) -> Iterator[None]:
        self.uow_entries += 1
        yield

    @property
    def indexing(self) -> _FakeEngine:
        return self

    def canonical_entity_revision(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(state=self.state)

    def canonical_revision_payload(self, **_: object) -> str:
        return json.dumps({"source_fingerprint": self.fingerprint})

    def enqueue_index_job(self, **values: JsonValue) -> str:
        self.enqueued.append(values)
        return "requeued-upsert"


def _projection_record(status: str = "tombstoned") -> JsonObject:
    return decode_json_object(
        {
            "id": "delete-target",
            "document": "retained canonical graph document",
            "metadata": {
                "lifecycle_status": status,
                "_kogwistar_stage2_ready": True,
                "_kogwistar_source_fingerprint": "old-fingerprint",
            },
            "embedding": [0.1, 0.2, 0.3],
        }
    )


def test_rust_postgres_delete_clears_vector_but_retains_tombstone_in_uow() -> None:
    engine = _FakeEngine(state="deleted", fingerprint="old-fingerprint")
    meta = _FakeMeta(_projection_record())
    adapter = RustPostgresTwoStageProjectionAdapter(cast(Any, engine), cast(Any, meta))

    adapter.apply_embedding_job(
        entity_kind="node",
        entity_id="delete-target",
        op="DELETE",
        payload_json=json.dumps({"source_fingerprint": "old-fingerprint"}),
    )

    assert engine.uow_entries == 1
    assert meta.last_read_for_update is True
    assert meta.record["id"] == "delete-target"
    assert meta.record["document"] == "retained canonical graph document"
    assert meta.record["embedding"] is None
    metadata = cast(JsonObject, meta.record["metadata"])
    assert metadata["lifecycle_status"] == "tombstoned"
    assert metadata["_kogwistar_stage2_ready"] is False
    assert "_kogwistar_source_fingerprint" not in metadata


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.asyncio
async def test_async_rust_postgres_delete_uses_same_stale_revision_fence(
    batched: bool,
) -> None:
    async def provider(texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    engine = _FakeEngine(state="active", fingerprint="new-fingerprint")
    engine._ef = provider
    meta = _FakeMeta(_projection_record(status="active"))
    adapter = AsyncRustPostgresTwoStageProjectionAdapter(engine, meta)
    payload = json.dumps({"source_fingerprint": "old-fingerprint"})

    if batched:
        outcomes = await adapter.apply_embedding_jobs_batch(
            [{
                "job_id": "stale-delete",
                "entity_kind": "node",
                "entity_id": "delete-target",
                "op": "DELETE",
                "payload_json": payload,
            }]
        )
        assert outcomes == {"stale-delete": None}
    else:
        await adapter.apply_embedding_job(
            entity_kind="node",
            entity_id="delete-target",
            op="DELETE",
            payload_json=payload,
        )

    assert engine.uow_entries == 1
    assert meta.last_read_for_update is True
    assert meta.record["embedding"] == [0.1, 0.2, 0.3]
    metadata = cast(JsonObject, meta.record["metadata"])
    assert metadata["_kogwistar_stage2_ready"] is True
    assert len(engine.enqueued) == 1
    assert engine.enqueued[0]["op"] == "UPSERT"
