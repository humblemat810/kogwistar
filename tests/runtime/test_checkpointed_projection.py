from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from kogwistar.id_provider import stable_id
from kogwistar.runtime import (
    BudgetAttribution,
    BudgetEvent,
    ProjectionCheckpoint,
    ProjectionLoadResult,
    refresh_checkpointed_named_projection,
)
from kogwistar.runtime.budget import budget_event_from_dict, budget_event_to_dict

pytestmark = [pytest.mark.ci, pytest.mark.runtime]


class _FakeStore:
    def __init__(self) -> None:
        self.events: list[tuple[int, str, str, str, str]] = []
        self.next_seq = 1
        self.projection_row: dict[str, object] | None = None

    def append_entity_event(
        self,
        *,
        namespace: str = "default",
        event_id: str,
        entity_kind: str,
        entity_id: str,
        op: str,
        payload_json: str,
    ) -> int:
        seq = self.next_seq
        self.next_seq += 1
        self.events.append((seq, entity_kind, entity_id, op, payload_json))
        return seq

    def get_latest_entity_event_seq(self, *, namespace: str = "default") -> int:
        return len(self.events)

    def iter_entity_events(
        self,
        *,
        namespace: str = "default",
        from_seq: int = 1,
        to_seq: int | None = None,
        batch_size: int = 500,
    ):
        for seq, entity_kind, entity_id, op, payload_json in self.events:
            if seq < from_seq:
                continue
            if to_seq is not None and seq > to_seq:
                continue
            yield seq, entity_kind, entity_id, op, payload_json

    def get_named_projection(self, namespace: str, key: str) -> dict[str, object] | None:
        if self.projection_row is None:
            return None
        return dict(self.projection_row)

    def replace_named_projection(
        self,
        namespace: str,
        key: str,
        payload: dict[str, object],
        *,
        last_authoritative_seq: int,
        last_materialized_seq: int,
        projection_schema_version: int,
        materialization_status: str,
    ) -> None:
        self.projection_row = {
            "namespace": namespace,
            "key": key,
            "payload": dict(payload),
            "last_authoritative_seq": int(last_authoritative_seq),
            "last_materialized_seq": int(last_materialized_seq),
            "projection_schema_version": int(projection_schema_version),
            "materialization_status": str(materialization_status),
        }


@dataclass(frozen=True)
class _CounterSnapshot:
    total: int
    checkpoint: ProjectionCheckpoint


def _append_event(store: _FakeStore, *, event: BudgetEvent) -> None:
    payload = budget_event_to_dict(event)
    payload["event_id"] = str(event.event_id or stable_id("evt", event.run_id, event.unit, event.amount))
    store.append_entity_event(
        namespace="usage",
        event_id=str(payload["event_id"]),
        entity_kind="usage_event",
        entity_id=str(payload["event_id"]),
        op="ADD",
        payload_json=json.dumps(payload, sort_keys=True, separators=(",", ":")),
    )


def _load_current(row: dict[str, object]) -> ProjectionLoadResult[dict[str, int]]:
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
        last_source_event_ts_ms=(
            int(payload["last_source_event_ts_ms"]) if payload.get("last_source_event_ts_ms") is not None else None
        ),
        snapshot_id=str(payload["snapshot_id"]),
        raw_event_count=int(payload["raw_event_count"]),
        materialization_status=str(row["materialization_status"]),
        rebuild_reason=str(payload.get("rebuild_reason")) if payload.get("rebuild_reason") else None,
        processed_event_ids=tuple(str(item) for item in payload.get("processed_event_ids") or []),
    )
    return ProjectionLoadResult(state={"total": int(payload["total"])}, checkpoint=checkpoint, payload=payload)


def _create_state() -> dict[str, int]:
    return {"total": 0}


def _decode_event(payload_json: str) -> BudgetEvent:
    return budget_event_from_dict(json.loads(payload_json))


def _event_key(event: BudgetEvent) -> str:
    return str(event.event_id)


def _apply_event(state: dict[str, int], event: BudgetEvent, _seq: int) -> None:
    state["total"] += int(event.amount or 0)


def _build_payload(state: dict[str, int], checkpoint: ProjectionCheckpoint, processed_event_ids: tuple[str, ...]) -> dict[str, object]:
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


def _build_snapshot(row: dict[str, object]) -> _CounterSnapshot:
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
        last_source_event_ts_ms=(
            int(payload["last_source_event_ts_ms"]) if payload.get("last_source_event_ts_ms") is not None else None
        ),
        snapshot_id=str(payload["snapshot_id"]),
        raw_event_count=int(payload["raw_event_count"]),
        materialization_status=str(row["materialization_status"]),
        rebuild_reason=str(payload.get("rebuild_reason")) if payload.get("rebuild_reason") else None,
        processed_event_ids=tuple(str(item) for item in payload.get("processed_event_ids") or []),
    )
    return _CounterSnapshot(total=int(payload["total"]), checkpoint=checkpoint)


def test_refresh_checkpointed_named_projection_materializes_and_resumes() -> None:
    store = _FakeStore()
    _append_event(
        store,
        event=BudgetEvent(
            event_id="evt-1",
            run_id="run-1",
            source="provider",
            kind="token",
            amount=5,
            unit="input_tokens",
            attribution=BudgetAttribution(source_document_id="doc-1"),
        ),
    )
    _append_event(
        store,
        event=BudgetEvent(
            event_id="evt-2",
            run_id="run-2",
            source="provider",
            kind="token",
            amount=7,
            unit="output_tokens",
        ),
    )

    first = refresh_checkpointed_named_projection(
        store,
        namespace="usage_projection",
        key="ws-1",
        projection_id="usage",
        workspace_id="ws-1",
        source_namespace="usage_events",
        projection_schema_version=1,
        decode_current=_load_current,
        create_state=_create_state,
        decode_event=_decode_event,
        event_key=_event_key,
        apply_event=_apply_event,
        build_payload=_build_payload,
        build_snapshot=_build_snapshot,
    )

    assert first.total == 12
    assert first.checkpoint.materialization_status == "materialized"
    assert first.checkpoint.last_materialized_seq == 2

    _append_event(
        store,
        event=BudgetEvent(
            event_id="evt-3",
            run_id="run-3",
            source="runtime",
            kind="debit",
            amount=3,
            unit="ms",
        ),
    )

    second = refresh_checkpointed_named_projection(
        store,
        namespace="usage_projection",
        key="ws-1",
        projection_id="usage",
        workspace_id="ws-1",
        source_namespace="usage_events",
        projection_schema_version=1,
        decode_current=_load_current,
        create_state=_create_state,
        decode_event=_decode_event,
        event_key=_event_key,
        apply_event=_apply_event,
        build_payload=_build_payload,
        build_snapshot=_build_snapshot,
    )

    assert second.total == 15
    assert second.checkpoint.source_from_seq == 3
    assert second.checkpoint.last_materialized_seq == 3
    assert second.checkpoint.processed_event_ids == ("evt-1", "evt-2", "evt-3")


def test_refresh_checkpointed_named_projection_failure_preserves_previous_checkpoint() -> None:
    store = _FakeStore()
    _append_event(
        store,
        event=BudgetEvent(
            event_id="evt-1",
            run_id="run-1",
            source="provider",
            kind="token",
            amount=5,
            unit="input_tokens",
        ),
    )
    first = refresh_checkpointed_named_projection(
        store,
        namespace="usage_projection",
        key="ws-1",
        projection_id="usage",
        workspace_id="ws-1",
        source_namespace="usage_events",
        projection_schema_version=1,
        decode_current=_load_current,
        create_state=_create_state,
        decode_event=_decode_event,
        event_key=_event_key,
        apply_event=_apply_event,
        build_payload=_build_payload,
        build_snapshot=_build_snapshot,
    )
    assert first.total == 5

    _append_event(
        store,
        event=BudgetEvent(
            event_id="evt-2",
            run_id="run-2",
            source="provider",
            kind="token",
            amount=9,
            unit="output_tokens",
        ),
    )

    def fail_apply(_state: dict[str, int], _event: BudgetEvent, _seq: int) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        refresh_checkpointed_named_projection(
            store,
            namespace="usage_projection",
            key="ws-1",
            projection_id="usage",
            workspace_id="ws-1",
            source_namespace="usage_events",
            projection_schema_version=1,
            decode_current=_load_current,
            create_state=_create_state,
            decode_event=_decode_event,
            event_key=_event_key,
            apply_event=fail_apply,
            build_payload=_build_payload,
            build_snapshot=_build_snapshot,
        )

    snapshot = _build_snapshot(store.get_named_projection("usage_projection", "ws-1"))
    assert snapshot.total == 5
    assert snapshot.checkpoint.materialization_status == "failed"


def test_refresh_checkpointed_named_projection_can_skip_unrelated_events() -> None:
    store = _FakeStore()
    _append_event(
        store,
        event=BudgetEvent(
            event_id="usage-1",
            run_id="run-1",
            source="provider",
            kind="token",
            amount=5,
            unit="input_tokens",
        ),
    )
    store.events.append((2, "other_event", "other-1", "ADD", "not usage json"))

    result = refresh_checkpointed_named_projection(
        store,
        namespace="usage_projection",
        key="ws-1",
        projection_id="usage",
        workspace_id="ws-1",
        source_namespace="usage_events",
        projection_schema_version=1,
        decode_current=_load_current,
        create_state=_create_state,
        decode_event=_decode_event,
        event_key=_event_key,
        apply_event=_apply_event,
        build_payload=_build_payload,
        build_snapshot=_build_snapshot,
        include_event=lambda entity_kind, _entity_id, _op, _payload: entity_kind == "usage_event",
    )

    assert result.total == 5
    assert result.checkpoint.last_materialized_seq == 2
    assert result.checkpoint.raw_event_count == 1
