from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from kogwistar.id_provider import stable_id


TState = TypeVar("TState")
TEvent = TypeVar("TEvent")
TSnapshot = TypeVar("TSnapshot")
_PROCESSED_EVENT_ID_TAIL_LIMIT = 1024


class ProjectionConflictError(RuntimeError):
    """Raised when another projector advanced the named projection first."""


class CheckpointedProjectionStore(Protocol):
    def get_latest_entity_event_seq(self, *, namespace: str = "default") -> int: ...

    def iter_entity_events(
        self,
        *,
        namespace: str = "default",
        from_seq: int = 1,
        to_seq: int | None = None,
        batch_size: int = 500,
    ) -> Iterable[tuple[int, str, str, str, str]]: ...

    def get_named_projection(self, namespace: str, key: str) -> dict[str, Any] | None: ...

    def replace_named_projection(
        self,
        namespace: str,
        key: str,
        payload: dict[str, Any],
        *,
        last_authoritative_seq: int,
        last_materialized_seq: int,
        projection_schema_version: int,
        materialization_status: str,
    ) -> None: ...

    def compare_and_swap_named_projection(
        self,
        namespace: str,
        key: str,
        payload: dict[str, Any],
        *,
        expected_last_authoritative_seq: int | None,
        expected_last_materialized_seq: int | None,
        last_authoritative_seq: int,
        last_materialized_seq: int,
        projection_schema_version: int,
        materialization_status: str,
    ) -> bool: ...

    def compare_and_swap_named_projections(self, updates: list[dict[str, Any]]) -> bool: ...


@dataclass(frozen=True, slots=True)
class ProjectionCheckpoint:
    projection_id: str
    workspace_id: str
    projection_schema_version: int
    source_namespace: str
    source_from_seq: int
    source_to_seq: int
    last_authoritative_seq: int
    last_materialized_seq: int
    projected_at_ms: int | None
    last_source_event_ts_ms: int | None
    snapshot_id: str
    raw_event_count: int
    materialization_status: str
    rebuild_reason: str | None
    processed_event_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectionLoadResult(Generic[TState]):
    state: TState
    checkpoint: ProjectionCheckpoint
    payload: dict[str, Any]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _default_snapshot_id(*, workspace_id: str, projection_id: str, source_to_seq: int) -> str:
    return str(stable_id("projection_snapshot", workspace_id, projection_id, source_to_seq))


def refresh_checkpointed_named_projection(
    store: CheckpointedProjectionStore,
    *,
    namespace: str,
    key: str,
    projection_id: str,
    workspace_id: str,
    source_namespace: str,
    projection_schema_version: int,
    decode_current: Callable[[Mapping[str, Any]], ProjectionLoadResult[TState]],
    create_state: Callable[[], TState],
    decode_event: Callable[[str], TEvent],
    event_key: Callable[[TEvent], str],
    apply_event: Callable[[TState, TEvent, int], None],
    build_payload: Callable[[TState, ProjectionCheckpoint, Sequence[str]], dict[str, Any]],
    build_snapshot: Callable[[Mapping[str, Any]], TSnapshot],
    include_event: Callable[[str, str, str, str], bool] | None = None,
    rebuild_from_scratch: bool = False,
) -> TSnapshot:
    current_row = store.get_named_projection(namespace, key)
    expected_last_authoritative_seq = (
        int(current_row["last_authoritative_seq"])
        if current_row is not None and current_row.get("last_authoritative_seq") is not None
        else None
    )
    expected_last_materialized_seq = (
        int(current_row["last_materialized_seq"])
        if current_row is not None and current_row.get("last_materialized_seq") is not None
        else None
    )
    current: ProjectionLoadResult[TState] | None = None
    rebuild_reason: str | None = None
    if current_row:
        try:
            current = decode_current(current_row)
        except (KeyError, TypeError, ValueError):
            current = None
            rebuild_reason = "incompatible_projection"
    if rebuild_from_scratch:
        current = None
        rebuild_reason = "explicit_rebuild"
    elif current is None and rebuild_reason is None:
        rebuild_reason = "missing_projection"

    def mark_failed(exc: Exception) -> None:
        if current is None:
            return
        failure_payload = dict(current.payload)
        failure_payload["rebuild_reason"] = f"projection_failed:{type(exc).__name__}"
        failure_payload["processed_event_ids"] = list(current.checkpoint.processed_event_ids)
        store.replace_named_projection(
            namespace,
            key,
            failure_payload,
            last_authoritative_seq=current.checkpoint.last_authoritative_seq,
            last_materialized_seq=current.checkpoint.last_materialized_seq,
            projection_schema_version=current.checkpoint.projection_schema_version,
            materialization_status="failed",
        )

    try:
        latest = int(store.get_latest_entity_event_seq(namespace=source_namespace))
    except Exception as exc:
        mark_failed(exc)
        raise
    if current is not None and current.checkpoint.last_authoritative_seq > latest:
        current = None
        rebuild_reason = "source_sequence_regressed"

    from_seq = 1 if current is None else int(current.checkpoint.last_materialized_seq) + 1
    to_seq = latest
    state = create_state() if current is None else current.state
    processed_event_ids = list(current.checkpoint.processed_event_ids if current is not None else ())
    seen_event_ids = set(processed_event_ids)
    raw_event_count = int(current.checkpoint.raw_event_count if current is not None else 0)
    last_source_event_ts_ms = (
        int(current.checkpoint.last_source_event_ts_ms)
        if current is not None and current.checkpoint.last_source_event_ts_ms is not None
        else None
    )

    try:
        for _seq, entity_kind, entity_id, op, payload_json in store.iter_entity_events(
            namespace=source_namespace,
            from_seq=from_seq,
            to_seq=to_seq,
        ):
            if include_event is not None and not include_event(entity_kind, entity_id, op, payload_json):
                continue
            event = decode_event(payload_json)
            event_id = str(event_key(event))
            if event_id in seen_event_ids:
                continue
            seen_event_ids.add(event_id)
            processed_event_ids.append(event_id)
            raw_event_count += 1
            if hasattr(event, "ts_ms"):
                ts_value = getattr(event, "ts_ms")
                if ts_value is not None:
                    ts_int = int(ts_value)
                    last_source_event_ts_ms = max(last_source_event_ts_ms or ts_int, ts_int)
            apply_event(state, event, _seq)

        projected_at_ms = _now_ms()
        authoritative_after = int(store.get_latest_entity_event_seq(namespace=source_namespace))
        status = "materialized" if authoritative_after <= to_seq else "catching_up"
        checkpoint = ProjectionCheckpoint(
            projection_id=projection_id,
            workspace_id=workspace_id,
            projection_schema_version=projection_schema_version,
            source_namespace=source_namespace,
            source_from_seq=from_seq,
            source_to_seq=to_seq,
            last_authoritative_seq=authoritative_after,
            last_materialized_seq=to_seq,
            projected_at_ms=projected_at_ms,
            last_source_event_ts_ms=last_source_event_ts_ms,
            snapshot_id=_default_snapshot_id(
                workspace_id=workspace_id,
                projection_id=projection_id,
                source_to_seq=to_seq,
            ),
            raw_event_count=raw_event_count,
            materialization_status=status,
            rebuild_reason=rebuild_reason,
            processed_event_ids=tuple(processed_event_ids[-_PROCESSED_EVENT_ID_TAIL_LIMIT:]),
        )
        payload = build_payload(state, checkpoint, checkpoint.processed_event_ids)
        compare_and_swap = getattr(store, "compare_and_swap_named_projection", None)
        if callable(compare_and_swap):
            committed = compare_and_swap(
                namespace,
                key,
                payload,
                expected_last_authoritative_seq=(
                    expected_last_authoritative_seq
                ),
                expected_last_materialized_seq=(
                    expected_last_materialized_seq
                ),
                last_authoritative_seq=authoritative_after,
                last_materialized_seq=to_seq,
                projection_schema_version=projection_schema_version,
                materialization_status=status,
            )
            if not committed:
                raise ProjectionConflictError(
                    f"projection advanced concurrently: namespace={namespace!r} key={key!r}"
                )
        else:
            store.replace_named_projection(
                namespace,
                key,
                payload,
                last_authoritative_seq=authoritative_after,
                last_materialized_seq=to_seq,
                projection_schema_version=projection_schema_version,
                materialization_status=status,
            )
        stored_row = store.get_named_projection(namespace, key)
        if stored_row is None:
            raise RuntimeError("projection disappeared after materialization")
        return build_snapshot(stored_row)
    except ProjectionConflictError:
        raise
    except Exception as exc:
        mark_failed(exc)
        raise
