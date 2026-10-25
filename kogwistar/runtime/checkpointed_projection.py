from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from kogwistar.id_provider import stable_id
from kogwistar.json_types import JsonValue

TState = TypeVar("TState")
TEvent = TypeVar("TEvent")
TSnapshot = TypeVar("TSnapshot")
ProjectionPayload = dict[str, JsonValue]
_PROCESSED_EVENT_ID_TAIL_LIMIT = 1024
PROJECTION_CHECKPOINT_WIRE_VERSION = 1
_PROJECTION_CHECKPOINT_WIRE_FIELDS = frozenset(
    {
        "contract_version",
        "projection_id",
        "workspace_id",
        "projection_schema_version",
        "source_namespace",
        "source_from_seq",
        "source_to_seq",
        "last_authoritative_seq",
        "last_materialized_seq",
        "projected_at_ms",
        "last_source_event_ts_ms",
        "snapshot_id",
        "raw_event_count",
        "materialization_status",
        "rebuild_reason",
        "processed_event_ids",
    }
)


class ProjectionConflictError(RuntimeError):
    """Raised when another projector advanced the named projection first."""


@runtime_checkable
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

    def get_named_projection(self, namespace: str, key: str) -> ProjectionPayload | None: ...

    def replace_named_projection(
        self,
        namespace: str,
        key: str,
        payload: ProjectionPayload,
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
        payload: ProjectionPayload,
        *,
        expected_last_authoritative_seq: int | None,
        expected_last_materialized_seq: int | None,
        last_authoritative_seq: int,
        last_materialized_seq: int,
        projection_schema_version: int,
        materialization_status: str,
    ) -> bool: ...

    def compare_and_swap_named_projections(self, updates: list[ProjectionPayload]) -> bool: ...


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

    def to_wire(self) -> dict[str, JsonValue]:
        """Return the strict versioned checkpoint payload shared with Rust."""
        return {
            "contract_version": PROJECTION_CHECKPOINT_WIRE_VERSION,
            "projection_id": self.projection_id,
            "workspace_id": self.workspace_id,
            "projection_schema_version": self.projection_schema_version,
            "source_namespace": self.source_namespace,
            "source_from_seq": self.source_from_seq,
            "source_to_seq": self.source_to_seq,
            "last_authoritative_seq": self.last_authoritative_seq,
            "last_materialized_seq": self.last_materialized_seq,
            "projected_at_ms": self.projected_at_ms,
            "last_source_event_ts_ms": self.last_source_event_ts_ms,
            "snapshot_id": self.snapshot_id,
            "raw_event_count": self.raw_event_count,
            "materialization_status": self.materialization_status,
            "rebuild_reason": self.rebuild_reason,
            "processed_event_ids": list(self.processed_event_ids),
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, object]) -> "ProjectionCheckpoint":
        """Decode a Rust-compatible checkpoint and reject ambiguous payloads."""
        unknown = set(payload) - _PROJECTION_CHECKPOINT_WIRE_FIELDS
        if unknown:
            raise ValueError(f"unknown projection checkpoint fields: {sorted(unknown)!r}")
        contract_version = payload.get("contract_version", PROJECTION_CHECKPOINT_WIRE_VERSION)
        if contract_version != PROJECTION_CHECKPOINT_WIRE_VERSION:
            raise ValueError("unsupported projection checkpoint contract version")

        def required_text(name: str) -> str:
            value = payload.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"projection checkpoint {name} must be non-empty text")
            return value

        def required_uint(name: str) -> int:
            value = payload.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"projection checkpoint {name} must be a non-negative integer")
            return value

        def optional_int(name: str) -> int | None:
            value = payload.get(name)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"projection checkpoint {name} must be an integer or null")
            return value

        def optional_text(name: str) -> str | None:
            value = payload.get(name)
            if value is None:
                return None
            if not isinstance(value, str):
                raise ValueError(f"projection checkpoint {name} must be text or null")
            return value

        processed = payload.get("processed_event_ids", ())
        if not isinstance(processed, (list, tuple)):
            raise ValueError("projection checkpoint processed_event_ids must be a list")
        event_ids = tuple(processed)
        if any(not isinstance(event_id, str) or not event_id.strip() for event_id in event_ids):
            raise ValueError("projection checkpoint event IDs must be non-empty text")
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("projection checkpoint event IDs must be unique")

        return cls(
            projection_id=required_text("projection_id"),
            workspace_id=required_text("workspace_id"),
            projection_schema_version=required_uint("projection_schema_version"),
            source_namespace=required_text("source_namespace"),
            source_from_seq=required_uint("source_from_seq"),
            source_to_seq=required_uint("source_to_seq"),
            last_authoritative_seq=required_uint("last_authoritative_seq"),
            last_materialized_seq=required_uint("last_materialized_seq"),
            projected_at_ms=optional_int("projected_at_ms"),
            last_source_event_ts_ms=optional_int("last_source_event_ts_ms"),
            snapshot_id=required_text("snapshot_id"),
            raw_event_count=required_uint("raw_event_count"),
            materialization_status=required_text("materialization_status"),
            rebuild_reason=optional_text("rebuild_reason"),
            processed_event_ids=event_ids,
        )


@dataclass(frozen=True, slots=True)
class ProjectionLoadResult(Generic[TState]):
    state: TState
    checkpoint: ProjectionCheckpoint
    payload: ProjectionPayload


def _now_ms() -> int:
    return int(time.time() * 1000)


def _json_int(value: JsonValue | None, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def _default_snapshot_id(*, workspace_id: str, projection_id: str, source_to_seq: int) -> str:
    return str(stable_id("projection_snapshot", workspace_id, projection_id, str(source_to_seq)))


def refresh_checkpointed_named_projection(
    store: CheckpointedProjectionStore,
    *,
    namespace: str,
    key: str,
    projection_id: str,
    workspace_id: str,
    source_namespace: str,
    projection_schema_version: int,
    decode_current: Callable[[Mapping[str, JsonValue]], ProjectionLoadResult[TState]],
    create_state: Callable[[], TState],
    decode_event: Callable[[str], TEvent],
    event_key: Callable[[TEvent], str],
    apply_event: Callable[[TState, TEvent, int], None],
    build_payload: Callable[[TState, ProjectionCheckpoint, Sequence[str]], ProjectionPayload],
    build_snapshot: Callable[[Mapping[str, JsonValue]], TSnapshot],
    include_event: Callable[[str, str, str, str], bool] | None = None,
    migrate_current: (
        Callable[[ProjectionLoadResult[TState], int], ProjectionLoadResult[TState]]
        | None
    ) = None,
    rebuild_from_scratch: bool = False,
) -> TSnapshot:
    current_row = store.get_named_projection(namespace, key)
    expected_last_authoritative_seq = (
        _json_int(current_row.get("last_authoritative_seq"))
        if current_row is not None and current_row.get("last_authoritative_seq") is not None
        else None
    )
    expected_last_materialized_seq = (
        _json_int(current_row.get("last_materialized_seq"))
        if current_row is not None and current_row.get("last_materialized_seq") is not None
        else None
    )
    current: ProjectionLoadResult[TState] | None = None
    rebuild_reason: str | None = None
    if current_row:
        try:
            current = decode_current(current_row)
            if current.checkpoint.projection_schema_version != projection_schema_version:
                if migrate_current is None:
                    current = None
                    rebuild_reason = "incompatible_projection_schema"
                else:
                    migrated = migrate_current(current, projection_schema_version)
                    if (
                        migrated.checkpoint.projection_schema_version
                        != projection_schema_version
                    ):
                        raise ValueError(
                            "projection migration returned an incompatible schema version"
                        )
                    current = migrated
                    rebuild_reason = "migrated_projection"
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
            ts_value = getattr(event, "ts_ms", None)
            if ts_value is not None:
                ts_int = _json_int(ts_value)
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
