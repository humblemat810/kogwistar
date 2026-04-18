from __future__ import annotations

import contextvars
import copy
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from .engine_sqlite import IndexJobRow
from .meta_lane_messages import LaneMessageMetaStoreMixin
from ..messaging.models import ProjectedLaneMessageRow


_active_in_memory_meta_txn: contextvars.ContextVar["_TxnView | None"] = contextvars.ContextVar(
    "gke_in_memory_meta_txn", default=None
)


def _now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


@dataclass
class _EntityEventRow:
    seq: int
    entity_kind: str
    entity_id: str
    op: str
    payload_json: str


@dataclass
class _JobState:
    job_id: str
    namespace: str
    entity_kind: str
    entity_id: str
    index_kind: str
    coalesce_key: str
    op: str
    status: str
    lease_until: int | None
    next_run_at: int | None
    max_retries: int
    retry_count: int
    last_error: str | None
    payload_json: str | None
    created_at: int
    updated_at: int

    def as_row(self) -> IndexJobRow:
        return IndexJobRow(
            job_id=self.job_id,
            namespace=self.namespace,
            entity_kind=self.entity_kind,
            entity_id=self.entity_id,
            index_kind=self.index_kind,
            coalesce_key=self.coalesce_key,
            op=self.op,
            status=self.status,
            lease_until=self.lease_until,
            next_run_at=self.next_run_at,
            max_retries=self.max_retries,
            retry_count=self.retry_count,
            last_error=self.last_error,
            payload_json=self.payload_json,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


@dataclass
class _WorkflowSnapshotRow:
    workflow_id: str
    version: int
    seq: int
    payload_json: str
    schema_version: int
    created_at_ms: int


@dataclass
class _WorkflowDeltaRow:
    workflow_id: str
    version: int
    prev_version: int
    target_seq: int
    forward_json: str
    inverse_json: str
    schema_version: int
    created_at_ms: int


@dataclass
class _ProjectedLaneMessageState:
    message_id: str
    namespace: str
    inbox_id: str
    conversation_id: str
    recipient_id: str
    sender_id: str
    msg_type: str
    status: str
    seq: int
    conversation_seq: int
    claimed_by: str | None
    lease_until: int | None
    retry_count: int
    created_at: int
    available_at: int
    run_id: str | None
    step_id: str | None
    correlation_id: str | None
    payload_json: str | None
    error_json: str | None

    def as_row(self) -> ProjectedLaneMessageRow:
        return ProjectedLaneMessageRow(
            message_id=self.message_id,
            namespace=self.namespace,
            inbox_id=self.inbox_id,
            conversation_id=self.conversation_id,
            recipient_id=self.recipient_id,
            sender_id=self.sender_id,
            msg_type=self.msg_type,
            status=self.status,
            seq=self.seq,
            conversation_seq=self.conversation_seq,
            claimed_by=self.claimed_by,
            lease_until=self.lease_until,
            retry_count=self.retry_count,
            created_at=self.created_at,
            available_at=self.available_at,
            run_id=self.run_id,
            step_id=self.step_id,
            correlation_id=self.correlation_id,
            payload_json=self.payload_json,
            error_json=self.error_json,
        )


@dataclass
class _ServerRunRow:
    run_id: str
    conversation_id: str
    workflow_id: str
    user_id: str | None
    user_turn_node_id: str | None
    assistant_turn_node_id: str | None
    status: str
    cancel_requested: bool
    result_json: str | None
    error_json: str | None
    created_at_ms: int
    updated_at_ms: int
    started_at_ms: int | None
    finished_at_ms: int | None


@dataclass
class _ServerRunEventRow:
    seq: int
    run_id: str
    event_type: str
    payload_json: str
    created_at_ms: int


@dataclass
class _MetaState:
    global_seq: int = 0
    user_seq: dict[str, int] = field(default_factory=dict)
    namespace_next_seq: dict[str, int] = field(default_factory=dict)
    index_jobs: dict[str, _JobState] = field(default_factory=dict)
    lane_messages: dict[str, _ProjectedLaneMessageState] = field(default_factory=dict)
    applied_fingerprints: dict[tuple[str, str], tuple[str | None, str | None, int]] = field(
        default_factory=dict
    )
    entity_events: dict[str, list[_EntityEventRow]] = field(default_factory=dict)
    replay_cursors: dict[tuple[str, str], int] = field(default_factory=dict)
    named_projections: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    workflow_snapshots: dict[tuple[str, int], _WorkflowSnapshotRow] = field(default_factory=dict)
    workflow_deltas: dict[tuple[str, int, int], _WorkflowDeltaRow] = field(default_factory=dict)
    server_runs: dict[str, _ServerRunRow] = field(default_factory=dict)
    server_run_events: dict[str, list[_ServerRunEventRow]] = field(default_factory=dict)


class _TxnView:
    def __init__(self, owner: "InMemoryMetaStore", state: _MetaState) -> None:
        self.owner = owner
        self.state = state


class _ResultShim:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = list(rows)

    def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[Any]:
        return list(self._rows)


class _InMemoryMetaConnection:
    def __init__(self, store: "InMemoryMetaStore") -> None:
        self._store = store

    def __enter__(self) -> "_InMemoryMetaConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> _ResultShim:
        normalized = " ".join(str(sql).strip().split()).upper()
        if normalized == "UPDATE INDEX_JOBS SET LEASE_UNTIL = 0 WHERE JOB_ID = ?":
            if not params:
                raise ValueError("job_id param required")
            self._store._debug_force_job_lease(job_id=str(params[0]), lease_until=0)
            return _ResultShim([])
        raise NotImplementedError(f"In-memory meta connect() does not support SQL: {sql!r}")


class InMemoryMetaStore(LaneMessageMetaStoreMixin):
    """Lock-backed in-memory metastore with EngineSQLite-like behavior."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = _MetaState()
        self.conn_str = ":memory:"

    def ensure_initialized(self) -> None:
        return None

    def connect(self) -> _InMemoryMetaConnection:
        return _InMemoryMetaConnection(self)

    def _view(self) -> _TxnView:
        active = _active_in_memory_meta_txn.get()
        if active is not None and active.owner is self:
            return active
        return _TxnView(self, self._state)

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[_TxnView]:
        _ = immediate
        active = _active_in_memory_meta_txn.get()
        if active is not None and active.owner is self:
            yield active
            return

        with self._lock:
            working = copy.deepcopy(self._state)
            txn = _TxnView(self, working)
            token = _active_in_memory_meta_txn.set(txn)
            try:
                yield txn
                self._state = working
            finally:
                _active_in_memory_meta_txn.reset(token)

    def _debug_force_job_lease(self, *, job_id: str, lease_until: int | None) -> None:
        with self.transaction() as txn:
            job = txn.state.index_jobs.get(str(job_id))
            if job is None:
                return
            job.lease_until = None if lease_until is None else int(lease_until)
            job.updated_at = _now_epoch()

    def next_global_seq(self) -> int:
        with self.transaction() as txn:
            txn.state.global_seq += 1
            return int(txn.state.global_seq)

    def current_global_seq(self) -> int:
        with self._lock:
            return int(self._state.global_seq)

    def next_user_seq(self, user_id: str) -> int:
        with self.transaction() as txn:
            value = int(txn.state.user_seq.get(str(user_id), 0)) + 1
            txn.state.user_seq[str(user_id)] = value
            return value

    def next_scoped_seq(self, scope_id: str) -> int:
        return self.next_user_seq(scope_id)

    def current_user_seq(self, user_id: str) -> int:
        with self._lock:
            return int(self._state.user_seq.get(str(user_id), 0))

    def current_scoped_seq(self, scope_id: str) -> int:
        return self.current_user_seq(scope_id)

    def set_user_seq(self, user_id: str, value: int) -> None:
        if int(value) < 0:
            raise ValueError("value must be >= 0")
        with self.transaction() as txn:
            txn.state.user_seq[str(user_id)] = int(value)

    def set_scoped_seq(self, scope_id: str, value: int) -> None:
        self.set_user_seq(scope_id, value)

    def enqueue_index_job(
        self,
        *,
        job_id: str,
        entity_kind: str,
        entity_id: str,
        index_kind: str,
        op: str,
        payload_json: Optional[str] = None,
        max_retries: int = 10,
        namespace: str = "default",
    ) -> str:
        now = _now_epoch()
        coalesce_key = f"{entity_kind}:{entity_id}:{index_kind}"
        with self.transaction() as txn:
            pending = sorted(
                (
                    job
                    for job in txn.state.index_jobs.values()
                    if job.namespace == str(namespace)
                    and job.coalesce_key == coalesce_key
                    and job.status == "PENDING"
                ),
                key=lambda item: item.created_at,
            )
            if pending:
                existing = pending[0]
                existing.op = "DELETE" if (op == "DELETE" or existing.op == "DELETE") else op
                existing.payload_json = payload_json
                existing.updated_at = now
                return existing.job_id

            txn.state.index_jobs[str(job_id)] = _JobState(
                job_id=str(job_id),
                namespace=str(namespace),
                entity_kind=str(entity_kind),
                entity_id=str(entity_id),
                index_kind=str(index_kind),
                coalesce_key=coalesce_key,
                op=str(op),
                status="PENDING",
                lease_until=None,
                next_run_at=None,
                max_retries=int(max_retries),
                retry_count=0,
                last_error=None,
                payload_json=payload_json,
                created_at=now,
                updated_at=now,
            )
            return str(job_id)

    def claim_index_jobs(
        self,
        *,
        limit: int = 50,
        lease_seconds: int = 60,
        namespace: Optional[str] = "default",
    ) -> list[IndexJobRow]:
        if int(limit) <= 0:
            return []
        now = _now_epoch()
        lease_until = now + int(lease_seconds)
        with self.transaction() as txn:
            candidates = []
            for job in txn.state.index_jobs.values():
                if namespace is not None and job.namespace != str(namespace):
                    continue
                eligible = (
                    job.status == "PENDING"
                    and (job.next_run_at is None or job.next_run_at <= now)
                ) or (
                    job.status == "DOING"
                    and job.lease_until is not None
                    and job.lease_until < now
                )
                if eligible:
                    candidates.append(job)
            candidates.sort(key=lambda item: item.created_at)
            claimed = candidates[: int(limit)]
            rows: list[IndexJobRow] = []
            for job in claimed:
                job.status = "DOING"
                job.lease_until = lease_until
                job.updated_at = now
                rows.append(job.as_row())
            return rows

    def mark_index_job_done(self, job_id: str) -> None:
        now = _now_epoch()
        with self.transaction() as txn:
            job = txn.state.index_jobs.get(str(job_id))
            if job is None:
                return
            job.status = "DONE"
            job.lease_until = None
            job.updated_at = now

    def mark_index_job_failed(self, job_id: str, error: str, *, final: bool = True) -> None:
        now = _now_epoch()
        with self.transaction() as txn:
            job = txn.state.index_jobs.get(str(job_id))
            if job is None:
                return
            if final:
                job.status = "FAILED"
            job.lease_until = None
            job.last_error = str(error or "")[:2000]
            job.updated_at = now

    def bump_retry_and_requeue(
        self, job_id: str, error: str, *, next_run_at_seconds: int
    ) -> None:
        now = _now_epoch()
        delay = max(0, int(next_run_at_seconds))
        next_run_at = now + delay
        with self.transaction() as txn:
            job = txn.state.index_jobs.get(str(job_id))
            if job is None:
                return
            job.retry_count += 1
            job.last_error = str(error or "")[:2000]
            job.updated_at = now
            job.lease_until = None
            if job.retry_count >= job.max_retries:
                job.status = "FAILED"
                job.next_run_at = None
            else:
                job.status = "PENDING"
                job.next_run_at = next_run_at

    def list_index_jobs(
        self,
        *,
        status: Optional[str] = None,
        entity_kind: Optional[str] = None,
        entity_id: Optional[str] = None,
        index_kind: Optional[str] = None,
        namespace: Optional[str] = "default",
        limit: int = 1000,
    ) -> list[IndexJobRow]:
        with self._lock:
            rows = list(self._state.index_jobs.values())
        out = []
        for job in rows:
            if status is not None and job.status != str(status):
                continue
            if entity_kind is not None and job.entity_kind != str(entity_kind):
                continue
            if entity_id is not None and job.entity_id != str(entity_id):
                continue
            if index_kind is not None and job.index_kind != str(index_kind):
                continue
            if namespace is not None and job.namespace != str(namespace):
                continue
            out.append(job)
        out.sort(key=lambda item: item.created_at)
        return [job.as_row() for job in out[: int(limit)]]

    def _lane_message_get_row(self, *, message_id: str) -> ProjectedLaneMessageRow | None:
        with self._lock:
            row = self._state.lane_messages.get(str(message_id))
            if row is None:
                return None
            return row.as_row()

    def _lane_message_insert_row(self, *, row: ProjectedLaneMessageRow) -> None:
        with self.transaction() as txn:
            txn.state.lane_messages[str(row.message_id)] = _ProjectedLaneMessageState(
                message_id=str(row.message_id),
                namespace=str(row.namespace),
                inbox_id=str(row.inbox_id),
                conversation_id=str(row.conversation_id),
                recipient_id=str(row.recipient_id),
                sender_id=str(row.sender_id),
                msg_type=str(row.msg_type),
                status=str(row.status),
                seq=int(row.seq),
                conversation_seq=int(row.conversation_seq),
                claimed_by=None if row.claimed_by is None else str(row.claimed_by),
                lease_until=None if row.lease_until is None else int(row.lease_until),
                retry_count=int(row.retry_count),
                created_at=int(row.created_at),
                available_at=int(row.available_at),
                run_id=None if row.run_id is None else str(row.run_id),
                step_id=None if row.step_id is None else str(row.step_id),
                correlation_id=None if row.correlation_id is None else str(row.correlation_id),
                payload_json=row.payload_json,
                error_json=row.error_json,
            )

    def _lane_message_update_row(self, *, row: ProjectedLaneMessageRow) -> None:
        self._lane_message_insert_row(row=row)

    def _lane_message_list_rows(
        self,
        *,
        namespace: str = "default",
        inbox_id: str | None = None,
        status: str | None = None,
        conversation_id: str | None = None,
    ) -> list[ProjectedLaneMessageRow]:
        with self._lock:
            rows = list(self._state.lane_messages.values())
        out: list[ProjectedLaneMessageRow] = []
        for row in rows:
            if row.namespace != str(namespace):
                continue
            if inbox_id is not None and row.inbox_id != str(inbox_id):
                continue
            if conversation_id is not None and row.conversation_id != str(conversation_id):
                continue
            if status is not None and row.status != str(status):
                continue
            out.append(row.as_row())
        return out

    def get_index_applied_fingerprint(
        self, *, namespace: str = "default", coalesce_key: str
    ) -> Optional[str]:
        with self._lock:
            row = self._state.applied_fingerprints.get((str(namespace), str(coalesce_key)))
        if row is None:
            return None
        return row[0]

    def set_index_applied_fingerprint(
        self,
        *,
        namespace: str = "default",
        coalesce_key: str,
        applied_fingerprint: Optional[str],
        last_job_id: Optional[str] = None,
    ) -> None:
        with self.transaction() as txn:
            txn.state.applied_fingerprints[(str(namespace), str(coalesce_key))] = (
                None if applied_fingerprint is None else str(applied_fingerprint),
                None if last_job_id is None else str(last_job_id),
                _now_epoch(),
            )

    def alloc_event_seq(self, namespace: str = "default") -> int:
        with self.transaction() as txn:
            next_seq = int(txn.state.namespace_next_seq.get(str(namespace), 1))
            txn.state.namespace_next_seq[str(namespace)] = next_seq + 1
            return next_seq

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
        _ = event_id
        seq = self.alloc_event_seq(namespace)
        with self.transaction() as txn:
            events = txn.state.entity_events.setdefault(str(namespace), [])
            events.append(
                _EntityEventRow(
                    seq=seq,
                    entity_kind=str(entity_kind),
                    entity_id=str(entity_id),
                    op=str(op),
                    payload_json=str(payload_json),
                )
            )
        return seq

    def iter_entity_events(
        self,
        *,
        namespace: str = "default",
        from_seq: int = 1,
        to_seq: int | None = None,
        batch_size: int = 500,
    ):
        next_seq = int(from_seq)
        while True:
            with self._lock:
                rows = [
                    event
                    for event in self._state.entity_events.get(str(namespace), [])
                    if int(event.seq) >= next_seq
                    and (to_seq is None or int(event.seq) <= int(to_seq))
                ]
            rows.sort(key=lambda item: item.seq)
            rows = rows[: int(batch_size)]
            if not rows:
                break
            for row in rows:
                yield (
                    int(row.seq),
                    str(row.entity_kind),
                    str(row.entity_id),
                    str(row.op),
                    str(row.payload_json),
                )
            next_seq = int(rows[-1].seq) + 1

    def prune_entity_events_after(self, *, namespace: str = "default", to_seq: int) -> int:
        with self.transaction() as txn:
            events = txn.state.entity_events.get(str(namespace), [])
            kept = [row for row in events if int(row.seq) <= int(to_seq)]
            deleted = len(events) - len(kept)
            txn.state.entity_events[str(namespace)] = kept
            next_seq = max([row.seq for row in kept], default=0) + 1
            txn.state.namespace_next_seq[str(namespace)] = max(
                int(txn.state.namespace_next_seq.get(str(namespace), 1)),
                int(next_seq),
            )
            return deleted

    def cursor_get(self, *, namespace: str, consumer: str) -> int:
        with self._lock:
            return int(self._state.replay_cursors.get((str(namespace), str(consumer)), 0))

    def cursor_set(self, *, namespace: str, consumer: str, last_seq: int) -> None:
        with self.transaction() as txn:
            txn.state.replay_cursors[(str(namespace), str(consumer))] = int(last_seq)

    def get_latest_entity_event_seq(self, *, namespace: str = "default") -> int:
        with self._lock:
            events = self._state.entity_events.get(str(namespace), [])
            return max((int(row.seq) for row in events), default=0)

    @staticmethod
    def _decode_named_projection_payload(raw_payload: Any) -> dict[str, Any]:
        payload = json.loads(str(raw_payload)) if raw_payload is not None else {}
        if not isinstance(payload, dict):
            raise ValueError("named projection payload must deserialize to a dict")
        return payload

    def get_named_projection(self, namespace: str, key: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._state.named_projections.get((str(namespace), str(key)))
            if row is None:
                return None
            return copy.deepcopy(row)

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
    ) -> None:
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        with self.transaction() as txn:
            txn.state.named_projections[(str(namespace), str(key))] = {
                "namespace": str(namespace),
                "key": str(key),
                "payload": copy.deepcopy(payload),
                "last_authoritative_seq": int(last_authoritative_seq),
                "last_materialized_seq": int(last_materialized_seq),
                "projection_schema_version": int(projection_schema_version),
                "materialization_status": str(materialization_status),
                "updated_at_ms": _now_ms(),
            }

    def list_named_projections(self, namespace: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = [
                copy.deepcopy(row)
                for (row_namespace, _key), row in self._state.named_projections.items()
                if row_namespace == str(namespace)
            ]
        rows.sort(key=lambda item: str(item.get("key") or ""))
        return rows

    def clear_named_projection(self, namespace: str, key: str) -> None:
        with self.transaction() as txn:
            txn.state.named_projections.pop((str(namespace), str(key)), None)

    def clear_projection_namespace(self, namespace: str) -> None:
        target = str(namespace)
        with self.transaction() as txn:
            for row_key in list(txn.state.named_projections.keys()):
                if row_key[0] == target:
                    txn.state.named_projections.pop(row_key, None)

    def get_workflow_design_projection(self, *, workflow_id: str) -> Optional[dict[str, Any]]:
        projection = self.get_named_projection("workflow_design", str(workflow_id))
        if projection is None:
            return None
        payload = projection.get("payload") or {}
        versions = payload.get("versions") or []
        dropped_ranges = payload.get("dropped_ranges") or []
        return {
            "workflow_id": str(workflow_id),
            "current_version": int(payload.get("current_version") or 0),
            "active_tip_version": int(payload.get("active_tip_version") or 0),
            "last_authoritative_seq": int(projection.get("last_authoritative_seq") or 0),
            "last_materialized_seq": int(projection.get("last_materialized_seq") or 0),
            "projection_schema_version": int(projection.get("projection_schema_version") or 1),
            "snapshot_schema_version": int(payload.get("snapshot_schema_version") or 1),
            "materialization_status": str(projection.get("materialization_status") or "ready"),
            "updated_at_ms": int(projection.get("updated_at_ms") or 0),
            "versions": [
                {
                    "version": int(item.get("version") or 0),
                    "prev_version": int(item.get("prev_version") or 0),
                    "target_seq": int(item.get("target_seq") or 0),
                    "created_at_ms": int(item.get("created_at_ms") or 0),
                }
                for item in versions
                if isinstance(item, dict)
            ],
            "dropped_ranges": [
                {
                    "start_seq": int(item.get("start_seq") or 0),
                    "end_seq": int(item.get("end_seq") or 0),
                    "start_version": int(item.get("start_version") or 0),
                    "end_version": int(item.get("end_version") or 0),
                }
                for item in dropped_ranges
                if isinstance(item, dict)
            ],
        }

    def replace_workflow_design_projection(
        self,
        *,
        workflow_id: str,
        head: dict[str, Any],
        versions: list[dict[str, Any]],
        dropped_ranges: list[dict[str, Any]],
    ) -> None:
        payload = {
            "current_version": int(head.get("current_version") or 0),
            "active_tip_version": int(head.get("active_tip_version") or 0),
            "snapshot_schema_version": int(head.get("snapshot_schema_version") or 1),
            "versions": [
                {
                    "version": int(item.get("version") or 0),
                    "prev_version": int(item.get("prev_version") or 0),
                    "target_seq": int(item.get("target_seq") or 0),
                    "created_at_ms": int(item.get("created_at_ms") or 0),
                }
                for item in versions
            ],
            "dropped_ranges": [
                {
                    "start_seq": int(item.get("start_seq") or 0),
                    "end_seq": int(item.get("end_seq") or 0),
                    "start_version": int(item.get("start_version") or 0),
                    "end_version": int(item.get("end_version") or 0),
                }
                for item in dropped_ranges
            ],
        }
        self.replace_named_projection(
            "workflow_design",
            str(workflow_id),
            payload,
            last_authoritative_seq=int(head.get("last_authoritative_seq") or 0),
            last_materialized_seq=int(head.get("last_materialized_seq") or 0),
            projection_schema_version=int(head.get("projection_schema_version") or 1),
            materialization_status=str(head.get("materialization_status") or "ready"),
        )

    def clear_workflow_design_projection(self, *, workflow_id: str) -> None:
        self.clear_named_projection("workflow_design", str(workflow_id))

    def put_workflow_design_snapshot(
        self,
        *,
        workflow_id: str,
        version: int,
        seq: int,
        payload_json: str,
        schema_version: int,
    ) -> None:
        with self.transaction() as txn:
            txn.state.workflow_snapshots[(str(workflow_id), int(version))] = _WorkflowSnapshotRow(
                workflow_id=str(workflow_id),
                version=int(version),
                seq=int(seq),
                payload_json=str(payload_json),
                schema_version=int(schema_version),
                created_at_ms=_now_ms(),
            )

    def get_workflow_design_snapshot(
        self,
        *,
        workflow_id: str,
        max_version: int,
        schema_version: int,
    ) -> Optional[dict[str, Any]]:
        with self._lock:
            rows = [
                row
                for row in self._state.workflow_snapshots.values()
                if row.workflow_id == str(workflow_id)
                and int(row.version) <= int(max_version)
                and int(row.schema_version) == int(schema_version)
            ]
        if not rows:
            return None
        row = sorted(rows, key=lambda item: item.version, reverse=True)[0]
        return {
            "workflow_id": str(row.workflow_id),
            "version": int(row.version),
            "seq": int(row.seq),
            "payload_json": str(row.payload_json),
            "schema_version": int(row.schema_version),
            "created_at_ms": int(row.created_at_ms),
        }

    def clear_workflow_design_snapshots(self, *, workflow_id: str) -> None:
        with self.transaction() as txn:
            for key in list(txn.state.workflow_snapshots.keys()):
                if key[0] == str(workflow_id):
                    txn.state.workflow_snapshots.pop(key, None)

    def put_workflow_design_delta(
        self,
        *,
        workflow_id: str,
        version: int,
        prev_version: int,
        target_seq: int,
        forward_json: str,
        inverse_json: str,
        schema_version: int,
    ) -> None:
        with self.transaction() as txn:
            txn.state.workflow_deltas[(str(workflow_id), int(version), int(schema_version))] = _WorkflowDeltaRow(
                workflow_id=str(workflow_id),
                version=int(version),
                prev_version=int(prev_version),
                target_seq=int(target_seq),
                forward_json=str(forward_json),
                inverse_json=str(inverse_json),
                schema_version=int(schema_version),
                created_at_ms=_now_ms(),
            )

    def get_workflow_design_delta(
        self,
        *,
        workflow_id: str,
        version: int,
        schema_version: int,
    ) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._state.workflow_deltas.get(
                (str(workflow_id), int(version), int(schema_version))
            )
        if row is None:
            return None
        return {
            "workflow_id": str(row.workflow_id),
            "version": int(row.version),
            "prev_version": int(row.prev_version),
            "target_seq": int(row.target_seq),
            "forward_json": str(row.forward_json),
            "inverse_json": str(row.inverse_json),
            "schema_version": int(row.schema_version),
            "created_at_ms": int(row.created_at_ms),
        }

    def clear_workflow_design_deltas(self, *, workflow_id: str) -> None:
        with self.transaction() as txn:
            for key in list(txn.state.workflow_deltas.keys()):
                if key[0] == str(workflow_id):
                    txn.state.workflow_deltas.pop(key, None)

    @staticmethod
    def _decode_run_json(raw: Any) -> Any:
        if raw in (None, ""):
            return None
        return json.loads(str(raw))

    def create_server_run(
        self,
        *,
        run_id: str,
        conversation_id: str,
        workflow_id: str,
        user_id: str | None,
        user_turn_node_id: str,
        status: str = "queued",
    ) -> None:
        now = _now_ms()
        with self.transaction() as txn:
            txn.state.server_runs[str(run_id)] = _ServerRunRow(
                run_id=str(run_id),
                conversation_id=str(conversation_id),
                workflow_id=str(workflow_id),
                user_id=None if user_id is None else str(user_id),
                user_turn_node_id=str(user_turn_node_id),
                assistant_turn_node_id=None,
                status=str(status),
                cancel_requested=False,
                result_json=None,
                error_json=None,
                created_at_ms=now,
                updated_at_ms=now,
                started_at_ms=None,
                finished_at_ms=None,
            )

    def get_server_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._state.server_runs.get(str(run_id))
        if row is None:
            return None
        status = str(row.status)
        return {
            "run_id": str(row.run_id),
            "conversation_id": str(row.conversation_id),
            "workflow_id": str(row.workflow_id),
            "user_id": row.user_id,
            "user_turn_node_id": row.user_turn_node_id,
            "assistant_turn_node_id": row.assistant_turn_node_id,
            "status": status,
            "cancel_requested": bool(row.cancel_requested),
            "result": self._decode_run_json(row.result_json),
            "error": self._decode_run_json(row.error_json),
            "created_at_ms": int(row.created_at_ms),
            "updated_at_ms": int(row.updated_at_ms),
            "started_at_ms": row.started_at_ms,
            "finished_at_ms": row.finished_at_ms,
            "terminal": status in {"succeeded", "failed", "cancelled"},
        }

    def list_server_run_events(
        self, run_id: str, *, after_seq: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = [
                row
                for row in self._state.server_run_events.get(str(run_id), [])
                if int(row.seq) > int(after_seq)
            ]
        rows.sort(key=lambda item: item.seq)
        rows = rows[: int(limit)]
        return [
            {
                "seq": int(row.seq),
                "run_id": str(row.run_id),
                "event_type": str(row.event_type),
                "payload": self._decode_run_json(row.payload_json) or {},
                "created_at_ms": int(row.created_at_ms),
            }
            for row in rows
        ]

    def append_server_run_event(
        self,
        run_id: str,
        event_type: str,
        payload_json: str,
    ) -> dict[str, Any]:
        now = _now_ms()
        with self.transaction() as txn:
            events = txn.state.server_run_events.setdefault(str(run_id), [])
            seq = (events[-1].seq + 1) if events else 1
            row = _ServerRunEventRow(
                seq=int(seq),
                run_id=str(run_id),
                event_type=str(event_type),
                payload_json=str(payload_json),
                created_at_ms=now,
            )
            events.append(row)
        return {
            "seq": int(seq),
            "run_id": str(run_id),
            "event_type": str(event_type),
            "payload": self._decode_run_json(payload_json) or {},
            "created_at_ms": now,
        }

    def update_server_run(
        self,
        *,
        run_id: str,
        status: str,
        assistant_turn_node_id: str | None,
        result_json: str | None,
        error_json: str | None,
        started_at_ms: int | None,
        finished_at_ms: int | None,
        cancel_requested: bool | None = None,
    ) -> None:
        now = _now_ms()
        with self.transaction() as txn:
            row = txn.state.server_runs.get(str(run_id))
            if row is None:
                return
            row.status = str(status)
            row.assistant_turn_node_id = assistant_turn_node_id
            row.result_json = result_json
            row.error_json = error_json
            row.started_at_ms = started_at_ms
            row.finished_at_ms = finished_at_ms
            if cancel_requested is not None:
                row.cancel_requested = bool(cancel_requested)
            row.updated_at_ms = now

    def request_server_run_cancel(self, *, run_id: str) -> None:
        now = _now_ms()
        with self.transaction() as txn:
            row = txn.state.server_runs.get(str(run_id))
            if row is None:
                return
            row.cancel_requested = True
            if row.status not in {"cancelled", "failed", "succeeded"}:
                row.status = "cancelling"
            row.updated_at_ms = now


__all__ = ["InMemoryMetaStore"]
