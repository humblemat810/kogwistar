from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterator
import uuid

from kogwistar._rust_bridge import store_sqlite
from kogwistar.engine_core.engine_sqlite import IndexJobRow, ProjectedLaneMessageSqlRow


_INDEX_JOB_FIELDS = {field.name for field in fields(IndexJobRow)}
_LANE_MESSAGE_FIELDS = {field.name for field in fields(ProjectedLaneMessageSqlRow)}


def _index_job_row(value: dict[str, Any]) -> IndexJobRow:
    return IndexJobRow(**{key: item for key, item in value.items() if key in _INDEX_JOB_FIELDS})


def _lane_message_row(value: dict[str, Any]) -> ProjectedLaneMessageSqlRow:
    return ProjectedLaneMessageSqlRow(
        **{key: item for key, item in value.items() if key in _LANE_MESSAGE_FIELDS}
    )


class RustSQLiteConnectionUnavailable(RuntimeError):
    """Raised when Rust authority code asks for a raw Python SQLite writer."""


class _RustTransactionToken:
    def __init__(self, value: str) -> None:
        self.value = value

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise RustSQLiteConnectionUnavailable(
            "raw SQL is unavailable while KOGWISTAR_IMPL_META_STORE=rust; "
            "use a meta-store capability method"
        )


class RustEngineSQLite:
    """Python-compatible meta facade with Rust as sole SQLite writer."""

    def __init__(self, persistent_directory: Path, filename: str = "engine.db") -> None:
        self.persistent_directory = Path(persistent_directory)
        self.db_path = self.persistent_directory / filename
        self._transaction_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"kogwistar_rust_sqlite_transaction_{id(self)}", default=None
        )

    def _call(self, kind: str, **values: Any) -> Any:
        return store_sqlite(
            path=self.db_path,
            operation={"kind": kind, **values},
            transaction_id=self._transaction_id.get(),
        )

    def ensure_initialized(self) -> None:
        self.persistent_directory.mkdir(parents=True, exist_ok=True)
        self._call("open_init")

    def connect(self) -> None:
        raise RustSQLiteConnectionUnavailable(
            "raw Python SQLite connection would create a second writer while "
            "KOGWISTAR_IMPL_META_STORE=rust"
        )

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[_RustTransactionToken]:
        del immediate
        current = self._transaction_id.get()
        if current is not None:
            yield _RustTransactionToken(current)
            return
        transaction_id = uuid.uuid4().hex
        token = self._transaction_id.set(transaction_id)
        try:
            self._call("begin_transaction")
            try:
                yield _RustTransactionToken(transaction_id)
            except BaseException:
                self._call("rollback_transaction")
                raise
            else:
                self._call("commit_transaction")
        finally:
            self._transaction_id.reset(token)

    def next_global_seq(self) -> int:
        return int(self._call("next_global_seq"))

    def next_global_seq_conn(self, conn: _RustTransactionToken) -> int:
        self._require_token(conn)
        return self.next_global_seq()

    def current_global_seq(self) -> int:
        return int(self._call("current_global_seq"))

    def next_user_seq(self, user_id: str) -> int:
        return int(self._call("next_user_seq", user_id=user_id))

    def next_scoped_seq(self, scope_id: str) -> int:
        return int(self._call("next_scoped_seq", scope_id=scope_id))

    def next_user_seq_conn(self, conn: _RustTransactionToken, user_id: str) -> int:
        self._require_token(conn)
        return self.next_user_seq(user_id)

    def current_user_seq(self, user_id: str) -> int:
        return int(self._call("current_user_seq", user_id=user_id))

    def current_scoped_seq(self, scope_id: str) -> int:
        return int(self._call("current_scoped_seq", scope_id=scope_id))

    def set_user_seq(self, user_id: str, value: int) -> None:
        self._call("set_user_seq", user_id=user_id, value=int(value))

    def set_scoped_seq(self, scope_id: str, value: int) -> None:
        self._call("set_scoped_seq", scope_id=scope_id, value=int(value))

    def set_user_seq_conn(
        self, conn: _RustTransactionToken, user_id: str, value: int
    ) -> None:
        self._require_token(conn)
        self.set_user_seq(user_id, value)

    def _require_token(self, conn: _RustTransactionToken) -> None:
        active = self._transaction_id.get()
        if not isinstance(conn, _RustTransactionToken) or conn.value != active:
            raise RustSQLiteConnectionUnavailable("stale Rust SQLite transaction token")

    def enqueue_index_job(self, **values: Any) -> str:
        return str(self._call("enqueue_index_job", **values))

    def claim_index_jobs(
        self, *, limit: int = 50, lease_seconds: int = 60, namespace: str | None = "default"
    ) -> list[IndexJobRow]:
        rows = self._call(
            "claim_index_jobs", limit=limit, lease_seconds=lease_seconds, namespace=namespace
        )
        return [_index_job_row(row) for row in rows]

    def mark_index_job_done(self, job_id: str, *, claim_token: str | None = None) -> bool:
        return bool(self._call("mark_index_job_done", job_id=job_id, claim_token=claim_token))

    def mark_index_job_failed(
        self,
        job_id: str,
        error: str,
        *,
        final: bool = True,
        claim_token: str | None = None,
    ) -> None:
        self._call(
            "mark_index_job_failed",
            job_id=job_id,
            error=error,
            final=final,
            claim_token=claim_token,
        )

    def bump_retry_and_requeue(
        self,
        job_id: str,
        error: str,
        *,
        next_run_at_seconds: int,
        claim_token: str | None = None,
    ) -> None:
        self._call(
            "bump_retry_and_requeue",
            job_id=job_id,
            error=error,
            next_run_at_seconds=next_run_at_seconds,
            claim_token=claim_token,
        )

    def renew_index_job_lease(
        self, job_id: str, *, claim_token: str, lease_seconds: int
    ) -> bool:
        return bool(
            self._call(
                "renew_index_job_lease",
                job_id=job_id,
                claim_token=claim_token,
                lease_seconds=lease_seconds,
            )
        )

    def requeue_index_job_at_tail(
        self,
        job_id: str,
        *,
        payload_json: str,
        delay_seconds: int = 0,
        claim_token: str | None = None,
    ) -> None:
        self._call(
            "requeue_index_job_at_tail",
            job_id=job_id,
            payload_json=payload_json,
            delay_seconds=delay_seconds,
            claim_token=claim_token,
        )

    def list_index_jobs(
        self,
        *,
        status: str | None = None,
        entity_kind: str | None = None,
        entity_id: str | None = None,
        index_kind: str | None = None,
        namespace: str | None = "default",
        limit: int = 1000,
    ) -> list[IndexJobRow]:
        rows = self._call(
            "list_index_jobs",
            status=status,
            entity_kind=entity_kind,
            entity_id=entity_id,
            index_kind=index_kind,
            namespace=namespace,
            limit=limit,
        )
        return [_index_job_row(row) for row in rows]

    def get_index_applied_fingerprint(
        self, *, namespace: str = "default", coalesce_key: str
    ) -> str | None:
        value = self._call(
            "get_index_applied_fingerprint",
            namespace=namespace,
            coalesce_key=coalesce_key,
        )
        return None if value is None else str(value)

    def set_index_applied_fingerprint(
        self,
        *,
        namespace: str = "default",
        coalesce_key: str,
        applied_fingerprint: str | None,
        last_job_id: str | None = None,
    ) -> None:
        self._call(
            "set_index_applied_fingerprint",
            namespace=namespace,
            coalesce_key=coalesce_key,
            applied_fingerprint=applied_fingerprint,
            last_job_id=last_job_id,
        )

    def alloc_event_seq(self, namespace: str = "default") -> int:
        return int(self._call("alloc_event_seq", namespace=namespace))

    def append_entity_event(self, **values: Any) -> int:
        return int(self._call("raw_append", **values)["seq"])

    def iter_entity_events(
        self,
        *,
        namespace: str = "default",
        from_seq: int = 1,
        to_seq: int | None = None,
        batch_size: int = 500,
    ) -> Iterator[tuple[int, str, str, str, str]]:
        after_seq = int(from_seq) - 1
        while True:
            rows = self._call(
                "exclusive_raw_replay",
                namespace=namespace,
                after_seq=after_seq,
                limit=batch_size,
            )
            if to_seq is not None:
                rows = [row for row in rows if int(row["seq"]) <= int(to_seq)]
            if not rows:
                return
            for row in rows:
                yield (
                    int(row["seq"]),
                    str(row["entity_kind"]),
                    str(row["entity_id"]),
                    str(row["op"]),
                    str(row["payload_json"]),
                )
            after_seq = int(rows[-1]["seq"])
            if len(rows) < batch_size or (to_seq is not None and after_seq >= to_seq):
                return

    def prune_entity_events_after(self, *, namespace: str = "default", to_seq: int) -> int:
        return int(
            self._call("prune_entity_events_after", namespace=namespace, to_seq=to_seq)
        )

    def cursor_get(self, *, namespace: str, consumer: str) -> int:
        return int(self._call("cursor_get", namespace=namespace, consumer=consumer)["last_seq"])

    def cursor_set(self, *, namespace: str, consumer: str, last_seq: int) -> None:
        self._call(
            "legacy_cursor_set",
            namespace=namespace,
            consumer=consumer,
            last_seq=last_seq,
        )

    def get_latest_entity_event_seq(self, *, namespace: str = "default") -> int:
        return int(self._call("latest_retained_event_seq", namespace=namespace))

    def get_named_projection(self, namespace: str, key: str) -> dict[str, Any] | None:
        return self._call("get_named_projection", namespace=namespace, key=key)

    def replace_named_projection(
        self, namespace: str, key: str, payload: dict[str, Any], **values: Any
    ) -> None:
        self._call(
            "replace_named_projection",
            namespace=namespace,
            key=key,
            payload=payload,
            **values,
        )

    def compare_and_swap_named_projection(
        self, namespace: str, key: str, payload: dict[str, Any], **values: Any
    ) -> bool:
        return bool(
            self._call(
                "compare_and_swap_named_projection",
                namespace=namespace,
                key=key,
                payload=payload,
                **values,
            )
        )

    def list_named_projections(self, namespace: str) -> list[dict[str, Any]]:
        return list(self._call("list_named_projections", namespace=namespace))

    def clear_named_projection(self, namespace: str, key: str) -> None:
        self._call("clear_named_projection", namespace=namespace, key=key)

    def clear_projection_namespace(self, namespace: str) -> None:
        self._call("clear_projection_namespace", namespace=namespace)

    def get_workflow_design_projection(self, *, workflow_id: str) -> dict[str, Any] | None:
        projection = self.get_named_projection("workflow_design", workflow_id)
        if projection is None:
            return None
        payload = projection.get("payload") or {}
        return {
            "workflow_id": workflow_id,
            "current_version": int(payload.get("current_version") or 0),
            "active_tip_version": int(payload.get("active_tip_version") or 0),
            "last_authoritative_seq": int(projection.get("last_authoritative_seq") or 0),
            "last_materialized_seq": int(projection.get("last_materialized_seq") or 0),
            "projection_schema_version": int(projection.get("projection_schema_version") or 1),
            "snapshot_schema_version": int(payload.get("snapshot_schema_version") or 1),
            "materialization_status": str(projection.get("materialization_status") or "ready"),
            "versions": list(payload.get("versions") or []),
            "dropped_ranges": list(payload.get("dropped_ranges") or []),
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
            "workflow_id": workflow_id,
            "current_version": int(head.get("current_version") or 0),
            "active_tip_version": int(head.get("active_tip_version") or 0),
            "snapshot_schema_version": int(head.get("snapshot_schema_version") or 1),
            "versions": versions,
            "dropped_ranges": dropped_ranges,
        }
        self.replace_named_projection(
            "workflow_design",
            workflow_id,
            payload,
            last_authoritative_seq=int(head.get("last_authoritative_seq") or 0),
            last_materialized_seq=int(head.get("last_materialized_seq") or 0),
            projection_schema_version=int(head.get("projection_schema_version") or 1),
            materialization_status=str(head.get("materialization_status") or "ready"),
        )

    def clear_workflow_design_projection(self, *, workflow_id: str) -> None:
        self.clear_named_projection("workflow_design", workflow_id)

    def put_workflow_design_snapshot(self, **values: Any) -> None:
        self._call("put_workflow_design_snapshot", **values)

    def get_workflow_design_snapshot(self, **values: Any) -> dict[str, Any] | None:
        return self._call("get_workflow_design_snapshot", **values)

    def clear_workflow_design_snapshots(self, *, workflow_id: str) -> None:
        self._call("clear_workflow_design_snapshots", workflow_id=workflow_id)

    def put_workflow_design_delta(self, **values: Any) -> None:
        self._call("put_workflow_design_delta", **values)

    def get_workflow_design_delta(self, **values: Any) -> dict[str, Any] | None:
        return self._call("get_workflow_design_delta", **values)

    def clear_workflow_design_deltas(self, *, workflow_id: str) -> None:
        self._call("clear_workflow_design_deltas", workflow_id=workflow_id)

    def create_server_run(self, **values: Any) -> None:
        self._call("create_server_run", **values)

    def get_server_run(self, run_id: str) -> dict[str, Any] | None:
        return self._call("get_server_run", run_id=run_id)

    def list_server_runs(self, **values: Any) -> list[dict[str, Any]]:
        return list(self._call("list_server_runs", **values))

    def list_server_run_events(
        self, run_id: str, *, after_seq: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]:
        return list(
            self._call(
                "list_server_run_events",
                run_id=run_id,
                after_seq=after_seq,
                limit=limit,
            )
        )

    def append_server_run_event(
        self, run_id: str, event_type: str, payload_json: str
    ) -> dict[str, Any]:
        return dict(
            self._call(
                "append_server_run_event",
                run_id=run_id,
                event_type=event_type,
                payload_json=payload_json,
            )
        )

    def update_server_run(self, **values: Any) -> None:
        self._call("update_server_run", **values)

    def request_server_run_cancel(self, *, run_id: str) -> None:
        self._call("request_server_run_cancel", run_id=run_id)

    def project_lane_message(self, **values: Any) -> None:
        self._call("project_lane_message", **values)

    def update_projected_lane_message_status(self, **values: Any) -> None:
        self._call("update_projected_lane_message_status", **values)

    def update_projected_lane_message_links(self, **values: Any) -> None:
        self._call("update_projected_lane_message_links", **values)

    def claim_projected_lane_messages(self, **values: Any) -> list[ProjectedLaneMessageSqlRow]:
        return [
            _lane_message_row(row)
            for row in self._call("claim_projected_lane_messages", **values)
        ]

    def ack_projected_lane_message(self, **values: Any) -> None:
        self._call("ack_projected_lane_message", **values)

    def requeue_projected_lane_message(self, **values: Any) -> None:
        self._call("requeue_projected_lane_message", **values)

    def clear_projected_lane_messages(self, namespace: str) -> int:
        return int(self._call("clear_projected_lane_messages", namespace=namespace))

    def list_projected_lane_messages(self, **values: Any) -> list[ProjectedLaneMessageSqlRow]:
        return [
            _lane_message_row(row)
            for row in self._call("list_projected_lane_messages", **values)
        ]


def build_sqlite_meta_store(
    persistent_directory: Path, filename: str = "engine.db"
) -> RustEngineSQLite | Any:
    from kogwistar._rust_bridge import meta_store_implementation_mode
    from kogwistar.engine_core.engine_sqlite import EngineSQLite

    mode = meta_store_implementation_mode()
    if mode == "rust":
        return RustEngineSQLite(persistent_directory, filename)
    return EngineSQLite(persistent_directory, filename)


__all__ = ["RustEngineSQLite", "RustSQLiteConnectionUnavailable", "build_sqlite_meta_store"]
