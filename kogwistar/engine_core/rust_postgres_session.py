from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator
import uuid

from kogwistar._rust_bridge import store_postgres
from kogwistar.engine_core.rust_meta_sqlite import RustEngineSQLite


class RustPostgresConnectionUnavailable(RuntimeError):
    """Raised when Rust authority code asks for a Python PostgreSQL writer."""


class _RustPostgresTransactionToken:
    def __init__(self, value: str) -> None:
        self.value = value

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise RustPostgresConnectionUnavailable(
            "raw SQL is unavailable inside a Rust PostgreSQL transaction; "
            "use a native capability method"
        )


class RustPostgresSession:
    """Nest-safe Python facade over one native PostgreSQL transaction owner."""

    def __init__(self, *, dsn: str, schema: str) -> None:
        self.dsn = dsn
        self.schema = schema
        self._transaction_id: contextvars.ContextVar[str | None] = (
            contextvars.ContextVar(
                f"kogwistar_rust_postgres_transaction_{id(self)}", default=None
            )
        )
        self._transaction_depth: contextvars.ContextVar[int] = (
            contextvars.ContextVar(
                f"kogwistar_rust_postgres_transaction_depth_{id(self)}", default=0
            )
        )

    def call(self, kind: str, **values: Any) -> Any:
        return store_postgres(
            dsn=self.dsn,
            schema=self.schema,
            operation={"kind": kind, **values},
            transaction_id=self._transaction_id.get(),
        )

    @property
    def transaction_active(self) -> bool:
        return self._transaction_id.get() is not None

    def ensure_initialized(self) -> None:
        self.call("ensure_schema")

    def connect(self) -> None:
        raise RustPostgresConnectionUnavailable(
            "raw Python PostgreSQL connection would create a second writer"
        )

    @contextmanager
    def transaction(self) -> Iterator[_RustPostgresTransactionToken]:
        current = self._transaction_id.get()
        if current is not None:
            depth_token = self._transaction_depth.set(
                self._transaction_depth.get() + 1
            )
            try:
                yield _RustPostgresTransactionToken(current)
            finally:
                self._transaction_depth.reset(depth_token)
            return

        transaction_id = uuid.uuid4().hex
        transaction_token = self._transaction_id.set(transaction_id)
        depth_token = self._transaction_depth.set(1)
        try:
            self.call("begin_transaction")
            try:
                yield _RustPostgresTransactionToken(transaction_id)
            except BaseException:
                self.call("rollback_transaction")
                raise
            else:
                self.call("commit_transaction")
        finally:
            self._transaction_depth.reset(depth_token)
            self._transaction_id.reset(transaction_token)

    def require_token(self, token: _RustPostgresTransactionToken) -> None:
        active = self._transaction_id.get()
        if (
            not isinstance(token, _RustPostgresTransactionToken)
            or token.value != active
        ):
            raise RustPostgresConnectionUnavailable(
                "stale Rust PostgreSQL transaction token"
            )


class RustEnginePostgresMetaStore(RustEngineSQLite):
    """Python meta-store surface backed by one native PostgreSQL session."""

    def __init__(self, *, dsn: str, schema: str) -> None:
        self.session = RustPostgresSession(dsn=dsn, schema=schema)
        self.schema = schema

    def _call(self, kind: str, **values: Any) -> Any:
        return self.session.call(kind, **values)

    @property
    def transaction_active(self) -> bool:
        return self.session.transaction_active

    def apply_graph_mutation(self, **values: Any) -> dict[str, Any]:
        return dict(self._call("graph_mutation", **values))

    def apply_graph_metadata_patch_mutation(
        self, **values: Any
    ) -> dict[str, Any] | None:
        result = self._call("graph_metadata_patch_mutation", **values)
        return None if result is None else dict(result)

    def apply_graph_delete_mutation(self, **values: Any) -> dict[str, Any] | None:
        result = self._call("graph_delete_mutation", **values)
        return None if result is None else dict(result)

    def upsert_graph_projection(self, **values: Any) -> None:
        self._call("upsert_graph_projection", **values)

    def patch_graph_projection_metadata(self, **values: Any) -> bool:
        return bool(self._call("patch_graph_projection_metadata", **values))

    def graph_projection_records(self, **values: Any) -> list[dict[str, Any]]:
        result = self._call("graph_projection_records", **values)
        if not isinstance(result, list) or not all(
            isinstance(record, dict) for record in result
        ):
            raise RuntimeError("native graph projection read returned invalid records")
        return [dict(record) for record in result]

    def graph_projection_vector_query(self, **values: Any) -> list[dict[str, Any]]:
        result = self._call("graph_projection_vector_query", **values)
        if not isinstance(result, list) or not all(
            isinstance(match, dict) for match in result
        ):
            raise RuntimeError("native graph vector query returned invalid matches")
        return [dict(match) for match in result]

    def ensure_initialized(self) -> None:
        self.session.ensure_initialized()

    def close(self) -> None:
        """Rust PostgreSQL calls own their pool lifetime per operation.

        Unlike the SQLite bridge, the PostgreSQL ABI has no cached-handle
        ``close`` operation. Keep engine shutdown explicit and harmless.
        """
        return None

    def connect(self) -> None:
        self.session.connect()

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[_RustPostgresTransactionToken]:
        del immediate
        with self.session.transaction() as token:
            yield token

    def _require_token(self, conn: _RustPostgresTransactionToken) -> None:
        self.session.require_token(conn)

    def next_scoped_seq(self, scope_id: str) -> int:
        return self.next_user_seq(scope_id)

    def next_global_seq_conn(self, conn: _RustPostgresTransactionToken) -> int:
        self._require_token(conn)
        return self.next_global_seq()

    def current_scoped_seq(self, scope_id: str) -> int:
        return self.current_user_seq(scope_id)

    def set_scoped_seq(self, scope_id: str, value: int) -> None:
        self.set_user_seq(scope_id, value)

    def next_user_seq_conn(
        self, conn: _RustPostgresTransactionToken, user_id: str
    ) -> int:
        self._require_token(conn)
        return self.next_user_seq(user_id)

    def set_user_seq_conn(
        self, conn: _RustPostgresTransactionToken, user_id: str, value: int
    ) -> None:
        self._require_token(conn)
        self.set_user_seq(user_id, value)

    def get_projected_lane_message(self, message_id: str) -> Any:
        return self._call("get_projected_lane_message", message_id=message_id)

    def dead_letter_projected_lane_message(self, **values: Any) -> None:
        self._call("dead_letter_projected_lane_message", **values)


__all__ = [
    "RustEnginePostgresMetaStore",
    "RustPostgresConnectionUnavailable",
    "RustPostgresSession",
]
