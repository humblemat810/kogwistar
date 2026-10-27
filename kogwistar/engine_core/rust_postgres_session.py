from __future__ import annotations

import contextvars
import math
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import NoReturn, cast

from kogwistar._rust_bridge import store_postgres
from kogwistar.json_types import JsonObject, JsonValue
from kogwistar.implementation_selection import transaction_authority_scope
from kogwistar.engine_core.rust_meta_sqlite import (
    RustEngineSQLite,
    _TransactionToken,
)
from kogwistar.engine_core.graph_mutation_contract import (
    validate_graph_mutation_batch,
)


class RustPostgresConnectionUnavailable(RuntimeError):
    """Raised when Rust authority code asks for a Python PostgreSQL writer."""


class _RustPostgresTransactionToken:
    def __init__(self, value: str, owner_id: str) -> None:
        self.value = value
        self.owner_id = owner_id

    def execute(self, *args: object, **kwargs: object) -> NoReturn:
        raise RustPostgresConnectionUnavailable(
            "raw SQL is unavailable inside a Rust PostgreSQL transaction; "
            "use a native capability method"
        )


def _valid_graph_projection_record(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    if (
        not isinstance(record.get("id"), str)
        or not record["id"]
        or not isinstance(record.get("document"), str)
        or not isinstance(record.get("metadata"), dict)
    ):
        return False
    embedding = record.get("embedding")
    if embedding is None:
        return True
    return isinstance(embedding, list) and all(
        _finite_projection_number(value)
        for value in embedding
    )


def _finite_projection_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _valid_graph_projection_match(match: object) -> bool:
    if not isinstance(match, dict):
        return False
    distance = match.get("distance")
    return (
        _valid_graph_projection_record(match.get("record"))
        and _finite_projection_number(distance)
    )


class RustPostgresSession:
    """Nest-safe Python facade over one native PostgreSQL transaction owner."""

    def __init__(self, *, dsn: str, schema: str) -> None:
        self.dsn = dsn
        self.schema = schema
        self.context_id = uuid.uuid4().hex
        self._closed = False
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
        self._session_lock = threading.RLock()
        self._active_sessions = 0

    def call(self, kind: str, **values: JsonValue) -> JsonValue:
        if self._closed and kind != "close_store":
            raise RustPostgresConnectionUnavailable("Rust PostgreSQL store is closed")
        with transaction_authority_scope("rust"):
            return store_postgres(
                dsn=self.dsn,
                schema=self.schema,
                context_id=self.context_id,
                operation={"kind": kind, **values},
                transaction_id=self._transaction_id.get(),
            )

    @property
    def transaction_active(self) -> bool:
        return self._transaction_id.get() is not None

    def ensure_initialized(self) -> None:
        self._closed = False
        self.call("ensure_schema")

    @contextmanager
    def execution_session(self) -> Iterator[RustPostgresSession]:
        """Keep the shared native store open until all overlapping scopes exit."""
        with self._session_lock:
            if self._active_sessions == 0:
                self.ensure_initialized()
            self._active_sessions += 1
        try:
            yield self
        finally:
            with self._session_lock:
                self._active_sessions -= 1
                if self._active_sessions == 0:
                    self.close()

    def graph_authority_capability(self) -> JsonObject:
        result = self.call("graph_authority_capability")
        if not isinstance(result, dict):
            raise RuntimeError("native graph authority capability returned invalid data")
        return cast(JsonObject, result)

    def close(self) -> None:
        with self._session_lock:
            if self._closed:
                return
            if self._active_sessions:
                raise RustPostgresConnectionUnavailable(
                    "cannot close Rust PostgreSQL store during an active execution session"
                )
            if self.transaction_active:
                raise RustPostgresConnectionUnavailable(
                    "cannot close Rust PostgreSQL store during an active transaction"
                )
            self.call("close_store")
            self._closed = True

    def connect(self) -> None:
        raise RustPostgresConnectionUnavailable(
            "raw Python PostgreSQL connection would create a second writer"
        )

    @contextmanager
    def transaction(self) -> Iterator[_RustPostgresTransactionToken]:
        with transaction_authority_scope("rust"):
            with self._transaction_impl() as token:
                yield token

    @contextmanager
    def _transaction_impl(self) -> Iterator[_RustPostgresTransactionToken]:
        current = self._transaction_id.get()
        if current is not None:
            depth_token = self._transaction_depth.set(
                self._transaction_depth.get() + 1
            )
            try:
                yield _RustPostgresTransactionToken(current, self.context_id)
            finally:
                self._transaction_depth.reset(depth_token)
            return

        transaction_id = uuid.uuid4().hex
        transaction_token = self._transaction_id.set(transaction_id)
        depth_token = self._transaction_depth.set(1)
        try:
            self.call("begin_transaction")
            try:
                yield _RustPostgresTransactionToken(transaction_id, self.context_id)
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
            or token.owner_id != self.context_id
        ):
            raise RustPostgresConnectionUnavailable(
                "stale Rust PostgreSQL transaction token"
            )


class RustEnginePostgresMetaStore(RustEngineSQLite):
    """Python meta-store surface backed by one native PostgreSQL session."""

    def __init__(self, *, dsn: str, schema: str) -> None:
        self._native_session = RustPostgresSession(dsn=dsn, schema=schema)
        self.schema = schema

    def _call(self, kind: str, **values: JsonValue) -> JsonValue:
        return self._native_session.call(kind, **values)

    @property
    def transaction_active(self) -> bool:
        return self._native_session.transaction_active

    @property
    def native_session(self) -> RustPostgresSession:
        """Expose the Rust worker handle without shadowing the context API."""
        return self._native_session

    @contextmanager
    def session(self) -> Iterator[RustEnginePostgresMetaStore]:
        """Match the SQLite facade's context-managed execution-session API."""
        with self._native_session.execution_session():
            yield self

    def apply_graph_mutation(self, **values: JsonValue) -> JsonObject:
        result = self._call("graph_mutation", **values)
        if not isinstance(result, dict):
            raise RuntimeError("native graph mutation returned a non-object")
        return result

    def apply_graph_mutations_atomic(self, batch: object) -> list[JsonObject]:
        """Apply a versioned graph batch and its index intents in one Rust call.

        This is an additive facade API. Existing single-mutation calls retain
        their wire shape; batch callers opt into the shared atomic contract.
        """
        validated = validate_graph_mutation_batch(batch)
        operations: list[JsonObject] = []
        for mutation in validated.mutations:
            operation: JsonObject = {
                "kind": "graph_mutation",
                "namespace": mutation.scope.namespace,
                "workspace_id": mutation.scope.workspace_id,
                "graph_space": mutation.scope.graph_space,
                "table": mutation.table,
                "entity_kind": mutation.entity_kind,
                "event_id": mutation.event_id,
                "op": mutation.op,
                "record": mutation.record.model_dump(mode="json"),
                "payload": mutation.payload,
                "embedding_dim": mutation.embedding_dim,
                "model_envelope": mutation.model_envelope,
            }
            operations.append(operation)
        for job in validated.index_jobs:
            operations.append(
                {
                    "kind": "enqueue_index_job",
                    "job_id": job.job_id,
                    "namespace": job.namespace,
                    "entity_kind": job.entity_kind,
                    "entity_id": job.entity_id,
                    "index_kind": job.index_kind,
                    "op": job.op,
                    "payload_json": job.payload_json,
                    "max_retries": job.max_retries,
                }
            )
        result = self.execute_batch(operations)
        if (
            not isinstance(result, list)
            or len(result) != len(operations)
            or not all(
                isinstance(item, dict)
                for item in result[: len(validated.mutations)]
            )
        ):
            raise RuntimeError("native graph mutation batch returned invalid results")
        return [
            cast(JsonObject, item)
            for item in result[: len(validated.mutations)]
        ]

    def graph_authority_capability(self) -> JsonObject:
        return self._native_session.graph_authority_capability()

    def apply_graph_metadata_patch_mutation(
        self, **values: JsonValue
    ) -> JsonObject | None:
        result = self._call("graph_metadata_patch_mutation", **values)
        if result is None:
            return None
        if not isinstance(result, dict):
            raise RuntimeError("native graph metadata patch returned a non-object")
        return result

    def apply_graph_delete_mutation(self, **values: JsonValue) -> JsonObject | None:
        result = self._call("graph_delete_mutation", **values)
        if result is None:
            return None
        if not isinstance(result, dict):
            raise RuntimeError("native graph delete returned a non-object")
        return result

    def upsert_graph_projection(self, **values: JsonValue) -> None:
        self._call("upsert_graph_projection", **values)

    def patch_graph_projection_metadata(self, **values: JsonValue) -> bool:
        return bool(self._call("patch_graph_projection_metadata", **values))

    def graph_projection_records(self, **values: JsonValue) -> list[JsonObject]:
        if values.get("for_update") and not self._native_session.transaction_active:
            raise RustPostgresConnectionUnavailable(
                "row-locking graph projection reads require an active Rust PostgreSQL UoW"
            )
        result = self._call("graph_projection_records", **values)
        if not isinstance(result, list) or not all(
            _valid_graph_projection_record(record) for record in result
        ):
            raise RuntimeError("native graph projection read returned invalid records")
        return [cast(JsonObject, record) for record in result]

    def graph_projection_vector_query(self, **values: JsonValue) -> list[JsonObject]:
        result = self._call("graph_projection_vector_query", **values)
        if not isinstance(result, list) or not all(
            _valid_graph_projection_match(match) for match in result
        ):
            raise RuntimeError("native graph vector query returned invalid matches")
        return [cast(JsonObject, match) for match in result]

    def graph_projection_vector_query_excluding_metadata(
        self, **values: JsonValue
    ) -> list[JsonObject]:
        result = self._call("graph_projection_vector_query", **values)
        if not isinstance(result, list) or not all(
            _valid_graph_projection_match(match) for match in result
        ):
            raise RuntimeError("native graph vector query returned invalid matches")
        return [cast(JsonObject, match) for match in result]

    def graph_projection_vector_query_batch(
        self, **values: JsonValue
    ) -> list[list[JsonObject]]:
        result = self._call("graph_projection_vector_query_batch", **values)
        if not isinstance(result, list) or not all(
            isinstance(row, list)
            and all(_valid_graph_projection_match(match) for match in row)
            for row in result
        ):
            raise RuntimeError("native batched graph vector query returned invalid matches")
        return [[cast(JsonObject, match) for match in row] for row in result]

    def graph_projection_vector_query_batch_excluding_metadata(
        self, **values: JsonValue
    ) -> list[list[JsonObject]]:
        result = self._call("graph_projection_vector_query_batch", **values)
        if not isinstance(result, list) or not all(
            isinstance(row, list)
            and all(_valid_graph_projection_match(match) for match in row)
            for row in result
        ):
            raise RuntimeError("native batched graph vector query returned invalid matches")
        return [[cast(JsonObject, match) for match in row] for row in result]

    def ensure_initialized(self) -> None:
        self._native_session.ensure_initialized()

    def close(self) -> None:
        """Release the native PostgreSQL worker and its connection pool."""
        self._native_session.close()

    def connect(self) -> None:
        self._native_session.connect()

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[_RustPostgresTransactionToken]:
        del immediate
        with self._native_session.transaction() as token:
            yield token

    def _require_token(self, conn: _TransactionToken) -> None:
        if not isinstance(conn, _RustPostgresTransactionToken):
            raise RustPostgresConnectionUnavailable("invalid Rust PostgreSQL transaction token")
        self._native_session.require_token(cast(_RustPostgresTransactionToken, conn))

    def next_scoped_seq(self, scope_id: str) -> int:
        return self.next_user_seq(scope_id)

    def next_global_seq_conn(self, conn: _TransactionToken) -> int:
        self._require_token(conn)
        return self.next_global_seq()

    def current_scoped_seq(self, scope_id: str) -> int:
        return self.current_user_seq(scope_id)

    def set_scoped_seq(self, scope_id: str, value: int) -> None:
        self.set_user_seq(scope_id, value)

    def next_user_seq_conn(
        self, conn: _TransactionToken, user_id: str
    ) -> int:
        self._require_token(conn)
        return self.next_user_seq(user_id)

    def set_user_seq_conn(
        self, conn: _TransactionToken, user_id: str, value: int
    ) -> None:
        self._require_token(conn)
        self.set_user_seq(user_id, value)

    def get_projected_lane_message(self, message_id: str) -> JsonValue:
        return self._call("get_projected_lane_message", message_id=message_id)

    def dead_letter_projected_lane_message(self, **values: JsonValue) -> None:
        self._call("dead_letter_projected_lane_message", **values)


__all__ = [
    "RustEnginePostgresMetaStore",
    "RustPostgresConnectionUnavailable",
    "RustPostgresSession",
]
