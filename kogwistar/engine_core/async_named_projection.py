"""Native async durable named-projection storage primitives."""

from __future__ import annotations

import json
import re
import time
import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class AsyncPostgresNamedProjectionStore:
    """Native async implementation of ``AsyncNamedProjectionStore``.

    This is deliberately only the named-projection surface. It does not turn
    the synchronous engine into an async facade, and it keeps absent-row CAS
    serialized with the same PostgreSQL advisory-lock rule as the sync store.
    """

    def __init__(self, engine: "AsyncEngine", *, schema: str = "public") -> None:
        if not callable(getattr(engine, "begin", None)) or not callable(
            getattr(engine, "connect", None)
        ):
            raise TypeError("AsyncPostgresNamedProjectionStore requires an async engine")
        if not _SCHEMA_RE.match(schema):
            raise ValueError(f"invalid schema: {schema!r}")
        self.engine = engine
        self.schema = schema
        self._table = f"{schema}.named_projections"

    async def ensure_initialized(self) -> None:
        """Create only this adapter's table when the wider meta store is absent."""

        import sqlalchemy as sa

        async with self.engine.begin() as conn:
            await conn.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {self.schema}"))
            await conn.execute(
                sa.text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        namespace TEXT NOT NULL,
                        key TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        last_authoritative_seq BIGINT NOT NULL,
                        last_materialized_seq BIGINT NOT NULL,
                        projection_schema_version INTEGER NOT NULL,
                        materialization_status TEXT NOT NULL,
                        updated_at_ms BIGINT NOT NULL,
                        PRIMARY KEY(namespace, key)
                    )
                    """
                )
            )

    @staticmethod
    def _decode_payload(raw: Any) -> dict[str, Any]:
        value = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(value, dict):
            raise ValueError("named projection payload must decode to an object")
        return value

    @classmethod
    def _row(cls, row: Any) -> dict[str, Any]:
        return {
            "namespace": str(row[0]),
            "key": str(row[1]),
            "payload": cls._decode_payload(row[2]),
            "last_authoritative_seq": int(row[3]),
            "last_materialized_seq": int(row[4]),
            "projection_schema_version": int(row[5]),
            "materialization_status": str(row[6]),
            "updated_at_ms": int(row[7]),
        }

    async def get_named_projection(
        self, namespace: str, key: str
    ) -> dict[str, Any] | None:
        import sqlalchemy as sa

        async with self.engine.connect() as conn:
            result = await conn.execute(
                sa.text(
                    f"""
                    SELECT namespace, key, payload_json,
                           last_authoritative_seq, last_materialized_seq,
                           projection_schema_version, materialization_status,
                           updated_at_ms
                    FROM {self._table}
                    WHERE namespace = :namespace AND key = :key
                    """
                ),
                {"namespace": str(namespace), "key": str(key)},
            )
            row = result.first()
        return None if row is None else self._row(row)

    async def list_named_projections(self, namespace: str) -> list[dict[str, Any]]:
        import sqlalchemy as sa

        async with self.engine.connect() as conn:
            result = await conn.execute(
                sa.text(
                    f"""
                    SELECT namespace, key, payload_json,
                           last_authoritative_seq, last_materialized_seq,
                           projection_schema_version, materialization_status,
                           updated_at_ms
                    FROM {self._table}
                    WHERE namespace = :namespace
                    ORDER BY key ASC
                    """
                ),
                {"namespace": str(namespace)},
            )
            rows = result.fetchall()
        return [self._row(row) for row in rows]

    async def compare_and_swap_named_projection(
        self,
        namespace: str,
        key: str,
        payload: dict[str, Any],
        **values: Any,
    ) -> bool:
        update = {
            "namespace": str(namespace),
            "key": str(key),
            "payload": payload,
            **values,
        }
        return await self.compare_and_swap_named_projections([update])

    async def compare_and_swap_named_projections(
        self, updates: list[dict[str, Any]]
    ) -> bool:
        if not updates:
            return True
        rows = sorted(updates, key=lambda item: (str(item["namespace"]), str(item["key"])))
        identities = {(str(item["namespace"]), str(item["key"])) for item in rows}
        if len(identities) != len(rows):
            raise ValueError("duplicate named projection key")
        import sqlalchemy as sa

        now = int(time.time() * 1000)
        async with self.engine.begin() as conn:
            for item in rows:
                lock_key = json.dumps(
                    [str(item["namespace"]), str(item["key"])],
                    separators=(",", ":"),
                )
                await conn.execute(
                    sa.text(
                        "SELECT pg_advisory_xact_lock(hashtextextended(:lock_key, 0))"
                    ),
                    {"lock_key": lock_key},
                )
                result = await conn.execute(
                    sa.text(
                        f"""
                        SELECT last_authoritative_seq, last_materialized_seq
                        FROM {self._table}
                        WHERE namespace = :namespace AND key = :key
                        FOR UPDATE
                        """
                    ),
                    {"namespace": str(item["namespace"]), "key": str(item["key"])},
                )
                current = result.first()
                expected_a = item.get("expected_last_authoritative_seq")
                expected_m = item.get("expected_last_materialized_seq")
                if expected_a is None and expected_m is None:
                    if current is not None:
                        return False
                elif (
                    current is None
                    or int(current[0]) != int(expected_a)
                    or int(current[1]) != int(expected_m)
                ):
                    return False
            for item in rows:
                await conn.execute(
                    sa.text(
                        f"""
                        INSERT INTO {self._table}(
                            namespace, key, payload_json,
                            last_authoritative_seq, last_materialized_seq,
                            projection_schema_version, materialization_status,
                            updated_at_ms
                        ) VALUES (
                            :namespace, :key, :payload_json, :authoritative,
                            :materialized, :schema_version, :status, :updated_at_ms
                        )
                        ON CONFLICT(namespace, key) DO UPDATE SET
                            payload_json = EXCLUDED.payload_json,
                            last_authoritative_seq = EXCLUDED.last_authoritative_seq,
                            last_materialized_seq = EXCLUDED.last_materialized_seq,
                            projection_schema_version = EXCLUDED.projection_schema_version,
                            materialization_status = EXCLUDED.materialization_status,
                            updated_at_ms = EXCLUDED.updated_at_ms
                        """
                    ),
                    {
                        "namespace": str(item["namespace"]),
                        "key": str(item["key"]),
                        "payload_json": json.dumps(
                            item["payload"], sort_keys=True, separators=(",", ":")
                        ),
                        "authoritative": int(item.get("last_authoritative_seq", 0)),
                        "materialized": int(item.get("last_materialized_seq", 0)),
                        "schema_version": int(item.get("projection_schema_version", 1)),
                        "status": str(item.get("materialization_status", "ready")),
                        "updated_at_ms": now,
                    },
                )
        return True


class AsyncSQLiteNamedProjectionStore:
    """Async contract adapter for the existing synchronous SQLite provider.

    SQLite operations run in ``asyncio.to_thread`` because Python's bundled
    sqlite3 driver is synchronous. The underlying implementation retains its
    ``BEGIN IMMEDIATE`` batch-CAS transaction; this is not a pretend native
    async SQLite driver.
    """

    def __init__(self, metadata: Any) -> None:
        required = (
            "get_named_projection",
            "compare_and_swap_named_projection",
            "compare_and_swap_named_projections",
            "list_named_projections",
        )
        if any(not callable(getattr(metadata, name, None)) for name in required):
            raise TypeError("metadata must implement named projection operations")
        self.metadata = metadata

    async def ensure_initialized(self) -> None:
        initializer = getattr(self.metadata, "ensure_initialized", None)
        if callable(initializer):
            await asyncio.to_thread(initializer)

    async def get_named_projection(
        self, namespace: str, key: str
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(
            self.metadata.get_named_projection, namespace, key
        )

    async def list_named_projections(self, namespace: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self.metadata.list_named_projections, namespace
        )

    async def compare_and_swap_named_projection(
        self,
        namespace: str,
        key: str,
        payload: dict[str, Any],
        **values: Any,
    ) -> bool:
        return await asyncio.to_thread(
            self.metadata.compare_and_swap_named_projection,
            namespace,
            key,
            payload,
            **values,
        )

    async def compare_and_swap_named_projections(
        self, updates: list[dict[str, Any]]
    ) -> bool:
        return await asyncio.to_thread(
            self.metadata.compare_and_swap_named_projections, updates
        )


__all__ = [
    "AsyncPostgresNamedProjectionStore",
    "AsyncSQLiteNamedProjectionStore",
]
