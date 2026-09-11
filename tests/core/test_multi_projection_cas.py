from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import uuid

import pytest

from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.in_memory_meta import InMemoryMetaStore
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore


def _row(namespace: str, key: str, value: int, *, expected: int | None = None) -> dict:
    row = {
        "namespace": namespace,
        "key": key,
        "payload": {"value": value},
        "last_authoritative_seq": (value + 1),
        "last_materialized_seq": (value + 1),
        "projection_schema_version": 1,
        "materialization_status": "ready",
    }
    row["expected_last_authoritative_seq"] = expected
    row["expected_last_materialized_seq"] = expected
    return row


def _seed(store, namespace: str) -> None:
    assert store.compare_and_swap_named_projections([_row(namespace, "a", 0), _row(namespace, "b", 0)])


def _snapshot(store, namespace: str) -> dict[str, dict]:
    return {row["key"]: row for row in store.list_named_projections(namespace)}


@pytest.fixture(params=["memory", "sqlite"])
def local_store(request, tmp_path):
    if request.param == "memory":
        return InMemoryMetaStore()
    store = EngineSQLite(tmp_path / "meta")
    store.ensure_initialized()
    return store


def test_multi_projection_cas_is_all_or_nothing(local_store):
    namespace = f"cas-test-{uuid.uuid4().hex}"
    _seed(local_store, namespace)
    before = _snapshot(local_store, namespace)

    stale = [_row(namespace, "a", 1, expected=1), _row(namespace, "b", 1, expected=99)]
    assert local_store.compare_and_swap_named_projections(stale) is False
    assert _snapshot(local_store, namespace) == before

    # Seed sequence is 1; the failed attempt above must not have changed either row.
    assert local_store.compare_and_swap_named_projections([_row(namespace, "a", 1, expected=1), _row(namespace, "b", 1, expected=1)]) is True
    assert {k: v["payload"]["value"] for k, v in _snapshot(local_store, namespace).items()} == {"a": 1, "b": 1}


def test_multi_projection_cas_concurrency_has_one_winner(local_store):
    namespace = f"cas-race-{uuid.uuid4().hex}"
    _seed(local_store, namespace)

    def attempt(value: int) -> bool:
        return bool(local_store.compare_and_swap_named_projections([_row(namespace, "a", value, expected=1), _row(namespace, "b", value, expected=1)]))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, (2, 3)))
    assert sorted(results) == [False, True]
    snapshot = _snapshot(local_store, namespace)
    assert snapshot["a"]["payload"]["value"] == snapshot["b"]["payload"]["value"]


def test_postgres_multi_projection_cas_is_atomic(sa_engine, pg_schema):
    if pg_schema is None:
        pytest.skip("PostgreSQL fixture unavailable")
    store = EnginePostgresMetaStore(engine=sa_engine, schema=pg_schema)
    store.ensure_initialized()
    namespace = f"cas-pg-{uuid.uuid4().hex}"
    _seed(store, namespace)
    before = _snapshot(store, namespace)
    assert store.compare_and_swap_named_projections([_row(namespace, "a", 1, expected=1), _row(namespace, "b", 1, expected=99)]) is False
    assert _snapshot(store, namespace) == before
    assert store.compare_and_swap_named_projections([_row(namespace, "a", 1, expected=1), _row(namespace, "b", 1, expected=1)]) is True

    def attempt(value: int) -> bool:
        return bool(store.compare_and_swap_named_projections([_row(namespace, "a", value, expected=2), _row(namespace, "b", value, expected=2)]))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, (2, 3)))
    assert sorted(results) == [False, True]
