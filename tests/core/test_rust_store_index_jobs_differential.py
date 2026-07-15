from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import time
import uuid
from typing import Any

import pytest

from kogwistar._rust_bridge import RustParityError, store_postgres, store_sqlite
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.engine_sqlite import EngineSQLite


pytestmark = [pytest.mark.ci, pytest.mark.core]


@pytest.fixture(scope="module", autouse=True)
def _native_extension() -> Any:
    return pytest.importorskip("kogwistar._rust")


def _sqlite(path: Path, operation: dict[str, Any]) -> Any:
    return store_sqlite(path=path, operation=operation)


def _job(job_id: str, namespace: str = "ns", op: str = "UPSERT", **extra: Any) -> dict[str, Any]:
    return {
        "kind": "enqueue_index_job",
        "job_id": job_id,
        "namespace": namespace,
        "entity_kind": "node",
        "entity_id": extra.pop("entity_id", "n"),
        "index_kind": "doc",
        "op": op,
        **extra,
    }


def test_sqlite_python_to_rust_and_rust_to_python_queue_contract(tmp_path: Path) -> None:
    python = EngineSQLite(tmp_path)
    python.ensure_initialized()
    assert python.enqueue_index_job(job_id="python", namespace="ns", entity_kind="node", entity_id="n", index_kind="doc", op="UPSERT") == "python"
    row = _sqlite(python.db_path, {"kind": "list_index_jobs", "namespace": "ns"})[0]
    assert row["job_id"] == "python" and isinstance(row["created_at"], int)

    assert _sqlite(python.db_path, _job("rust", namespace="other", payload_json='{"from":"rust"}')) == "rust"
    rows = python.list_index_jobs(namespace="other")
    assert rows[0].job_id == "rust" and rows[0].payload_json == '{"from":"rust"}'


def test_sqlite_coalesce_claim_tokens_retry_tail_namespace_and_rollback(tmp_path: Path) -> None:
    path = tmp_path / "engine.db"
    assert _sqlite(path, _job("one", payload_json='{"old":true}')) == "one"
    assert _sqlite(path, _job("two", op="DELETE", payload_json='{"delete":true}')) == "one"
    assert _sqlite(path, {"kind": "list_index_jobs", "namespace": "ns"})[0]["op"] == "DELETE"
    assert _sqlite(path, _job("isolated", namespace="other")) == "isolated"
    claimed = _sqlite(path, {"kind": "claim_index_jobs", "namespace": "ns", "limit": 1, "lease_seconds": 0})[0]
    old = claimed["claim_token"]
    time.sleep(1.05)
    reclaimed = _sqlite(path, {"kind": "claim_index_jobs", "namespace": "ns", "limit": 1, "lease_seconds": 30})[0]
    assert reclaimed["claim_token"] != old
    assert _sqlite(path, {"kind": "mark_index_job_done", "job_id": "one", "claim_token": old}) is False
    assert _sqlite(path, {"kind": "renew_index_job_lease", "job_id": "one", "claim_token": old, "lease_seconds": 30}) is False
    assert _sqlite(path, {"kind": "bump_retry_and_requeue", "job_id": "one", "error": "retry", "next_run_at_seconds": 0, "claim_token": reclaimed["claim_token"]}) is None
    retry = _sqlite(path, {"kind": "claim_index_jobs", "namespace": "ns", "limit": 1, "lease_seconds": 30})[0]
    assert retry["retry_count"] == 1
    assert _sqlite(path, {"kind": "requeue_index_job_at_tail", "job_id": "one", "payload_json": "{}", "claim_token": retry["claim_token"]}) is None
    assert _sqlite(path, {"kind": "list_index_jobs", "namespace": "other"})[0]["job_id"] == "isolated"

    with pytest.raises(RustParityError) as aborted:
        _sqlite(path, {"kind": "batch", "operations": [_job("rolled", entity_id="rollback"), {"kind": "raw_append", "namespace": "ns", "event_id": "rollback", "entity_kind": "node", "entity_id": "n", "op": "UPSERT", "payload_json": "{}"}], "abort": True})
    assert aborted.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    assert not any(row["job_id"] == "rolled" for row in _sqlite(path, {"kind": "list_index_jobs", "namespace": "ns"}))


def test_sqlite_concurrent_same_key_enqueue_and_claim_are_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "engine.db"
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda n: _sqlite(path, _job(f"job-{n}")), range(8)))
    assert len(set(ids)) == 1
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: _sqlite(path, {"kind": "claim_index_jobs", "namespace": "ns", "limit": 1, "lease_seconds": 30}), range(8)))
    assert sum(len(rows) for rows in claims) == 1


def test_sqlite_retry_terminal_unknown_fields_and_raw_durable_effect(tmp_path: Path) -> None:
    path = tmp_path / "engine.db"
    _sqlite(path, _job("terminal", max_retries=1))
    claim = _sqlite(path, {"kind": "claim_index_jobs", "namespace": "ns", "limit": 1, "lease_seconds": 30})[0]
    _sqlite(path, {"kind": "bump_retry_and_requeue", "job_id": "terminal", "error": "terminal", "next_run_at_seconds": 0, "claim_token": claim["claim_token"]})
    row = _sqlite(path, {"kind": "list_index_jobs", "namespace": "ns", "status": "FAILED"})[0]
    assert row["retry_count"] == 1 and row["last_error"] == "terminal"
    _sqlite(path, _job("final", entity_id="final"))
    final_claim = _sqlite(path, {"kind": "claim_index_jobs", "namespace": "ns", "limit": 10, "lease_seconds": 30})
    final = next(item for item in final_claim if item["job_id"] == "final")
    assert _sqlite(path, {"kind": "mark_index_job_failed", "job_id": "final", "error": "final-error", "final": True, "claim_token": final["claim_token"]}) is None
    with sqlite3.connect(path) as conn:
        raw = conn.execute("SELECT status, last_error, claim_token FROM index_jobs WHERE job_id='final'").fetchone()
        columns = {row[1] for row in conn.execute("PRAGMA table_info(index_jobs)")}
    assert raw == ("FAILED", "final-error", None)
    assert "claim_attempts" not in columns
    with pytest.raises(RustParityError) as invalid:
        _sqlite(path, {"kind": "list_index_jobs", "namespace": "ns", "unexpected": True})
    assert invalid.value.code == "KOGWISTAR_STORE_INVALID_PAYLOAD"


def _pg(dsn: str, schema: str, operation: dict[str, Any]) -> Any:
    return store_postgres(dsn=dsn, schema=schema, operation=operation)


def _require_pg(pg_dsn: str | None, pg_schema: str | None) -> tuple[str, str]:
    if not pg_dsn or not pg_schema:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    return pg_dsn, pg_schema


def test_postgres_python_rust_queue_differential_live(sa_engine, pg_dsn: str | None, pg_schema: str | None) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    py = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    py.ensure_initialized()
    assert py.enqueue_index_job(job_id="python", namespace="ns", entity_kind="node", entity_id="n", index_kind="doc", op="UPSERT") == "python"
    assert _pg(dsn, schema, {"kind": "list_index_jobs", "namespace": "ns"})[0]["job_id"] == "python"
    assert _pg(dsn, schema, _job("rust", namespace="other")) == "rust"
    assert py.list_index_jobs(namespace="other")[0].job_id == "rust"
    with sa_engine.connect() as conn:
        assert conn.exec_driver_sql(f'SELECT status FROM "{schema}".index_jobs WHERE job_id=\'rust\'').scalar_one() == "PENDING"

    assert _pg(dsn, schema, _job("first", namespace="queue", entity_id="same")) == "first"
    assert _pg(dsn, schema, _job("second", namespace="queue", entity_id="same", op="DELETE")) == "first"
    claim = _pg(dsn, schema, {"kind": "claim_index_jobs", "namespace": "queue", "limit": 1, "lease_seconds": 30})[0]
    assert _pg(dsn, schema, {"kind": "mark_index_job_done", "job_id": "first", "claim_token": "stale"}) is False
    assert _pg(dsn, schema, {"kind": "renew_index_job_lease", "job_id": "first", "claim_token": claim["claim_token"], "lease_seconds": 30}) is True
    with pytest.raises(RustParityError) as aborted:
        _pg(dsn, schema, {"kind": "batch", "operations": [_job("rolled", namespace="queue", entity_id="rollback"), {"kind": "raw_append", "namespace": "queue", "event_id": f"rollback-{uuid.uuid4().hex}", "entity_kind": "node", "entity_id": "n", "op": "UPSERT", "payload_json": "{}"}], "abort": True})
    assert aborted.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    with pytest.raises(RustParityError) as invalid:
        _pg(dsn, schema, {"kind": "list_index_jobs", "namespace": "queue", "unexpected": True})
    assert invalid.value.code == "KOGWISTAR_STORE_INVALID_PAYLOAD"


def test_postgres_queue_concurrency_reclaim_and_terminal_retry_live(
    pg_dsn: str | None, pg_schema: str | None
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    assert _pg(dsn, schema, {"kind": "ensure_schema"}) == {"initialized": True}
    key = f"queue-{uuid.uuid4().hex}"
    def enqueue(n: int) -> str:
        return _pg(dsn, schema, _job(f"{key}-{n}", namespace=key, entity_id="same"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert len(set(pool.map(enqueue, range(8)))) == 1
    def claim(_: int) -> list[dict[str, Any]]:
        return _pg(dsn, schema, {"kind": "claim_index_jobs", "namespace": key, "limit": 1, "lease_seconds": 30})
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(len(rows) for rows in pool.map(claim, range(8))) == 1
    first = _pg(dsn, schema, {"kind": "list_index_jobs", "namespace": key})[0]
    assert _pg(dsn, schema, {"kind": "mark_index_job_done", "job_id": first["job_id"], "claim_token": first["claim_token"]}) is True

    _pg(dsn, schema, _job(f"{key}-reclaim", namespace=key, entity_id="reclaim"))
    old = _pg(dsn, schema, {"kind": "claim_index_jobs", "namespace": key, "limit": 1, "lease_seconds": 0})[0]
    time.sleep(1.05)
    new = _pg(dsn, schema, {"kind": "claim_index_jobs", "namespace": key, "limit": 1, "lease_seconds": 30})[0]
    assert new["claim_token"] != old["claim_token"]
    assert _pg(dsn, schema, {"kind": "mark_index_job_done", "job_id": old["job_id"], "claim_token": old["claim_token"]}) is False
    _pg(dsn, schema, {"kind": "bump_retry_and_requeue", "job_id": new["job_id"], "error": "terminal", "next_run_at_seconds": 0, "claim_token": new["claim_token"]})
    # Default max_retries allows a retry; force terminal on a single-attempt row.
    _pg(dsn, schema, _job(f"{key}-terminal", namespace=key, entity_id="terminal", max_retries=1))
    terminal = _pg(dsn, schema, {"kind": "claim_index_jobs", "namespace": key, "limit": 10, "lease_seconds": 30})
    row = next(item for item in terminal if item["entity_id"] == "terminal")
    _pg(dsn, schema, {"kind": "bump_retry_and_requeue", "job_id": row["job_id"], "error": "terminal", "next_run_at_seconds": 0, "claim_token": row["claim_token"]})
    assert any(item["entity_id"] == "terminal" and item["status"] == "FAILED" for item in _pg(dsn, schema, {"kind": "list_index_jobs", "namespace": key}))

    # PostgreSQL alone limits expired-lease takeovers. Initial claim is attempt 0;
    # three reclaims reach 3, then the next expired claim moves the row to FAILED.
    _pg(dsn, schema, _job(f"{key}-takeover", namespace=key, entity_id="takeover"))
    current = _pg(dsn, schema, {"kind": "claim_index_jobs", "namespace": key, "limit": 1, "lease_seconds": -1})[0]
    for expected in range(1, 4):
        current = _pg(dsn, schema, {"kind": "claim_index_jobs", "namespace": key, "limit": 1, "lease_seconds": -1})[0]
        assert current["claim_attempts"] == expected
    assert _pg(dsn, schema, {"kind": "claim_index_jobs", "namespace": key, "limit": 1, "lease_seconds": 30}) == []
    takeover = next(item for item in _pg(dsn, schema, {"kind": "list_index_jobs", "namespace": key}) if item["entity_id"] == "takeover")
    assert takeover["status"] == "FAILED" and takeover["last_error"] == "lease ownership exceeded"
