from __future__ import annotations

import pytest

from kogwistar.engine_core.rust_postgres_session import (
    RustEnginePostgresMetaStore,
    RustPostgresConnectionUnavailable,
    RustPostgresSession,
)


pytestmark = [pytest.mark.ci, pytest.mark.core]


def test_postgres_session_nested_commit_uses_one_native_transaction(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    def fake_store_postgres(*, operation, transaction_id, **kwargs):
        del kwargs
        calls.append((operation["kind"], transaction_id))
        return None

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    session = RustPostgresSession(dsn="postgresql://example/db", schema="test")

    with session.transaction() as outer:
        session.call("raw_append")
        with session.transaction() as inner:
            session.require_token(inner)
            assert inner.value == outer.value
            session.call("replace_named_projection")

    transaction_ids = {value for _, value in calls if value is not None}
    assert len(transaction_ids) == 1
    assert [kind for kind, _ in calls] == [
        "begin_transaction",
        "raw_append",
        "replace_named_projection",
        "commit_transaction",
    ]


def test_postgres_session_rolls_back_and_rejects_raw_or_stale_writer(monkeypatch):
    calls: list[str] = []

    def fake_store_postgres(*, operation, **kwargs):
        del kwargs
        calls.append(operation["kind"])
        return None

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    session = RustPostgresSession(dsn="postgresql://example/db", schema="test")
    with pytest.raises(RustPostgresConnectionUnavailable):
        session.connect()

    stale = None
    with pytest.raises(RuntimeError, match="stop"):
        with session.transaction() as token:
            stale = token
            raise RuntimeError("stop")
    assert calls == ["begin_transaction", "rollback_transaction"]
    assert stale is not None
    with pytest.raises(RustPostgresConnectionUnavailable, match="stale"):
        session.require_token(stale)


def test_postgres_meta_facade_routes_nested_sequence_and_queue_calls(monkeypatch):
    calls: list[tuple[str, str | None]] = []
    results = {
        "next_global_seq": 1,
        "next_user_seq": 1,
        "enqueue_index_job": "job",
    }

    def fake_store_postgres(*, operation, transaction_id, **kwargs):
        del kwargs
        calls.append((operation["kind"], transaction_id))
        return results.get(operation["kind"])

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    store = RustEnginePostgresMetaStore(
        dsn="postgresql://example/db", schema="test"
    )
    with store.transaction() as token:
        assert store.next_global_seq_conn(token) == 1
        assert store.next_scoped_seq("scope") == 1
        assert store.enqueue_index_job(
            job_id="job",
            entity_kind="node",
            entity_id="n",
            index_kind="node_docs",
            op="UPSERT",
        ) == "job"
    assert [kind for kind, _ in calls] == [
        "begin_transaction",
        "next_global_seq",
        "next_user_seq",
        "enqueue_index_job",
        "commit_transaction",
    ]
    assert len({value for _, value in calls if value is not None}) == 1
