from __future__ import annotations

import pytest

from kogwistar.messaging.models import (
    LaneMessageSendResult,
    ProjectedLaneMessageRow,
)


@pytest.fixture
def lane_message_contract_sample() -> dict[str, object]:
    return {
        "send_result": LaneMessageSendResult(
            message_id="msg-sample",
            conversation_anchor_id="anchor-conv",
            inbox_anchor_id="anchor-inbox",
            sender_anchor_id="anchor-sender",
            recipient_anchor_id="anchor-recipient",
        ),
        "projected_row": ProjectedLaneMessageRow(
            message_id="msg-sample",
            namespace="ns-sample",
            purpose="user_visible",
            inbox_id="inbox:worker:sample",
            conversation_id="conv-sample",
            recipient_id="lane:worker:sample",
            sender_id="lane:foreground",
            msg_type="request.sample",
            status="pending",
            seq=1,
            conversation_seq=1,
            claimed_by=None,
            lease_until=None,
            retry_count=0,
            created_at=1,
            available_at=1,
            run_id=None,
            step_id=None,
            correlation_id="corr-sample",
            payload_json='{"sample":true}',
            error_json=None,
        ),
    }
import os
import sqlite3
import sys
from hashlib import sha256
from pathlib import Path

import pytest


def _sqlite_diag_sidecars(database: str) -> str:
    path = Path(database)
    details = []
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            stat = candidate.stat()
        except FileNotFoundError:
            details.append(f"{candidate.name}=absent")
        else:
            details.append(f"{candidate.name}=size:{stat.st_size},mtime_ns:{stat.st_mtime_ns}")
    return " ".join(details)


class _DiagnosticSQLiteMixin:
    database: str

    def execute(self, sql, parameters=(), /):
        try:
            return super().execute(sql, parameters)
        except sqlite3.Error as error:
            try:
                database = super().execute("PRAGMA database_list").fetchone()[2]
            except sqlite3.Error:
                database = self.database
            print(
                f"[sqlite-python-error] conn={id(self)} path={database} "
                f"code={getattr(error, 'sqlite_errorcode', None)} "
                f"name={getattr(error, 'sqlite_errorname', None)} "
                f"sql_sha256={sha256(sql.encode('utf-8')).hexdigest()} "
                f"{_sqlite_diag_sidecars(database)}",
                file=sys.stderr,
                flush=True,
            )
            raise

    def close(self):
        if os.getenv("KOGWISTAR_SQLITE_DIAGNOSTICS") == "1":
            print(
                f"[sqlite-python-close] conn={id(self)} path={self.database} "
                f"{_sqlite_diag_sidecars(self.database)}",
                file=sys.stderr,
                flush=True,
            )
        return super().close()

    def __del__(self):
        if os.getenv("KOGWISTAR_SQLITE_DIAGNOSTICS") == "1":
            print(
                f"[sqlite-python-destroy] conn={id(self)} path={self.database} "
                f"{_sqlite_diag_sidecars(self.database)}",
                file=sys.stderr,
                flush=True,
            )


@pytest.fixture(autouse=True)
def _sqlite_diagnostics(monkeypatch):
    if os.getenv("KOGWISTAR_SQLITE_DIAGNOSTICS") != "1":
        return
    original_connect = sqlite3.connect

    def diagnostic_connect(*args, **kwargs):
        base_factory = kwargs.get("factory", sqlite3.Connection)
        diagnostic_factory = type(
            "DiagnosticSQLiteConnection",
            (_DiagnosticSQLiteMixin, base_factory),
            {},
        )
        kwargs["factory"] = diagnostic_factory
        connection = original_connect(*args, **kwargs)
        database = str(args[0] if args else kwargs.get("database", ""))
        connection.database = database
        print(
            f"[sqlite-python-open] conn={id(connection)} path={database} "
            f"python_sqlite={sqlite3.sqlite_version} {_sqlite_diag_sidecars(database)}",
            file=sys.stderr,
            flush=True,
        )
        return connection

    monkeypatch.setattr(sqlite3, "connect", diagnostic_connect)
