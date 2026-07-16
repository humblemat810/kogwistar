"""Python callback worker for ADR-015 Rust-owned durable runtime lanes.

Worker callbacks run outside Rust/SQLite transactions.  A local journal makes
completed callback results replayable after response loss.  A process that
finds an ``executing`` journal row refuses to run that callback again because
its side-effect outcome is ambiguous; operators may then reconcile it using
the stable lane ``message_id`` idempotency key.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx

from kogwistar.id_provider import stable_id


class RustWorkerError(RuntimeError):
    """Rust worker protocol or transport failure."""


class AmbiguousWorkerExecution(RustWorkerError):
    """Callback began before restart, but no durable result was recorded."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _work_digest(work: Mapping[str, Any]) -> str:
    immutable = {
        key: work.get(key)
        for key in (
            "message_id",
            "run_id",
            "step_id",
            "correlation_id",
            "expected_event_seq",
            "payload",
        )
    }
    return hashlib.sha256(_canonical(immutable).encode("utf-8")).hexdigest()


class WorkerResultJournal:
    """Small SQLite journal shared across worker process restarts."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rust_worker_results(
                        message_id TEXT PRIMARY KEY,
                        work_digest TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status IN ('executing', 'completed')),
                        result_json TEXT
                    )
                    """
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def begin(self, *, message_id: str, work_digest: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            with connection:
                row = connection.execute(
                    "SELECT work_digest, status, result_json FROM rust_worker_results "
                    "WHERE message_id = ?",
                    (message_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO rust_worker_results(message_id, work_digest, status) "
                        "VALUES (?, ?, 'executing')",
                        (message_id, work_digest),
                    )
                    return None
                stored_digest, status, result_json = row
                if stored_digest != work_digest:
                    raise RustWorkerError(
                        f"lane message {message_id!r} changed across retry"
                    )
                if status == "executing":
                    raise AmbiguousWorkerExecution(
                        f"lane message {message_id!r} may already have run; "
                        "reconcile using message_id before retry"
                    )
                if not isinstance(result_json, str):
                    raise RustWorkerError(
                        f"lane message {message_id!r} has completed journal row without result"
                    )
                result = json.loads(result_json)
                if not isinstance(result, dict):
                    raise RustWorkerError(
                        f"lane message {message_id!r} journal result is not an object"
                    )
                return result

    def complete(self, *, message_id: str, result: Mapping[str, Any]) -> None:
        result_json = _canonical(dict(result))
        with closing(self._connect()) as connection:
            with connection:
                changed = connection.execute(
                    "UPDATE rust_worker_results SET status = 'completed', result_json = ? "
                    "WHERE message_id = ? AND status = 'executing'",
                    (result_json, message_id),
                ).rowcount
                if changed != 1:
                    raise RustWorkerError(
                        f"lane message {message_id!r} journal ownership changed"
                    )


class RustRuntimeWorker:
    """Claim Rust runtime work, execute Python callback, submit durable result."""

    def __init__(
        self,
        *,
        base_url: str,
        worker_id: str,
        journal_path: str | os.PathLike[str],
        execute: Callable[[dict[str, Any]], Mapping[str, Any]],
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        self.worker_id = worker_id
        self.execute = execute
        self.journal = WorkerResultJournal(journal_path)
        self._owns_client = client is None
        self.client = client or httpx.Client(
            base_url=base_url.rstrip("/"), headers=dict(headers or {}), timeout=timeout
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> RustRuntimeWorker:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = self.client.post(path, json=dict(payload))
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise RustWorkerError(str(error)) from error
        value = response.json()
        if not isinstance(value, dict):
            raise RustWorkerError(f"{path} returned non-object JSON")
        return value

    def claim(self, *, limit: int = 1, lease_seconds: int = 60) -> list[dict[str, Any]]:
        if limit <= 0 or lease_seconds <= 0:
            raise ValueError("limit and lease_seconds must be positive")
        value = self._post(
            "/internal/runtime/claim",
            {
                "claimed_by": self.worker_id,
                "limit": limit,
                "lease_seconds": lease_seconds,
            },
        )
        work = value.get("work")
        if not isinstance(work, list) or not all(isinstance(item, dict) for item in work):
            raise RustWorkerError("runtime claim response has invalid work list")
        return work

    @staticmethod
    def _result_envelope(
        work: Mapping[str, Any], effect: Mapping[str, Any]
    ) -> dict[str, Any]:
        payload = work.get("payload")
        if not isinstance(payload, dict):
            raise RustWorkerError("claimed work payload must be an object")
        message_id = str(work.get("message_id") or "")
        run_id = str(work.get("run_id") or "")
        step_id = str(work.get("step_id") or "")
        claimed_by = str(work.get("claimed_by") or "")
        correlation_id = str(work.get("correlation_id") or "")
        required = {
            "message_id": message_id,
            "run_id": run_id,
            "step_id": step_id,
            "claimed_by": claimed_by,
            "correlation_id": correlation_id,
        }
        empty = sorted(key for key, value in required.items() if not value)
        if empty:
            raise RustWorkerError(f"claimed work missing identity fields: {empty!r}")

        restricted = {
            "contract_version": 1,
            "effect_id": str(
                stable_id("runtime.worker.result", message_id, run_id, step_id)
            ),
            **dict(effect),
        }
        forbidden = {
            "run_id",
            "workflow_id",
            "conversation_id",
            "step_seq",
            "node_id",
            "token_id",
            "parent_token_id",
            "expected_event_seq",
            "kind",
            "frontier",
            "transition_id",
        }.intersection(restricted)
        if forbidden:
            raise RustWorkerError(
                f"worker effect may not override scheduler fields: {sorted(forbidden)!r}"
            )
        allowed = {
            "contract_version",
            "effect_id",
            "status",
            "state_update",
            "update",
            "state_schema",
            "successors",
            "result",
            "errors",
            "wait_reason",
            "resume_payload",
            "usage",
            "trace_events",
        }
        unknown = set(restricted).difference(allowed)
        if unknown:
            raise RustWorkerError(f"worker effect has unknown fields: {sorted(unknown)!r}")
        fixed = {
            "contract_version": 1,
            "effect_id": str(
                stable_id("runtime.worker.result", message_id, run_id, step_id)
            ),
        }
        for key, value in fixed.items():
            if key in restricted and restricted[key] != value:
                raise RustWorkerError(
                    f"worker effect may not override scheduler field {key!r}"
                )
            restricted[key] = value
        successors = restricted.get("successors", [])
        if not isinstance(successors, list):
            raise RustWorkerError("worker effect successors must be an array")
        return {
            "handoff": required,
            "effect": restricted,
        }

    def process(self, work: dict[str, Any]) -> dict[str, Any]:
        message_id = str(work.get("message_id") or "")
        if not message_id:
            raise RustWorkerError("claimed work missing message_id")
        digest = _work_digest(work)
        envelope = self.journal.begin(message_id=message_id, work_digest=digest)
        if envelope is None:
            effect = self.execute(dict(work))
            if not isinstance(effect, Mapping):
                raise RustWorkerError("worker callback must return an object")
            envelope = self._result_envelope(work, effect)
            self.journal.complete(message_id=message_id, result=envelope)
        else:
            handoff = envelope.get("handoff")
            if not isinstance(handoff, dict):
                raise RustWorkerError("journaled result lacks handoff object")
            handoff["claimed_by"] = work.get("claimed_by")
        return self._post("/internal/runtime/results", envelope)

    def poll_once(self, *, limit: int = 1, lease_seconds: int = 60) -> int:
        work = self.claim(limit=limit, lease_seconds=lease_seconds)
        for item in work:
            self.process(item)
        return len(work)


__all__ = [
    "AmbiguousWorkerExecution",
    "RustRuntimeWorker",
    "RustWorkerError",
    "WorkerResultJournal",
]
