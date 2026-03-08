from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


def _now_ms() -> int:
    return int(time.time() * 1000)


def _stable_json(payload: dict[str, Any] | None) -> str:
    return json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class RunRegistry:
    TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS server_runs (
                    run_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    user_id TEXT,
                    user_turn_node_id TEXT,
                    assistant_turn_node_id TEXT,
                    status TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error_json TEXT,
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    started_at_ms INTEGER,
                    finished_at_ms INTEGER
                );

                CREATE TABLE IF NOT EXISTS server_run_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_server_runs_status
                    ON server_runs(status, updated_at_ms);
                CREATE INDEX IF NOT EXISTS idx_server_run_events_run_seq
                    ON server_run_events(run_id, seq);
                """
            )

    def _row_to_run(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = json.loads(row["result_json"]) if row["result_json"] else None
        error = json.loads(row["error_json"]) if row["error_json"] else None
        return {
            "run_id": row["run_id"],
            "conversation_id": row["conversation_id"],
            "workflow_id": row["workflow_id"],
            "user_id": row["user_id"],
            "user_turn_node_id": row["user_turn_node_id"],
            "assistant_turn_node_id": row["assistant_turn_node_id"],
            "status": row["status"],
            "cancel_requested": bool(row["cancel_requested"]),
            "result": result,
            "error": error,
            "created_at_ms": row["created_at_ms"],
            "updated_at_ms": row["updated_at_ms"],
            "started_at_ms": row["started_at_ms"],
            "finished_at_ms": row["finished_at_ms"],
            "terminal": str(row["status"]) in self.TERMINAL_STATUSES,
        }

    def create_run(
        self,
        *,
        run_id: str,
        conversation_id: str,
        workflow_id: str,
        user_id: str | None,
        user_turn_node_id: str,
        status: str = "queued",
    ) -> dict[str, Any]:
        now = _now_ms()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO server_runs (
                    run_id, conversation_id, workflow_id, user_id,
                    user_turn_node_id, assistant_turn_node_id, status,
                    cancel_requested, result_json, error_json,
                    created_at_ms, updated_at_ms, started_at_ms, finished_at_ms
                ) VALUES (?, ?, ?, ?, ?, NULL, ?, 0, NULL, NULL, ?, ?, NULL, NULL)
                """,
                (run_id, conversation_id, workflow_id, user_id, user_turn_node_id, status, now, now),
            )
        return self.get_run(run_id) or {}

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM server_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._row_to_run(row)

    def list_events(self, run_id: str, *, after_seq: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT seq, run_id, event_type, payload_json, created_at_ms
                FROM server_run_events
                WHERE run_id = ? AND seq > ?
                ORDER BY seq ASC
                LIMIT ?
                """,
                (run_id, int(after_seq), int(limit)),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "seq": int(row["seq"]),
                    "run_id": row["run_id"],
                    "event_type": row["event_type"],
                    "payload": json.loads(row["payload_json"]),
                    "created_at_ms": int(row["created_at_ms"]),
                }
            )
        return out

    def append_event(self, run_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        now = _now_ms()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO server_run_events (run_id, event_type, payload_json, created_at_ms)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, event_type, _stable_json(payload), now),
            )
            seq = int(cur.lastrowid)
        return {
            "seq": seq,
            "run_id": run_id,
            "event_type": event_type,
            "payload": payload or {},
            "created_at_ms": now,
        }

    def update_status(
        self,
        run_id: str,
        *,
        status: str,
        assistant_turn_node_id: str | None = None,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> dict[str, Any]:
        now = _now_ms()
        existing = self.get_run(run_id)
        if existing is None:
            raise KeyError(f"Unknown run_id: {run_id}")

        started_at_ms = existing.get("started_at_ms")
        finished_at_ms = existing.get("finished_at_ms")
        if started and started_at_ms is None:
            started_at_ms = now
        if finished:
            finished_at_ms = now

        assistant_turn_node_id = assistant_turn_node_id or existing.get("assistant_turn_node_id")
        result_json = _stable_json(result) if result is not None else (
            _stable_json(existing.get("result")) if existing.get("result") is not None else None
        )
        error_json = _stable_json(error) if error is not None else (
            _stable_json(existing.get("error")) if existing.get("error") is not None else None
        )

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE server_runs
                SET status = ?,
                    assistant_turn_node_id = ?,
                    result_json = ?,
                    error_json = ?,
                    started_at_ms = ?,
                    finished_at_ms = ?,
                    updated_at_ms = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    assistant_turn_node_id,
                    result_json,
                    error_json,
                    started_at_ms,
                    finished_at_ms,
                    now,
                    run_id,
                ),
            )
        return self.get_run(run_id) or {}

    def request_cancel(self, run_id: str) -> dict[str, Any]:
        existing = self.get_run(run_id)
        if existing is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        if existing["terminal"]:
            return existing

        status = existing["status"]
        if status not in {"cancelled", "failed", "succeeded"}:
            status = "cancelling"

        now = _now_ms()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE server_runs
                SET cancel_requested = 1,
                    status = ?,
                    updated_at_ms = ?
                WHERE run_id = ?
                """,
                (status, now, run_id),
            )
        return self.get_run(run_id) or {}

    def is_cancel_requested(self, run_id: str) -> bool:
        run = self.get_run(run_id)
        if run is None:
            return False
        return bool(run["cancel_requested"])


class RunRegistryTraceBridge:
    _STAGES: dict[str, tuple[str, str]] = {
        "prepare": ("prepare", "Preparing the answer run."),
        "view": ("build_context", "Building the conversation context."),
        "retrieve": ("retrieve", "Retrieving candidate evidence."),
        "select": ("select_evidence", "Selecting the most relevant evidence."),
        "materialize": ("materialize", "Materializing the evidence pack."),
        "answer": ("draft_answer", "Drafting the answer with citations."),
        "repair": ("citation_repair", "Validating and repairing citations."),
        "eval": ("evaluate", "Evaluating the draft answer."),
        "project": ("project", "Projecting supporting references."),
        "persist": ("persist", "Persisting the assistant response."),
    }

    def __init__(self, *, registry: RunRegistry, run_id: str, delegate: Any | None = None) -> None:
        self.registry = registry
        self.run_id = run_id
        self.delegate = delegate

    def emit(self, evt: dict[str, Any]) -> None:
        if self.delegate is not None:
            self.delegate.emit(evt)

        if str(evt.get("run_id") or "") != self.run_id:
            return

        evt_type = str(evt.get("type") or "")
        node_id = str(evt.get("node_id") or "")
        suffix = node_id.rsplit(":", 1)[-1]
        stage = self._STAGES.get(suffix)

        if evt_type == "step_attempt_started" and stage is not None:
            stage_name, summary = stage
            payload = {
                "stage": stage_name,
                "workflow_node_id": node_id,
                "step_seq": int(evt.get("step_seq") or 0),
            }
            self.registry.append_event(self.run_id, "run.stage", payload)
            self.registry.append_event(
                self.run_id,
                "reasoning.summary",
                {
                    **payload,
                    "summary": summary,
                },
            )
        elif evt_type == "workflow_run_cancelling":
            self.registry.append_event(
                self.run_id,
                "run.cancelling",
                {
                    "run_id": self.run_id,
                    "status": "cancelling",
                },
            )
