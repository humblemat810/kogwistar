from __future__ import annotations

import json
import threading
import time
from typing import Any


def _now_ms() -> int:
    return int(time.time() * 1000)


def _stable_json(payload: dict[str, Any] | None) -> str:
    return json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class RunRegistry:
    TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}

    def __init__(self, meta_store: Any) -> None:
        self.meta_store = meta_store
        self._lock = threading.Lock()
        ensure_initialized = getattr(self.meta_store, "ensure_initialized", None)
        if callable(ensure_initialized):
            ensure_initialized()

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
        with self._lock:
            self.meta_store.create_server_run(
                run_id=run_id,
                conversation_id=conversation_id,
                workflow_id=workflow_id,
                user_id=user_id,
                user_turn_node_id=user_turn_node_id,
                status=status,
            )
        return self.get_run(run_id) or {}

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = self.meta_store.get_server_run(run_id)
        if run is None:
            return None
        run["terminal"] = str(run.get("status") or "") in self.TERMINAL_STATUSES
        return run

    def list_events(self, run_id: str, *, after_seq: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        return list(self.meta_store.list_server_run_events(run_id, after_seq=int(after_seq), limit=int(limit)))

    def append_event(self, run_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            return self.meta_store.append_server_run_event(run_id, event_type, _stable_json(payload))

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
        existing = self.get_run(run_id)
        if existing is None:
            raise KeyError(f"Unknown run_id: {run_id}")

        now = _now_ms()
        started_at_ms = existing.get("started_at_ms")
        finished_at_ms = existing.get("finished_at_ms")
        if started and started_at_ms is None:
            started_at_ms = now
        if finished:
            finished_at_ms = now

        resolved_assistant_turn_node_id = assistant_turn_node_id or existing.get("assistant_turn_node_id")
        resolved_result = result if result is not None else existing.get("result")
        resolved_error = error if error is not None else existing.get("error")
        with self._lock:
            self.meta_store.update_server_run(
                run_id=run_id,
                status=status,
                assistant_turn_node_id=(str(resolved_assistant_turn_node_id) if resolved_assistant_turn_node_id else None),
                result_json=(None if resolved_result is None else _stable_json(resolved_result)),
                error_json=(None if resolved_error is None else _stable_json(resolved_error)),
                started_at_ms=(None if started_at_ms is None else int(started_at_ms)),
                finished_at_ms=(None if finished_at_ms is None else int(finished_at_ms)),
            )
        return self.get_run(run_id) or {}

    def request_cancel(self, run_id: str) -> dict[str, Any]:
        existing = self.get_run(run_id)
        if existing is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        if existing["terminal"]:
            return existing
        with self._lock:
            self.meta_store.request_server_run_cancel(run_id=run_id)
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
