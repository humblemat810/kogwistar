"""Run inspection helpers for workflow trace lookup, checkpoints, and replay.

This module serves the read-only inspection surface for workflow runs. It
retrieves persisted step execution and checkpoint artifacts from the
conversation graph and delegates replay reconstruction to the runtime replay
helpers without owning run execution itself.
"""

from __future__ import annotations

import json
from typing import Any

from kogwistar.runtime.projections import workflow_checkpoint_latest_projection_namespace
from kogwistar.runtime.replay import load_checkpoint, replay_to

from .chat_service_shared import _BaseComponent


class _RunInspectionService(_BaseComponent):
    """Owns step/checkpoint lookup and replay helpers."""

    _WAIT_REASONS = [
        "approval",
        "message",
        "schedule_delay",
        "external_callback",
        "dependency",
        "rate_window",
    ]

    def resume_contract(self, run_id: str) -> dict[str, Any]:
        checkpoints = self.list_checkpoints(run_id)
        latest = checkpoints[-1] if checkpoints else None
        state = latest.get("state") if isinstance(latest, dict) else None
        latest_node = self._latest_checkpoint_node(run_id)
        resume_keys = {
            "run_id": run_id,
            "latest_checkpoint_step_seq": None if latest is None else latest.get("step_seq"),
            "latest_wait_reason": None
            if latest is None
            else (latest.get("state") or {}).get("wait_reason"),
            "checkpoint_schema_version": None
            if latest_node is None
            else int((getattr(latest_node, "metadata", {}) or {}).get("checkpoint_schema_version", 1)),
            "persisted_keys": [
                "run_id",
                "workflow_id",
                "step_seq",
                "state_json",
                "checkpoint_schema_version",
            ],
            "ephemeral_keys": ["_deps", "_rt_join", "_rt"],
            "supported_wait_reasons": list(self._WAIT_REASONS),
            "compatible": bool(checkpoints),
            "state_keys": sorted(state.keys()) if isinstance(state, dict) else [],
        }
        return resume_keys

    def _latest_checkpoint_node(self, run_id: str) -> Any | None:
        meta = getattr(self._conversation_engine(), "meta_sqlite", None)
        get_projection = getattr(meta, "get_named_projection", None)
        if callable(get_projection):
            namespace = str(getattr(self._conversation_engine(), "namespace", "") or "")
            row = get_projection(
                workflow_checkpoint_latest_projection_namespace(namespace),
                str(run_id),
            )
            payload = dict((row or {}).get("payload") or {}) if row else {}
            node_id = str(payload.get("node_id") or "")
            if node_id:
                nodes = self._conversation_engine().read.get_nodes(ids=[node_id], limit=1)
                if nodes:
                    return nodes[0]

        nodes = self._workflow_nodes(entity_type="workflow_checkpoint", run_id=run_id)
        if not nodes:
            return None
        return max(
            nodes,
            key=lambda n: int((getattr(n, "metadata", {}) or {}).get("step_seq", -1)),
        )

    def _workflow_nodes(self, *, entity_type: str, run_id: str) -> list[Any]:
        try:
            return self._conversation_engine().get_nodes(
                where={"$and": [{"entity_type": entity_type}, {"run_id": run_id}]},
                limit=200_000,
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "Nothing found on disk" in msg or "hnsw segment reader" in msg:
                return []
            raise

    def list_steps(self, run_id: str) -> list[dict[str, Any]]:
        nodes = self._workflow_nodes(entity_type="workflow_step_exec", run_id=run_id)
        out: list[dict[str, Any]] = []
        for node in nodes:
            metadata = getattr(node, "metadata", {}) or {}
            raw = metadata.get("result_json")
            out.append(
                {
                    "node_id": str(getattr(node, "id", "") or ""),
                    "step_seq": int(metadata.get("step_seq", 0) or 0),
                    "workflow_id": str(metadata.get("workflow_id") or ""),
                    "workflow_node_id": str(metadata.get("workflow_node_id") or ""),
                    "op": str(metadata.get("op") or ""),
                    "status": str(metadata.get("status") or ""),
                    "duration_ms": int(metadata.get("duration_ms", 0) or 0),
                    "result": None if not raw else json.loads(str(raw)),
                }
            )
        out.sort(key=lambda item: int(item["step_seq"]))
        return out

    def list_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        nodes = self._workflow_nodes(entity_type="workflow_checkpoint", run_id=run_id)
        out: list[dict[str, Any]] = []
        for node in nodes:
            metadata = getattr(node, "metadata", {}) or {}
            out.append(
                {
                    "node_id": str(getattr(node, "id", "") or ""),
                    "step_seq": int(metadata.get("step_seq", 0) or 0),
                    "workflow_id": str(metadata.get("workflow_id") or ""),
                    "state": json.loads(str(metadata.get("state_json") or "{}")),
                }
            )
        out.sort(key=lambda item: int(item["step_seq"]))
        return out

    def get_checkpoint(self, run_id: str, step_seq: int) -> dict[str, Any]:
        state = load_checkpoint(
            conversation_engine=self._conversation_engine(),
            run_id=run_id,
            step_seq=step_seq,
        )
        return {
            "run_id": run_id,
            "step_seq": int(step_seq),
            "state": state,
        }

    def replay_run(self, run_id: str, target_step_seq: int) -> dict[str, Any]:
        state = replay_to(
            conversation_engine=self._conversation_engine(),
            run_id=run_id,
            target_step_seq=int(target_step_seq),
        )
        return {
            "run_id": run_id,
            "target_step_seq": int(target_step_seq),
            "state": state,
        }
