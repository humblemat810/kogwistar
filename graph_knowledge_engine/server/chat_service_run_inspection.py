"""Run inspection helpers for workflow trace lookup, checkpoints, and replay.

This module serves the read-only inspection surface for workflow runs. It
retrieves persisted step execution and checkpoint artifacts from the
conversation graph and delegates replay reconstruction to the runtime replay
helpers without owning run execution itself.
"""

from __future__ import annotations

import json
from typing import Any

from graph_knowledge_engine.runtime.replay import load_checkpoint, replay_to

from .chat_service_shared import _BaseComponent


class _RunInspectionService(_BaseComponent):
    """Owns step/checkpoint lookup and replay helpers."""

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
        state = load_checkpoint(conversation_engine=self._conversation_engine(), run_id=run_id, step_seq=step_seq)
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
