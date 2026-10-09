"""Run inspection helpers for workflow trace lookup, checkpoints, and replay.

This module serves the read-only inspection surface for workflow runs. It
retrieves persisted step execution and checkpoint artifacts from the
conversation graph and delegates replay reconstruction to the runtime replay
helpers without owning run execution itself.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from kogwistar.engine_core.models import Node
from kogwistar.json_types import JsonObject, JsonValue
from kogwistar.runtime.projections import (
    workflow_checkpoint_latest_projection_namespace,
)
from kogwistar.runtime.replay import load_checkpoint, replay_to

from .chat_service_shared import _BaseComponent, json_safe


def _node_metadata(node: Node) -> JsonObject:
    """Normalize legacy untyped node metadata at the JSON response boundary."""
    raw = getattr(node, "metadata", {})
    return _as_json_object(raw if isinstance(raw, Mapping) else {})


def _as_json_object(value: object) -> JsonObject:
    """Narrow a dynamic persisted value to the engine's JSON object contract."""
    converted = json_safe(value)
    return cast(JsonObject, converted) if isinstance(converted, dict) else {}


def _json_int(value: object, default: int = 0) -> int:
    """Convert persisted scalar metadata without accepting containers as integers."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            pass
    return default


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

    def resume_contract(self, run_id: str) -> JsonObject:
        checkpoints = self.list_checkpoints(run_id)
        latest = checkpoints[-1] if checkpoints else None
        state = latest.get("state") if latest is not None else None
        state_object = _as_json_object(state)
        latest_node = self._latest_checkpoint_node(run_id)
        resume_options: list[JsonObject] = []
        if state_object:
            suspended = (
                state_object.get("suspended_tokens")
                or state_object.get("_suspended_tokens")
                or {}
            )
            if isinstance(suspended, dict):
                for token_id, item in suspended.items():
                    if isinstance(item, (list, tuple)) and item:
                        resume_options.append(
                            {
                                "suspended_node_id": str(item[0]),
                                "suspended_token_id": str(token_id),
                            }
                        )
        resume_keys = {
            "run_id": run_id,
            "latest_checkpoint_step_seq": None if latest is None else latest.get("step_seq"),
            "latest_wait_reason": None
            if latest is None
            else state_object.get("wait_reason"),
            "checkpoint_schema_version": None
            if latest_node is None
            else _json_int(_node_metadata(latest_node).get("checkpoint_schema_version"), 1),
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
            "state_keys": sorted(state_object),
            "resume_options": resume_options,
        }
        return resume_keys

    def _latest_checkpoint_node(self, run_id: str) -> Node | None:
        meta = getattr(self._conversation_engine(), "meta_sqlite", None)
        get_projection = getattr(meta, "get_named_projection", None)
        if callable(get_projection):
            namespace = str(getattr(self._conversation_engine(), "namespace", "") or "")
            row = get_projection(
                workflow_checkpoint_latest_projection_namespace(namespace),
                str(run_id),
            )
            row_map = row if isinstance(row, Mapping) else {}
            raw_payload = row_map.get("payload")
            payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
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
            key=lambda n: _json_int(_node_metadata(n).get("step_seq"), -1),
        )

    def _workflow_nodes(self, *, entity_type: str, run_id: str) -> list[Node]:
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

    def list_steps(self, run_id: str) -> list[JsonObject]:
        nodes = self._workflow_nodes(entity_type="workflow_step_exec", run_id=run_id)
        out: list[JsonObject] = []
        for node in nodes:
            metadata = _node_metadata(node)
            raw = metadata.get("result_json")
            out.append(
                {
                    "node_id": str(getattr(node, "id", "") or ""),
                    "step_seq": _json_int(metadata.get("step_seq")),
                    "workflow_id": str(metadata.get("workflow_id") or ""),
                    "workflow_node_id": str(metadata.get("workflow_node_id") or ""),
                    "op": str(metadata.get("op") or ""),
                    "status": str(metadata.get("status") or ""),
                    "duration_ms": _json_int(metadata.get("duration_ms")),
                    "result": None if not raw else json.loads(str(raw)),
                }
            )
        out.sort(key=lambda item: _json_int(item.get("step_seq")))
        return out

    def workflow_run_lineage(self, run_id: str) -> JsonObject:
        """Return the persisted parent chain without exposing graph contents."""
        lineage: list[JsonObject] = []
        current = str(run_id)
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            rows = self._conversation_engine().read.get_nodes(
                ids=[f"wf_run|{current}"], limit=1
            )
            if not rows:
                if not lineage:
                    raise KeyError(f"Workflow run not found: {run_id}")
                break
            metadata = _node_metadata(rows[0])
            lineage.append(
                {
                    "run_id": current,
                    "workflow_id": str(metadata.get("workflow_id") or ""),
                    "parent_run_id": metadata.get("parent_run_id"),
                    "status": str(metadata.get("status") or ""),
                    "wf_invoked": metadata.get("wf_invoked"),
                }
            )
            current = str(metadata.get("parent_run_id") or "")
        return {
            "run_id": str(run_id),
            "lineage": cast(JsonValue, lineage),
        }

    def list_checkpoints(self, run_id: str) -> list[JsonObject]:
        nodes = self._workflow_nodes(entity_type="workflow_checkpoint", run_id=run_id)
        out: list[JsonObject] = []
        for node in nodes:
            metadata = _node_metadata(node)
            out.append(
                {
                    "node_id": str(getattr(node, "id", "") or ""),
                    "step_seq": _json_int(metadata.get("step_seq")),
                    "workflow_id": str(metadata.get("workflow_id") or ""),
                    "state": json.loads(str(metadata.get("state_json") or "{}")),
                }
            )
        out.sort(key=lambda item: _json_int(item.get("step_seq")))
        return out

    def get_checkpoint(self, run_id: str, step_seq: int) -> JsonObject:
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

    def replay_run(self, run_id: str, target_step_seq: int) -> JsonObject:
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
