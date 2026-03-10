from __future__ import annotations

import contextlib
import json
import logging
import pathlib
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from graph_knowledge_engine.conversation.agentic_answering import AgenticAnsweringAgent
from graph_knowledge_engine.conversation.models import (
    ConversationNode,
    FilteringResult,
    MetaFromLastSummary,
)
from graph_knowledge_engine.conversation.service import ConversationService
from graph_knowledge_engine.engine_core.models import Edge as CoreEdge
from graph_knowledge_engine.engine_core.models import Grounding, Node as CoreNode, Span
from graph_knowledge_engine.id_provider import new_id_str
from graph_knowledge_engine.runtime.replay import load_checkpoint, replay_to
from graph_knowledge_engine.runtime.models import WorkflowEdge, WorkflowNode
from graph_knowledge_engine.runtime.runtime import _get_shared_sqlite_sink
from graph_knowledge_engine.runtime.telemetry import EventEmitter

from .run_registry import RunRegistry, RunRegistryTraceBridge


class RunCancelledError(RuntimeError):
    """Raised when a submitted chat run is cancelled cooperatively."""


class WorkflowProjectionRebuildingError(RuntimeError):
    """Raised when a workflow design projection is being rebuilt."""


@dataclass(frozen=True)
class AnswerRunRequest:
    run_id: str
    conversation_id: str
    user_id: str
    user_text: str
    user_turn_node_id: str
    workflow_id: str
    knowledge_engine: Any
    conversation_engine: Any
    workflow_engine: Any
    prev_turn_meta_summary: MetaFromLastSummary
    registry: RunRegistry
    publish: Callable[[str, dict[str, Any] | None], dict[str, Any]]
    is_cancel_requested: Callable[[], bool]


@dataclass(frozen=True)
class RuntimeRunRequest:
    run_id: str
    workflow_id: str
    conversation_id: str
    turn_node_id: str
    user_id: str | None
    initial_state: dict[str, Any]
    knowledge_engine: Any
    conversation_engine: Any
    workflow_engine: Any
    registry: RunRegistry
    publish: Callable[[str, dict[str, Any] | None], dict[str, Any]]
    is_cancel_requested: Callable[[], bool]


class ChatRunService:
    _DESIGN_CONTROL_KIND = "design_control"
    _CTRL_UNDO_APPLIED = "UNDO_APPLIED"
    _CTRL_REDO_APPLIED = "REDO_APPLIED"
    _CTRL_BRANCH_DROPPED = "BRANCH_DROPPED"
    _CTRL_MUTATION_COMMITTED = "MUTATION_COMMITTED"
    _PROJECTION_SCHEMA_VERSION = 1
    _SNAPSHOT_SCHEMA_VERSION = 1
    _DELTA_SCHEMA_VERSION = 1
    _SNAPSHOT_INTERVAL = 50

    def __init__(
        self,
        *,
        get_knowledge_engine: Callable[[], Any],
        get_conversation_engine: Callable[[], Any],
        get_workflow_engine: Callable[[], Any],
        run_registry: RunRegistry,
        answer_runner: Callable[[AnswerRunRequest], dict[str, Any]] | None = None,
        runtime_runner: Callable[[RuntimeRunRequest], dict[str, Any]] | None = None,
    ) -> None:
        self._get_knowledge_engine = get_knowledge_engine
        self._get_conversation_engine = get_conversation_engine
        self._get_workflow_engine = get_workflow_engine
        self.run_registry = run_registry
        self.answer_runner = answer_runner or self._default_answer_runner
        self.runtime_runner = runtime_runner or self._default_runtime_runner
        self._workflow_history_lock = threading.Lock()

    def _knowledge_engine(self) -> Any:
        return self._get_knowledge_engine()

    def _conversation_engine(self) -> Any:
        return self._get_conversation_engine()

    def _workflow_engine(self) -> Any:
        return self._get_workflow_engine()

    def _conversation_service(self) -> ConversationService:
        return ConversationService.from_engine(
            self._conversation_engine(),
            knowledge_engine=self._knowledge_engine(),
            workflow_engine=self._workflow_engine(),
        )

    def _publish(self, run_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.run_registry.append_event(run_id, event_type, payload)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _workflow_namespace(workflow_id: str) -> str:
        return f"wf_design:{str(workflow_id)}"

    @contextlib.contextmanager
    def _workflow_namespace_scope(self, workflow_id: str):
        eng = self._workflow_engine()
        prev_ns = str(getattr(eng, "namespace", "default") or "default")
        target_ns = self._workflow_namespace(workflow_id)
        eng.namespace = target_ns
        try:
            yield eng
        finally:
            eng.namespace = prev_ns

    def _assert_designer_identity(self, *, designer_id: str, actor_sub: str | None) -> str:
        resolved_designer = str(designer_id or "").strip()
        if not resolved_designer:
            raise ValueError("designer_id is required")
        subject = str(actor_sub or "").strip()
        if subject and resolved_designer != subject:
            raise PermissionError("designer_id must match authenticated subject")
        return resolved_designer

    def _iter_entity_events(self, *, namespace: str, from_seq: int = 1, to_seq: int | None = None):
        iter_events = getattr(self._workflow_engine().meta_sqlite, "iter_entity_events", None)
        if not callable(iter_events):
            return
        kwargs: dict[str, Any] = {"namespace": str(namespace), "from_seq": int(from_seq)}
        if to_seq is not None:
            kwargs["to_seq"] = int(to_seq)
        try:
            yield from iter_events(**kwargs)
            return
        except TypeError:
            pass
        rows = iter_events(namespace=str(namespace), from_seq=int(from_seq))
        max_seq = None if to_seq is None else int(to_seq)
        for row in rows:
            seq = int(row[0])
            if max_seq is not None and seq > max_seq:
                break
            yield row

    @staticmethod
    def _parse_event_payload(payload_raw: Any) -> dict[str, Any]:
        if isinstance(payload_raw, dict):
            return dict(payload_raw)
        if isinstance(payload_raw, str):
            try:
                loaded = json.loads(payload_raw)
                return loaded if isinstance(loaded, dict) else {}
            except Exception:
                return {}
        return {}

    def _append_design_control_event(
        self,
        *,
        workflow_id: str,
        op: str,
        designer_id: str,
        source: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        append = getattr(self._workflow_engine().meta_sqlite, "append_entity_event", None)
        if not callable(append):
            return 0
        body = {
            "workflow_id": str(workflow_id),
            "designer_id": str(designer_id),
            "ts_ms": self._now_ms(),
            "source": str(source),
        }
        body.update(dict(payload or {}))
        return int(
            append(
                namespace=self._workflow_namespace(workflow_id),
                event_id=str(uuid.uuid4()),
                entity_kind=self._DESIGN_CONTROL_KIND,
                entity_id=str(workflow_id),
                op=str(op),
                payload_json=json.dumps(body, sort_keys=True, separators=(",", ":")),
            )
        )

    def _append_workflow_entity_event(
        self,
        *,
        workflow_id: str,
        entity_kind: str,
        entity_id: str,
        op: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        append = getattr(self._workflow_engine().meta_sqlite, "append_entity_event", None)
        if not callable(append):
            return 0
        return int(
            append(
                namespace=self._workflow_namespace(workflow_id),
                event_id=str(uuid.uuid4()),
                entity_kind=str(entity_kind),
                entity_id=str(entity_id),
                op=str(op),
                payload_json=json.dumps(dict(payload or {}), sort_keys=True, separators=(",", ":")),
            )
        )

    def _workflow_meta_store(self) -> Any:
        return self._workflow_engine().meta_sqlite

    def _workflow_projection(self, *, workflow_id: str) -> dict[str, Any] | None:
        getter = getattr(self._workflow_meta_store(), "get_workflow_design_projection", None)
        if not callable(getter):
            return None
        return getter(workflow_id=str(workflow_id))

    @staticmethod
    def _workflow_empty_delta() -> dict[str, Any]:
        return {
            "upsert_nodes": [],
            "delete_node_ids": [],
            "upsert_edges": [],
            "delete_edge_ids": [],
        }

    @staticmethod
    def _workflow_compute_visible_delta(*, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        before_nodes = {str(item.get("id") or ""): item for item in list(before.get("nodes") or []) if str(item.get("id") or "")}
        after_nodes = {str(item.get("id") or ""): item for item in list(after.get("nodes") or []) if str(item.get("id") or "")}
        before_edges = {str(item.get("id") or ""): item for item in list(before.get("edges") or []) if str(item.get("id") or "")}
        after_edges = {str(item.get("id") or ""): item for item in list(after.get("edges") or []) if str(item.get("id") or "")}
        return {
            "upsert_nodes": [
                after_nodes[node_id]
                for node_id in sorted(after_nodes.keys())
                if before_nodes.get(node_id) != after_nodes[node_id]
            ],
            "delete_node_ids": sorted(node_id for node_id in before_nodes.keys() if node_id not in after_nodes),
            "upsert_edges": [
                after_edges[edge_id]
                for edge_id in sorted(after_edges.keys())
                if before_edges.get(edge_id) != after_edges[edge_id]
            ],
            "delete_edge_ids": sorted(edge_id for edge_id in before_edges.keys() if edge_id not in after_edges),
        }

    def _workflow_store_version_delta(
        self,
        *,
        workflow_id: str,
        version: int,
        prev_version: int,
        target_seq: int,
        forward: dict[str, Any],
        inverse: dict[str, Any],
    ) -> None:
        putter = getattr(self._workflow_meta_store(), "put_workflow_design_delta", None)
        if not callable(putter):
            return
        putter(
            workflow_id=workflow_id,
            version=int(version),
            prev_version=int(prev_version),
            target_seq=int(target_seq),
            forward_json=json.dumps(dict(forward or self._workflow_empty_delta()), sort_keys=True, separators=(",", ":")),
            inverse_json=json.dumps(dict(inverse or self._workflow_empty_delta()), sort_keys=True, separators=(",", ":")),
            schema_version=self._DELTA_SCHEMA_VERSION,
        )

    def _workflow_version_delta(self, *, workflow_id: str, version: int) -> dict[str, Any] | None:
        getter = getattr(self._workflow_meta_store(), "get_workflow_design_delta", None)
        if not callable(getter):
            return None
        row = getter(
            workflow_id=str(workflow_id),
            version=int(version),
            schema_version=self._DELTA_SCHEMA_VERSION,
        )
        if row is None:
            return None
        return {
            "workflow_id": str(row.get("workflow_id") or workflow_id),
            "version": int(row.get("version") or version),
            "prev_version": int(row.get("prev_version") or 0),
            "target_seq": int(row.get("target_seq") or 0),
            "forward": self._parse_event_payload(str(row.get("forward_json") or "{}")),
            "inverse": self._parse_event_payload(str(row.get("inverse_json") or "{}")),
            "schema_version": int(row.get("schema_version") or self._DELTA_SCHEMA_VERSION),
            "created_at_ms": int(row.get("created_at_ms") or 0),
        }

    def _workflow_apply_visible_delta(self, *, workflow_id: str, delta: dict[str, Any] | None) -> None:
        payload = dict(delta or self._workflow_empty_delta())
        eng = self._workflow_engine()
        prev_log = getattr(eng, "_disable_event_log", False)
        prev_idx = getattr(eng, "_phase1_enable_index_jobs", False)
        eng._disable_event_log = True
        eng._phase1_enable_index_jobs = False
        try:
            with eng.uow():
                edge_delete_ids = sorted({str(item) for item in list(payload.get("delete_edge_ids") or []) if str(item)})
                node_delete_ids = sorted({str(item) for item in list(payload.get("delete_node_ids") or []) if str(item)})
                if edge_delete_ids:
                    eng.backend.edge_delete(ids=edge_delete_ids)
                if node_delete_ids:
                    eng.backend.node_delete(ids=node_delete_ids)
                for raw in list(payload.get("upsert_nodes") or []):
                    eng.write.add_node(WorkflowNode.model_validate(raw))
                for raw in list(payload.get("upsert_edges") or []):
                    eng.write.add_edge(WorkflowEdge.model_validate(raw))
                self._workflow_remove_orphan_edges(workflow_id=workflow_id)
        finally:
            eng._disable_event_log = prev_log
            eng._phase1_enable_index_jobs = prev_idx

    def _workflow_control_timeline(self, *, namespace: str, limit: int = 500) -> list[dict[str, Any]]:
        keep = max(1, int(limit))
        out: deque[dict[str, Any]] = deque(maxlen=keep)
        for seq, entity_kind, _entity_id, op, payload_raw in self._iter_entity_events(
            namespace=namespace,
            from_seq=1,
        ):
            if str(entity_kind) != self._DESIGN_CONTROL_KIND:
                continue
            payload = self._parse_event_payload(payload_raw)
            item: dict[str, Any] = {
                "seq": int(seq),
                "op": str(op),
                "designer_id": str(payload.get("designer_id") or ""),
                "ts_ms": int(payload.get("ts_ms") or 0),
            }
            for key, value in payload.items():
                if key in {"designer_id", "ts_ms"}:
                    continue
                item[key] = value
            out.append(item)
        return list(out)

    def _workflow_lineage_path(
        self,
        *,
        commits: dict[int, dict[str, Any]],
        version: int,
    ) -> list[dict[str, Any]]:
        if int(version) <= 0:
            return [{"version": 0, "prev_version": 0, "target_seq": 0, "created_at_ms": 0}]
        path: list[dict[str, Any]] = []
        seen: set[int] = set()
        current = int(version)
        while current > 0:
            if current in seen:
                raise RuntimeError(f"Workflow design lineage loop detected at version={current}")
            seen.add(current)
            commit = commits.get(current)
            if commit is None:
                raise RuntimeError(f"Workflow design history missing committed version={current}")
            path.append(commit)
            current = int(commit.get("prev_version") or 0)
        path.reverse()
        return [{"version": 0, "prev_version": 0, "target_seq": 0, "created_at_ms": 0}] + path

    def _workflow_fold_history(self, *, workflow_id: str) -> dict[str, Any]:
        namespace = self._workflow_namespace(workflow_id)
        commits: dict[int, dict[str, Any]] = {}
        dropped_ranges: list[dict[str, Any]] = []
        current_version = 0
        active_tip_version = 0
        allocated_max_version = 0
        latest_seq = 0

        for seq, entity_kind, _entity_id, op, payload_raw in self._iter_entity_events(namespace=namespace, from_seq=1):
            latest_seq = max(latest_seq, int(seq))
            if str(entity_kind) != self._DESIGN_CONTROL_KIND:
                continue
            payload = self._parse_event_payload(payload_raw)
            op_s = str(op or "")
            if op_s == self._CTRL_MUTATION_COMMITTED:
                version = int(payload.get("version") or 0)
                prev_version = int(payload.get("prev_version") or 0)
                target_seq = int(payload.get("target_seq") or payload.get("seq") or 0)
                commit = {
                    "version": version,
                    "prev_version": prev_version,
                    "target_seq": target_seq,
                    "created_at_ms": int(payload.get("ts_ms") or 0),
                    "entity_id": str(payload.get("entity_id") or ""),
                    "action": str(payload.get("action") or ""),
                }
                commits[version] = commit
                allocated_max_version = max(allocated_max_version, version)
                current_version = version
                active_tip_version = version
            elif op_s == self._CTRL_UNDO_APPLIED:
                current_version = int(payload.get("to_version") or current_version)
            elif op_s == self._CTRL_REDO_APPLIED:
                current_version = int(payload.get("to_version") or current_version)
            elif op_s == self._CTRL_BRANCH_DROPPED:
                start_version = int(payload.get("drop_from_version") or 0)
                end_version = int(payload.get("drop_to_version") or -1)
                start_seq = int(payload.get("drop_from_seq") or 0)
                end_seq = int(payload.get("drop_to_seq") or -1)
                if end_version >= start_version >= 0 and end_seq >= start_seq >= 0:
                    dropped_ranges.append(
                        {
                            "start_version": start_version,
                            "end_version": end_version,
                            "start_seq": start_seq,
                            "end_seq": end_seq,
                        }
                    )
                    if start_version <= active_tip_version <= end_version:
                        active_tip_version = current_version

        active_lineage = self._workflow_lineage_path(commits=commits, version=active_tip_version)
        selected_lineage = self._workflow_lineage_path(commits=commits, version=current_version)
        active_versions = [
            {
                "version": int(item.get("version") or 0),
                "seq": int(item.get("target_seq") or 0),
                "created_at_ms": int(item.get("created_at_ms") or 0),
            }
            for item in active_lineage
        ]
        selected_versions = [
            {
                "version": int(item.get("version") or 0),
                "seq": int(item.get("target_seq") or 0),
                "created_at_ms": int(item.get("created_at_ms") or 0),
                "prev_version": int(item.get("prev_version") or 0),
                "target_seq": int(item.get("target_seq") or 0),
            }
            for item in selected_lineage
        ]
        active_ids = [int(item.get("version") or 0) for item in active_lineage]
        current_seq = 0
        if current_version > 0:
            current_seq = int(commits.get(current_version, {}).get("target_seq") or 0)
        return {
            "workflow_id": workflow_id,
            "namespace": namespace,
            "current_version": int(current_version),
            "active_tip_version": int(active_tip_version),
            "max_version": int(active_tip_version),
            "allocated_max_version": int(allocated_max_version),
            "current_seq": int(current_seq),
            "can_undo": int(current_version) > 0,
            "can_redo": int(current_version) in active_ids and active_ids.index(int(current_version)) < len(active_ids) - 1,
            "versions": active_versions,
            "selected_versions": selected_versions,
            "dropped_ranges": dropped_ranges,
            "latest_seq": int(latest_seq),
            "timeline": self._workflow_control_timeline(namespace=namespace, limit=500),
            "commits": commits,
        }

    def _workflow_projection_stale(self, *, state: dict[str, Any], projection: dict[str, Any] | None) -> bool:
        if projection is None:
            return True
        if int(projection.get("projection_schema_version") or 0) != self._PROJECTION_SCHEMA_VERSION:
            return True
        if int(projection.get("snapshot_schema_version") or 0) != self._SNAPSHOT_SCHEMA_VERSION:
            return True
        if int(projection.get("last_authoritative_seq") or 0) < int(state.get("latest_seq") or 0):
            return True
        if str(projection.get("materialization_status") or "") != "ready":
            return True
        return False

    def _workflow_projection_head(self, *, state: dict[str, Any], materialization_status: str = "ready") -> dict[str, Any]:
        return {
            "current_version": int(state.get("current_version") or 0),
            "active_tip_version": int(state.get("active_tip_version") or 0),
            "last_authoritative_seq": int(state.get("latest_seq") or 0),
            "last_materialized_seq": int(state.get("current_seq") or 0),
            "projection_schema_version": self._PROJECTION_SCHEMA_VERSION,
            "snapshot_schema_version": self._SNAPSHOT_SCHEMA_VERSION,
            "materialization_status": str(materialization_status),
            "updated_at_ms": self._now_ms(),
        }

    def _store_workflow_projection(self, *, workflow_id: str, state: dict[str, Any], materialization_status: str = "ready") -> None:
        replace = getattr(self._workflow_meta_store(), "replace_workflow_design_projection", None)
        if not callable(replace):
            return
        replace(
            workflow_id=workflow_id,
            head=self._workflow_projection_head(state=state, materialization_status=materialization_status),
            versions=[
                {
                    "version": int(item.get("version") or 0),
                    "prev_version": int(item.get("prev_version") or 0),
                    "target_seq": int(item.get("target_seq") or 0),
                    "created_at_ms": int(item.get("created_at_ms") or 0),
                }
                for item in state.get("selected_versions") or []
            ]
            + [
                {
                    "version": int(item.get("version") or 0),
                    "prev_version": int((state.get("commits") or {}).get(int(item.get("version") or 0), {}).get("prev_version") or 0),
                    "target_seq": int(item.get("seq") or 0),
                    "created_at_ms": int(item.get("created_at_ms") or 0),
                }
                for item in (state.get("versions") or [])
                if int(item.get("version") or 0) not in {int(v.get("version") or 0) for v in (state.get("selected_versions") or [])}
            ],
            dropped_ranges=list(state.get("dropped_ranges") or []),
        )

    def _workflow_projection_status(self, *, workflow_id: str) -> str | None:
        projection = self._workflow_projection(workflow_id=workflow_id)
        if projection is None:
            return None
        return str(projection.get("materialization_status") or "")

    def _assert_workflow_projection_not_rebuilding(self, *, workflow_id: str) -> None:
        if self._workflow_projection_status(workflow_id=workflow_id) == "rebuilding":
            raise WorkflowProjectionRebuildingError(f"workflow_id={workflow_id!r} is rebuilding; retry later")

    def _workflow_latest_seq(self, *, namespace: str, from_seq: int = 1) -> int:
        getter = getattr(self._workflow_meta_store(), "get_latest_entity_event_seq", None)
        if callable(getter) and int(from_seq) <= 1:
            return int(getter(namespace=namespace))
        last = 0
        for seq, _ek, _eid, _op, _payload in self._iter_entity_events(namespace=namespace, from_seq=max(1, int(from_seq))):
            last = int(seq)
        return last

    def _workflow_collect_visible_entity_ids(self, *, workflow_id: str) -> tuple[set[str], set[str]]:
        nodes = self._workflow_engine().get_nodes(
            where={"$and": [{"entity_type": "workflow_node"}, {"workflow_id": workflow_id}]},
            limit=5000,
            node_type=WorkflowNode,
        )
        edges = self._workflow_engine().get_edges(
            where={"$and": [{"entity_type": "workflow_edge"}, {"workflow_id": workflow_id}]},
            limit=20_000,
            edge_type=WorkflowEdge,
        )
        return (
            {str(node.id) for node in nodes},
            {str(edge.id) for edge in edges},
        )

    @staticmethod
    def _workflow_seq_in_dropped_ranges(seq: int, dropped_ranges: list[dict[str, Any]]) -> bool:
        for item in dropped_ranges:
            start_seq = int(item.get("start_seq") or 0)
            end_seq = int(item.get("end_seq") or -1)
            if start_seq <= seq <= end_seq:
                return True
        return False

    def _workflow_replay_entity_range(
        self,
        *,
        namespace: str,
        from_seq: int,
        to_seq: int,
        dropped_ranges: list[dict[str, Any]] | None = None,
    ) -> None:
        eng = self._workflow_engine()
        prev_log = getattr(eng, "_disable_event_log", False)
        prev_idx = getattr(eng, "_phase1_enable_index_jobs", False)
        eng._disable_event_log = True
        eng._phase1_enable_index_jobs = False
        try:
            for seq, entity_kind, entity_id, op, payload_raw in self._iter_entity_events(
                namespace=namespace,
                from_seq=max(1, int(from_seq)),
                to_seq=int(to_seq),
            ):
                if self._workflow_seq_in_dropped_ranges(int(seq), list(dropped_ranges or [])):
                    continue
                entity_kind_s = str(entity_kind)
                if entity_kind_s == self._DESIGN_CONTROL_KIND:
                    continue
                payload = self._parse_event_payload(payload_raw)
                op_s = str(op)
                if entity_kind_s == "node":
                    if op_s in {"ADD", "REPLACE"}:
                        try:
                            node = CoreNode.model_validate(payload)
                        except Exception:
                            node = CoreNode.model_validate_json(json.dumps(payload))
                        eng.write.add_node(node)
                    elif op_s in {"TOMBSTONE", "DELETE"}:
                        eng.backend.node_delete(ids=[str(entity_id)])
                elif entity_kind_s == "edge":
                    if op_s in {"ADD", "REPLACE"}:
                        try:
                            edge = CoreEdge.model_validate(payload)
                        except Exception:
                            edge = CoreEdge.model_validate_json(json.dumps(payload))
                        eng.write.add_edge(edge)
                    elif op_s in {"TOMBSTONE", "DELETE"}:
                        eng.backend.edge_delete(ids=[str(entity_id)])
        finally:
            eng._disable_event_log = prev_log
            eng._phase1_enable_index_jobs = prev_idx

    def _workflow_capture_visible_snapshot(self, *, workflow_id: str) -> dict[str, Any]:
        nodes = self._workflow_engine().get_nodes(
            where={"$and": [{"entity_type": "workflow_node"}, {"workflow_id": workflow_id}]},
            limit=5000,
            node_type=WorkflowNode,
        )
        edges = self._workflow_engine().get_edges(
            where={"$and": [{"entity_type": "workflow_edge"}, {"workflow_id": workflow_id}]},
            limit=20_000,
            edge_type=WorkflowEdge,
        )
        return {
            "nodes": [node.model_dump(field_mode="backend", exclude={"embedding"}) for node in nodes],
            "edges": [edge.model_dump(field_mode="backend", exclude={"embedding"}) for edge in edges],
        }

    def _workflow_restore_snapshot(self, *, snapshot_payload: dict[str, Any]) -> None:
        eng = self._workflow_engine()
        prev_log = getattr(eng, "_disable_event_log", False)
        prev_idx = getattr(eng, "_phase1_enable_index_jobs", False)
        eng._disable_event_log = True
        eng._phase1_enable_index_jobs = False
        try:
            for raw in list(snapshot_payload.get("nodes") or []):
                node = WorkflowNode.model_validate(raw)
                eng.write.add_node(node)
            for raw in list(snapshot_payload.get("edges") or []):
                edge = WorkflowEdge.model_validate(raw)
                eng.write.add_edge(edge)
        finally:
            eng._disable_event_log = prev_log
            eng._phase1_enable_index_jobs = prev_idx

    def _workflow_remove_orphan_edges(self, *, workflow_id: str) -> None:
        nodes = self._workflow_engine().get_nodes(
            where={"$and": [{"entity_type": "workflow_node"}, {"workflow_id": workflow_id}]},
            limit=5000,
            node_type=WorkflowNode,
        )
        node_ids = {str(node.id) for node in nodes}
        edges = self._workflow_engine().get_edges(
            where={"$and": [{"entity_type": "workflow_edge"}, {"workflow_id": workflow_id}]},
            limit=20_000,
            edge_type=WorkflowEdge,
        )
        orphan_ids = [
            str(edge.id)
            for edge in edges
            if not getattr(edge, "source_ids", None)
            or not getattr(edge, "target_ids", None)
            or str(edge.source_ids[0]) not in node_ids
            or str(edge.target_ids[0]) not in node_ids
        ]
        if orphan_ids:
            self._workflow_engine().backend.edge_delete(ids=sorted(orphan_ids))

    def _workflow_store_snapshot_if_needed(self, *, workflow_id: str, state: dict[str, Any]) -> None:
        current_version = int(state.get("current_version") or 0)
        if current_version <= 0 or current_version % self._SNAPSHOT_INTERVAL != 0:
            return
        put_snapshot = getattr(self._workflow_meta_store(), "put_workflow_design_snapshot", None)
        if not callable(put_snapshot):
            return
        payload = self._workflow_capture_visible_snapshot(workflow_id=workflow_id)
        put_snapshot(
            workflow_id=workflow_id,
            version=current_version,
            seq=int(state.get("current_seq") or 0),
            payload_json=json.dumps(payload, sort_keys=True, separators=(",", ":")),
            schema_version=self._SNAPSHOT_SCHEMA_VERSION,
        )

    def _workflow_rebuild_namespace_for_state(self, *, workflow_id: str, state: dict[str, Any]) -> None:
        namespace = self._workflow_namespace(workflow_id)
        eng = self._workflow_engine()
        node_ids, edge_ids = self._workflow_collect_visible_entity_ids(workflow_id=workflow_id)
        if edge_ids:
            try:
                eng.backend.edge_delete(ids=sorted(edge_ids))
            except Exception:
                logging.getLogger(__name__).exception("failed clearing workflow edges during rebuild: namespace=%s", namespace)
        if node_ids:
            try:
                eng.backend.node_delete(ids=sorted(node_ids))
            except Exception:
                logging.getLogger(__name__).exception("failed clearing workflow nodes during rebuild: namespace=%s", namespace)
        snapshot_get = getattr(self._workflow_meta_store(), "get_workflow_design_snapshot", None)
        selected_versions = list(state.get("selected_versions") or [])
        selected_by_version = {int(item.get("version") or 0): item for item in selected_versions}
        restored_version = 0
        if callable(snapshot_get) and int(state.get("current_version") or 0) > 0:
            snapshot = snapshot_get(
                workflow_id=workflow_id,
                max_version=int(state.get("current_version") or 0),
                schema_version=self._SNAPSHOT_SCHEMA_VERSION,
            )
            if snapshot is not None and int(snapshot.get("version") or 0) in selected_by_version:
                try:
                    payload = self._parse_event_payload(str(snapshot.get("payload_json") or "{}"))
                    self._workflow_restore_snapshot(snapshot_payload=payload)
                    restored_version = int(snapshot.get("version") or 0)
                except Exception:
                    logging.getLogger(__name__).exception("failed restoring workflow snapshot: workflow_id=%s", workflow_id)
                    restored_version = 0
        for item in selected_versions:
            version = int(item.get("version") or 0)
            if version <= 0 or version <= restored_version:
                continue
            lower = int(item.get("prev_version") or 0)
            lower_seq = 0
            if lower > 0:
                lower_seq = int(selected_by_version.get(lower, {}).get("target_seq") or 0)
            upper_seq = int(item.get("target_seq") or 0)
            if upper_seq > lower_seq:
                self._workflow_replay_entity_range(
                    namespace=namespace,
                    from_seq=lower_seq + 1,
                    to_seq=upper_seq,
                    dropped_ranges=list(state.get("dropped_ranges") or []),
                )
        self._workflow_remove_orphan_edges(workflow_id=workflow_id)

    def _workflow_sync_projection_locked(
        self,
        *,
        workflow_id: str,
        force_rebuild: bool = False,
        materialize: bool = True,
    ) -> dict[str, Any]:
        state = self._workflow_fold_history(workflow_id=workflow_id)
        projection = self._workflow_projection(workflow_id=workflow_id)
        if force_rebuild or materialize and self._workflow_projection_stale(state=state, projection=projection):
            self._workflow_rebuild_namespace_for_state(workflow_id=workflow_id, state=state)
            self._workflow_store_snapshot_if_needed(workflow_id=workflow_id, state=state)
        self._store_workflow_projection(workflow_id=workflow_id, state=state, materialization_status="ready")
        return state

    def workflow_design_history(self, *, workflow_id: str) -> dict[str, Any]:
        workflow_id = str(workflow_id or "").strip()
        if not workflow_id:
            raise ValueError("workflow_id is required")
        with self._workflow_history_lock:
            return self._workflow_sync_projection_locked(workflow_id=workflow_id, materialize=False)

    def _workflow_append_branch_drop_if_needed_locked(
        self,
        *,
        workflow_id: str,
        state: dict[str, Any],
        designer_id: str,
        source: str,
    ) -> bool:
        current_version = int(state.get("current_version") or 0)
        dropped_versions = [item for item in (state.get("versions") or []) if int(item.get("version") or 0) > current_version]
        if not dropped_versions:
            return False
        self._append_design_control_event(
            workflow_id=workflow_id,
            op=self._CTRL_BRANCH_DROPPED,
            designer_id=designer_id,
            source=source,
            payload={
                "drop_from_version": int(dropped_versions[0].get("version") or 0),
                "drop_to_version": int(dropped_versions[-1].get("version") or 0),
                "drop_from_seq": int(dropped_versions[0].get("seq") or 0),
                "drop_to_seq": int(dropped_versions[-1].get("seq") or 0),
                "reason": "new_edit_after_undo",
            },
        )
        return True

    def _workflow_finalize_design_state_locked(self, *, workflow_id: str, rebuild: bool) -> dict[str, Any]:
        state = self._workflow_fold_history(workflow_id=workflow_id)
        if rebuild:
            self._workflow_rebuild_namespace_for_state(workflow_id=workflow_id, state=state)
        self._store_workflow_projection(workflow_id=workflow_id, state=state, materialization_status="ready")
        self._workflow_store_snapshot_if_needed(workflow_id=workflow_id, state=state)
        return state

    def refresh_workflow_design_projection(self, *, workflow_id: str) -> dict[str, Any]:
        workflow_id = str(workflow_id or "").strip()
        if not workflow_id:
            raise ValueError("workflow_id is required")
        clear_snapshots = getattr(self._workflow_meta_store(), "clear_workflow_design_snapshots", None)
        with self._workflow_history_lock:
            state = self._workflow_fold_history(workflow_id=workflow_id)
            self._store_workflow_projection(workflow_id=workflow_id, state=state, materialization_status="rebuilding")
            if callable(clear_snapshots):
                clear_snapshots(workflow_id=workflow_id)
            out = self._workflow_finalize_design_state_locked(workflow_id=workflow_id, rebuild=True)
            out["status"] = "ok"
            return out

    def workflow_design_upsert_node(
        self,
        *,
        workflow_id: str,
        designer_id: str,
        node_id: str | None,
        label: str,
        op: str | None = None,
        start: bool = False,
        terminal: bool = False,
        fanout: bool = False,
        metadata: dict[str, Any] | None = None,
        actor_sub: str | None = None,
        source: str = "rest",
    ) -> dict[str, Any]:
        workflow_id = str(workflow_id or "").strip()
        if not workflow_id:
            raise ValueError("workflow_id is required")
        resolved_designer_id = self._assert_designer_identity(designer_id=designer_id, actor_sub=actor_sub)
        self._assert_workflow_projection_not_rebuilding(workflow_id=workflow_id)
        resolved_node_id = str(node_id or "").strip() or f"wf|{workflow_id}|n|{new_id_str()}"
        resolved_label = str(label or "").strip()
        if not resolved_label:
            raise ValueError("label is required")
        wf_op = str(op or ("end" if bool(terminal) else "noop"))
        md = dict(metadata or {})
        md.update(
            {
                "entity_type": "workflow_node",
                "workflow_id": workflow_id,
                "wf_op": wf_op,
                "wf_start": bool(start),
                "wf_terminal": bool(terminal),
                "wf_fanout": bool(fanout),
                "designer_id": resolved_designer_id,
            }
        )
        n = WorkflowNode(
            id=resolved_node_id,
            label=resolved_label,
            type="entity",
            summary=f"workflow node {resolved_label}",
            doc_id=f"workflow:{workflow_id}",
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            properties={},
            metadata=md,
            level_from_root=0,
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
        namespace = self._workflow_namespace(workflow_id)
        with self._workflow_history_lock:
            state_before = self._workflow_sync_projection_locked(workflow_id=workflow_id, materialize=True)
            before_visible = self._workflow_capture_visible_snapshot(workflow_id=workflow_id)
            known_node_ids = {str(item.get("id") or "") for item in list(before_visible.get("nodes") or [])}
            current_version = int(state_before.get("current_version") or 0)
            new_version = int(state_before.get("allocated_max_version") or 0) + 1
            branch_dropped = False
            with self._workflow_namespace_scope(workflow_id) as eng, eng.uow():
                branch_dropped = self._workflow_append_branch_drop_if_needed_locked(
                    workflow_id=workflow_id,
                    state=state_before,
                    designer_id=resolved_designer_id,
                    source=source,
                )
                eng.write.add_node(n)
                latest_seq = self._workflow_latest_seq(namespace=namespace, from_seq=1)
                self._append_design_control_event(
                    workflow_id=workflow_id,
                    op=self._CTRL_MUTATION_COMMITTED,
                    designer_id=resolved_designer_id,
                    source=source,
                    payload={
                        "action": "node_upsert",
                        "entity_id": resolved_node_id,
                        "version": new_version,
                        "prev_version": current_version,
                        "target_seq": int(latest_seq),
                    },
                )
            after_visible = self._workflow_capture_visible_snapshot(workflow_id=workflow_id)
            self._workflow_store_version_delta(
                workflow_id=workflow_id,
                version=new_version,
                prev_version=current_version,
                target_seq=int(latest_seq),
                forward=self._workflow_compute_visible_delta(before=before_visible, after=after_visible),
                inverse=self._workflow_compute_visible_delta(before=after_visible, after=before_visible),
            )
            history = self._workflow_finalize_design_state_locked(workflow_id=workflow_id, rebuild=branch_dropped)
        return {
            "workflow_id": workflow_id,
            "namespace": namespace,
            "node_id": resolved_node_id,
            "designer_id": resolved_designer_id,
            "version": int(history["current_version"]),
            "seq": int(history["current_seq"]),
            "can_undo": bool(history["can_undo"]),
            "can_redo": bool(history["can_redo"]),
        }

    def workflow_design_upsert_edge(
        self,
        *,
        workflow_id: str,
        designer_id: str,
        edge_id: str | None,
        src: str,
        dst: str,
        relation: str = "wf_next",
        predicate: str | None = None,
        priority: int = 100,
        is_default: bool = False,
        multiplicity: str = "one",
        metadata: dict[str, Any] | None = None,
        actor_sub: str | None = None,
        source: str = "rest",
    ) -> dict[str, Any]:
        workflow_id = str(workflow_id or "").strip()
        if not workflow_id:
            raise ValueError("workflow_id is required")
        resolved_designer_id = self._assert_designer_identity(designer_id=designer_id, actor_sub=actor_sub)
        self._assert_workflow_projection_not_rebuilding(workflow_id=workflow_id)
        src = str(src or "").strip()
        dst = str(dst or "").strip()
        if not src or not dst:
            raise ValueError("src and dst are required")
        resolved_edge_id = str(edge_id or "").strip() or f"wf|{workflow_id}|e|{new_id_str()}"
        md = dict(metadata or {})
        md.update(
            {
                "entity_type": "workflow_edge",
                "workflow_id": workflow_id,
                "wf_predicate": (None if predicate is None else str(predicate)),
                "wf_priority": int(priority),
                "wf_is_default": bool(is_default),
                "wf_multiplicity": str(multiplicity or "one"),
                "designer_id": resolved_designer_id,
            }
        )
        e = WorkflowEdge(
            id=resolved_edge_id,
            source_ids=[src],
            target_ids=[dst],
            relation=str(relation or "wf_next"),
            label=str(relation or "wf_next"),
            type="relationship",
            summary=f"workflow edge {src} -> {dst}",
            doc_id=f"workflow:{workflow_id}",
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            properties={},
            metadata=md,
            source_edge_ids=[],
            target_edge_ids=[],
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
        namespace = self._workflow_namespace(workflow_id)
        with self._workflow_history_lock:
            state_before = self._workflow_sync_projection_locked(workflow_id=workflow_id, materialize=True)
            before_visible = self._workflow_capture_visible_snapshot(workflow_id=workflow_id)
            known_node_ids = {str(item.get("id") or "") for item in list(before_visible.get("nodes") or [])}
            current_version = int(state_before.get("current_version") or 0)
            new_version = int(state_before.get("allocated_max_version") or 0) + 1
            branch_dropped = False
            with self._workflow_namespace_scope(workflow_id) as eng, eng.uow():
                branch_dropped = self._workflow_append_branch_drop_if_needed_locked(
                    workflow_id=workflow_id,
                    state=state_before,
                    designer_id=resolved_designer_id,
                    source=source,
                )
                eng.write.add_edge(e)
                latest_seq = self._workflow_latest_seq(namespace=namespace, from_seq=1)
                self._append_design_control_event(
                    workflow_id=workflow_id,
                    op=self._CTRL_MUTATION_COMMITTED,
                    designer_id=resolved_designer_id,
                    source=source,
                    payload={
                        "action": "edge_upsert",
                        "entity_id": resolved_edge_id,
                        "version": new_version,
                        "prev_version": current_version,
                        "target_seq": int(latest_seq),
                    },
                )
            after_visible = self._workflow_capture_visible_snapshot(workflow_id=workflow_id)
            self._workflow_store_version_delta(
                workflow_id=workflow_id,
                version=new_version,
                prev_version=current_version,
                target_seq=int(latest_seq),
                forward=self._workflow_compute_visible_delta(before=before_visible, after=after_visible),
                inverse=self._workflow_compute_visible_delta(before=after_visible, after=before_visible),
            )
            history = self._workflow_finalize_design_state_locked(workflow_id=workflow_id, rebuild=branch_dropped)
        return {
            "workflow_id": workflow_id,
            "namespace": namespace,
            "edge_id": resolved_edge_id,
            "designer_id": resolved_designer_id,
            "version": int(history["current_version"]),
            "seq": int(history["current_seq"]),
            "can_undo": bool(history["can_undo"]),
            "can_redo": bool(history["can_redo"]),
        }

    def workflow_design_delete_node(
        self,
        *,
        workflow_id: str,
        node_id: str,
        designer_id: str,
        actor_sub: str | None = None,
        source: str = "rest",
    ) -> dict[str, Any]:
        workflow_id = str(workflow_id or "").strip()
        node_id = str(node_id or "").strip()
        if not workflow_id:
            raise ValueError("workflow_id is required")
        if not node_id:
            raise ValueError("node_id is required")
        resolved_designer_id = self._assert_designer_identity(designer_id=designer_id, actor_sub=actor_sub)
        self._assert_workflow_projection_not_rebuilding(workflow_id=workflow_id)
        namespace = self._workflow_namespace(workflow_id)
        with self._workflow_history_lock:
            state_before = self._workflow_sync_projection_locked(workflow_id=workflow_id, materialize=True)
            before_visible = self._workflow_capture_visible_snapshot(workflow_id=workflow_id)
            known_node_ids = {str(item.get("id") or "") for item in list(before_visible.get("nodes") or [])}
            current_version = int(state_before.get("current_version") or 0)
            new_version = int(state_before.get("allocated_max_version") or 0) + 1
            branch_dropped = False
            with self._workflow_namespace_scope(workflow_id) as eng, eng.uow():
                branch_dropped = self._workflow_append_branch_drop_if_needed_locked(
                    workflow_id=workflow_id,
                    state=state_before,
                    designer_id=resolved_designer_id,
                    source=source,
                )
                if node_id not in known_node_ids:
                    raise KeyError(f"Unknown node_id: {node_id}")
                self._append_workflow_entity_event(
                    workflow_id=workflow_id,
                    entity_kind="node",
                    entity_id=node_id,
                    op="TOMBSTONE",
                    payload={
                        "entity_id": node_id,
                        "reason": "workflow_design_delete",
                        "deleted_by": resolved_designer_id,
                    },
                )
                eng.backend.node_delete(ids=[node_id])
                self._workflow_remove_orphan_edges(workflow_id=workflow_id)
                latest_seq = self._workflow_latest_seq(namespace=namespace, from_seq=1)
                self._append_design_control_event(
                    workflow_id=workflow_id,
                    op=self._CTRL_MUTATION_COMMITTED,
                    designer_id=resolved_designer_id,
                    source=source,
                    payload={
                        "action": "node_delete",
                        "entity_id": node_id,
                        "version": new_version,
                        "prev_version": current_version,
                        "target_seq": int(latest_seq),
                    },
                )
            after_visible = self._workflow_capture_visible_snapshot(workflow_id=workflow_id)
            self._workflow_store_version_delta(
                workflow_id=workflow_id,
                version=new_version,
                prev_version=current_version,
                target_seq=int(latest_seq),
                forward=self._workflow_compute_visible_delta(before=before_visible, after=after_visible),
                inverse=self._workflow_compute_visible_delta(before=after_visible, after=before_visible),
            )
            history = self._workflow_finalize_design_state_locked(workflow_id=workflow_id, rebuild=branch_dropped)
        return {
            "workflow_id": workflow_id,
            "namespace": namespace,
            "node_id": node_id,
            "designer_id": resolved_designer_id,
            "deleted": True,
            "version": int(history["current_version"]),
            "seq": int(history["current_seq"]),
            "can_undo": bool(history["can_undo"]),
            "can_redo": bool(history["can_redo"]),
        }

    def workflow_design_delete_edge(
        self,
        *,
        workflow_id: str,
        edge_id: str,
        designer_id: str,
        actor_sub: str | None = None,
        source: str = "rest",
    ) -> dict[str, Any]:
        workflow_id = str(workflow_id or "").strip()
        edge_id = str(edge_id or "").strip()
        if not workflow_id:
            raise ValueError("workflow_id is required")
        if not edge_id:
            raise ValueError("edge_id is required")
        resolved_designer_id = self._assert_designer_identity(designer_id=designer_id, actor_sub=actor_sub)
        self._assert_workflow_projection_not_rebuilding(workflow_id=workflow_id)
        namespace = self._workflow_namespace(workflow_id)
        with self._workflow_history_lock:
            state_before = self._workflow_sync_projection_locked(workflow_id=workflow_id, materialize=True)
            before_visible = self._workflow_capture_visible_snapshot(workflow_id=workflow_id)
            known_edge_ids = {str(item.get("id") or "") for item in list(before_visible.get("edges") or [])}
            current_version = int(state_before.get("current_version") or 0)
            new_version = int(state_before.get("allocated_max_version") or 0) + 1
            branch_dropped = False
            with self._workflow_namespace_scope(workflow_id) as eng, eng.uow():
                branch_dropped = self._workflow_append_branch_drop_if_needed_locked(
                    workflow_id=workflow_id,
                    state=state_before,
                    designer_id=resolved_designer_id,
                    source=source,
                )
                if edge_id not in known_edge_ids:
                    raise KeyError(f"Unknown edge_id: {edge_id}")
                self._append_workflow_entity_event(
                    workflow_id=workflow_id,
                    entity_kind="edge",
                    entity_id=edge_id,
                    op="TOMBSTONE",
                    payload={
                        "entity_id": edge_id,
                        "reason": "workflow_design_delete",
                        "deleted_by": resolved_designer_id,
                    },
                )
                eng.backend.edge_delete(ids=[edge_id])
                latest_seq = self._workflow_latest_seq(namespace=namespace, from_seq=1)
                self._append_design_control_event(
                    workflow_id=workflow_id,
                    op=self._CTRL_MUTATION_COMMITTED,
                    designer_id=resolved_designer_id,
                    source=source,
                    payload={
                        "action": "edge_delete",
                        "entity_id": edge_id,
                        "version": new_version,
                        "prev_version": current_version,
                        "target_seq": int(latest_seq),
                    },
                )
            after_visible = self._workflow_capture_visible_snapshot(workflow_id=workflow_id)
            self._workflow_store_version_delta(
                workflow_id=workflow_id,
                version=new_version,
                prev_version=current_version,
                target_seq=int(latest_seq),
                forward=self._workflow_compute_visible_delta(before=before_visible, after=after_visible),
                inverse=self._workflow_compute_visible_delta(before=after_visible, after=before_visible),
            )
            history = self._workflow_finalize_design_state_locked(workflow_id=workflow_id, rebuild=branch_dropped)
        return {
            "workflow_id": workflow_id,
            "namespace": namespace,
            "edge_id": edge_id,
            "designer_id": resolved_designer_id,
            "deleted": True,
            "version": int(history["current_version"]),
            "seq": int(history["current_seq"]),
            "can_undo": bool(history["can_undo"]),
            "can_redo": bool(history["can_redo"]),
        }

    def workflow_design_undo(
        self,
        *,
        workflow_id: str,
        designer_id: str,
        actor_sub: str | None = None,
        source: str = "rest",
    ) -> dict[str, Any]:
        workflow_id = str(workflow_id or "").strip()
        if not workflow_id:
            raise ValueError("workflow_id is required")
        resolved_designer_id = self._assert_designer_identity(designer_id=designer_id, actor_sub=actor_sub)
        self._assert_workflow_projection_not_rebuilding(workflow_id=workflow_id)
        with self._workflow_history_lock:
            state = self._workflow_sync_projection_locked(workflow_id=workflow_id, materialize=True)
            current_version = int(state.get("current_version") or 0)
            if current_version <= 0:
                state["status"] = "noop"
                self._store_workflow_projection(workflow_id=workflow_id, state=state, materialization_status="ready")
                return state
            selected_versions = list(state.get("selected_versions") or [])
            current_entry = next((item for item in selected_versions if int(item.get("version") or 0) == current_version), None)
            if current_entry is None:
                raise RuntimeError(f"Current workflow version missing from selected lineage: {current_version}")
            target_version = int(current_entry.get("prev_version") or 0)
            target_seq = 0
            if target_version > 0:
                target_entry = next(
                    (item for item in selected_versions if int(item.get("version") or 0) == target_version),
                    None,
                )
                if target_entry is None:
                    raise RuntimeError(f"Undo target missing from selected lineage: {target_version}")
                target_seq = int(target_entry.get("target_seq") or 0)
            with self._workflow_engine().uow():
                self._append_design_control_event(
                    workflow_id=workflow_id,
                    op=self._CTRL_UNDO_APPLIED,
                    designer_id=resolved_designer_id,
                    source=source,
                    payload={
                        "from_version": current_version,
                        "to_version": target_version,
                        "target_seq": int(target_seq),
                    },
                )
            out = self._workflow_fold_history(workflow_id=workflow_id)
            applied_delta = False
            delta = self._workflow_version_delta(workflow_id=workflow_id, version=current_version)
            if delta is not None:
                try:
                    self._workflow_apply_visible_delta(workflow_id=workflow_id, delta=dict(delta.get("inverse") or {}))
                    applied_delta = True
                except Exception:
                    logging.getLogger(__name__).exception(
                        "failed applying workflow inverse delta; falling back to rebuild: workflow_id=%s version=%s",
                        workflow_id,
                        current_version,
                    )
            if not applied_delta:
                self._workflow_rebuild_namespace_for_state(workflow_id=workflow_id, state=out)
            self._store_workflow_projection(workflow_id=workflow_id, state=out, materialization_status="ready")
            self._workflow_store_snapshot_if_needed(workflow_id=workflow_id, state=out)
            out["status"] = "ok"
            return out

    def workflow_design_redo(
        self,
        *,
        workflow_id: str,
        designer_id: str,
        actor_sub: str | None = None,
        source: str = "rest",
    ) -> dict[str, Any]:
        workflow_id = str(workflow_id or "").strip()
        if not workflow_id:
            raise ValueError("workflow_id is required")
        resolved_designer_id = self._assert_designer_identity(designer_id=designer_id, actor_sub=actor_sub)
        self._assert_workflow_projection_not_rebuilding(workflow_id=workflow_id)
        with self._workflow_history_lock:
            state = self._workflow_sync_projection_locked(workflow_id=workflow_id, materialize=True)
            current_version = int(state.get("current_version") or 0)
            active_versions = list(state.get("versions") or [])
            active_ids = [int(item.get("version") or 0) for item in active_versions]
            if current_version not in active_ids:
                raise RuntimeError(f"Current workflow version missing from active lineage: {current_version}")
            current_index = active_ids.index(current_version)
            if current_index >= len(active_ids) - 1:
                state["status"] = "noop"
                self._store_workflow_projection(workflow_id=workflow_id, state=state, materialization_status="ready")
                return state
            target_entry = active_versions[current_index + 1]
            target_version = int(target_entry.get("version") or 0)
            target_seq = int(target_entry.get("seq") or 0)
            with self._workflow_engine().uow():
                self._append_design_control_event(
                    workflow_id=workflow_id,
                    op=self._CTRL_REDO_APPLIED,
                    designer_id=resolved_designer_id,
                    source=source,
                    payload={
                        "from_version": current_version,
                        "to_version": target_version,
                        "target_seq": int(target_seq),
                    },
                )
            out = self._workflow_fold_history(workflow_id=workflow_id)
            applied_delta = False
            delta = self._workflow_version_delta(workflow_id=workflow_id, version=target_version)
            if delta is not None:
                try:
                    self._workflow_apply_visible_delta(workflow_id=workflow_id, delta=dict(delta.get("forward") or {}))
                    applied_delta = True
                except Exception:
                    logging.getLogger(__name__).exception(
                        "failed applying workflow forward delta; falling back to rebuild: workflow_id=%s version=%s",
                        workflow_id,
                        target_version,
                    )
            if not applied_delta:
                self._workflow_rebuild_namespace_for_state(workflow_id=workflow_id, state=out)
            self._store_workflow_projection(workflow_id=workflow_id, state=out, materialization_status="ready")
            self._workflow_store_snapshot_if_needed(workflow_id=workflow_id, state=out)
            out["status"] = "ok"
            return out

    def create_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        start_node_id: str | None = None,
    ) -> dict[str, Any]:
        svc = self._conversation_service()
        conv_id, start_id = svc.create_conversation(user_id=user_id, conv_id=conversation_id, node_id=start_node_id)
        return self.get_conversation(conv_id) | {"start_node_id": start_id}

    def _conversation_nodes(self, conversation_id: str) -> list[ConversationNode]:
        nodes = self._conversation_engine().get_nodes(
            where={"conversation_id": conversation_id},
            node_type=ConversationNode,
            limit=10_000,
        )
        if not nodes:
            raise KeyError(f"Unknown conversation_id: {conversation_id}")
        return nodes

    def _conversation_owner(self, conversation_id: str) -> str | None:
        starts = [
            node
            for node in self._conversation_nodes(conversation_id)
            if str((getattr(node, "metadata", {}) or {}).get("entity_type") or "") == "conversation_start"
        ]
        if not starts:
            return None
        starts.sort(key=lambda node: int(getattr(node, "turn_index", -1) or -1))
        return str(getattr(starts[0], "user_id", None) or "")

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        nodes = self._conversation_nodes(conversation_id)
        svc = self._conversation_service()
        tail = svc.get_conversation_tail(conversation_id=conversation_id)
        starts = [
            node
            for node in nodes
            if str((getattr(node, "metadata", {}) or {}).get("entity_type") or "") == "conversation_start"
        ]
        turns = self.list_transcript(conversation_id)
        start_node = starts[0] if starts else None
        return {
            "conversation_id": conversation_id,
            "user_id": str(getattr(start_node, "user_id", None) or ""),
            "status": str((getattr(start_node, "properties", {}) or {}).get("status") or "active"),
            "start_node_id": str(getattr(start_node, "id", None) or ""),
            "tail_node_id": str(getattr(tail, "id", None) or ""),
            "turn_count": len(turns),
        }

    def list_transcript(self, conversation_id: str) -> list[dict[str, Any]]:
        nodes = self._conversation_nodes(conversation_id)
        turns: list[dict[str, Any]] = []
        for node in nodes:
            metadata = getattr(node, "metadata", {}) or {}
            entity_type = str(metadata.get("entity_type") or "")
            if entity_type not in {"conversation_turn", "assistant_turn"}:
                continue
            turn_index = getattr(node, "turn_index", None)
            if turn_index is None:
                continue
            turns.append(
                {
                    "node_id": str(getattr(node, "id", "") or ""),
                    "turn_index": int(turn_index),
                    "role": str(getattr(node, "role", "") or ""),
                    "content": str(getattr(node, "summary", "") or ""),
                    "entity_type": entity_type,
                }
            )
        turns.sort(key=lambda item: (int(item["turn_index"]), str(item["node_id"])))
        return turns

    def latest_snapshot(
        self,
        conversation_id: str,
        *,
        run_id: str | None = None,
        stage: str | None = None,
    ) -> dict[str, Any]:
        svc = self._conversation_service()
        snap = svc.latest_context_snapshot_node(conversation_id=conversation_id, run_id=run_id, stage=stage)
        if snap is None:
            raise KeyError(f"No context snapshot found for conversation_id={conversation_id!r}")
        payload = svc.get_context_snapshot_payload(snapshot_node_id=str(snap.id))
        return {
            "snapshot_node_id": str(snap.id),
            "conversation_id": conversation_id,
            "metadata": payload.get("metadata") or {},
            "properties": payload.get("properties") or {},
        }

    def submit_turn_for_answer(
        self,
        *,
        conversation_id: str,
        user_id: str | None,
        text: str,
        workflow_id: str = "agentic_answering.v2",
    ) -> dict[str, Any]:
        text = str(text or "").strip()
        if not text:
            raise ValueError("text must be non-empty")

        resolved_user_id = str(user_id or self._conversation_owner(conversation_id) or "")
        if not resolved_user_id:
            raise ValueError("user_id is required for this conversation")

        svc = self._conversation_service()
        prev_turn_meta_summary = MetaFromLastSummary(0, 0)
        add_turn = svc.add_conversation_turn(
            user_id=resolved_user_id,
            conversation_id=conversation_id,
            turn_id=str(new_id_str()),
            mem_id=str(new_id_str()),
            role="user",
            content=text,
            ref_knowledge_engine=self._knowledge_engine(),
            filtering_callback=lambda *args, **kwargs: (FilteringResult(node_ids=[], edge_ids=[]), ""),
            prev_turn_meta_summary=prev_turn_meta_summary,
            add_turn_only=True,
        )

        run_id = str(new_id_str())
        self.run_registry.create_run(
            run_id=run_id,
            conversation_id=conversation_id,
            workflow_id=workflow_id,
            user_id=resolved_user_id,
            user_turn_node_id=str(add_turn.user_turn_node_id),
            status="queued",
        )
        self._publish(
            run_id,
            "run.created",
            {
                "run_id": run_id,
                "conversation_id": conversation_id,
                "workflow_id": workflow_id,
                "status": "queued",
                "user_turn_node_id": str(add_turn.user_turn_node_id),
            },
        )

        req = AnswerRunRequest(
            run_id=run_id,
            conversation_id=conversation_id,
            user_id=resolved_user_id,
            user_text=text,
            user_turn_node_id=str(add_turn.user_turn_node_id),
            workflow_id=workflow_id,
            knowledge_engine=self._knowledge_engine(),
            conversation_engine=self._conversation_engine(),
            workflow_engine=self._workflow_engine(),
            prev_turn_meta_summary=add_turn.prev_turn_meta_summary,
            registry=self.run_registry,
            publish=lambda event_type, payload=None: self._publish(run_id, event_type, payload),
            is_cancel_requested=lambda: self.run_registry.is_cancel_requested(run_id),
        )

        thread = threading.Thread(
            target=self._run_answer,
            args=(req,),
            daemon=True,
            name=f"chat-run-{run_id}",
        )
        thread.start()

        return {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "workflow_id": workflow_id,
            "status": "queued",
            "user_turn_node_id": str(add_turn.user_turn_node_id),
        }

    def _output_chunks(self, text: str, *, chunk_size: int = 160) -> list[str]:
        if not text:
            return []
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _run_answer(self, req: AnswerRunRequest) -> None:
        self._publish(req.run_id, "run.started", {"run_id": req.run_id, "status": "running"})
        self.run_registry.update_status(req.run_id, status="running", started=True)
        try:
            out = self.answer_runner(req) or {}
            workflow_status = str(out.get("workflow_status") or "succeeded")
            if workflow_status == "cancelled":
                self._publish(req.run_id, "run.cancelled", {"run_id": req.run_id, "status": "cancelled"})
                self.run_registry.update_status(req.run_id, status="cancelled", result=out, finished=True)
                return

            assistant_text = str(out.get("assistant_text") or "")
            for idx, chunk in enumerate(self._output_chunks(assistant_text)):
                self._publish(
                    req.run_id,
                    "output.delta",
                    {
                        "run_id": req.run_id,
                        "delta": chunk,
                        "chunk_index": idx,
                    },
                )
            self._publish(
                req.run_id,
                "output.completed",
                {
                    "run_id": req.run_id,
                    "assistant_text": assistant_text,
                    "assistant_turn_node_id": str(out.get("assistant_turn_node_id") or ""),
                },
            )
            self._publish(
                req.run_id,
                "run.completed",
                {
                    "run_id": req.run_id,
                    "status": "succeeded",
                    "assistant_turn_node_id": str(out.get("assistant_turn_node_id") or ""),
                },
            )
            self.run_registry.update_status(
                req.run_id,
                status="succeeded",
                assistant_turn_node_id=str(out.get("assistant_turn_node_id") or "") or None,
                result=out,
                finished=True,
            )
        except RunCancelledError:
            self._publish(req.run_id, "run.cancelled", {"run_id": req.run_id, "status": "cancelled"})
            self.run_registry.update_status(req.run_id, status="cancelled", finished=True)
        except Exception as exc:
            err = {"message": str(exc)}
            logging.getLogger(__name__).exception("chat run failed: run_id=%s", req.run_id)
            self._publish(req.run_id, "run.failed", {"run_id": req.run_id, "status": "failed", "error": err})
            self.run_registry.update_status(req.run_id, status="failed", error=err, finished=True)

    def _default_answer_runner(self, req: AnswerRunRequest) -> dict[str, Any]:
        trace_db_path = pathlib.Path(str(getattr(req.workflow_engine, "persist_directory", "."))) / "wf_trace.sqlite"
        shared_sink = _get_shared_sqlite_sink(str(trace_db_path), drop_when_full=True)
        sink = RunRegistryTraceBridge(registry=req.registry, run_id=req.run_id, delegate=shared_sink)
        events = EventEmitter(sink=sink, logger=logging.getLogger("workflow.trace"))
        agent = AgenticAnsweringAgent(
            conversation_engine=req.conversation_engine,
            knowledge_engine=req.knowledge_engine,
            llm_tasks=req.conversation_engine.llm_tasks,
        )
        return agent.answer_workflow_v2(
            conversation_id=req.conversation_id,
            user_id=req.user_id,
            prev_turn_meta_summary=req.prev_turn_meta_summary,
            workflow_engine=req.workflow_engine,
            workflow_id=req.workflow_id,
            run_id=req.run_id,
            events=events,
            trace=True,
            cancel_requested=lambda rid: req.is_cancel_requested(),
        )

    def submit_workflow_run(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        initial_state: dict[str, Any] | None = None,
        turn_node_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        workflow_id = str(workflow_id or "").strip()
        if not workflow_id:
            raise ValueError("workflow_id is required")
        self._assert_workflow_projection_not_rebuilding(workflow_id=workflow_id)
        conversation_id = str(conversation_id or "").strip()
        if not conversation_id:
            raise ValueError("conversation_id is required")

        # Validate conversation existence early.
        self._conversation_nodes(conversation_id)

        resolved_turn_node_id = str(turn_node_id or "").strip()
        if not resolved_turn_node_id:
            tail = self._conversation_service().get_conversation_tail(conversation_id=conversation_id)
            resolved_turn_node_id = str(getattr(tail, "id", None) or "").strip()
        if not resolved_turn_node_id:
            resolved_turn_node_id = f"wf_turn|{new_id_str()}"

        run_id = str(new_id_str())
        self.run_registry.create_run(
            run_id=run_id,
            conversation_id=conversation_id,
            workflow_id=workflow_id,
            user_id=(str(user_id) if user_id is not None else None),
            user_turn_node_id=resolved_turn_node_id,
            status="queued",
        )
        self._publish(
            run_id,
            "run.created",
            {
                "run_id": run_id,
                "run_kind": "workflow_runtime",
                "conversation_id": conversation_id,
                "workflow_id": workflow_id,
                "status": "queued",
                "turn_node_id": resolved_turn_node_id,
            },
        )

        req = RuntimeRunRequest(
            run_id=run_id,
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id=resolved_turn_node_id,
            user_id=(str(user_id) if user_id is not None else None),
            initial_state=dict(initial_state or {}),
            knowledge_engine=self._knowledge_engine(),
            conversation_engine=self._conversation_engine(),
            workflow_engine=self._workflow_engine(),
            registry=self.run_registry,
            publish=lambda event_type, payload=None: self._publish(run_id, event_type, payload),
            is_cancel_requested=lambda: self.run_registry.is_cancel_requested(run_id),
        )
        thread = threading.Thread(
            target=self._run_workflow,
            args=(req,),
            daemon=True,
            name=f"workflow-run-{run_id}",
        )
        thread.start()
        return {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "workflow_id": workflow_id,
            "turn_node_id": resolved_turn_node_id,
            "status": "queued",
        }

    def _run_workflow(self, req: RuntimeRunRequest) -> None:
        self._publish(
            req.run_id,
            "run.started",
            {
                "run_id": req.run_id,
                "run_kind": "workflow_runtime",
                "status": "running",
            },
        )
        self.run_registry.update_status(req.run_id, status="running", started=True)
        try:
            out = self._json_safe(self.runtime_runner(req) or {})
            workflow_status = str(out.get("workflow_status") or out.get("status") or "succeeded")
            if workflow_status == "cancelled":
                self._publish(req.run_id, "run.cancelled", {"run_id": req.run_id, "status": "cancelled"})
                self.run_registry.update_status(req.run_id, status="cancelled", result=out, finished=True)
                return
            if workflow_status in {"failed", "error"}:
                err = out.get("error")
                if not isinstance(err, dict):
                    err = {"message": f"Workflow runtime failed: status={workflow_status}"}
                self._publish(req.run_id, "run.failed", {"run_id": req.run_id, "status": "failed", "error": err})
                self.run_registry.update_status(req.run_id, status="failed", result=out, error=err, finished=True)
                return
            self._publish(req.run_id, "run.completed", {"run_id": req.run_id, "status": "succeeded"})
            self.run_registry.update_status(req.run_id, status="succeeded", result=out, finished=True)
        except RunCancelledError:
            self._publish(req.run_id, "run.cancelled", {"run_id": req.run_id, "status": "cancelled"})
            self.run_registry.update_status(req.run_id, status="cancelled", finished=True)
        except Exception as exc:
            err = {"message": str(exc)}
            logging.getLogger(__name__).exception("workflow run failed: run_id=%s", req.run_id)
            self._publish(req.run_id, "run.failed", {"run_id": req.run_id, "status": "failed", "error": err})
            self.run_registry.update_status(req.run_id, status="failed", error=err, finished=True)

    def _default_runtime_runner(self, req: RuntimeRunRequest) -> dict[str, Any]:
        from graph_knowledge_engine.conversation.resolvers import default_resolver
        from graph_knowledge_engine.runtime.runtime import WorkflowRuntime

        def predicate_always(_workflow_info, _state, _last_result):
            return True

        initial_state = dict(req.initial_state or {})
        deps = initial_state.get("_deps")
        if not isinstance(deps, dict):
            deps = {}
        deps.setdefault("conversation_engine", req.conversation_engine)
        deps.setdefault("knowledge_engine", req.knowledge_engine)
        deps.setdefault("ref_knowledge_engine", req.knowledge_engine)
        deps.setdefault("workflow_engine", req.workflow_engine)
        deps.setdefault("agentic_workflow_engine", req.workflow_engine)
        initial_state["_deps"] = deps

        runtime = WorkflowRuntime(
            workflow_engine=req.workflow_engine,
            conversation_engine=req.conversation_engine,
            step_resolver=default_resolver,
            predicate_registry={"always": predicate_always},
            checkpoint_every_n_steps=1,
            max_workers=1,
            cancel_requested=lambda _rid: req.is_cancel_requested(),
        )
        run_result = runtime.run(
            workflow_id=req.workflow_id,
            conversation_id=req.conversation_id,
            turn_node_id=req.turn_node_id,
            initial_state=initial_state,
            run_id=req.run_id,
        )
        final_state = self._json_safe(dict(getattr(run_result, "final_state", {}) or {}))
        final_state.pop("_deps", None)
        return {
            "workflow_status": str(getattr(run_result, "status", "succeeded") or "succeeded"),
            "final_state": final_state,
        }

    def get_run(self, run_id: str) -> dict[str, Any]:
        run = self.run_registry.get_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        steps = self.list_steps(run_id)
        if steps:
            run["last_step_seq"] = int(steps[-1]["step_seq"])
            run["step_count"] = len(steps)
        else:
            run["last_step_seq"] = None
            run["step_count"] = 0
        return run

    def list_run_events(self, run_id: str, *, after_seq: int = 0) -> list[dict[str, Any]]:
        if self.run_registry.get_run(run_id) is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        return self.run_registry.list_events(run_id, after_seq=after_seq)

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        run = self.run_registry.get_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        if run["terminal"]:
            return run
        try:
            self._conversation_service().persist_workflow_cancel_request(
                conversation_id=str(run.get("conversation_id") or ""),
                run_id=str(run_id),
                workflow_id=str(run.get("workflow_id") or ""),
                requested_by="api",
                reason="api_cancel",
            )
        except Exception:
            logging.getLogger(__name__).exception("failed to persist cancel request node: run_id=%s", run_id)
        self._publish(run_id, "run.cancelling", {"run_id": run_id, "status": "cancelling"})
        return self.run_registry.request_cancel(run_id)

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

    def workflow_design_graph(self, workflow_id: str, refresh: bool = False) -> dict[str, object]:
        """Thin public wrapper around the existing internal projection + visible snapshot flow."""
        if refresh:
            self.refresh_workflow_design_projection(workflow_id=workflow_id)
        snapshot = self._workflow_capture_visible_snapshot(workflow_id=workflow_id)
        history = self.workflow_design_history(workflow_id=workflow_id)
        return {
            "workflow_id": workflow_id,
            "current_version": history.get("current_version", 0),
            "active_tip_version": history.get("active_tip_version", 0),
            "can_undo": history.get("can_undo", False),
            "can_redo": history.get("can_redo", False),
            "materialization_status": history.get("materialization_status", "unknown"),
            "nodes": snapshot.get("nodes", []),
            "edges": snapshot.get("edges", []),
        }


    def workflow_catalog_ops(self) -> list[dict[str, object]]:
        return [
            {
                "op": "start",
                "label": "Start",
                "description": "Entry point for workflow execution.",
                "input_schema": {},
                "output_schema": {"type": "object"},
                "config_schema": {"type": "object"},
            },
            {
                "op": "llm_call",
                "label": "LLM Call",
                "description": "Calls an LLM and returns structured output.",
                "input_schema": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]},
                "output_schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
                "config_schema": {"type": "object", "properties": {"model": {"type": "string"}, "temperature": {"type": "number"}}},
            },
        ]
