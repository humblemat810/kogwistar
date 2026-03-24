"""Workflow design service for mutations, history reads, and projection sync.

This module exposes the workflow-design-facing operations that the top-level
chat service routes to: create/update/delete workflow design artifacts,
undo/redo, history inspection, and projection refresh. It composes the history
helpers and keeps workflow design management separate from run execution.
"""

from __future__ import annotations

import logging
from typing import Any

from kogwistar.engine_core.models import Grounding, Span
from kogwistar.id_provider import new_id_str
from kogwistar.runtime.models import WorkflowEdge, WorkflowNode

from .chat_service_workflow_history import _WorkflowDesignHistoryMixin


class _WorkflowDesignService(_WorkflowDesignHistoryMixin):
    """Owns workflow design history, projections, snapshots, and undo/redo."""

    def _assert_designer_identity(
        self, *, designer_id: str, actor_sub: str | None
    ) -> str:
        resolved_designer = str(designer_id or "").strip()
        if not resolved_designer:
            raise ValueError("designer_id is required")
        subject = str(actor_sub or "").strip()
        if subject and resolved_designer != subject:
            raise PermissionError("designer_id must match authenticated subject")
        return resolved_designer

    def workflow_design_history(self, *, workflow_id: str) -> dict[str, Any]:
        workflow_id = str(workflow_id or "").strip()
        if not workflow_id:
            raise ValueError("workflow_id is required")
        with self._workflow_history_lock:
            state = self._workflow_sync_projection_locked(
                workflow_id=workflow_id, materialize=False
            )
            projection_status = self._workflow_projection_status(workflow_id=workflow_id)
            if projection_status:
                state = dict(state)
                state["materialization_status"] = projection_status
            return state

    def refresh_workflow_design_projection(self, *, workflow_id: str) -> dict[str, Any]:
        workflow_id = str(workflow_id or "").strip()
        if not workflow_id:
            raise ValueError("workflow_id is required")
        clear_snapshots = getattr(
            self._workflow_meta_store(), "clear_workflow_design_snapshots", None
        )
        with self._workflow_history_lock:
            state = self._workflow_fold_history(workflow_id=workflow_id)
            self._store_workflow_projection(
                workflow_id=workflow_id,
                state=state,
                materialization_status="rebuilding",
            )
            if callable(clear_snapshots):
                clear_snapshots(workflow_id=workflow_id)
            out = self._workflow_finalize_design_state_locked(
                workflow_id=workflow_id, rebuild=True
            )
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
        resolved_designer_id = self._assert_designer_identity(
            designer_id=designer_id, actor_sub=actor_sub
        )
        self._assert_workflow_projection_not_rebuilding(workflow_id=workflow_id)
        resolved_node_id = (
            str(node_id or "").strip() or f"wf|{workflow_id}|n|{new_id_str()}"
        )
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
            state_before = self._workflow_sync_projection_locked(
                workflow_id=workflow_id, materialize=True
            )
            before_visible = self._workflow_capture_visible_snapshot(
                workflow_id=workflow_id
            )
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
                    op=self._ctrl_mutation_committed,
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
            after_visible = self._workflow_capture_visible_snapshot(
                workflow_id=workflow_id
            )
            self._workflow_store_version_delta(
                workflow_id=workflow_id,
                version=new_version,
                prev_version=current_version,
                target_seq=int(latest_seq),
                forward=self._workflow_compute_visible_delta(
                    before=before_visible, after=after_visible
                ),
                inverse=self._workflow_compute_visible_delta(
                    before=after_visible, after=before_visible
                ),
            )
            history = self._workflow_finalize_design_state_locked(
                workflow_id=workflow_id, rebuild=branch_dropped
            )
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
        resolved_designer_id = self._assert_designer_identity(
            designer_id=designer_id, actor_sub=actor_sub
        )
        self._assert_workflow_projection_not_rebuilding(workflow_id=workflow_id)
        src = str(src or "").strip()
        dst = str(dst or "").strip()
        if not src or not dst:
            raise ValueError("src and dst are required")
        resolved_edge_id = (
            str(edge_id or "").strip() or f"wf|{workflow_id}|e|{new_id_str()}"
        )
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
            state_before = self._workflow_sync_projection_locked(
                workflow_id=workflow_id, materialize=True
            )
            before_visible = self._workflow_capture_visible_snapshot(
                workflow_id=workflow_id
            )
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
                    op=self._ctrl_mutation_committed,
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
            after_visible = self._workflow_capture_visible_snapshot(
                workflow_id=workflow_id
            )
            self._workflow_store_version_delta(
                workflow_id=workflow_id,
                version=new_version,
                prev_version=current_version,
                target_seq=int(latest_seq),
                forward=self._workflow_compute_visible_delta(
                    before=before_visible, after=after_visible
                ),
                inverse=self._workflow_compute_visible_delta(
                    before=after_visible, after=before_visible
                ),
            )
            history = self._workflow_finalize_design_state_locked(
                workflow_id=workflow_id, rebuild=branch_dropped
            )
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
        resolved_designer_id = self._assert_designer_identity(
            designer_id=designer_id, actor_sub=actor_sub
        )
        self._assert_workflow_projection_not_rebuilding(workflow_id=workflow_id)
        namespace = self._workflow_namespace(workflow_id)
        with self._workflow_history_lock:
            state_before = self._workflow_sync_projection_locked(
                workflow_id=workflow_id, materialize=True
            )
            before_visible = self._workflow_capture_visible_snapshot(
                workflow_id=workflow_id
            )
            known_node_ids = {
                str(item.get("id") or "")
                for item in list(before_visible.get("nodes") or [])
            }
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
                    op=self._ctrl_mutation_committed,
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
            after_visible = self._workflow_capture_visible_snapshot(
                workflow_id=workflow_id
            )
            self._workflow_store_version_delta(
                workflow_id=workflow_id,
                version=new_version,
                prev_version=current_version,
                target_seq=int(latest_seq),
                forward=self._workflow_compute_visible_delta(
                    before=before_visible, after=after_visible
                ),
                inverse=self._workflow_compute_visible_delta(
                    before=after_visible, after=before_visible
                ),
            )
            history = self._workflow_finalize_design_state_locked(
                workflow_id=workflow_id, rebuild=branch_dropped
            )
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
        resolved_designer_id = self._assert_designer_identity(
            designer_id=designer_id, actor_sub=actor_sub
        )
        self._assert_workflow_projection_not_rebuilding(workflow_id=workflow_id)
        namespace = self._workflow_namespace(workflow_id)
        with self._workflow_history_lock:
            state_before = self._workflow_sync_projection_locked(
                workflow_id=workflow_id, materialize=True
            )
            before_visible = self._workflow_capture_visible_snapshot(
                workflow_id=workflow_id
            )
            known_edge_ids = {
                str(item.get("id") or "")
                for item in list(before_visible.get("edges") or [])
            }
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
                    op=self._ctrl_mutation_committed,
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
            after_visible = self._workflow_capture_visible_snapshot(
                workflow_id=workflow_id
            )
            self._workflow_store_version_delta(
                workflow_id=workflow_id,
                version=new_version,
                prev_version=current_version,
                target_seq=int(latest_seq),
                forward=self._workflow_compute_visible_delta(
                    before=before_visible, after=after_visible
                ),
                inverse=self._workflow_compute_visible_delta(
                    before=after_visible, after=before_visible
                ),
            )
            history = self._workflow_finalize_design_state_locked(
                workflow_id=workflow_id, rebuild=branch_dropped
            )
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
        resolved_designer_id = self._assert_designer_identity(
            designer_id=designer_id, actor_sub=actor_sub
        )
        self._assert_workflow_projection_not_rebuilding(workflow_id=workflow_id)
        with self._workflow_history_lock:
            state = self._workflow_sync_projection_locked(
                workflow_id=workflow_id, materialize=True
            )
            current_version = int(state.get("current_version") or 0)
            if current_version <= 0:
                state["status"] = "noop"
                self._store_workflow_projection(
                    workflow_id=workflow_id, state=state, materialization_status="ready"
                )
                return state
            selected_versions = list(state.get("selected_versions") or [])
            current_entry = next(
                (
                    item
                    for item in selected_versions
                    if int(item.get("version") or 0) == current_version
                ),
                None,
            )
            if current_entry is None:
                raise RuntimeError(
                    f"Current workflow version missing from selected lineage: {current_version}"
                )
            target_version = int(current_entry.get("prev_version") or 0)
            target_seq = 0
            if target_version > 0:
                target_entry = next(
                    (
                        item
                        for item in selected_versions
                        if int(item.get("version") or 0) == target_version
                    ),
                    None,
                )
                if target_entry is None:
                    raise RuntimeError(
                        f"Undo target missing from selected lineage: {target_version}"
                    )
                target_seq = int(target_entry.get("target_seq") or 0)
            with self._workflow_engine().uow():
                self._append_design_control_event(
                    workflow_id=workflow_id,
                    op=self._ctrl_undo_applied,
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
            delta = self._workflow_version_delta(
                workflow_id=workflow_id, version=current_version
            )
            if delta is not None:
                try:
                    self._workflow_apply_visible_delta(
                        workflow_id=workflow_id, delta=dict(delta.get("inverse") or {})
                    )
                    applied_delta = True
                except Exception:
                    logging.getLogger(__name__).exception(
                        "failed applying workflow inverse delta; falling back to rebuild: workflow_id=%s version=%s",
                        workflow_id,
                        current_version,
                    )
            if not applied_delta:
                self._workflow_rebuild_namespace_for_state(
                    workflow_id=workflow_id, state=out
                )
            self._store_workflow_projection(
                workflow_id=workflow_id, state=out, materialization_status="ready"
            )
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
        resolved_designer_id = self._assert_designer_identity(
            designer_id=designer_id, actor_sub=actor_sub
        )
        self._assert_workflow_projection_not_rebuilding(workflow_id=workflow_id)
        with self._workflow_history_lock:
            state = self._workflow_sync_projection_locked(
                workflow_id=workflow_id, materialize=True
            )
            current_version = int(state.get("current_version") or 0)
            active_versions = list(state.get("versions") or [])
            active_ids = [int(item.get("version") or 0) for item in active_versions]
            if current_version not in active_ids:
                raise RuntimeError(
                    f"Current workflow version missing from active lineage: {current_version}"
                )
            current_index = active_ids.index(current_version)
            if current_index >= len(active_ids) - 1:
                state["status"] = "noop"
                self._store_workflow_projection(
                    workflow_id=workflow_id, state=state, materialization_status="ready"
                )
                return state
            target_entry = active_versions[current_index + 1]
            target_version = int(target_entry.get("version") or 0)
            target_seq = int(target_entry.get("seq") or 0)
            with self._workflow_engine().uow():
                self._append_design_control_event(
                    workflow_id=workflow_id,
                    op=self._ctrl_redo_applied,
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
            delta = self._workflow_version_delta(
                workflow_id=workflow_id, version=target_version
            )
            if delta is not None:
                try:
                    self._workflow_apply_visible_delta(
                        workflow_id=workflow_id, delta=dict(delta.get("forward") or {})
                    )
                    applied_delta = True
                except Exception:
                    logging.getLogger(__name__).exception(
                        "failed applying workflow forward delta; falling back to rebuild: workflow_id=%s version=%s",
                        workflow_id,
                        target_version,
                    )
            if not applied_delta:
                self._workflow_rebuild_namespace_for_state(
                    workflow_id=workflow_id, state=out
                )
            self._store_workflow_projection(
                workflow_id=workflow_id, state=out, materialization_status="ready"
            )
            self._workflow_store_snapshot_if_needed(workflow_id=workflow_id, state=out)
            out["status"] = "ok"
            return out

    def workflow_design_graph(
        self, workflow_id: str, refresh: bool = False
    ) -> dict[str, object]:
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
                "input_schema": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "temperature": {"type": "number"},
                    },
                },
            },
        ]
