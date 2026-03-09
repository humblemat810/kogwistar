from __future__ import annotations

import contextlib
import json
import logging
import pathlib
import sqlite3
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
    _CTRL_HISTORY_BACKFILL = "HISTORY_BACKFILL"
    _CTRL_UNDO_APPLIED = "UNDO_APPLIED"
    _CTRL_REDO_APPLIED = "REDO_APPLIED"
    _CTRL_BRANCH_DROPPED = "BRANCH_DROPPED"
    _CTRL_MUTATION_COMMITTED = "MUTATION_COMMITTED"

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

    def _workflow_history_db_path(self) -> pathlib.Path:
        root = pathlib.Path(str(getattr(self._workflow_engine(), "persist_directory", ".") or "."))
        root.mkdir(parents=True, exist_ok=True)
        return root / "workflow_design_history.sqlite"

    def _history_connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._workflow_history_db_path()), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS workflow_design_history (
                workflow_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                seq INTEGER NOT NULL,
                created_at_ms INTEGER NOT NULL,
                PRIMARY KEY (workflow_id, version)
            );
            CREATE TABLE IF NOT EXISTS workflow_design_pointer (
                workflow_id TEXT PRIMARY KEY,
                current_version INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );
            """
        )
        return conn

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

    def _history_ensure_initialized(self, conn: sqlite3.Connection, workflow_id: str) -> None:
        now = self._now_ms()
        conn.execute(
            """
            INSERT OR IGNORE INTO workflow_design_history(workflow_id, version, seq, created_at_ms)
            VALUES (?, 0, 0, ?)
            """,
            (workflow_id, now),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO workflow_design_pointer(workflow_id, current_version, updated_at_ms)
            VALUES (?, 0, ?)
            """,
            (workflow_id, now),
        )

    def _history_state_locked(self, conn: sqlite3.Connection, workflow_id: str) -> dict[str, Any]:
        self._history_ensure_initialized(conn, workflow_id)
        ptr = conn.execute(
            "SELECT current_version FROM workflow_design_pointer WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
        current_version = int(ptr["current_version"]) if ptr is not None else 0
        max_row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS max_version FROM workflow_design_history WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
        max_version = int(max_row["max_version"]) if max_row is not None else 0
        seq_row = conn.execute(
            "SELECT seq FROM workflow_design_history WHERE workflow_id = ? AND version = ?",
            (workflow_id, current_version),
        ).fetchone()
        current_seq = int(seq_row["seq"]) if seq_row is not None else 0
        versions_rows = conn.execute(
            """
            SELECT version, seq, created_at_ms
            FROM workflow_design_history
            WHERE workflow_id = ?
            ORDER BY version ASC
            """,
            (workflow_id,),
        ).fetchall()
        versions = [
            {
                "version": int(row["version"]),
                "seq": int(row["seq"]),
                "created_at_ms": int(row["created_at_ms"]),
            }
            for row in versions_rows
        ]
        return {
            "workflow_id": workflow_id,
            "namespace": self._workflow_namespace(workflow_id),
            "current_version": current_version,
            "max_version": max_version,
            "current_seq": current_seq,
            "can_undo": current_version > 0,
            "can_redo": current_version < max_version,
            "versions": versions,
        }

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

    def _workflow_has_control_events(self, *, namespace: str) -> bool:
        for _seq, entity_kind, _entity_id, _op, _payload_raw in self._iter_entity_events(
            namespace=namespace,
            from_seq=1,
        ):
            if str(entity_kind) == self._DESIGN_CONTROL_KIND:
                return True
        return False

    def _ensure_control_backfill_locked(self, *, workflow_id: str, state: dict[str, Any]) -> None:
        namespace = self._workflow_namespace(workflow_id)
        if self._workflow_has_control_events(namespace=namespace):
            return
        self._append_design_control_event(
            workflow_id=workflow_id,
            op=self._CTRL_HISTORY_BACKFILL,
            designer_id="__migration__",
            source="migration",
            payload={
                "current_version": int(state.get("current_version", 0)),
                "current_seq": int(state.get("current_seq", 0)),
                "versions": list(state.get("versions") or []),
            },
        )

    def _history_record_seq(
        self,
        workflow_id: str,
        seq: int,
        *,
        designer_id: str,
        source: str,
    ) -> dict[str, Any]:
        with self._workflow_history_lock, self._history_connect() as conn:
            state = self._history_state_locked(conn, workflow_id)
            self._ensure_control_backfill_locked(workflow_id=workflow_id, state=state)
            state = self._history_state_locked(conn, workflow_id)
            current_version = int(state["current_version"])
            current_seq = int(state["current_seq"])
            max_version = int(state["max_version"])
            seq = int(seq)
            if seq <= current_seq:
                return state
            if current_version < max_version:
                dropped_versions = [v for v in state.get("versions", []) if int(v.get("version", -1)) > current_version]
                if dropped_versions:
                    drop_from_seq = min(int(v.get("seq", 0)) for v in dropped_versions)
                    drop_to_seq = max(int(v.get("seq", 0)) for v in dropped_versions)
                    drop_evt_seq = self._append_design_control_event(
                        workflow_id=workflow_id,
                        op=self._CTRL_BRANCH_DROPPED,
                        designer_id=designer_id,
                        source=source,
                        payload={
                            "drop_from_seq": int(drop_from_seq),
                            "drop_to_seq": int(drop_to_seq),
                            "reason": "new_edit_after_undo",
                        },
                    )
                    seq = max(seq, int(drop_evt_seq))
                conn.execute(
                    "DELETE FROM workflow_design_history WHERE workflow_id = ? AND version > ?",
                    (workflow_id, current_version),
                )
            new_version = current_version + 1
            now = self._now_ms()
            conn.execute(
                """
                INSERT INTO workflow_design_history(workflow_id, version, seq, created_at_ms)
                VALUES (?, ?, ?, ?)
                """,
                (workflow_id, new_version, seq, now),
            )
            conn.execute(
                """
                UPDATE workflow_design_pointer
                SET current_version = ?, updated_at_ms = ?
                WHERE workflow_id = ?
                """,
                (new_version, now, workflow_id),
            )
            return self._history_state_locked(conn, workflow_id)

    def _workflow_latest_seq(self, *, namespace: str, from_seq: int = 1) -> int:
        last = 0
        for seq, _ek, _eid, _op, _payload in self._iter_entity_events(
            namespace=namespace,
            from_seq=max(1, int(from_seq)),
        ):
            last = int(seq)
        return last

    def _workflow_collect_entity_ids(self, *, namespace: str) -> tuple[set[str], set[str]]:
        node_ids: set[str] = set()
        edge_ids: set[str] = set()
        for _seq, entity_kind, entity_id, _op, _payload in self._iter_entity_events(namespace=namespace, from_seq=1):
            if str(entity_kind) == "node":
                node_ids.add(str(entity_id))
            elif str(entity_kind) == "edge":
                edge_ids.add(str(entity_id))
        return node_ids, edge_ids

    @staticmethod
    def _seq_in_dropped_ranges(seq: int, ranges: list[tuple[int, int]]) -> bool:
        for start, end in ranges:
            if start <= seq <= end:
                return True
        return False

    def _workflow_replay_masked_to_seq(self, *, namespace: str, to_seq: int) -> None:
        eng = self._workflow_engine()
        dropped_ranges: list[tuple[int, int]] = []
        for _seq, entity_kind, _entity_id, op, payload_raw in self._iter_entity_events(
            namespace=namespace,
            from_seq=1,
            to_seq=int(to_seq),
        ):
            if str(entity_kind) != self._DESIGN_CONTROL_KIND or str(op) != self._CTRL_BRANCH_DROPPED:
                continue
            payload = self._parse_event_payload(payload_raw)
            start = int(payload.get("drop_from_seq", -1))
            end = int(payload.get("drop_to_seq", -1))
            if start >= 0 and end >= start:
                dropped_ranges.append((start, end))

        prev_log = getattr(eng, "_disable_event_log", False)
        prev_idx = getattr(eng, "_phase1_enable_index_jobs", False)
        eng._disable_event_log = True
        eng._phase1_enable_index_jobs = False
        try:
            for seq, entity_kind, entity_id, op, payload_raw in self._iter_entity_events(
                namespace=namespace,
                from_seq=1,
                to_seq=int(to_seq),
            ):
                seq_i = int(seq)
                if self._seq_in_dropped_ranges(seq_i, dropped_ranges):
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
                        try:
                            eng.lifecycle.tombstone_node(str(entity_id))
                        except Exception:
                            pass
                elif entity_kind_s == "edge":
                    if op_s in {"ADD", "REPLACE"}:
                        try:
                            edge = CoreEdge.model_validate(payload)
                        except Exception:
                            edge = CoreEdge.model_validate_json(json.dumps(payload))
                        eng.write.add_edge(edge)
                    elif op_s in {"TOMBSTONE", "DELETE"}:
                        try:
                            eng.lifecycle.tombstone_edge(str(entity_id))
                        except Exception:
                            pass
        finally:
            eng._disable_event_log = prev_log
            eng._phase1_enable_index_jobs = prev_idx

    def _workflow_rebuild_namespace_to_seq(self, *, namespace: str, to_seq: int) -> None:
        eng = self._workflow_engine()
        node_ids, edge_ids = self._workflow_collect_entity_ids(namespace=namespace)
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
        if int(to_seq) > 0:
            self._workflow_replay_masked_to_seq(namespace=namespace, to_seq=int(to_seq))

    def workflow_design_history(self, *, workflow_id: str) -> dict[str, Any]:
        workflow_id = str(workflow_id or "").strip()
        if not workflow_id:
            raise ValueError("workflow_id is required")
        with self._workflow_history_lock, self._history_connect() as conn:
            state = self._history_state_locked(conn, workflow_id)
            self._ensure_control_backfill_locked(workflow_id=workflow_id, state=state)
            state = self._history_state_locked(conn, workflow_id)
        namespace = self._workflow_namespace(workflow_id)
        state["latest_seq"] = self._workflow_latest_seq(namespace=namespace, from_seq=1)
        state["timeline"] = self._workflow_control_timeline(namespace=namespace, limit=500)
        return state

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
        with self._workflow_namespace_scope(workflow_id) as eng:
            eng.write.add_node(n)
        latest_seq = self._workflow_latest_seq(namespace=namespace, from_seq=1)
        history = self._history_record_seq(
            workflow_id,
            latest_seq,
            designer_id=resolved_designer_id,
            source=source,
        )
        self._append_design_control_event(
            workflow_id=workflow_id,
            op=self._CTRL_MUTATION_COMMITTED,
            designer_id=resolved_designer_id,
            source=source,
            payload={
                "action": "node_upsert",
                "entity_id": resolved_node_id,
                "seq": int(history["current_seq"]),
                "version": int(history["current_version"]),
            },
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
        resolved_designer_id = self._assert_designer_identity(designer_id=designer_id, actor_sub=actor_sub)
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
        with self._workflow_namespace_scope(workflow_id) as eng:
            eng.write.add_edge(e)
        latest_seq = self._workflow_latest_seq(namespace=namespace, from_seq=1)
        history = self._history_record_seq(
            workflow_id,
            latest_seq,
            designer_id=resolved_designer_id,
            source=source,
        )
        self._append_design_control_event(
            workflow_id=workflow_id,
            op=self._CTRL_MUTATION_COMMITTED,
            designer_id=resolved_designer_id,
            source=source,
            payload={
                "action": "edge_upsert",
                "entity_id": resolved_edge_id,
                "seq": int(history["current_seq"]),
                "version": int(history["current_version"]),
            },
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
        resolved_designer_id = self._assert_designer_identity(designer_id=designer_id, actor_sub=actor_sub)
        namespace = self._workflow_namespace(workflow_id)
        with self._workflow_namespace_scope(workflow_id) as eng:
            ok = bool(eng.tombstone_node(node_id, reason="workflow_design_delete", deleted_by=resolved_designer_id))
        if not ok:
            raise KeyError(f"Unknown node_id: {node_id}")
        latest_seq = self._workflow_latest_seq(namespace=namespace, from_seq=1)
        history = self._history_record_seq(
            workflow_id,
            latest_seq,
            designer_id=resolved_designer_id,
            source=source,
        )
        self._append_design_control_event(
            workflow_id=workflow_id,
            op=self._CTRL_MUTATION_COMMITTED,
            designer_id=resolved_designer_id,
            source=source,
            payload={
                "action": "node_delete",
                "entity_id": node_id,
                "seq": int(history["current_seq"]),
                "version": int(history["current_version"]),
            },
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
        resolved_designer_id = self._assert_designer_identity(designer_id=designer_id, actor_sub=actor_sub)
        namespace = self._workflow_namespace(workflow_id)
        with self._workflow_namespace_scope(workflow_id) as eng:
            ok = bool(eng.tombstone_edge(edge_id, reason="workflow_design_delete", deleted_by=resolved_designer_id))
        if not ok:
            raise KeyError(f"Unknown edge_id: {edge_id}")
        latest_seq = self._workflow_latest_seq(namespace=namespace, from_seq=1)
        history = self._history_record_seq(
            workflow_id,
            latest_seq,
            designer_id=resolved_designer_id,
            source=source,
        )
        self._append_design_control_event(
            workflow_id=workflow_id,
            op=self._CTRL_MUTATION_COMMITTED,
            designer_id=resolved_designer_id,
            source=source,
            payload={
                "action": "edge_delete",
                "entity_id": edge_id,
                "seq": int(history["current_seq"]),
                "version": int(history["current_version"]),
            },
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
        resolved_designer_id = self._assert_designer_identity(designer_id=designer_id, actor_sub=actor_sub)
        namespace = self._workflow_namespace(workflow_id)
        with self._workflow_history_lock, self._history_connect() as conn:
            state = self._history_state_locked(conn, workflow_id)
            self._ensure_control_backfill_locked(workflow_id=workflow_id, state=state)
            state = self._history_state_locked(conn, workflow_id)
            current_version = int(state["current_version"])
            if current_version <= 0:
                state["status"] = "noop"
                state["timeline"] = self._workflow_control_timeline(namespace=namespace, limit=500)
                state["latest_seq"] = self._workflow_latest_seq(namespace=namespace, from_seq=1)
                return state
            target_version = current_version - 1
            row = conn.execute(
                "SELECT seq FROM workflow_design_history WHERE workflow_id = ? AND version = ?",
                (workflow_id, target_version),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"History version missing for workflow_id={workflow_id!r} version={target_version}")
            target_seq = int(row["seq"])
            self._workflow_rebuild_namespace_to_seq(namespace=namespace, to_seq=target_seq)
            conn.execute(
                """
                UPDATE workflow_design_pointer
                SET current_version = ?, updated_at_ms = ?
                WHERE workflow_id = ?
                """,
                (target_version, self._now_ms(), workflow_id),
            )
            self._append_design_control_event(
                workflow_id=workflow_id,
                op=self._CTRL_UNDO_APPLIED,
                designer_id=resolved_designer_id,
                source=source,
                payload={
                    "from_version": current_version,
                    "to_version": target_version,
                    "target_seq": target_seq,
                },
            )
            out = self._history_state_locked(conn, workflow_id)
            out["status"] = "ok"
            out["timeline"] = self._workflow_control_timeline(namespace=namespace, limit=500)
            out["latest_seq"] = self._workflow_latest_seq(namespace=namespace, from_seq=1)
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
        namespace = self._workflow_namespace(workflow_id)
        with self._workflow_history_lock, self._history_connect() as conn:
            state = self._history_state_locked(conn, workflow_id)
            self._ensure_control_backfill_locked(workflow_id=workflow_id, state=state)
            state = self._history_state_locked(conn, workflow_id)
            current_version = int(state["current_version"])
            max_version = int(state["max_version"])
            if current_version >= max_version:
                state["status"] = "noop"
                state["timeline"] = self._workflow_control_timeline(namespace=namespace, limit=500)
                state["latest_seq"] = self._workflow_latest_seq(namespace=namespace, from_seq=1)
                return state
            target_version = current_version + 1
            row = conn.execute(
                "SELECT seq FROM workflow_design_history WHERE workflow_id = ? AND version = ?",
                (workflow_id, target_version),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"History version missing for workflow_id={workflow_id!r} version={target_version}")
            target_seq = int(row["seq"])
            self._workflow_rebuild_namespace_to_seq(namespace=namespace, to_seq=target_seq)
            conn.execute(
                """
                UPDATE workflow_design_pointer
                SET current_version = ?, updated_at_ms = ?
                WHERE workflow_id = ?
                """,
                (target_version, self._now_ms(), workflow_id),
            )
            self._append_design_control_event(
                workflow_id=workflow_id,
                op=self._CTRL_REDO_APPLIED,
                designer_id=resolved_designer_id,
                source=source,
                payload={
                    "from_version": current_version,
                    "to_version": target_version,
                    "target_seq": target_seq,
                },
            )
            out = self._history_state_locked(conn, workflow_id)
            out["status"] = "ok"
            out["timeline"] = self._workflow_control_timeline(namespace=namespace, limit=500)
            out["latest_seq"] = self._workflow_latest_seq(namespace=namespace, from_seq=1)
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
