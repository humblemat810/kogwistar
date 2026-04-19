"""Workflow design history helpers for event-sourced workflow editing.

This module implements the low-level history folding, projection rebuild,
snapshotting, and visible-delta application used by the workflow designer
surface. It operates on workflow design nodes and edges only; it does not
manage workflow execution, scheduler/runtime concerns, or conversation
lifecycle state.

Two authoring patterns are supported.

1. Designer-history pattern
   Designer actions are recorded as append-only control events
   (for example: mutation committed, undo applied, redo applied, branch
   dropped). Those control events define the authoritative design history.
   The currently visible workflow graph is then treated as a materialized
   projection derived from that history.

2. Direct graph mutation pattern
   Callers may bypass the designer-history helpers and operate directly on
   workflow node/edge primitives using the underlying workflow graph APIs.
   In that mode, the workflow graph behaves like the broader CR-style graph
   model used elsewhere (KG-graph CR) in the system. In this case, the graph
   is source of truth but not projected

In the designer-history pattern, CR-like graph behavior is preserved by
recording control-plane events separately, computing deltas between visible
snapshots, and materializing those deltas back into workflow node/edge
entities. As a result, visible workflow node/edge changes may be emitted
again during projection update or rebuild. This is intentional: the graph is
the projection, while design control events remain the authoritative editing
history.

These two patterns are related but distinct. This module is concerned only
with the first pattern and with the projection machinery needed to keep the
visible workflow graph consistent with design history.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any, Generator, NotRequired, TypedDict, cast

if TYPE_CHECKING:
    from sqlalchemy import Row as SQLAlchemyRow
else:
    SQLAlchemyRow = Any

from kogwistar.runtime.models import WorkflowEdge, WorkflowNode

from .chat_service_shared import WorkflowProjectionRebuildingError, _BaseComponent


class WorkflowVisibleSnapshot(TypedDict):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


class WorkflowVisibleDelta(TypedDict):
    upsert_nodes: list[dict[str, Any]]
    delete_node_ids: list[str]
    upsert_edges: list[dict[str, Any]]
    delete_edge_ids: list[str]


class WorkflowVersionCommit(TypedDict):
    version: int
    prev_version: int
    target_seq: int
    created_at_ms: int
    entity_id: str
    action: str


class WorkflowVersionRef(TypedDict):
    version: int
    seq: int
    created_at_ms: int


class WorkflowSelectedVersionRef(TypedDict):
    version: int
    seq: int
    created_at_ms: int
    prev_version: int
    target_seq: int


class WorkflowDroppedRange(TypedDict):
    start_version: int
    end_version: int
    start_seq: int
    end_seq: int


class WorkflowControlTimelineItem(TypedDict, total=False):
    seq: int
    op: str
    designer_id: str
    ts_ms: int
    workflow_id: NotRequired[str]
    source: NotRequired[str]
    action: NotRequired[str]
    entity_id: NotRequired[str]
    version: NotRequired[int]
    prev_version: NotRequired[int]
    target_seq: NotRequired[int]
    from_version: NotRequired[int]
    to_version: NotRequired[int]
    drop_from_version: NotRequired[int]
    drop_to_version: NotRequired[int]
    drop_from_seq: NotRequired[int]
    drop_to_seq: NotRequired[int]
    reason: NotRequired[str]


class WorkflowProjectionRow(TypedDict, total=False):
    current_version: int
    active_tip_version: int
    last_authoritative_seq: int
    last_materialized_seq: int
    projection_schema_version: int
    snapshot_schema_version: int
    materialization_status: str
    updated_at_ms: int


class WorkflowProjectionPayload(TypedDict, total=False):
    current_version: int
    active_tip_version: int
    snapshot_schema_version: int
    versions: list[dict[str, Any]]
    dropped_ranges: list[WorkflowDroppedRange]


class WorkflowVersionDeltaRecord(TypedDict):
    workflow_id: str
    version: int
    prev_version: int
    target_seq: int
    forward: WorkflowVisibleDelta
    inverse: WorkflowVisibleDelta
    schema_version: int
    created_at_ms: int


class WorkflowFoldState(TypedDict):
    workflow_id: str
    namespace: str
    current_version: int
    active_tip_version: int
    max_version: int
    allocated_max_version: int
    current_seq: int
    can_undo: bool
    can_redo: bool
    versions: list[WorkflowVersionRef]
    selected_versions: list[WorkflowSelectedVersionRef]
    dropped_ranges: list[WorkflowDroppedRange]
    latest_seq: int
    timeline: list[WorkflowControlTimelineItem]
    commits: dict[int, WorkflowVersionCommit]


class _WorkflowDesignHistoryMixin(_BaseComponent):
    """Designer-side workflow history and projection helper mixin.

    This mixin treats the visible workflow graph as a materialized projection
    of append-only design control events plus workflow entity events within a
    workflow-specific namespace.

    Responsibilities include:
    - reading entity/control events from the workflow namespace
    - folding control history into an undo/redo/branch state model
    - rebuilding the visible workflow graph from snapshots and event replay
    - storing version deltas and snapshots for efficient materialization
    - applying visible deltas while suppressing recursive event-log feedback

    It does not execute workflows and does not define conversation/runtime
    behavior.
    """
    def _safe_workflow_nodes(self, *, workflow_id: str) -> list[WorkflowNode]:
        """Return all visible workflow nodes for a workflow.

        This is a defensive wrapper around the workflow engine's node query API.
        Some backends may raise transient/read-path exceptions when a collection
        exists logically but has not yet been materialized on disk. In those
        cases this helper returns an empty list instead of failing the designer
        surface.

        Args:
            workflow_id: Workflow whose visible node projection should be read.

        Returns:
            The current visible workflow nodes in the workflow namespace.

        Notes:
            This reads the materialized graph projection, not the authoritative
            design-control history.
        """        
        try:
            return self._workflow_engine().read.get_nodes(
                where={
                    "$and": [
                        {"entity_type": "workflow_node"},
                        {"workflow_id": workflow_id},
                    ]
                },
                limit=5000,
                node_type=WorkflowNode,
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "Nothing found on disk" in msg or "hnsw segment reader" in msg:
                return []
            raise

    def _safe_workflow_edges(self, *, workflow_id: str) -> list[WorkflowEdge]:
        """Return all visible workflow edges for a workflow.

        This is the edge equivalent of `_safe_workflow_nodes`, with the same
        defensive behavior for partially materialized or backend-specific empty
        states.

        Args:
            workflow_id: Workflow whose visible edge projection should be read.

        Returns:
            The current visible workflow edges in the workflow namespace.
        """
        try:
            return self._workflow_engine().read.get_edges(
                where={
                    "$and": [
                        {"entity_type": "workflow_edge"},
                        {"workflow_id": workflow_id},
                    ]
                },
                limit=20_000,
                edge_type=WorkflowEdge,
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "Nothing found on disk" in msg or "hnsw segment reader" in msg:
                return []
            raise

    def _iter_entity_events(
        self, *, namespace: str, from_seq: int = 1, to_seq: int | None = None
    ):
        """Iterate append-only entity events for a namespace.

        The underlying metadata store may expose slightly different iterator
        signatures across implementations. This helper normalizes that access
        and yields rows in increasing sequence order.

        Args:
            namespace: Event namespace to read from.
            from_seq: Inclusive lower bound sequence number.
            to_seq: Optional inclusive upper bound sequence number.

        Yields:
            Rows shaped like `(seq, entity_kind, entity_id, op, payload_raw)`.

        Notes:
            - Both control-plane designer events and data-plane node/edge events
            live in the same namespace stream.
            - Callers are responsible for filtering by `entity_kind`.

        Example:
            A rebuild path may replay only entity events between two committed
            versions by calling this helper with the lower and upper sequence
            bounds derived from version metadata.
        """
        iter_events = cast(
            Generator[SQLAlchemyRow, Any, None],
            getattr(self._workflow_engine().meta_sqlite, "iter_entity_events", None),
        )
        if not callable(iter_events):
            return
        kwargs: dict[str, Any] = {
            "namespace": str(namespace),
            "from_seq": int(from_seq),
        }
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

    def _parse_event_payload(self, payload_raw: Any) -> dict[str, Any]:
        """Best-effort parse an event payload into a dictionary.

        Payloads may already be decoded dicts or raw JSON strings depending on
        the metadata backend. Invalid or non-dict payloads degrade to `{}`.

        Args:
            payload_raw: Raw payload returned from the event store.

        Returns:
            A dictionary payload suitable for downstream folding/replay logic.
        """
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
        """Append a designer control-plane event to the workflow namespace.

        Control events encode editing history such as commit, undo, redo, and
        branch-drop operations. These events are the authoritative source of
        versioned designer state.

        Args:
            workflow_id: Workflow being edited.
            op: Control operation name.
            designer_id: Logical actor performing the change.
            source: Caller/source label for traceability.
            payload: Additional operation-specific fields.

        Returns:
            The appended event sequence number, or `0` if the metadata store does
            not support appends.

        Example:
            After a designer mutation is accepted, the service may append a
            `mutation_committed` control event carrying version, prev_version,
            target_seq, entity_id, and action metadata.
        """
        append = self._workflow_engine().meta_sqlite.append_entity_event
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
                entity_kind=self._design_control_kind,
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
        """Append a workflow data-plane entity event to the workflow namespace.

        This records node/edge-level changes in the same namespace as designer
        control events, but under a distinct `entity_kind`.

        Args:
            workflow_id: Workflow namespace owner.
            entity_kind: `node` or `edge`.
            entity_id: Entity identifier.
            op: Entity operation such as ADD, REPLACE, DELETE, or TOMBSTONE.
            payload: Serialized entity body or mutation payload.

        Returns:
            The appended event sequence number, or `0` if unsupported.

        Notes:
            In the designer-history pattern, these events are replayed to
            reconstruct the visible graph between committed versions.
        """
        append = self._workflow_engine().meta_sqlite.append_entity_event
        if not callable(append):
            return 0
        return int(
            append(
                namespace=self._workflow_namespace(workflow_id),
                event_id=str(uuid.uuid4()),
                entity_kind=str(entity_kind),
                entity_id=str(entity_id),
                op=str(op),
                payload_json=json.dumps(
                    dict(payload or {}), sort_keys=True, separators=(",", ":")
                ),
            )
        )

    def _workflow_meta_store(self):
        """Return the workflow metadata store backing design history state."""
        return self._workflow_engine().meta_sqlite

    def _workflow_projection(self, *, workflow_id: str):
        """Load the persisted workflow design projection metadata.

        This reads projection head/version metadata from the meta store, not the
        visible node/edge graph itself.

        Args:
            workflow_id: Workflow whose projection metadata is requested.

        Returns:
            A projection row dict if present, else `None`.
        """
        getter = self._workflow_meta_store().get_named_projection
        if not callable(getter):
            return None
        row = getter(namespace="workflow_design", key=str(workflow_id))
        if row is None:
            return None
        payload = row.get("payload")
        if not isinstance(payload, dict):
            return None
        return {
            "current_version": int(payload.get("current_version") or 0),
            "active_tip_version": int(payload.get("active_tip_version") or 0),
            "last_authoritative_seq": int(row.get("last_authoritative_seq") or 0),
            "last_materialized_seq": int(row.get("last_materialized_seq") or 0),
            "projection_schema_version": int(
                row.get("projection_schema_version") or self._projection_schema_version
            ),
            "snapshot_schema_version": int(
                payload.get("snapshot_schema_version") or self._snapshot_schema_version
            ),
            "materialization_status": str(row.get("materialization_status") or "ready"),
            "updated_at_ms": int(row.get("updated_at_ms") or self._now_ms()),
        }

    def _workflow_empty_delta(self) -> WorkflowVisibleDelta:
        """Return an empty visible-delta structure.

        The visible delta format describes graph materialization changes as:
        - node upserts
        - node deletes
        - edge upserts
        - edge deletes
        """
        return {
            "upsert_nodes": [],
            "delete_node_ids": [],
            "upsert_edges": [],
            "delete_edge_ids": [],
        }

    def _workflow_compute_visible_delta(
        self,
        *,
        before: WorkflowVisibleSnapshot,
        after: WorkflowVisibleSnapshot,
    ) -> WorkflowVisibleDelta:
        """Compute the materialized graph delta between two visible snapshots.

        Args:
            before: Previous visible workflow snapshot.
            after: New visible workflow snapshot.

        Returns:
            A delta containing upserts for changed/new entities and delete lists
            for removed entities.

        Notes:
            Identity is by entity `id`; payload equality is structural dict
            equality.

        Example:
            If a node label changes but its id stays the same, that node appears in
            `upsert_nodes`. If an edge is removed entirely, its id appears in
            `delete_edge_ids`.
        """
        before_nodes = {
            str(item.get("id") or ""): item
            for item in list(before.get("nodes") or [])
            if str(item.get("id") or "")
        }
        after_nodes = {
            str(item.get("id") or ""): item
            for item in list(after.get("nodes") or [])
            if str(item.get("id") or "")
        }
        before_edges = {
            str(item.get("id") or ""): item
            for item in list(before.get("edges") or [])
            if str(item.get("id") or "")
        }
        after_edges = {
            str(item.get("id") or ""): item
            for item in list(after.get("edges") or [])
            if str(item.get("id") or "")
        }
        return {
            "upsert_nodes": [
                after_nodes[node_id]
                for node_id in sorted(after_nodes.keys())
                if before_nodes.get(node_id) != after_nodes[node_id]
            ],
            "delete_node_ids": sorted(
                node_id for node_id in before_nodes.keys() if node_id not in after_nodes
            ),
            "upsert_edges": [
                after_edges[edge_id]
                for edge_id in sorted(after_edges.keys())
                if before_edges.get(edge_id) != after_edges[edge_id]
            ],
            "delete_edge_ids": sorted(
                edge_id for edge_id in before_edges.keys() if edge_id not in after_edges
            ),
        }

    def _workflow_store_version_delta(
        self,
        *,
        workflow_id: str,
        version: int,
        prev_version: int,
        target_seq: int,
        forward: WorkflowVisibleDelta,
        inverse: WorkflowVisibleDelta,
    ) -> None:
        """Persist the forward and inverse visible deltas for a committed version.

        These deltas allow efficient undo/redo or targeted materialization without
        always rebuilding from scratch.

        Args:
            workflow_id: Workflow owning the version.
            version: Newly committed version.
            prev_version: Parent version in lineage.
            target_seq: Authoritative event sequence associated with this version.
            forward: Delta from previous visible snapshot to this version.
            inverse: Delta that would revert this version back to its parent.
        """
        putter = getattr(self._workflow_meta_store(), "put_workflow_design_delta", None)
        if not callable(putter):
            return
        putter(
            workflow_id=workflow_id,
            version=int(version),
            prev_version=int(prev_version),
            target_seq=int(target_seq),
            forward_json=json.dumps(
                dict(forward or self._workflow_empty_delta()),
                sort_keys=True,
                separators=(",", ":"),
            ),
            inverse_json=json.dumps(
                dict(inverse or self._workflow_empty_delta()),
                sort_keys=True,
                separators=(",", ":"),
            ),
            schema_version=self._delta_schema_version,
        )

    def _workflow_version_delta(
        self, *, workflow_id: str, version: int
    ) -> WorkflowVersionDeltaRecord | None:
        """Load the stored visible delta for a committed version.

        Args:
            workflow_id: Workflow owning the version.
            version: Version to load.

        Returns:
            A normalized delta record, or `None` if no compatible stored delta
            exists.
        """
        getter = self._workflow_meta_store().get_workflow_design_delta
        if not callable(getter):
            return None
        row = getter(
            workflow_id=str(workflow_id),
            version=int(version),
            schema_version=self._delta_schema_version,
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
            "schema_version": int(
                row.get("schema_version") or self._delta_schema_version
            ),
            "created_at_ms": int(row.get("created_at_ms") or 0),
        }

    def _workflow_apply_visible_delta(
        self, *, workflow_id: str, delta: WorkflowVisibleDelta | None
    ) -> None:
        """Apply a visible delta directly to the materialized workflow graph.

        This mutates the visible node/edge projection while temporarily disabling
        recursive event-log/index side effects. It is intended for projection
        maintenance, not for authoritatively recording designer intent.

        Args:
            workflow_id: Workflow whose visible graph will be updated.
            delta: Delta to apply. `None` is treated as an empty delta.

        Behavior:
            1. delete edges
            2. delete nodes
            3. upsert nodes
            4. upsert edges
            5. remove orphan edges

        Why order matters:
            Edge deletion happens before node deletion to avoid dangling edge
            references during backend mutation.
        """
        payload: WorkflowVisibleDelta = dict(delta or self._workflow_empty_delta())
        eng = self._workflow_engine()
        prev_log = getattr(eng, "_disable_event_log", False)
        prev_idx = getattr(eng, "_phase1_enable_index_jobs", False)
        eng._disable_event_log = True
        eng._phase1_enable_index_jobs = False
        try:
            with eng.uow():
                edge_delete_ids = sorted(
                    {
                        str(item)
                        for item in list(payload.get("delete_edge_ids") or [])
                        if str(item)
                    }
                )
                node_delete_ids = sorted(
                    {
                        str(item)
                        for item in list(payload.get("delete_node_ids") or [])
                        if str(item)
                    }
                )
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

    def _workflow_control_timeline(
        self, *, namespace: str, limit: int = 500
    ) -> list[WorkflowControlTimelineItem]:
        """Return recent designer control events for a workflow namespace.

        Args:
            namespace: Workflow namespace.
            limit: Maximum number of most recent control events to retain.

        Returns:
            A bounded timeline of control-plane events suitable for UI/history
            inspection.

        Notes:
            This ignores node/edge entity events and includes only
            `self._design_control_kind`.
        """
        keep = max(1, int(limit))
        out: deque[WorkflowControlTimelineItem] = deque(maxlen=keep)
        for seq, entity_kind, _entity_id, op, payload_raw in self._iter_entity_events(
            namespace=namespace,
            from_seq=1,
        ):
            if str(entity_kind) != self._design_control_kind:
                continue
            payload = self._parse_event_payload(payload_raw)
            item: WorkflowControlTimelineItem = {
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
        commits: dict[int, WorkflowVersionCommit],
        version: int,
    ) -> list[WorkflowVersionCommit]:
        """Resolve the ancestry path from version 0 to a target committed version.

        Args:
            commits: Mapping of version id to committed version metadata.
            version: Target version to resolve.

        Returns:
            An ordered list beginning with synthetic version 0 and ending with the
            requested version.

        Raises:
            RuntimeError: If lineage loops or missing parent commits are detected.

        Example:
            If version 5 descends from 3, and 3 descends from 1, the returned path
            is `[0, 1, 3, 5]`.
        """
        if int(version) <= 0:
            return [
                {"version": 0, "prev_version": 0, "target_seq": 0, "created_at_ms": 0}
            ]
        path: list[WorkflowVersionCommit] = []
        seen: set[int] = set()
        current = int(version)
        while current > 0:
            if current in seen:
                raise RuntimeError(
                    f"Workflow design lineage loop detected at version={current}"
                )
            seen.add(current)
            commit = commits.get(current)
            if commit is None:
                raise RuntimeError(
                    f"Workflow design history missing committed version={current}"
                )
            path.append(commit)
            current = int(commit.get("prev_version") or 0)
        path.reverse()
        return [
            {"version": 0, "prev_version": 0, "target_seq": 0, "created_at_ms": 0}
        ] + path

    def _workflow_fold_history(self, *, workflow_id: str) -> WorkflowFoldState:
        """Fold the workflow namespace event stream into designer history state.

        This is the core history reducer for the workflow designer. It scans the
        namespace event log, interprets control events, and derives:

        - current selected version
        - active tip version
        - undo/redo availability
        - lineage-visible versions
        - dropped branch ranges
        - control timeline
        - committed version metadata

        Args:
            workflow_id: Workflow whose design history should be folded.

        Returns:
            A normalized `WorkflowFoldState`.

        Important distinction:
            - `current_version` is the currently selected version after undo/redo.
            - `active_tip_version` is the active branch tip after commit/drop logic.
            - `allocated_max_version` tracks the highest version number ever used,
            even if later dropped.

        Example:
            Suppose versions 1, 2, 3 are committed, then the user undoes to 2, then
            commits a new edit as version 4. The old forward branch from 3 may be
            marked dropped; `current_version` and `active_tip_version` both become
            4, while `allocated_max_version` may still reflect the highest allocated
            version id seen overall.
        """
        namespace = self._workflow_namespace(workflow_id)
        commits: dict[int, WorkflowVersionCommit] = {}
        dropped_ranges: list[WorkflowDroppedRange] = []
        current_version = 0
        active_tip_version = 0
        allocated_max_version = 0
        latest_seq = 0

        for seq, entity_kind, _entity_id, op, payload_raw in self._iter_entity_events(
            namespace=namespace, from_seq=1
        ):
            latest_seq = max(latest_seq, int(seq))
            if str(entity_kind) != self._design_control_kind:
                continue
            payload = self._parse_event_payload(payload_raw)
            op_s = str(op or "")
            if op_s == self._ctrl_mutation_committed:
                version = int(payload.get("version") or 0)
                prev_version = int(payload.get("prev_version") or 0)
                target_seq = int(payload.get("target_seq") or payload.get("seq") or 0)
                commit: WorkflowVersionCommit = {
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
            elif op_s == self._ctrl_undo_applied:
                current_version = int(payload.get("to_version") or current_version)
            elif op_s == self._ctrl_redo_applied:
                current_version = int(payload.get("to_version") or current_version)
            elif op_s == self._ctrl_branch_dropped:
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

        active_lineage = self._workflow_lineage_path(
            commits=commits, version=active_tip_version
        )
        selected_lineage = self._workflow_lineage_path(
            commits=commits, version=current_version
        )
        active_versions: list[WorkflowVersionRef] = [
            {
                "version": int(item.get("version") or 0),
                "seq": int(item.get("target_seq") or 0),
                "created_at_ms": int(item.get("created_at_ms") or 0),
            }
            for item in active_lineage
        ]
        selected_versions: list[WorkflowSelectedVersionRef] = [
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
            "can_redo": int(current_version) in active_ids
            and active_ids.index(int(current_version)) < len(active_ids) - 1,
            "versions": active_versions,
            "selected_versions": selected_versions,
            "dropped_ranges": dropped_ranges,
            "latest_seq": int(latest_seq),
            "timeline": self._workflow_control_timeline(namespace=namespace, limit=500),
            "commits": commits,
        }

    def _workflow_projection_stale(
        self, *, state: WorkflowFoldState, projection: WorkflowProjectionRow | None
    ) -> bool:
        """Return whether persisted projection metadata is stale.

        A projection is considered stale if it is missing, uses an incompatible
        schema version, lags behind the authoritative event sequence, or is not in
        the `ready` materialized state.
        """
        if projection is None:
            return True
        if (
            int(projection.get("projection_schema_version") or 0)
            != self._projection_schema_version
        ):
            return True
        if (
            int(projection.get("snapshot_schema_version") or 0)
            != self._snapshot_schema_version
        ):
            return True
        if int(projection.get("last_authoritative_seq") or 0) < int(
            state.get("latest_seq") or 0
        ):
            return True
        if str(projection.get("materialization_status") or "") != "ready":
            return True
        return False

    def _workflow_projection_payload(
        self,
        *,
        state: WorkflowFoldState,
    ) -> WorkflowProjectionPayload:
        return {
            "current_version": int(state.get("current_version") or 0),
            "active_tip_version": int(state.get("active_tip_version") or 0),
            "snapshot_schema_version": self._snapshot_schema_version,
            "versions": [
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
                    "prev_version": int(
                        (state.get("commits") or {})
                        .get(int(item.get("version") or 0), {})
                        .get("prev_version")
                        or 0
                    ),
                    "target_seq": int(item.get("seq") or 0),
                    "created_at_ms": int(item.get("created_at_ms") or 0),
                }
                for item in (state.get("versions") or [])
                if int(item.get("version") or 0)
                not in {
                    int(v.get("version") or 0)
                    for v in (state.get("selected_versions") or [])
                }
            ],
            "dropped_ranges": [
                {
                    "start_version": int(item.get("start_version") or 0),
                    "end_version": int(item.get("end_version") or 0),
                    "start_seq": int(item.get("start_seq") or 0),
                    "end_seq": int(item.get("end_seq") or 0),
                }
                for item in state.get("dropped_ranges") or []
            ],
        }

    def _store_workflow_projection(
        self,
        *,
        workflow_id: str,
        state: WorkflowFoldState,
        materialization_status: str = "ready",
    ) -> None:
        """Persist the workflow projection head and version metadata.

        This writes the projection metadata view used by the designer surface to
        reason about the selected lineage, active versions, and dropped ranges
        without rescanning the entire event log on every request.
        """
        replace = getattr(self._workflow_meta_store(), "replace_named_projection", None)
        if not callable(replace):
            return
        replace(
            "workflow_design",
            workflow_id,
            self._workflow_projection_payload(state=state),
            last_authoritative_seq=int(state.get("latest_seq") or 0),
            last_materialized_seq=int(state.get("current_seq") or 0),
            projection_schema_version=self._projection_schema_version,
            materialization_status=str(materialization_status),
        )

    def _workflow_projection_status(self, *, workflow_id: str) -> str | None:
        """Return the current projection materialization status, if any."""
        projection = self._workflow_projection(workflow_id=workflow_id)
        if projection is None:
            return None
        return str(projection.get("materialization_status") or "")

    def _assert_workflow_projection_not_rebuilding(self, *, workflow_id: str) -> None:
        """Raise if the workflow projection is currently marked as rebuilding.

        This protects read/write API surfaces that require a stable visible graph
        while a projection rebuild is in progress.
        """
        if self._workflow_projection_status(workflow_id=workflow_id) == "rebuilding":
            raise WorkflowProjectionRebuildingError(
                f"workflow_id={workflow_id!r} is rebuilding; retry later"
            )

    def _workflow_latest_seq(self, *, namespace: str, from_seq: int = 1) -> int:
        """Return the latest entity-event sequence for a namespace.

        Uses a direct metadata-store shortcut when available; otherwise falls back
        to iterating events from `from_seq`.
        """
        getter = self._workflow_meta_store().get_latest_entity_event_seq
        if callable(getter) and int(from_seq) <= 1:
            return int(getter(namespace=namespace))
        last = 0
        for seq, _ek, _eid, _op, _payload in self._iter_entity_events(
            namespace=namespace, from_seq=max(1, int(from_seq))
        ):
            last = int(seq)
        return last

    def _workflow_collect_visible_entity_ids(
        self, *, workflow_id: str
    ) -> tuple[set[str], set[str]]:
        nodes = self._safe_workflow_nodes(workflow_id=workflow_id)
        edges = self._safe_workflow_edges(workflow_id=workflow_id)
        """Collect ids of currently materialized workflow nodes and edges.

        Returns:
            `(node_ids, edge_ids)` for the visible workflow projection.
        """
        return (
            {str(node.id) for node in nodes},
            {str(edge.id) for edge in edges},
        )

    def _workflow_seq_in_dropped_ranges(
        self, seq: int, dropped_ranges: list[WorkflowDroppedRange]
    ) -> bool:
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
        dropped_ranges: list[WorkflowDroppedRange] | None = None,
    ) -> None:
        """Replay node/edge entity events from a namespace sequence range.

        This is the low-level materialization primitive used during projection
        rebuild. Control-plane designer events are ignored; only node/edge entity
        events are replayed. Events whose sequences lie inside dropped ranges are
        skipped.

        Args:
            namespace: Workflow namespace to replay from.
            from_seq: Inclusive lower bound.
            to_seq: Inclusive upper bound.
            dropped_ranges: Optional branch ranges to suppress during replay.

        Notes:
            Event logging and index jobs are temporarily disabled during replay so
            that reconstruction does not recursively append new history.

        Example:
            If version 7 descends from version 4, rebuild may restore a snapshot at
            4 and replay only the entity events between seq(4)+1 and seq(7), while
            skipping sequences that belong to a branch dropped after undo.
        """
        eng = self._workflow_engine()
        prev_log = getattr(eng, "_disable_event_log", False)
        prev_idx = getattr(eng, "_phase1_enable_index_jobs", False)
        eng._disable_event_log = True
        eng._phase1_enable_index_jobs = False
        try:
            for (
                seq,
                entity_kind,
                entity_id,
                op,
                payload_raw,
            ) in self._iter_entity_events(
                namespace=namespace,
                from_seq=max(1, int(from_seq)),
                to_seq=int(to_seq),
            ):
                if self._workflow_seq_in_dropped_ranges(
                    int(seq), list(dropped_ranges or [])
                ):
                    continue
                entity_kind_s = str(entity_kind)
                if entity_kind_s == self._design_control_kind:
                    continue
                payload = self._parse_event_payload(payload_raw)
                op_s = str(op)
                if entity_kind_s == "node":
                    if op_s in {"ADD", "REPLACE"}:
                        try:
                            node = WorkflowNode.model_validate(payload)
                        except Exception:
                            node = WorkflowNode.model_validate_json(
                                json.dumps(payload)
                            )
                        eng.write.add_node(node)
                    elif op_s in {"TOMBSTONE", "DELETE"}:
                        eng.backend.node_delete(ids=[str(entity_id)])
                elif entity_kind_s == "edge":
                    if op_s in {"ADD", "REPLACE"}:
                        try:
                            edge = WorkflowEdge.model_validate(payload)
                        except Exception:
                            edge = WorkflowEdge.model_validate_json(
                                json.dumps(payload)
                            )
                        eng.write.add_edge(edge)
                    elif op_s in {"TOMBSTONE", "DELETE"}:
                        eng.backend.edge_delete(ids=[str(entity_id)])
        finally:
            eng._disable_event_log = prev_log
            eng._phase1_enable_index_jobs = prev_idx

    def _workflow_capture_visible_snapshot(
        self, *, workflow_id: str
    ) -> WorkflowVisibleSnapshot:
        """Capture the currently visible workflow graph as a serializable snapshot.

        Returns:
            A snapshot containing node and edge payloads suitable for later restore
            or delta computation.

        Notes:
            Embeddings are excluded because this snapshot is concerned with visible
            structural design state, not derived vector data.
        """
        nodes = self._safe_workflow_nodes(workflow_id=workflow_id)
        edges = self._safe_workflow_edges(workflow_id=workflow_id)
        return {
            "nodes": [
                node.model_dump(field_mode="backend", exclude={"embedding"})
                for node in nodes
            ],
            "edges": [
                edge.model_dump(field_mode="backend", exclude={"embedding"})
                for edge in edges
            ],
        }

    def _workflow_restore_snapshot(
        self, *, snapshot_payload: WorkflowVisibleSnapshot
    ) -> None:
        """Restore a previously captured visible snapshot into the workflow graph.

        This writes nodes and edges back into the materialized projection while
        suppressing recursive event-log/index side effects.

        Notes:
            This does not clear existing entities first. Callers that need a clean
            rebuild should delete the visible projection before restoring.
        """
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
        """Delete visible edges whose endpoints are missing or malformed.

        An edge is considered orphaned if:
        - it has no source ids
        - it has no target ids
        - its first source id does not exist as a visible node
        - its first target id does not exist as a visible node

        This is a repair/sanity step used after delta application and rebuild.
        """
        nodes = self._safe_workflow_nodes(workflow_id=workflow_id)
        node_ids = {str(node.id) for node in nodes}
        edges = self._safe_workflow_edges(workflow_id=workflow_id)
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

    def _workflow_store_snapshot_if_needed(
        self, *, workflow_id: str, state: WorkflowFoldState
    ) -> None:
        """Persist a snapshot for the current selected version when eligible.

        Snapshots are stored only for positive versions that land on the configured
        snapshot interval. They are used to accelerate later rebuilds by reducing
        replay distance.
        """
        current_version = int(state.get("current_version") or 0)
        if current_version <= 0 or current_version % self._snapshot_interval != 0:
            return
        put_snapshot = getattr(
            self._workflow_meta_store(), "put_workflow_design_snapshot", None
        )
        if not callable(put_snapshot):
            return
        payload = self._workflow_capture_visible_snapshot(workflow_id=workflow_id)
        put_snapshot(
            workflow_id=workflow_id,
            version=current_version,
            seq=int(state.get("current_seq") or 0),
            payload_json=json.dumps(payload, sort_keys=True, separators=(",", ":")),
            schema_version=self._snapshot_schema_version,
        )

    def _workflow_rebuild_namespace_for_state(
        self, *, workflow_id: str, state: WorkflowFoldState
    ) -> None:
        """Rebuild the visible workflow namespace from folded designer state.

        Rebuild strategy:
            1. clear the currently materialized visible graph
            2. restore the nearest compatible snapshot on the selected lineage
            3. replay remaining entity events version-by-version up to the selected
            current version
            4. remove orphan edges

        Args:
            workflow_id: Workflow whose visible graph should be rebuilt.
            state: Folded designer state describing selected lineage and dropped
                branch ranges.

        Why this exists:
            The authoritative source of designer history is the append-only control
            stream, not the currently visible graph. When projection metadata is
            stale or a force rebuild is requested, the graph must be reconstructed
            from history.

        Example:
            If the current selected version is 12 and a snapshot exists at version
            10 on the selected lineage, rebuild restores version 10's snapshot and
            then replays only the entity events needed to reach versions 11 and 12.
        """
        namespace = self._workflow_namespace(workflow_id)
        eng = self._workflow_engine()
        node_ids, edge_ids = self._workflow_collect_visible_entity_ids(
            workflow_id=workflow_id
        )
        if edge_ids:
            try:
                eng.backend.edge_delete(ids=sorted(edge_ids))
            except Exception:
                logging.getLogger(__name__).exception(
                    "failed clearing workflow edges during rebuild: namespace=%s",
                    namespace,
                )
        if node_ids:
            try:
                eng.backend.node_delete(ids=sorted(node_ids))
            except Exception:
                logging.getLogger(__name__).exception(
                    "failed clearing workflow nodes during rebuild: namespace=%s",
                    namespace,
                )
        snapshot_get = self._workflow_meta_store().get_workflow_design_snapshot
        selected_versions = list(state.get("selected_versions") or [])
        selected_by_version = {
            int(item.get("version") or 0): item for item in selected_versions
        }
        restored_version = 0
        if callable(snapshot_get) and int(state.get("current_version") or 0) > 0:
            snapshot = snapshot_get(
                workflow_id=workflow_id,
                max_version=int(state.get("current_version") or 0),
                schema_version=self._snapshot_schema_version,
            )
            if (
                snapshot is not None
                and int(snapshot.get("version") or 0) in selected_by_version
            ):
                try:
                    payload = self._parse_event_payload(
                        str(snapshot.get("payload_json") or "{}")
                    )
                    self._workflow_restore_snapshot(snapshot_payload=payload)
                    restored_version = int(snapshot.get("version") or 0)
                except Exception:
                    logging.getLogger(__name__).exception(
                        "failed restoring workflow snapshot: workflow_id=%s",
                        workflow_id,
                    )
                    restored_version = 0
        for item in selected_versions:
            version = int(item.get("version") or 0)
            if version <= 0 or version <= restored_version:
                continue
            lower = int(item.get("prev_version") or 0)
            lower_seq = 0
            if lower > 0:
                lower_seq = int(
                    selected_by_version.get(lower, {}).get("target_seq") or 0
                )
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
    ) -> WorkflowFoldState:
        """Fold history and ensure projection metadata/materialization are aligned.

        Args:
            workflow_id: Workflow to synchronize.
            force_rebuild: If true, always rebuild the visible graph projection.
            materialize: If true, rebuild when the stored projection is stale.

        Returns:
            The freshly folded workflow state.

        Notes:
            This method assumes the caller already holds whatever higher-level lock
            or serialization discipline protects concurrent designer updates.
        """
        state = self._workflow_fold_history(workflow_id=workflow_id)
        projection = self._workflow_projection(workflow_id=workflow_id)
        if (
            force_rebuild
            or materialize
            and self._workflow_projection_stale(state=state, projection=projection)
        ):
            self._workflow_rebuild_namespace_for_state(
                workflow_id=workflow_id, state=state
            )
            self._workflow_store_snapshot_if_needed(
                workflow_id=workflow_id, state=state
            )
        self._store_workflow_projection(
            workflow_id=workflow_id, state=state, materialization_status="ready"
        )
        return state

    def _workflow_append_branch_drop_if_needed_locked(
        self,
        *,
        workflow_id: str,
        state: WorkflowFoldState,
        designer_id: str,
        source: str,
    ) -> bool:
        """Append a branch-drop control event when editing after undo.

        If the selected version is not the active tip and a new edit is about to be
        committed, the forward branch beyond the selected version is considered
        dropped. This helper records that dropped sequence/version range.

        Returns:
            True if a branch-drop event was appended, else False.

        Example:
            History is 1 -> 2 -> 3. User undoes to 1, then edits again.
            Version 2..3 becomes a dropped forward range before the new commit is
            recorded on the new branch.
        """
        current_version = int(state.get("current_version") or 0)
        dropped_versions = [
            item
            for item in (state.get("versions") or [])
            if int(item.get("version") or 0) > current_version
        ]
        if not dropped_versions:
            return False
        self._append_design_control_event(
            workflow_id=workflow_id,
            op=self._ctrl_branch_dropped,
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

    def _workflow_finalize_design_state_locked(
        self, *, workflow_id: str, rebuild: bool
    ) -> WorkflowFoldState:
        """Finalize projection metadata after a designer mutation workflow.

        This helper refolds the workflow history, optionally rebuilds the visible
        namespace, persists projection metadata, and stores a snapshot if the
        current version lands on the snapshot interval.

        Args:
            workflow_id: Workflow to finalize.
            rebuild: Whether to force a visible-graph rebuild before storing
                projection metadata.

        Returns:
            The final folded workflow state after persistence.
        """
        state = self._workflow_fold_history(workflow_id=workflow_id)
        if rebuild:
            self._workflow_rebuild_namespace_for_state(
                workflow_id=workflow_id, state=state
            )
        self._store_workflow_projection(
            workflow_id=workflow_id, state=state, materialization_status="ready"
        )
        self._workflow_store_snapshot_if_needed(workflow_id=workflow_id, state=state)
        return state
