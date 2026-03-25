from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from kogwistar.runtime import MappingStepResolver, StepContext, WorkflowRuntime
from kogwistar.runtime.models import RunSuccess


warnings.filterwarnings(
    "ignore",
    message=r"Using advanced underscore state key '_deps'.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Using advanced underscore state key '_rt_join'.*",
    category=RuntimeWarning,
)


JsonValue = Any


def _jsonable_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _history_event(step: str, **payload: Any) -> dict[str, Any]:
    event = {"step": step}
    event.update(payload)
    return event


def _classify_mock_note(note: Mapping[str, Any]) -> str:
    text = " ".join(str(note.get(key, "")) for key in ("title", "text")).lower()
    if "invoice" in text:
        return "finance/"
    if "meeting" in text:
        return "meetings/"
    return "archive/"


def _collect_run_result(
    *,
    framework_name: str,
    step_order: list[str],
    transition_map: dict[str, dict[str, str]],
    run_result: Any,
    trace_sink: Any,
) -> dict[str, Any]:
    clean_state = {
        key: value
        for key, value in run_result.final_state.items()
        if key not in {"_deps", "_rt_join"}
    }
    final_state = _jsonable_copy(clean_state)

    step_execs = trace_sink.get_nodes(
        where={
            "$and": [
                {"entity_type": "workflow_step_exec"},
                {"run_id": run_result.run_id},
            ]
        },
        limit=1000,
    )
    step_execs = sorted(
        step_execs,
        key=lambda node: int((getattr(node, "metadata", {}) or {}).get("step_seq", 0)),
    )
    runtime_step_ops = [
        str((getattr(node, "metadata", {}) or {}).get("op", ""))
        for node in step_execs
    ]

    counts: dict[str, int] = {}
    for node in trace_sink.get_nodes(where={"run_id": run_result.run_id}, limit=1000):
        entity_type = str((getattr(node, "metadata", {}) or {}).get("entity_type", ""))
        counts[entity_type] = counts.get(entity_type, 0) + 1

    return {
        "framework_name": framework_name,
        "framework_step_order": list(step_order),
        "transition_map": transition_map,
        "run_status": run_result.status,
        "run_id": run_result.run_id,
        "final_state": final_state,
        "staged_moves": final_state.get("staged_moves", []),
        "execution_history": final_state.get("execution_history", []),
        "runtime_step_ops": runtime_step_ops,
        "trace_counts": counts,
    }


@dataclass(frozen=True)
class DemoWorkflowNode:
    id: str
    label: str
    summary: str
    metadata: dict[str, Any]

    def safe_get_id(self) -> str:
        return self.id

    @property
    def op(self) -> str:
        return str(self.metadata.get("wf_op") or "noop")

    @property
    def terminal(self) -> bool:
        return bool(self.metadata.get("wf_terminal", False))

    @property
    def start(self) -> bool:
        return bool(self.metadata.get("wf_start", False))

    @property
    def fanout(self) -> bool:
        return bool(self.metadata.get("wf_fanout", False))


@dataclass(frozen=True)
class DemoWorkflowEdge:
    id: str
    source_ids: list[str]
    target_ids: list[str]
    label: str
    summary: str
    metadata: dict[str, Any]
    relation: str = "wf_next"
    type: str = "relationship"
    source_edge_ids: list[str] = field(default_factory=list)
    target_edge_ids: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    doc_id: str = "framework_then_agent_demo"
    domain_id: str | None = None
    canonical_entity_id: str | None = None

    def safe_get_id(self) -> str:
        return self.id

    @property
    def predicate(self) -> str | None:
        return self.metadata.get("wf_predicate")

    @property
    def priority(self) -> int:
        return int(self.metadata.get("wf_priority", 100))

    @property
    def is_default(self) -> bool:
        return bool(self.metadata.get("wf_is_default", False))

    @property
    def multiplicity(self) -> str:
        return str(self.metadata.get("wf_multiplicity", "one"))


class DemoGraphStore:
    """Tiny in-memory graph store used for workflow design and runtime traces."""

    def __init__(self) -> None:
        self._nodes: list[Any] = []
        self._edges: list[Any] = []
        self.read = self
        self.write = self

    def add_node(self, node: Any) -> Any:
        self._nodes.append(node)
        return node

    def add_edge(self, edge: Any) -> Any:
        self._edges.append(edge)
        return edge

    @staticmethod
    def _metadata(entity: Any) -> dict[str, Any]:
        metadata = getattr(entity, "metadata", None)
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _entity_id(entity: Any) -> str:
        if hasattr(entity, "safe_get_id"):
            return str(entity.safe_get_id())
        return str(getattr(entity, "id", ""))

    def _matches(self, entity: Any, where: dict[str, Any] | None) -> bool:
        if where is None:
            return True
        if "$and" in where:
            parts = where.get("$and", [])
            return all(self._matches(entity, part) for part in parts if isinstance(part, dict))

        metadata = self._metadata(entity)
        for key, expected in where.items():
            if key == "id":
                if self._entity_id(entity) != str(expected):
                    return False
                continue
            if key == "entity_type":
                if metadata.get("entity_type") != expected:
                    return False
                continue
            if key == "workflow_id":
                if metadata.get("workflow_id") != expected:
                    return False
                continue
            if key == "run_id":
                if metadata.get("run_id") != expected:
                    return False
                continue
            if metadata.get(key) != expected:
                return False
        return True

    def _select(
        self,
        entities: list[Any],
        *,
        where: dict[str, Any] | None = None,
        ids: Sequence[str] | None = None,
        limit: int = 5000,
    ) -> list[Any]:
        selected = list(entities)
        if ids is not None:
            wanted = {str(item) for item in ids}
            selected = [item for item in selected if self._entity_id(item) in wanted]
        if where is not None:
            selected = [item for item in selected if self._matches(item, where)]
        return selected[: int(limit)]

    def get_nodes(
        self,
        where: dict[str, Any] | None = None,
        limit: int = 5000,
        ids: Sequence[str] | None = None,
        **_: Any,
    ) -> list[Any]:
        return self._select(self._nodes, where=where, ids=ids, limit=limit)

    def get_edges(
        self,
        where: dict[str, Any] | None = None,
        limit: int = 5000,
        ids: Sequence[str] | None = None,
        **_: Any,
    ) -> list[Any]:
        return self._select(self._edges, where=where, ids=ids, limit=limit)


def _wf_node(
    workflow_id: str,
    node_id: str,
    *,
    op: str,
    start: bool = False,
    terminal: bool = False,
) -> DemoWorkflowNode:
    return DemoWorkflowNode(
        id=node_id,
        label=op,
        summary=op,
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": op,
            "wf_version": "v1",
            "wf_start": start,
            "wf_terminal": terminal,
            "wf_fanout": False,
        },
    )


def _wf_edge(
    workflow_id: str,
    edge_id: str,
    *,
    src: str,
    dst: str,
    predicate: str | None,
    priority: int = 100,
    default: bool = False,
) -> DemoWorkflowEdge:
    return DemoWorkflowEdge(
        id=edge_id,
        source_ids=[src],
        target_ids=[dst],
        label=f"{src}->{dst}",
        summary=f"{src} -> {dst}",
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_predicate": predicate,
            "wf_priority": priority,
            "wf_is_default": default,
            "wf_multiplicity": "one",
        },
    )


@dataclass
class PlanActObserveFramework:
    """Reusable loop framework: plan, guard, act, observe, finish."""

    workflow_id: str = "framework_then_agent_demo"

    @property
    def step_order(self) -> list[str]:
        return ["plan", "approve", "act", "observe", "end"]

    def build_workflow(self) -> tuple[DemoGraphStore, dict[str, dict[str, str]]]:
        engine = DemoGraphStore()
        nodes = [
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|plan", op="plan", start=True),
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|approve", op="approve"),
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|act", op="act"),
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|observe", op="observe"),
            _wf_node(
                self.workflow_id,
                f"wf|{self.workflow_id}|end",
                op="end",
                terminal=True,
            ),
        ]
        for node in nodes:
            engine.add_node(node)

        edges = [
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|plan->approve",
                src=nodes[0].safe_get_id(),
                dst=nodes[1].safe_get_id(),
                predicate=None,
                default=True,
            ),
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|approve->act",
                src=nodes[1].safe_get_id(),
                dst=nodes[2].safe_get_id(),
                predicate="approved",
                priority=1,
            ),
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|approve->end",
                src=nodes[1].safe_get_id(),
                dst=nodes[4].safe_get_id(),
                predicate="denied",
                priority=0,
            ),
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|act->observe",
                src=nodes[2].safe_get_id(),
                dst=nodes[3].safe_get_id(),
                predicate=None,
                default=True,
            ),
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|observe->plan",
                src=nodes[3].safe_get_id(),
                dst=nodes[0].safe_get_id(),
                predicate="has_more",
                priority=1,
            ),
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|observe->end",
                src=nodes[3].safe_get_id(),
                dst=nodes[4].safe_get_id(),
                predicate=None,
                default=True,
            ),
        ]
        for edge in edges:
            engine.add_edge(edge)

        transition_map = {
            "plan": {"default": "approve"},
            "approve": {"approved": "act", "denied": "end"},
            "act": {"default": "observe"},
            "observe": {"has_more": "plan", "default": "end"},
            "end": {},
        }
        return engine, transition_map

    def build_resolver(self) -> MappingStepResolver:
        resolver = MappingStepResolver()
        resolver.set_state_schema(
            {
                "execution_history": "a",
                "staged_moves": "a",
            }
        )

        @resolver.register("plan")
        def _plan(ctx: StepContext):
            deps = dict(ctx.state_view.get("_deps") or {})
            pending = list(ctx.state_view.get("pending_notes") or [])
            next_note = dict(pending[0]) if pending else None
            event = _history_event(
                "plan",
                note_id=next_note.get("id") if next_note else None,
                pending_count=len(pending),
            )
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    (
                        "u",
                        {
                            "planned_note": next_note,
                            "current_note": next_note,
                            "blocked": False,
                            "blocked_reason": None,
                        },
                    ),
                    ("a", {"execution_history": event}),
                ],
                next_step_names=[],
            )

        @resolver.register("approve")
        def _approve(ctx: StepContext):
            deps = dict(ctx.state_view.get("_deps") or {})
            note = ctx.state_view.get("current_note")
            approval_policy = deps.get("approval_policy")
            approved = bool(approval_policy(note, ctx.state_view)) if callable(approval_policy) else False
            event = _history_event(
                "approve",
                note_id=(note or {}).get("id") if isinstance(note, dict) else None,
                approved=approved,
            )
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    (
                        "u",
                        {
                            "approval_granted": approved,
                            "blocked": not approved,
                            "blocked_reason": None if approved else "approval required",
                        },
                    ),
                    ("a", {"execution_history": event}),
                ],
            )

        @resolver.register("act")
        def _act(ctx: StepContext):
            deps = dict(ctx.state_view.get("_deps") or {})
            note = ctx.state_view.get("current_note")
            classify_note = deps.get("classify_note")
            bucket = (
                str(classify_note(note, ctx.state_view))
                if callable(classify_note)
                else "archive/"
            )
            pending = list(ctx.state_view.get("pending_notes") or [])
            remaining = pending[1:] if pending else []
            move = {
                "note_id": (note or {}).get("id") if isinstance(note, dict) else None,
                "note_title": (note or {}).get("title") if isinstance(note, dict) else None,
                "destination": bucket,
            }
            event = _history_event(
                "act",
                note_id=move["note_id"],
                destination=bucket,
            )
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    (
                        "u",
                        {
                            "pending_notes": remaining,
                            "planned_note": None,
                            "current_note": note,
                        },
                    ),
                    ("a", {"staged_moves": move}),
                    ("a", {"execution_history": event}),
                ],
            )

        @resolver.register("observe")
        def _observe(ctx: StepContext):
            pending = list(ctx.state_view.get("pending_notes") or [])
            event = _history_event("observe", remaining_count=len(pending))
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    ("a", {"execution_history": event}),
                ],
            )

        @resolver.register("end")
        def _end(ctx: StepContext):
            blocked = bool(ctx.state_view.get("blocked"))
            pending = list(ctx.state_view.get("pending_notes") or [])
            completed = (not blocked) and len(pending) == 0
            event = _history_event(
                "end",
                final_status="blocked" if blocked else "completed",
            )
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    (
                        "u",
                        {
                            "completed": completed,
                            "final_status": "blocked" if blocked else "completed",
                        },
                    ),
                    ("a", {"execution_history": event}),
                ],
            )

        return resolver

    def build_runtime(self) -> tuple[WorkflowRuntime, DemoGraphStore, DemoGraphStore]:
        workflow_engine, _transition_map = self.build_workflow()
        trace_sink = DemoGraphStore()
        resolver = self.build_resolver()

        runtime = WorkflowRuntime(
            workflow_engine=workflow_engine,
            conversation_engine=trace_sink,
            step_resolver=resolver,
            predicate_registry={
                "approved": lambda _e, state, _r: bool(state.get("approval_granted")),
                "denied": lambda _e, state, _r: not bool(state.get("approval_granted")),
                "has_more": lambda _e, state, _r: len(list(state.get("pending_notes") or [])) > 0,
            },
            checkpoint_every_n_steps=1,
            max_workers=1,
            transaction_mode="none",
            trace=True,
        )
        return runtime, workflow_engine, trace_sink

    def transition_summary(self) -> dict[str, dict[str, str]]:
        _engine, transition_map = self.build_workflow()
        return transition_map


@dataclass
class PlanActObserveNoApprovalFramework(PlanActObserveFramework):
    """Easy swap: same family, but approval is removed and the agent runs unchanged."""

    @property
    def step_order(self) -> list[str]:
        return ["plan", "act", "observe", "end"]

    def build_workflow(self) -> tuple[DemoGraphStore, dict[str, dict[str, str]]]:
        engine = DemoGraphStore()
        nodes = [
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|plan", op="plan", start=True),
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|act", op="act"),
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|observe", op="observe"),
            _wf_node(
                self.workflow_id,
                f"wf|{self.workflow_id}|end",
                op="end",
                terminal=True,
            ),
        ]
        for node in nodes:
            engine.add_node(node)

        edges = [
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|plan->act",
                src=nodes[0].safe_get_id(),
                dst=nodes[1].safe_get_id(),
                predicate=None,
                default=True,
            ),
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|act->observe",
                src=nodes[1].safe_get_id(),
                dst=nodes[2].safe_get_id(),
                predicate=None,
                default=True,
            ),
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|observe->plan",
                src=nodes[2].safe_get_id(),
                dst=nodes[0].safe_get_id(),
                predicate="has_more",
                priority=1,
            ),
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|observe->end",
                src=nodes[2].safe_get_id(),
                dst=nodes[3].safe_get_id(),
                predicate=None,
                default=True,
            ),
        ]
        for edge in edges:
            engine.add_edge(edge)

        transition_map = {
            "plan": {"default": "act"},
            "act": {"default": "observe"},
            "observe": {"has_more": "plan", "default": "end"},
            "end": {},
        }
        return engine, transition_map

    def build_resolver(self) -> MappingStepResolver:
        resolver = MappingStepResolver()
        resolver.set_state_schema(
            {
                "execution_history": "a",
                "staged_moves": "a",
            }
        )

        @resolver.register("plan")
        def _plan(ctx: StepContext):
            pending = list(ctx.state_view.get("pending_notes") or [])
            next_note = dict(pending[0]) if pending else None
            event = _history_event(
                "plan",
                note_id=next_note.get("id") if next_note else None,
                pending_count=len(pending),
            )
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    (
                        "u",
                        {
                            "planned_note": next_note,
                            "current_note": next_note,
                            "blocked": False,
                            "blocked_reason": None,
                        },
                    ),
                    ("a", {"execution_history": event}),
                ],
                next_step_names=[],
            )

        @resolver.register("act")
        def _act(ctx: StepContext):
            deps = dict(ctx.state_view.get("_deps") or {})
            note = ctx.state_view.get("current_note")
            classify_note = deps.get("classify_note")
            bucket = (
                str(classify_note(note, ctx.state_view))
                if callable(classify_note)
                else "archive/"
            )
            pending = list(ctx.state_view.get("pending_notes") or [])
            remaining = pending[1:] if pending else []
            move = {
                "note_id": (note or {}).get("id") if isinstance(note, dict) else None,
                "note_title": (note or {}).get("title") if isinstance(note, dict) else None,
                "destination": bucket,
            }
            event = _history_event("act", note_id=move["note_id"], destination=bucket)
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    (
                        "u",
                        {
                            "pending_notes": remaining,
                            "planned_note": None,
                            "current_note": note,
                        },
                    ),
                    ("a", {"staged_moves": move}),
                    ("a", {"execution_history": event}),
                ],
            )

        @resolver.register("observe")
        def _observe(ctx: StepContext):
            pending = list(ctx.state_view.get("pending_notes") or [])
            event = _history_event("observe", remaining_count=len(pending))
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("a", {"execution_history": event})],
            )

        @resolver.register("end")
        def _end(ctx: StepContext):
            pending = list(ctx.state_view.get("pending_notes") or [])
            completed = len(pending) == 0
            event = _history_event("end", final_status="completed")
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    ("u", {"completed": completed, "final_status": "completed"}),
                    ("a", {"execution_history": event}),
                ],
            )

        return resolver

    def build_runtime(self) -> tuple[WorkflowRuntime, DemoGraphStore, DemoGraphStore]:
        workflow_engine, _transition_map = self.build_workflow()
        trace_sink = DemoGraphStore()
        resolver = self.build_resolver()

        runtime = WorkflowRuntime(
            workflow_engine=workflow_engine,
            conversation_engine=trace_sink,
            step_resolver=resolver,
            predicate_registry={
                "has_more": lambda _e, state, _r: len(list(state.get("pending_notes") or []))
                > 0,
            },
            checkpoint_every_n_steps=1,
            max_workers=1,
            transaction_mode="none",
            trace=True,
        )
        return runtime, workflow_engine, trace_sink

    def transition_summary(self) -> dict[str, dict[str, str]]:
        _engine, transition_map = self.build_workflow()
        return transition_map


@dataclass(frozen=True)
class BatchNotesOrganizerAdapter:
    """Adapter that makes the same notes-organizer usable by a batch-style framework."""

    agent: "MockNotesOrganizerAgent"

    def batch_notes(self) -> list[dict[str, str]]:
        return [dict(note) for note in self.agent.notes]

    def classify_batch(self, notes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "note_id": str(note.get("id")),
                "note_title": str(note.get("title")),
                "destination": _classify_mock_note(note),
            }
            for note in notes
        ]

    def apply_batch(self, plan: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [dict(item) for item in plan]


@dataclass
class BatchClassifyThenApplyFramework:
    """Harder variant: batch classify first, then apply moves as a separate phase."""

    workflow_id: str = "framework_then_agent_demo_batch"

    @property
    def step_order(self) -> list[str]:
        return ["collect", "classify_batch", "apply_batch", "end"]

    def build_workflow(self) -> tuple[DemoGraphStore, dict[str, dict[str, str]]]:
        engine = DemoGraphStore()
        nodes = [
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|collect", op="collect", start=True),
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|classify", op="classify_batch"),
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|apply", op="apply_batch"),
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|end", op="end", terminal=True),
        ]
        for node in nodes:
            engine.add_node(node)

        edges = [
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|collect->classify",
                src=nodes[0].safe_get_id(),
                dst=nodes[1].safe_get_id(),
                predicate=None,
                default=True,
            ),
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|classify->apply",
                src=nodes[1].safe_get_id(),
                dst=nodes[2].safe_get_id(),
                predicate="has_plan",
                priority=1,
            ),
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|apply->end",
                src=nodes[2].safe_get_id(),
                dst=nodes[3].safe_get_id(),
                predicate=None,
                default=True,
            ),
        ]
        for edge in edges:
            engine.add_edge(edge)

        transition_map = {
            "collect": {"default": "classify_batch"},
            "classify_batch": {"has_plan": "apply_batch"},
            "apply_batch": {"default": "end"},
            "end": {},
        }
        return engine, transition_map

    def build_resolver(
        self, adapter: BatchNotesOrganizerAdapter
    ) -> MappingStepResolver:
        resolver = MappingStepResolver()
        resolver.set_state_schema(
            {
                "execution_history": "a",
                "classification_plan": "a",
                "staged_moves": "a",
            }
        )

        @resolver.register("collect")
        def _collect(ctx: StepContext):
            notes = [dict(note) for note in adapter.batch_notes()]
            event = _history_event("collect", note_count=len(notes))
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    ("u", {"batch_notes": notes, "batch_ready": False}),
                    ("a", {"execution_history": event}),
                ],
            )

        @resolver.register("classify_batch")
        def _classify(ctx: StepContext):
            notes = list(ctx.state_view.get("batch_notes") or [])
            plan = adapter.classify_batch(notes)
            event = _history_event("classify_batch", plan_count=len(plan))
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    ("u", {"classification_plan": plan, "batch_ready": True}),
                    ("a", {"execution_history": event}),
                ],
            )

        @resolver.register("apply_batch")
        def _apply(ctx: StepContext):
            plan = list(ctx.state_view.get("classification_plan") or [])
            applied = adapter.apply_batch(plan)
            event = _history_event("apply_batch", applied_count=len(applied))
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    (
                        "u",
                        {
                            "batch_notes": [],
                            "batch_applied": True,
                            "completed": True,
                            "final_status": "completed",
                            "staged_moves": applied,
                        },
                    ),
                    ("a", {"execution_history": event}),
                ],
            )

        @resolver.register("end")
        def _end(ctx: StepContext):
            event = _history_event(
                "end",
                final_status=str(ctx.state_view.get("final_status") or "completed"),
            )
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("a", {"execution_history": event})],
            )

        return resolver

    def build_runtime(
        self, adapter: BatchNotesOrganizerAdapter
    ) -> tuple[WorkflowRuntime, DemoGraphStore, DemoGraphStore]:
        workflow_engine, _transition_map = self.build_workflow()
        trace_sink = DemoGraphStore()
        resolver = self.build_resolver(adapter)

        runtime = WorkflowRuntime(
            workflow_engine=workflow_engine,
            conversation_engine=trace_sink,
            step_resolver=resolver,
            predicate_registry={
                "has_plan": lambda _e, state, _r: bool(state.get("classification_plan")),
            },
            checkpoint_every_n_steps=1,
            max_workers=1,
            transaction_mode="none",
            trace=True,
        )
        return runtime, workflow_engine, trace_sink

    def transition_summary(self) -> dict[str, dict[str, str]]:
        _engine, transition_map = self.build_workflow()
        return transition_map

    def run(self, agent: "MockNotesOrganizerAgent") -> dict[str, Any]:
        adapter = BatchNotesOrganizerAdapter(agent)
        runtime, _workflow_engine, trace_sink = self.build_runtime(adapter)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Using advanced underscore state key '_deps'.*",
                category=RuntimeWarning,
            )
            run_result = runtime.run(
                workflow_id=self.workflow_id,
                conversation_id="demo-conversation",
                turn_node_id="demo-turn",
                initial_state={
                    "_deps": agent.tool_deps(),
                    "batch_notes": adapter.batch_notes(),
                    "classification_plan": [],
                    "batch_ready": False,
                    "batch_applied": False,
                    "completed": False,
                    "final_status": "running",
                    "execution_history": [],
                    "staged_moves": [],
                },
            )
        return _collect_run_result(
            framework_name=self.__class__.__name__,
            step_order=list(self.step_order),
            transition_map=self.transition_summary(),
            run_result=run_result,
            trace_sink=trace_sink,
        )


@dataclass
class MockNotesOrganizerAgent:
    """Concrete agent built on top of the reusable framework."""

    framework: PlanActObserveFramework = field(
        default_factory=PlanActObserveFramework
    )
    notes: list[dict[str, str]] = field(
        default_factory=lambda: [
            {
                "id": "note-1",
                "title": "Vendor invoice",
                "text": "Invoice for subscription renewal",
            },
            {
                "id": "note-2",
                "title": "Team meeting",
                "text": "Meeting notes from product sync",
            },
            {
                "id": "note-3",
                "title": "Loose idea",
                "text": "Archive later if still useful",
            },
        ]
    )
    auto_approve: bool = True

    def approval_policy(
        self, note: Mapping[str, Any] | None, state: Mapping[str, Any]
    ) -> bool:
        _ = note, state
        return bool(self.auto_approve)

    def classify_note(
        self, note: Mapping[str, Any] | None, state: Mapping[str, Any]
    ) -> str:
        _ = state
        return _classify_mock_note(note or {})

    def tool_deps(self) -> dict[str, Any]:
        return {
            "approval_policy": self.approval_policy,
            "classify_note": self.classify_note,
        }

    def initial_state(self) -> dict[str, Any]:
        return {
            "_deps": self.tool_deps(),
            "pending_notes": [dict(note) for note in self.notes],
            "planned_note": None,
            "current_note": None,
            "approval_granted": False,
            "blocked": False,
            "blocked_reason": None,
            "completed": False,
            "final_status": "running",
            "execution_history": [],
            "staged_moves": [],
        }

    def run(self, framework: Any | None = None) -> dict[str, Any]:
        framework = framework or self.framework
        runtime, _workflow_engine, trace_sink = framework.build_runtime()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Using advanced underscore state key '_deps'.*",
                category=RuntimeWarning,
            )
            run_result = runtime.run(
                workflow_id=self.framework.workflow_id,
                conversation_id="demo-conversation",
                turn_node_id="demo-turn",
                initial_state=self.initial_state(),
            )
        return _collect_run_result(
            framework_name=framework.__class__.__name__,
            step_order=list(framework.step_order),
            transition_map=framework.transition_summary(),
            run_result=run_result,
            trace_sink=trace_sink,
        )


def run_framework_then_agent_demo(
    auto_approve: bool = True, variant: str = "original"
) -> dict[str, Any]:
    agent = MockNotesOrganizerAgent(auto_approve=auto_approve)
    if variant == "original":
        return agent.run()
    if variant == "easy":
        return agent.run(framework=PlanActObserveNoApprovalFramework())
    if variant == "harder":
        return BatchClassifyThenApplyFramework().run(agent)
    raise ValueError(
        "variant must be one of: 'original', 'easy', 'harder'"
    )


def run_framework_then_agent_demo_suite(
    auto_approve: bool = True,
) -> dict[str, Any]:
    agent = MockNotesOrganizerAgent(auto_approve=auto_approve)
    original = agent.run()
    easy = agent.run(framework=PlanActObserveNoApprovalFramework())
    harder = BatchClassifyThenApplyFramework().run(agent)
    return {
        "original": original,
        "easy": easy,
        "harder": harder,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the framework-first demo: define a loop, then instantiate a concrete agent."
    )
    parser.add_argument(
        "--variant",
        choices=["original", "easy", "harder", "all"],
        default="original",
        help="Choose which framework variant to run.",
    )
    parser.add_argument(
        "--deny-approval",
        action="store_true",
        help="Run the same demo with the guard step denying action.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    auto_approve = not args.deny_approval
    if args.variant == "all":
        result = run_framework_then_agent_demo_suite(auto_approve=auto_approve)
    else:
        result = run_framework_then_agent_demo(
            auto_approve=auto_approve, variant=args.variant
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
