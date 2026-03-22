from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from kogwistar.engine_core.models import Grounding, MentionVerification, Node, Span
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


def _jsonable_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _history_event(step: str, **payload: Any) -> dict[str, Any]:
    event = {"step": step}
    event.update(payload)
    return event


def _entity_to_dict(entity: Any) -> dict[str, Any]:
    metadata = dict(getattr(entity, "metadata", {}) or {})
    payload: dict[str, Any] = {
        "id": str(getattr(entity, "id", "")),
        "label": str(getattr(entity, "label", "")),
        "summary": str(getattr(entity, "summary", "")),
        "metadata": _jsonable_copy(metadata),
    }
    if hasattr(entity, "source_ids"):
        payload["source_ids"] = list(getattr(entity, "source_ids") or [])
    if hasattr(entity, "target_ids"):
        payload["target_ids"] = list(getattr(entity, "target_ids") or [])
    if hasattr(entity, "relation"):
        payload["relation"] = str(getattr(entity, "relation", ""))
    if hasattr(entity, "type"):
        payload["type"] = str(getattr(entity, "type", ""))
    if hasattr(entity, "mentions"):
        mentions = []
        for grounding in list(getattr(entity, "mentions") or []):
            spans = []
            for span in list(getattr(grounding, "spans", []) or []):
                spans.append(
                    {
                        "doc_id": str(getattr(span, "doc_id", "")),
                        "document_page_url": str(
                            getattr(span, "document_page_url", "")
                        ),
                        "page_number": int(getattr(span, "page_number", 1) or 1),
                        "start_char": int(getattr(span, "start_char", 0) or 0),
                        "end_char": int(getattr(span, "end_char", 0) or 0),
                        "excerpt": str(getattr(span, "excerpt", "")),
                        "context_before": str(getattr(span, "context_before", "")),
                        "context_after": str(getattr(span, "context_after", "")),
                    }
                )
            mentions.append({"spans": spans})
        payload["mentions"] = mentions
    return payload


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _slug(value: str) -> str:
    return "-".join(part for part in value.lower().split() if part)


def _infer_topic(note: Mapping[str, Any]) -> str:
    return str(_topic_judgment(note)["topic"])


def _topic_judgment(note: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic stand-in for a future LLM topic-classification call."""
    text = " ".join(str(note.get(key, "")) for key in ("title", "text")).lower()
    signals: list[str] = []
    topic = "general"
    if "invoice" in text or "budget" in text or "finance" in text:
        topic = "finance"
        signals = ["invoice", "budget", "finance"]
    elif "meeting" in text or "sync" in text or "standup" in text:
        topic = "team"
        signals = ["meeting", "sync", "standup"]

    rationale = (
        f"Detected signals {signals} in the note text." if signals else "No strong topic signal found."
    )
    return {
        "topic": topic,
        "confidence": 0.94 if signals else 0.58,
        "signals": signals,
        "rationale": rationale,
    }


def _provenance_response(
    *,
    note_id: str,
    topic: str,
    title: str,
    span: Mapping[str, Any] | None,
    normalized_note: Mapping[str, Any] | None,
    evidence_steps: Sequence[str],
) -> dict[str, Any]:
    """Deterministic stand-in for a future LLM provenance synthesis call."""
    if span is not None and normalized_note is not None:
        answer = (
            f"I would move {note_id} to {topic}/ because the grounded source span says "
            f"{span['excerpt']!r} and the stored normalize step classified \"{title}\" as "
            f"topic={normalized_note.get('topic')!r}. The commit step then persisted that grounded "
            f"artifact node in the shared graph."
        )
        reasoning = [
            f"Grounding points to excerpt {span['excerpt']!r} in {span['doc_id']}.",
            f"Normalize step classified {title!r} as topic={normalized_note.get('topic')!r}.",
            "Commit step persisted the grounded artifact node.",
        ]
    elif normalized_note is not None:
        answer = (
            f"I would move {note_id} to {topic}/ because the stored normalize step classified "
            f'"{title}" as topic={normalized_note.get("topic")!r}, and the commit step persisted that '
            f"artifact node in the shared graph."
        )
        reasoning = [
            f"Normalize step classified {title!r} as topic={normalized_note.get('topic')!r}.",
            "Commit step persisted the artifact node.",
        ]
    else:
        answer = (
            f"I would move {note_id} to {topic}/ because the grounded artifact node and stored step "
            f"history show that classification and commit completed in the same run."
        )
        reasoning = [
            "Artifact node links back to a completed workflow run.",
            "Stored step history shows classification and commit completed.",
        ]

    citations = []
    if span is not None:
        citations.append(
            {
                "kind": "grounding_span",
                "doc_id": str(span.get("doc_id") or ""),
                "excerpt": str(span.get("excerpt") or ""),
                "start_char": int(span.get("start_char") or 0),
                "end_char": int(span.get("end_char") or 0),
            }
        )
    citations.extend(
        {
            "kind": "workflow_step",
            "step": step_name,
        }
        for step_name in evidence_steps
    )
    return {
        "answer": answer,
        "confidence": 0.96 if span is not None else 0.88,
        "reasoning": reasoning,
        "citations": citations,
        "replace_with_llm": True,
    }


def _source_doc_id(note_id: str) -> str:
    return f"doc|graph_native_artifact_demo|{note_id}"


def _note_source_content(note: Mapping[str, Any]) -> str:
    return (
        f"Title: {str(note.get('title') or '').strip()}\n"
        f"Body: {str(note.get('text') or '').strip()}"
    )


def _surrounding_text(content: str, *, start: int, end: int, width: int = 24) -> tuple[str, str]:
    before = max(0, start - width)
    after = min(len(content), end + width)
    return content[before:start], content[end:after]


def _grounding_for_note(note: Mapping[str, Any], *, insertion_method: str) -> Grounding:
    note_id = str(note.get("id") or "").strip()
    excerpt = str(note.get("title") or "").strip() or str(note.get("text") or "").strip()
    content = _note_source_content(note)
    start_char = content.find(excerpt)
    if start_char < 0:
        excerpt = content
        start_char = 0
    end_char = start_char + max(1, len(excerpt))
    context_before, context_after = _surrounding_text(
        content,
        start=start_char,
        end=end_char,
    )
    return Grounding(
        spans=[
            Span(
                collection_page_url="demo://graph-native-artifact",
                document_page_url=f"demo://graph-native-artifact/{_source_doc_id(note_id)}",
                doc_id=_source_doc_id(note_id),
                insertion_method=insertion_method,
                page_number=1,
                start_char=start_char,
                end_char=end_char,
                excerpt=excerpt,
                context_before=context_before,
                context_after=context_after,
                verification=MentionVerification(
                    method="system",
                    is_verified=True,
                    score=1.0,
                    notes="demo-seeded grounding",
                ),
            )
        ]
    )


def _artifact_node_entity(
    workflow_id: str,
    note: Mapping[str, Any],
    *,
    run_id: str,
    conversation_id: str,
    status: str,
) -> Node:
    note_id = str(note.get("id") or "")
    return Node(
        id=f"artifact|{workflow_id}|{note_id}",
        label=str(note.get("title") or note_id or "note"),
        type="entity",
        summary=str(note.get("text") or ""),
        doc_id=_source_doc_id(note_id),
        mentions=[_grounding_for_note(note, insertion_method="demo-artifact-grounding")],
        properties={"topic": str(note.get("topic") or "general")},
        metadata={
            "entity_type": "artifact_note",
            "workflow_id": workflow_id,
            "note_id": note_id,
            "title": str(note.get("title") or ""),
            "text": str(note.get("text") or ""),
            "topic": str(note.get("topic") or "general"),
            "topic_judgment": _jsonable_copy(note.get("topic_judgment") or {}),
            "source_run": str(run_id),
            "conversation_id": str(conversation_id or ""),
            "status": status,
        },
        domain_id=None,
        canonical_entity_id=None,
        level_from_root=0,
        embedding=None,
    )


def _source_document_node(note: Mapping[str, Any], *, workflow_id: str) -> DemoNode:
    note_id = str(note.get("id") or "").strip()
    content = _note_source_content(note)
    return DemoNode(
        id=_source_doc_id(note_id),
        label=f"source:{note_id}",
        summary=content,
        metadata={
            "entity_type": "source_document",
            "workflow_id": workflow_id,
            "note_id": note_id,
            "content": content,
        },
    )


@dataclass(frozen=True)
class DemoNode:
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
class DemoEdge:
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
    doc_id: str = "graph_native_artifact_demo"
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
    """Minimal in-memory store used for the demo workflow graph and traces."""

    def __init__(self) -> None:
        self._nodes: list[Any] = []
        self._edges: list[Any] = []
        self.read = self
        self.write = self

    def add_node(self, node: Any) -> Any:
        node_id = self._entity_id(node)
        for idx, existing in enumerate(self._nodes):
            if self._entity_id(existing) == node_id:
                self._nodes[idx] = node
                return node
        self._nodes.append(node)
        return node

    def add_edge(self, edge: Any) -> Any:
        edge_id = self._entity_id(edge)
        for idx, existing in enumerate(self._edges):
            if self._entity_id(existing) == edge_id:
                self._edges[idx] = edge
                return edge
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
) -> DemoNode:
    return DemoNode(
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
) -> DemoEdge:
    return DemoEdge(
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


def _conversation_turn_node(
    conversation_id: str,
    node_id: str,
    *,
    role: str,
    turn_index: int,
    text: str,
) -> DemoNode:
    return DemoNode(
        id=node_id,
        label=f"{role}:{turn_index}",
        summary=text,
        metadata={
            "entity_type": "conversation_node",
            "conversation_id": conversation_id,
            "role": role,
            "turn_index": turn_index,
            "level_from_root": 0,
        },
    )


def _conversation_turn_edge(
    conversation_id: str,
    edge_id: str,
    *,
    src: str,
    dst: str,
    relation: str = "next_turn",
) -> DemoEdge:
    return DemoEdge(
        id=edge_id,
        source_ids=[src],
        target_ids=[dst],
        label=f"{src}->{dst}",
        summary=f"{src} -> {dst}",
        metadata={
            "entity_type": "conversation_edge",
            "conversation_id": conversation_id,
            "relation": relation,
            "causal_type": "chain" if relation == "next_turn" else "reference",
        },
        relation=relation,
    )


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
        "graph_snapshot": final_state.get("graph_snapshot", {"nodes": [], "edges": []}),
        "provenance_log": final_state.get("provenance_log", []),
        "execution_history": final_state.get("execution_history", []),
        "runtime_step_ops": runtime_step_ops,
        "trace_counts": counts,
    }


def _query_entities(
    store: DemoGraphStore,
    *,
    where: dict[str, Any] | None = None,
    entity_type: str | None = None,
) -> list[dict[str, Any]]:
    criteria = dict(where or {})
    if entity_type is not None:
        criteria["entity_type"] = entity_type
    return [_entity_to_dict(entity) for entity in store.get_nodes(where=criteria, limit=1000)]


def _query_edges(
    store: DemoGraphStore,
    *,
    where: dict[str, Any] | None = None,
    entity_type: str | None = None,
) -> list[dict[str, Any]]:
    criteria = dict(where or {})
    if entity_type is not None:
        criteria["entity_type"] = entity_type
    return [_entity_to_dict(edge) for edge in store.get_edges(where=criteria, limit=1000)]


@dataclass
class GraphArtifactPipelineFramework:
    """Reusable artifact pipeline: ingest, validate, normalize, link, commit."""

    workflow_id: str = "graph_native_artifact_demo"

    @property
    def step_order(self) -> list[str]:
        return ["ingest", "validate", "normalize", "link", "commit", "end"]

    def build_workflow(
        self, graph_store: DemoGraphStore
    ) -> dict[str, dict[str, str]]:
        nodes = [
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|ingest", op="ingest", start=True),
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|validate", op="validate"),
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|normalize", op="normalize"),
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|link", op="link"),
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|commit", op="commit"),
            _wf_node(self.workflow_id, f"wf|{self.workflow_id}|end", op="end", terminal=True),
        ]
        for node in nodes:
            graph_store.add_node(node)

        edges = [
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|ingest->validate",
                src=nodes[0].safe_get_id(),
                dst=nodes[1].safe_get_id(),
                predicate=None,
                default=True,
            ),
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|validate->normalize",
                src=nodes[1].safe_get_id(),
                dst=nodes[2].safe_get_id(),
                predicate="valid",
                priority=1,
            ),
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|validate->end",
                src=nodes[1].safe_get_id(),
                dst=nodes[5].safe_get_id(),
                predicate=None,
                default=True,
            ),
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|normalize->link",
                src=nodes[2].safe_get_id(),
                dst=nodes[3].safe_get_id(),
                predicate=None,
                default=True,
            ),
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|link->commit",
                src=nodes[3].safe_get_id(),
                dst=nodes[4].safe_get_id(),
                predicate=None,
                default=True,
            ),
            _wf_edge(
                self.workflow_id,
                f"wf|{self.workflow_id}|e|commit->end",
                src=nodes[4].safe_get_id(),
                dst=nodes[5].safe_get_id(),
                predicate=None,
                default=True,
            ),
        ]
        for edge in edges:
            graph_store.add_edge(edge)

        transition_map = {
            "ingest": {"default": "validate"},
            "validate": {"valid": "normalize", "default": "end"},
            "normalize": {"default": "link"},
            "link": {"default": "commit"},
            "commit": {"default": "end"},
            "end": {},
        }
        return transition_map

    def build_resolver(self, agent: "NoteGraphCuratorAgent") -> MappingStepResolver:
        resolver = MappingStepResolver()
        resolver.set_state_schema(
            {
                "execution_history": "a",
                "provenance_log": "a",
            }
        )

        @resolver.register("ingest")
        def _ingest(ctx: StepContext):
            raw_notes = [dict(note) for note in ctx.state_view.get("raw_notes") or []]
            deps = dict(ctx.state_view.get("_deps") or {})
            graph_store = deps.get("graph_store")
            if callable(getattr(graph_store, "add_node", None)):
                for note in raw_notes:
                    graph_store.add_node(
                        _source_document_node(note, workflow_id=self.workflow_id)
                    )
                    graph_store.add_node(
                        DemoNode(
                            id=str(note.get("id") or ""),
                            label=str(note.get("title") or note.get("id") or "note"),
                            summary=str(note.get("text") or ""),
                            metadata={
                                "entity_type": "artifact_note_input",
                                "workflow_id": self.workflow_id,
                                "note_id": str(note.get("id") or ""),
                                "raw_title": str(note.get("title") or ""),
                                "raw_text": str(note.get("text") or ""),
                            },
                        )
                    )
            event = _history_event("ingest", raw_count=len(raw_notes))
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    ("u", {"ingested_notes": raw_notes}),
                    ("a", {"provenance_log": _history_event("ingest", note_ids=[n.get("id") for n in raw_notes])}),
                    ("a", {"execution_history": event}),
                ],
            )

        @resolver.register("validate")
        def _validate(ctx: StepContext):
            notes = list(ctx.state_view.get("ingested_notes") or [])
            errors: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for note in notes:
                note_id = str(note.get("id") or "").strip()
                title = str(note.get("title") or "").strip()
                text = str(note.get("text") or "").strip()
                if not note_id:
                    errors.append({"id": None, "error": "missing id"})
                if not title:
                    errors.append({"id": note_id or None, "error": "missing title"})
                if not text:
                    errors.append({"id": note_id or None, "error": "missing text"})
                if note_id:
                    if note_id in seen_ids:
                        errors.append({"id": note_id, "error": "duplicate id"})
                    seen_ids.add(note_id)

            valid = len(errors) == 0
            event = _history_event("validate", valid=valid, error_count=len(errors))
            state_update: list[tuple[str, dict[str, Any]]] = [
                (
                    "u",
                    {
                        "validation_passed": valid,
                        "validation_errors": errors,
                        "blocked": not valid,
                        "blocked_reason": None if valid else "validation failed",
                        "final_status": "running" if valid else "blocked",
                    },
                ),
                ("a", {"provenance_log": _history_event("validate", valid=valid, error_count=len(errors))}),
                ("a", {"execution_history": event}),
            ]
            return RunSuccess(conversation_node_id=None, state_update=state_update)

        @resolver.register("normalize")
        def _normalize(ctx: StepContext):
            deps = dict(ctx.state_view.get("_deps") or {})
            infer_topic = deps.get("infer_topic")
            infer_topic_judgment = deps.get("infer_topic_judgment")
            notes = list(ctx.state_view.get("ingested_notes") or [])
            graph_store = deps.get("graph_store")
            normalized: list[dict[str, Any]] = []
            for note in notes:
                note_id = str(note.get("id") or "").strip()
                title = _normalize_text(note.get("title"))
                text = _normalize_text(note.get("text"))
                topic = (
                    str(infer_topic(note, ctx.state_view))
                    if callable(infer_topic)
                    else _infer_topic(note)
                )
                topic_judgment = (
                    dict(infer_topic_judgment(note, ctx.state_view))
                    if callable(infer_topic_judgment)
                    else _topic_judgment(note)
                )
                topic_judgment["topic"] = topic
                normalized.append(
                    {
                        "id": note_id,
                        "title": title,
                        "title_slug": _slug(title),
                        "text": text,
                        "topic": topic,
                        "topic_judgment": topic_judgment,
                    }
                )
                if callable(getattr(graph_store, "add_node", None)):
                    graph_store.add_node(
                        _artifact_node_entity(
                            self.workflow_id,
                            normalized[-1],
                            run_id=str(ctx.run_id),
                            conversation_id=str(ctx.conversation_id or ""),
                            status="normalized",
                        )
                    )
            event = _history_event("normalize", note_count=len(normalized))
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    ("u", {"normalized_notes": normalized}),
                    ("a", {"provenance_log": _history_event("normalize", note_ids=[n["id"] for n in normalized])}),
                    ("a", {"execution_history": event}),
                ],
            )

        @resolver.register("link")
        def _link(ctx: StepContext):
            deps = dict(ctx.state_view.get("_deps") or {})
            link_notes = deps.get("link_notes")
            normalized = list(ctx.state_view.get("normalized_notes") or [])
            graph_store = deps.get("graph_store")
            edges = (
                list(link_notes(normalized, ctx.state_view))
                if callable(link_notes)
                else []
            )
            event = _history_event("link", edge_count=len(edges))
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    ("u", {"draft_edges": edges}),
                    ("a", {"provenance_log": _history_event("link", edge_count=len(edges))}),
                    ("a", {"execution_history": event}),
                ],
            )

        @resolver.register("commit")
        def _commit(ctx: StepContext):
            nodes = list(ctx.state_view.get("normalized_notes") or [])
            edges = list(ctx.state_view.get("draft_edges") or [])
            deps = dict(ctx.state_view.get("_deps") or {})
            graph_store = deps.get("graph_store")
            if callable(getattr(graph_store, "add_node", None)):
                for node in nodes:
                    graph_store.add_node(
                        _artifact_node_entity(
                            self.workflow_id,
                            node,
                            run_id=str(ctx.run_id),
                            conversation_id=str(ctx.conversation_id or ""),
                            status="committed",
                        )
                    )
            if callable(getattr(graph_store, "add_edge", None)):
                for edge in edges:
                    source = str(edge.get("source") or "")
                    target = str(edge.get("target") or "")
                    topic = str(edge.get("topic") or "")
                    graph_store.add_edge(
                        DemoEdge(
                            id=f"artifact-commit-edge|{self.workflow_id}|{source}|{target}",
                            source_ids=[source],
                            target_ids=[target],
                            label=f"{source}->{target}",
                            summary=f"committed same_topic:{topic}",
                            metadata={
                                "entity_type": "artifact_edge",
                                "workflow_id": self.workflow_id,
                                "relation": str(edge.get("relation") or "same_topic"),
                                "topic": topic,
                                "source_run": str(ctx.run_id),
                                "conversation_id": str(ctx.conversation_id or ""),
                                "status": "committed",
                            },
                            relation=str(edge.get("relation") or "same_topic"),
                        )
                    )
            snapshot = {"nodes": nodes, "edges": edges}
            event = _history_event("commit", node_count=len(nodes), edge_count=len(edges))
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    (
                        "u",
                        {
                            "graph_snapshot": snapshot,
                            "committed_nodes": nodes,
                            "committed_edges": edges,
                            "completed": True,
                            "final_status": "committed",
                        },
                    ),
                    ("a", {"provenance_log": _history_event("commit", node_count=len(nodes), edge_count=len(edges))}),
                    ("a", {"execution_history": event}),
                ],
            )

        @resolver.register("end")
        def _end(ctx: StepContext):
            final_status = str(ctx.state_view.get("final_status") or "blocked")
            event = _history_event("end", final_status=final_status)
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    ("u", {"completed": final_status == "committed"}),
                    ("a", {"execution_history": event}),
                ],
            )

        return resolver

    def build_runtime(
        self,
        agent: "NoteGraphCuratorAgent",
        *,
        graph_store: DemoGraphStore,
        conversation_id: str,
        turn_node_id: str,
    ) -> WorkflowRuntime:
        self.build_workflow(graph_store)
        resolver = self.build_resolver(agent)

        runtime = WorkflowRuntime(
            workflow_engine=graph_store,
            conversation_engine=graph_store,
            step_resolver=resolver,
            predicate_registry={
                "valid": lambda _e, state, _r: bool(state.get("validation_passed")),
            },
            checkpoint_every_n_steps=1,
            max_workers=1,
            transaction_mode="none",
            trace=True,
        )
        return runtime

    def transition_summary(self) -> dict[str, dict[str, str]]:
        store = DemoGraphStore()
        return self.build_workflow(store)

    def run(
        self,
        agent: "NoteGraphCuratorAgent",
        *,
        graph_store: DemoGraphStore | None = None,
        conversation_id: str = "demo-conversation",
        turn_node_id: str = "demo-turn",
        initial_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        graph_store = graph_store or DemoGraphStore()
        runtime = self.build_runtime(
            agent,
            graph_store=graph_store,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
        )
        state = agent.initial_state()
        if initial_state:
            for key, value in initial_state.items():
                if key == "_deps" and isinstance(value, dict):
                    state.setdefault("_deps", {}).update(value)
                else:
                    state[key] = value
        state.setdefault("_deps", {})["graph_store"] = graph_store
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Using advanced underscore state key '_deps'.*",
                category=RuntimeWarning,
            )
            run_result = runtime.run(
                workflow_id=self.workflow_id,
                conversation_id=conversation_id,
                turn_node_id=turn_node_id,
                initial_state=state,
            )
        return _collect_run_result(
            framework_name=self.__class__.__name__,
            step_order=list(self.step_order),
            transition_map=self.transition_summary(),
            run_result=run_result,
            trace_sink=graph_store,
        )


@dataclass
class NoteGraphCuratorAgent:
    """Concrete policy object that plugs note-specific logic into the pipeline."""

    framework: GraphArtifactPipelineFramework = field(
        default_factory=GraphArtifactPipelineFramework
    )
    raw_notes: list[dict[str, str]] = field(
        default_factory=lambda: [
            {
                "id": "note-1",
                "title": "Vendor invoice",
                "text": "Invoice for subscription renewal",
            },
            {
                "id": "note-2",
                "title": "Budget check-in",
                "text": "Quarterly finance review",
            },
            {
                "id": "note-3",
                "title": "Team meeting",
                "text": "Meeting notes from product sync",
            },
        ]
    )

    def infer_topic(self, note: Mapping[str, Any], state: Mapping[str, Any]) -> str:
        _ = state
        return _infer_topic(note)

    def link_notes(
        self, normalized_notes: Sequence[Mapping[str, Any]], state: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        _ = state
        anchors: dict[str, Mapping[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        for note in normalized_notes:
            topic = str(note.get("topic") or "general")
            anchor = anchors.get(topic)
            if anchor is None:
                anchors[topic] = note
                continue
            edges.append(
                {
                    "source": str(anchor.get("id")),
                    "target": str(note.get("id")),
                    "relation": "same_topic",
                    "topic": topic,
                }
            )
        return edges

    def tool_deps(self) -> dict[str, Any]:
        return {
            "infer_topic": self.infer_topic,
            "infer_topic_judgment": self.infer_topic_judgment,
            "link_notes": self.link_notes,
        }

    def infer_topic_judgment(
        self, note: Mapping[str, Any], state: Mapping[str, Any]
    ) -> dict[str, Any]:
        _ = state
        return _topic_judgment(note)

    def initial_state(self) -> dict[str, Any]:
        return {
            "_deps": self.tool_deps(),
            "raw_notes": [dict(note) for note in self.raw_notes],
            "ingested_notes": [],
            "validation_passed": False,
            "validation_errors": [],
            "normalized_notes": [],
            "draft_edges": [],
            "graph_snapshot": {"nodes": [], "edges": []},
            "committed_nodes": [],
            "committed_edges": [],
            "blocked": False,
            "blocked_reason": None,
            "completed": False,
            "final_status": "running",
            "provenance_log": [],
            "execution_history": [],
        }

    def run(
        self,
        framework: Any | None = None,
        *,
        graph_store: DemoGraphStore | None = None,
        conversation_id: str = "demo-conversation",
        turn_node_id: str = "demo-turn",
        initial_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        framework = framework or self.framework
        return framework.run(
            self,
            graph_store=graph_store,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            initial_state=initial_state,
        )


def run_graph_native_artifact_demo(
    raw_notes: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    agent = NoteGraphCuratorAgent()
    if raw_notes is not None:
        agent.raw_notes = [dict(note) for note in raw_notes]
    return agent.run()


def _result_json_dict(step_node: Any) -> dict[str, Any]:
    metadata = dict(getattr(step_node, "metadata", {}) or {})
    raw = metadata.get("result_json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            return {"value": raw}
    return {}


def _step_results(store: DemoGraphStore, run_id: str) -> list[dict[str, Any]]:
    nodes = store.get_nodes(
        where={"$and": [{"entity_type": "workflow_step_exec"}, {"run_id": run_id}]},
        limit=1000,
    )
    nodes = sorted(
        nodes,
        key=lambda node: int((getattr(node, "metadata", {}) or {}).get("step_seq", 0)),
    )
    return [_entity_to_dict(node) | {"result_json": _result_json_dict(node)} for node in nodes]


def _artifact_nodes(store: DemoGraphStore, *, run_id: str | None = None) -> list[dict[str, Any]]:
    where: dict[str, Any] = {"entity_type": "artifact_note"}
    if run_id is not None:
        where["source_run"] = run_id
    return _query_entities(store, where=where)


def _artifact_edges(store: DemoGraphStore, *, run_id: str | None = None) -> list[dict[str, Any]]:
    where: dict[str, Any] = {"entity_type": "artifact_edge"}
    if run_id is not None:
        where["source_run"] = run_id
    return _query_edges(store, where=where)


def _source_documents(store: DemoGraphStore) -> list[dict[str, Any]]:
    return _query_entities(store, where={"entity_type": "source_document"})


def _conversation_graph(store: DemoGraphStore, conversation_id: str) -> dict[str, Any]:
    conv_nodes = _query_entities(
        store,
        where={"conversation_id": conversation_id},
        entity_type="conversation_node",
    )
    conv_edges = _query_edges(
        store,
        where={"conversation_id": conversation_id},
        entity_type="conversation_edge",
    )
    workflow_runs = _query_entities(
        store,
        where={"conversation_id": conversation_id},
        entity_type="workflow_run",
    )
    run_ids = [str(node["metadata"].get("run_id", "")) for node in workflow_runs]
    workflow_steps = [
        step
        for run_id in run_ids
        for step in _step_results(store, run_id)
    ]
    artifacts = [
        artifact
        for run_id in run_ids
        for artifact in _artifact_nodes(store, run_id=run_id)
    ]
    return {
        "conversation_nodes": conv_nodes,
        "conversation_edges": conv_edges,
        "workflow_runs": workflow_runs,
        "workflow_steps": workflow_steps,
        "artifact_nodes": artifacts,
    }


def _memory_ids_from_store(store: DemoGraphStore) -> list[str]:
    note_ids = []
    seen: set[str] = set()
    for node in _artifact_nodes(store):
        note_id = str(node["metadata"].get("note_id") or "")
        if note_id and note_id not in seen:
            seen.add(note_id)
            note_ids.append(note_id)
    return note_ids


def _count_entities(store: DemoGraphStore, *, conversation_id: str) -> dict[str, int]:
    return {
        "conversation_nodes": len(
            _query_entities(
                store,
                where={"conversation_id": conversation_id},
                entity_type="conversation_node",
            )
        ),
        "workflow_run_nodes": len(
            _query_entities(
                store,
                where={"conversation_id": conversation_id},
                entity_type="workflow_run",
            )
        ),
        "workflow_step_nodes": len(
            _query_entities(
                store,
                where={"conversation_id": conversation_id},
                entity_type="workflow_step_exec",
            )
        ),
        "artifact_nodes": len(
            _query_entities(
                store,
                where={"conversation_id": conversation_id},
                entity_type="artifact_note",
            )
        ),
    }


def build_demo_summary(
    execution_memory: dict[str, Any],
    conversation_workflow: dict[str, Any],
    provenance_reasoning: dict[str, Any],
) -> dict[str, Any]:
    return {
        "execution_memory": {
            "claim": "Execution history was reused as graph memory on the second run.",
            "first_run_processed": execution_memory["first_run_processed"],
            "second_run_skipped_as_known": execution_memory["second_run_skipped_as_known"],
            "second_run_processed": execution_memory["second_run_processed"],
        },
        "conversation_workflow": {
            "claim": "The conversation turn and workflow run are stored and queried in one graph.",
            "question": conversation_workflow["question"],
            "workflow_run_id": conversation_workflow["workflow_run_id"],
            "linked_entities": conversation_workflow["linked_entities"],
        },
        "provenance_reasoning": {
            "claim": "Stored step history explains why note-1 moved to finance.",
            "question": provenance_reasoning["question"],
            "answer": provenance_reasoning["answer"],
            "evidence_steps": provenance_reasoning["evidence_steps"],
            "grounding_excerpt": provenance_reasoning.get("grounding_excerpt"),
        },
    }


def _run_conversation_workflow_case(
    store: DemoGraphStore,
    *,
    conversation_id: str,
    user_turn_text: str,
    assistant_turn_text: str,
    assistant_turn_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    user_turn = _conversation_turn_node(
        conversation_id,
        f"turn|{conversation_id}|user-1",
        role="user",
        turn_index=1,
        text=user_turn_text,
    )
    assistant_turn = _conversation_turn_node(
        conversation_id,
        assistant_turn_id,
        role="assistant",
        turn_index=2,
        text=assistant_turn_text,
    )
    store.add_node(user_turn)
    store.add_node(assistant_turn)
    store.add_edge(
        _conversation_turn_edge(
            conversation_id,
            f"edge|{conversation_id}|turn-1|turn-2",
            src=user_turn.id,
            dst=assistant_turn.id,
        )
    )

    agent = NoteGraphCuratorAgent()
    result = agent.run(
        graph_store=store,
        conversation_id=conversation_id,
        turn_node_id=assistant_turn.id,
    )
    run_node_id = f"wf_run|{result['run_id']}"
    store.add_edge(
        _conversation_turn_edge(
            conversation_id,
            f"edge|{conversation_id}|turn-2|run-{result['run_id']}",
            src=assistant_turn.id,
            dst=run_node_id,
            relation="run_result",
        )
    )

    graph_view = _conversation_graph(store, conversation_id)
    graph_view.update(
        {
            "question": user_turn_text,
            "workflow_run_id": result["run_id"],
        }
    )
    return graph_view, result


def run_execution_memory_demo() -> dict[str, Any]:
    store = DemoGraphStore()
    first_agent = NoteGraphCuratorAgent()
    first_result = first_agent.run(
        graph_store=store,
        conversation_id="execution-memory",
        turn_node_id="execution-memory-turn-1",
    )

    memory_note_ids = _memory_ids_from_store(store)
    next_notes = [
        {"id": "note-2", "title": "Budget check-in", "text": "Quarterly finance review"},
        {"id": "note-3", "title": "Team meeting", "text": "Meeting notes from product sync"},
        {"id": "note-4", "title": "Invoice follow-up", "text": "Send invoice reminder"},
    ]
    filtered_notes = [note for note in next_notes if note["id"] not in memory_note_ids]

    second_agent = NoteGraphCuratorAgent(raw_notes=filtered_notes)
    second_result = second_agent.run(
        graph_store=store,
        conversation_id="execution-memory",
        turn_node_id="execution-memory-turn-2",
    )
    first_run_processed = [note["id"] for note in first_agent.raw_notes]
    second_run_skipped = [note["id"] for note in next_notes if note["id"] in memory_note_ids]
    second_run_processed = [note["id"] for note in filtered_notes]
    first_run_steps = [step["metadata"]["op"] for step in _step_results(store, first_result["run_id"])]
    second_run_steps = [step["metadata"]["op"] for step in _step_results(store, second_result["run_id"])]
    details = {
        "scenario": "execution_memory",
        "run_ids": {
            "first": first_result["run_id"],
            "second": second_result["run_id"],
        },
        "memory_query": {
            "processed_note_ids": memory_note_ids,
        },
        "runs": {
            "first_processed_note_ids": first_run_processed,
            "second_skipped_note_ids": second_run_skipped,
            "second_processed_note_ids": second_run_processed,
        },
        "step_ops": {
            "first_run": first_run_steps,
            "second_run": second_run_steps,
        },
        "graph_entity_counts": {
            "artifact_nodes": len(_artifact_nodes(store)),
            "workflow_step_nodes": len(
                _query_entities(
                    store,
                    where={"conversation_id": "execution-memory"},
                    entity_type="workflow_step_exec",
                )
            ),
        },
    }
    summary = {
        "claim": "Execution history was reused as graph memory on the second run.",
        "first_run_processed": first_run_processed,
        "second_run_skipped_as_known": second_run_skipped,
        "second_run_processed": second_run_processed,
    }
    return {"summary": summary, "details": details}


def run_conversation_workflow_demo() -> dict[str, Any]:
    store = DemoGraphStore()
    conversation_id = "conversation-workflow"
    graph_view, result = _run_conversation_workflow_case(
        store,
        conversation_id=conversation_id,
        user_turn_text="Organize my notes",
        assistant_turn_text="I will organize the notes with the graph-native pipeline.",
        assistant_turn_id="turn|conversation-workflow|assistant-1",
    )
    linked_entities = _count_entities(store, conversation_id=conversation_id)
    details = {
        "scenario": "conversation_workflow",
        "conversation_id": conversation_id,
        "question": graph_view["question"],
        "workflow_run_id": result["run_id"],
        "conversation_node_ids": [node["id"] for node in graph_view["conversation_nodes"]],
        "conversation_edge_ids": [edge["id"] for edge in graph_view["conversation_edges"]],
        "workflow_run_node_ids": [node["id"] for node in graph_view["workflow_runs"]],
        "workflow_step_ops": [step["metadata"]["op"] for step in graph_view["workflow_steps"]],
        "artifact_node_ids": [node["id"] for node in graph_view["artifact_nodes"]],
        "linked_entities": linked_entities,
    }
    summary = {
        "claim": "The conversation turn and workflow run are stored and queried in one graph.",
        "question": graph_view["question"],
        "workflow_run_id": result["run_id"],
        "linked_entities": linked_entities,
    }
    return {"summary": summary, "details": details}


def _explain_note_move(store: DemoGraphStore, *, note_id: str) -> dict[str, Any]:
    artifact_nodes = [node for node in _artifact_nodes(store) if node["metadata"].get("note_id") == note_id]
    if not artifact_nodes:
        return {
            "note_id": note_id,
            "explanation": f"No stored artifact node found for {note_id}.",
            "evidence": [],
        }

    artifact = artifact_nodes[-1]
    run_id = str(artifact["metadata"].get("source_run") or "")
    steps = _step_results(store, run_id) if run_id else []
    normalize_step = next((step for step in steps if step["metadata"].get("op") == "normalize"), None)
    link_step = next((step for step in steps if step["metadata"].get("op") == "link"), None)
    commit_step = next((step for step in steps if step["metadata"].get("op") == "commit"), None)

    normalized_note = None
    if normalize_step:
        for update_kind, update_payload in normalize_step.get("result_json", {}).get("state_update", []):
            if isinstance(update_payload, dict) and "normalized_notes" in update_payload:
                normalized_note = next(
                    (item for item in update_payload["normalized_notes"] if item.get("id") == note_id),
                    None,
                )
                break

    topic = str(artifact["metadata"].get("topic") or "general")
    title = str(artifact["metadata"].get("title") or note_id)
    grounding = None
    span = None
    mentions = artifact.get("mentions") or []
    if mentions:
        grounding = mentions[0]
        spans = grounding.get("spans") or []
        if spans:
            span = spans[0]
    source_document = None
    if span is not None:
        source_document = next(
            (
                doc
                for doc in _source_documents(store)
                if doc["id"] == str(span.get("doc_id") or "")
            ),
            None,
        )

    evidence = [
        (
            f"grounding span excerpt={span['excerpt']!r} at chars "
            f"{span['start_char']}:{span['end_char']} in {span['doc_id']}"
            if span is not None
            else f"artifact note {note_id} has no stored grounding span"
        ),
        f"normalize step stored topic={topic!r} for {note_id}",
        f"commit step stored artifact note {note_id} in the graph",
    ]
    if link_step:
        evidence.append("link step stored same-topic edges for the run")

    evidence_steps: list[str] = []
    if normalize_step is not None:
        evidence_steps.append("normalize")
    if link_step is not None:
        evidence_steps.append("link")
    if commit_step is not None:
        evidence_steps.append("commit")

    response = _provenance_response(
        note_id=note_id,
        topic=topic,
        title=title,
        span=span,
        normalized_note=normalized_note,
        evidence_steps=evidence_steps,
    )

    return {
        "note_id": note_id,
        "run_id": run_id,
        "explanation": response["answer"],
        "evidence": evidence,
        "evidence_steps": evidence_steps,
        "artifact_node": artifact,
        "grounding": grounding,
        "span": span,
        "source_document": source_document,
        "response": response,
        "normalize_step": normalize_step,
        "link_step": link_step,
        "commit_step": commit_step,
    }


def run_provenance_reasoning_demo() -> dict[str, Any]:
    store = DemoGraphStore()
    graph_view, workflow_result = _run_conversation_workflow_case(
        store,
        conversation_id="provenance-reasoning",
        user_turn_text="Organize my notes",
        assistant_turn_text="Explain why note-1 should move to finance.",
        assistant_turn_id="turn|provenance-reasoning|assistant-1",
    )
    explanation = _explain_note_move(store, note_id="note-1")
    summary = {
        "claim": "Stored step history explains why note-1 moved to finance.",
        "question": "Why did we move note-1 to finance?",
        "answer": explanation["explanation"],
        "evidence_steps": explanation["evidence_steps"],
        "grounding_excerpt": (
            explanation["span"]["excerpt"] if explanation.get("span") else None
        ),
    }
    details = {
        "scenario": "provenance_reasoning",
        "conversation_id": "provenance-reasoning",
        "workflow_run_id": workflow_result["run_id"],
        "conversation_node_ids": [node["id"] for node in graph_view["conversation_nodes"]],
        "workflow_step_ids": [step["id"] for step in graph_view["workflow_steps"]],
        "artifact_node_id": explanation["artifact_node"]["id"],
        "grounding_trace": {
            "source_document_id": (
                explanation["source_document"]["id"]
                if explanation.get("source_document")
                else None
            ),
            "span": explanation.get("span"),
        },
        "llm_style_response": explanation["response"],
        "answer": {
            "explanation": explanation["explanation"],
            "evidence": explanation["evidence"],
            "run_id": explanation["run_id"],
        },
    }
    return {"summary": summary, "details": details}


def run_unified_substrate_demo_suite() -> dict[str, Any]:
    execution_memory = run_execution_memory_demo()
    conversation_workflow = run_conversation_workflow_demo()
    provenance_reasoning = run_provenance_reasoning_demo()
    return {
        "summary": build_demo_summary(
            execution_memory["summary"],
            conversation_workflow["summary"],
            provenance_reasoning["summary"],
        ),
        "details": {
            "execution_memory": execution_memory["details"],
            "conversation_workflow": conversation_workflow["details"],
            "provenance_reasoning": provenance_reasoning["details"],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the graph-native artifact pipeline demo."
    )
    parser.add_argument(
        "--scenario",
        choices=["suite", "execution-memory", "conversation-workflow", "provenance-reasoning", "artifact"],
        default="suite",
        help="Which demo scenario to run.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the short summary section.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full summary-plus-details report.",
    )
    parser.add_argument(
        "--notes-json",
        help="Optional JSON array of raw notes to process.",
        default=None,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    raw_notes = json.loads(args.notes_json) if args.notes_json else None
    if args.scenario == "artifact":
        result = run_graph_native_artifact_demo(raw_notes=raw_notes)
    elif args.scenario == "execution-memory":
        result = run_execution_memory_demo()
    elif args.scenario == "conversation-workflow":
        result = run_conversation_workflow_demo()
    elif args.scenario == "provenance-reasoning":
        result = run_provenance_reasoning_demo()
    else:
        result = run_unified_substrate_demo_suite()
    payload = result["summary"] if args.summary_only else result
    if args.verbose and not args.summary_only:
        payload = result
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
