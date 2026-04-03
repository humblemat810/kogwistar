from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence

from pydantic import BaseModel, Field
from pydantic_extension.model_slicing import ModeSlicingMixin
from pydantic_extension.model_slicing.mixin import ExcludeMode

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, Node, Span
from kogwistar.runtime import MappingStepResolver, WorkflowRuntime
from kogwistar.runtime.design import load_workflow_design
from kogwistar.runtime.models import (
    RunFailure,
    RunSuccess,
    WorkflowEdge,
    WorkflowNode,
)
from kogwistar.runtime.replay import replay_to


PUBLIC_MODE = "public"


class ArtifactMetadata(ModeSlicingMixin, BaseModel):
    build_id: str
    release_channel: str
    commit_sha: str
    source_root: Annotated[str | None, ExcludeMode(PUBLIC_MODE)] = None
    source_map_manifest: Annotated[list[str], ExcludeMode(PUBLIC_MODE)] = Field(
        default_factory=list
    )
    internal_notes: Annotated[str | None, ExcludeMode(PUBLIC_MODE)] = None


class BuildArtifact(ModeSlicingMixin, BaseModel):
    artifact_id: str
    mode: Literal["internal", "public"] = "internal"
    files: list[str]
    metadata: ArtifactMetadata
    source_maps: Annotated[list[str], ExcludeMode(PUBLIC_MODE)] = Field(
        default_factory=list
    )
    raw_sources: Annotated[list[str], ExcludeMode(PUBLIC_MODE)] = Field(
        default_factory=list
    )


ArtifactMetadata.register_mode(PUBLIC_MODE)
ArtifactMetadata.include_unmarked_for_modes = (
    set(ArtifactMetadata.include_unmarked_for_modes) | {PUBLIC_MODE}
)
BuildArtifact.register_mode(PUBLIC_MODE)
BuildArtifact.include_unmarked_for_modes = (
    set(BuildArtifact.include_unmarked_for_modes) | {PUBLIC_MODE}
)


PREDEFINED_ARTIFACT_BLUEPRINT: dict[str, Any] = {
    "artifact_id": "artifact-demo-001",
    "mode": "internal",
    "files": ["dist/app.js", "dist/app.css"],
    "metadata": {
        "build_id": "build-2026-04-03",
        "release_channel": "stable",
        "commit_sha": "abc1234",
        "source_root": "/src",
        "source_map_manifest": ["dist/app.js.map"],
        "internal_notes": "Internal debug traces retained before the public boundary.",
    },
    "source_maps": ["dist/app.js.map"],
    "raw_sources": ["/src/index.ts"],
}


class _DemoEmbeddingFunction:
    _name = "artifact-governance-demo-embedding-v1"

    def name(self):
        return self._name

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in input:
            value = str(text or "")
            length = float((len(value) % 19) + 1)
            checksum = float((sum(ord(ch) for ch in value) % 23) + 1)
            vectors.append([length, checksum])
        return vectors


def _grounding(doc_id: str) -> Grounding:
    return Grounding(spans=[Span.from_dummy_for_conversation(doc_id)])


def _workflow_node(
    *,
    workflow_id: str,
    node_id: str,
    op: str,
    start: bool = False,
    terminal: bool = False,
) -> WorkflowNode:
    return WorkflowNode(
        id=node_id,
        label=node_id.split("|")[-1],
        type="entity",
        doc_id=node_id,
        summary=op,
        properties={},
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": op,
            "wf_start": bool(start),
            "wf_terminal": bool(terminal),
            "wf_version": "v_demo",
        },
        mentions=[_grounding(f"wf:{workflow_id}")],
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )


def _workflow_edge(
    *,
    workflow_id: str,
    edge_id: str,
    src: str,
    dst: str,
    is_default: bool = True,
) -> WorkflowEdge:
    return WorkflowEdge(
        id=edge_id,
        label="wf_next",
        type="relationship",
        doc_id=edge_id,
        summary="next",
        properties={},
        source_ids=[src],
        target_ids=[dst],
        source_edge_ids=[],
        target_edge_ids=[],
        relation="wf_next",
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_priority": 100,
            "wf_is_default": bool(is_default),
            "wf_multiplicity": "one",
            "wf_version": "v_demo",
        },
        mentions=[_grounding(f"wf:{workflow_id}")],
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )


def _persist_design(
    workflow_engine: GraphKnowledgeEngine,
    *,
    workflow_id: str,
    nodes: list[WorkflowNode],
    edges: list[WorkflowEdge],
) -> None:
    _ = workflow_id
    for node in nodes:
        workflow_engine.write.add_node(node)
    for edge in edges:
        workflow_engine.write.add_edge(edge)


def _build_engines(
    *,
    data_dir: Path,
    backend_factory: Any | None = None,
) -> tuple[GraphKnowledgeEngine, GraphKnowledgeEngine]:
    embedding = _DemoEmbeddingFunction()
    kwargs: dict[str, Any] = {"embedding_function": embedding}
    if backend_factory is not None:
        kwargs["backend_factory"] = backend_factory
    workflow_engine = GraphKnowledgeEngine(
        persist_directory=str(data_dir / "workflow"),
        kg_graph_type="workflow",
        **kwargs,
    )
    conversation_engine = GraphKnowledgeEngine(
        persist_directory=str(data_dir / "conversation"),
        kg_graph_type="conversation",
        **kwargs,
    )
    return workflow_engine, conversation_engine


def _collect_string_leaks(value: Any, *, prefix: str = "") -> list[str]:
    leaks: list[str] = []
    if isinstance(value, dict):
        for key, inner in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            leaks.extend(_collect_string_leaks(inner, prefix=path))
        return leaks
    if isinstance(value, list):
        for index, inner in enumerate(value):
            path = f"{prefix}[{index}]"
            leaks.extend(_collect_string_leaks(inner, prefix=path))
        return leaks
    if isinstance(value, str):
        normalized = value.lower()
        if normalized.endswith(".map"):
            leaks.append(f"{prefix}: contains source map path {value!r}")
        if "/src" in normalized or "\\src" in normalized:
            leaks.append(f"{prefix}: contains source root path {value!r}")
    return leaks


def _public_artifact_violations(payload: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    mode = str(payload.get("mode") or "")
    if mode != PUBLIC_MODE:
        violations.append(f"artifact.mode must be {PUBLIC_MODE!r}, found {mode!r}")

    for field_name in ("source_maps", "raw_sources"):
        if field_name in payload:
            violations.append(f"public artifact leaked top-level field {field_name!r}")

    metadata = payload.get("metadata") or {}
    if isinstance(metadata, dict):
        for field_name in (
            "source_root",
            "source_map_manifest",
            "internal_notes",
        ):
            if field_name in metadata:
                violations.append(f"public artifact leaked metadata field {field_name!r}")

    violations.extend(_collect_string_leaks(payload))
    return violations


def _assert_public_artifact_clean(payload: dict[str, Any]) -> None:
    violations = _public_artifact_violations(payload)
    if violations:
        raise ValueError("; ".join(violations))


def _sanitize_public_artifact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = json.loads(json.dumps(payload))
    sanitized["mode"] = PUBLIC_MODE
    sanitized.pop("source_maps", None)
    sanitized.pop("raw_sources", None)
    metadata = sanitized.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("source_root", None)
        metadata.pop("source_map_manifest", None)
        metadata.pop("internal_notes", None)
    return sanitized


def _dump_backend(model: ModeSlicingMixin) -> dict[str, Any]:
    return json.loads(
        json.dumps(model.model_dump(field_mode="backend", dump_format="python"))
    )


def _dump_public(model: ModeSlicingMixin) -> dict[str, Any]:
    return json.loads(
        json.dumps(model.model_dump(field_mode=PUBLIC_MODE, dump_format="python"))
    )


def _project_public_artifact(artifact: BuildArtifact) -> tuple[dict[str, Any], str]:
    base_payload = _dump_backend(artifact)
    candidates: list[tuple[str, dict[str, Any]]] = []

    try:
        sliced_cls = BuildArtifact[PUBLIC_MODE]
        payload = _dump_public(artifact)
        sliced_cls.model_validate(payload)
        if isinstance(payload, dict):
            candidates.append(("BuildArtifact['public']", payload))
    except Exception:
        pass

    for strategy, payload in candidates:
        payload["mode"] = PUBLIC_MODE
        try:
            _assert_public_artifact_clean(payload)
            return payload, strategy
        except Exception:
            sanitized = _sanitize_public_artifact_payload(payload)
            _assert_public_artifact_clean(sanitized)
            return sanitized, strategy

    fallback_payload = _sanitize_public_artifact_payload(
        base_payload
    )
    _assert_public_artifact_clean(fallback_payload)
    return fallback_payload, "BuildArtifact['public']"

    raise RuntimeError("Public mode projection failed unexpectedly.")


def _classify_sensitive_components(artifact: BuildArtifact) -> list[dict[str, Any]]:
    classification: list[dict[str, Any]] = []
    if artifact.source_maps:
        classification.append(
            {
                "path": "source_maps",
                "kind": "sensitive_field",
                "reason": "Source maps reveal build internals and symbol mappings.",
                "values": list(artifact.source_maps),
            }
        )
    if artifact.raw_sources:
        classification.append(
            {
                "path": "raw_sources",
                "kind": "sensitive_field",
                "reason": "Raw sources cross the public boundary only after filtering.",
                "values": list(artifact.raw_sources),
            }
        )
    if artifact.metadata.source_root:
        classification.append(
            {
                "path": "metadata.source_root",
                "kind": "sensitive_metadata",
                "reason": "Internal source roots expose repository layout.",
                "values": [artifact.metadata.source_root],
            }
        )
    if artifact.metadata.source_map_manifest:
        classification.append(
            {
                "path": "metadata.source_map_manifest",
                "kind": "sensitive_metadata",
                "reason": "Source map manifests advertise private debug artifacts.",
                "values": list(artifact.metadata.source_map_manifest),
            }
        )
    if artifact.metadata.internal_notes:
        classification.append(
            {
                "path": "metadata.internal_notes",
                "kind": "sensitive_metadata",
                "reason": "Internal operator notes must not cross the boundary.",
                "values": [artifact.metadata.internal_notes],
            }
        )
    return classification


def _diff_payload(before: Any, after: Any, *, prefix: str = "") -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    if isinstance(before, dict) and isinstance(after, dict):
        keys = sorted(set(before.keys()) | set(after.keys()))
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in after:
                diffs.append(
                    {
                        "path": path,
                        "change": "removed",
                        "before": before.get(key),
                        "after": None,
                    }
                )
                continue
            if key not in before:
                diffs.append(
                    {
                        "path": path,
                        "change": "added",
                        "before": None,
                        "after": after.get(key),
                    }
                )
                continue
            diffs.extend(_diff_payload(before[key], after[key], prefix=path))
        return diffs

    if before != after:
        diffs.append(
            {"path": prefix or "$", "change": "changed", "before": before, "after": after}
        )
    return diffs


def _artifact_payload_for_event(
    *,
    state: dict[str, Any],
    event_type: str,
) -> dict[str, Any]:
    if event_type in {
        "artifact_filtered",
        "artifact_validated",
        "artifact_published",
    } and isinstance(state.get("public_artifact"), dict):
        return dict(state.get("public_artifact") or {})
    if event_type == "artifact_rejected" and isinstance(state.get("public_artifact"), dict):
        return dict(state.get("public_artifact") or {})
    return dict(state.get("artifact_internal") or {})


def _emit_governance_event(
    conversation_engine: GraphKnowledgeEngine,
    *,
    run_id: str,
    workflow_id: str,
    conversation_id: str | None,
    step_seq: int,
    event_type: str,
    artifact_payload: dict[str, Any],
    filter_diff: list[dict[str, Any]],
    projection_strategy: str | None = None,
) -> str:
    artifact_id = str(artifact_payload.get("artifact_id") or "unknown-artifact")
    mode = str(artifact_payload.get("mode") or "internal")
    event_id = f"artifact_event|{run_id}|{step_seq}|{event_type}"
    node = Node(
        id=event_id,
        label=event_type,
        type="entity",
        doc_id=f"artifact-governance:{run_id}",
        summary=f"{event_type} artifact={artifact_id} mode={mode}",
        properties={},
        metadata={
            "entity_type": "artifact_governance_event",
            "event_type": event_type,
            "artifact_id": artifact_id,
            "mode": mode,
            "run_id": run_id,
            "workflow_id": workflow_id,
            "conversation_id": str(conversation_id or ""),
            "step_seq": int(step_seq),
            "diff_json": json.dumps(filter_diff, sort_keys=True),
            "artifact_json": json.dumps(artifact_payload, sort_keys=True),
            "projection_strategy": projection_strategy,
        },
        mentions=[_grounding(f"artifact:{artifact_id}")],
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )
    conversation_engine.write.add_node(node)
    return event_id


def _workflow_shape(
    workflow_engine: GraphKnowledgeEngine, *, workflow_id: str
) -> dict[str, Any]:
    start, nodes, adj, _rev_adj = load_workflow_design(
        workflow_engine=workflow_engine, workflow_id=workflow_id
    )
    return {
        "start_node_id": start.id,
        "node_ids": sorted(nodes.keys()),
        "edge_ids": sorted(
            edge.id for edges in adj.values() for edge in list(edges or [])
        ),
    }


def _step_execs(
    conversation_engine: GraphKnowledgeEngine, *, run_id: str
) -> list[Any]:
    nodes = conversation_engine.read.get_nodes(
        where={"$and": [{"entity_type": "workflow_step_exec"}, {"run_id": run_id}]},
        limit=10_000,
    )
    return sorted(
        list(nodes or []),
        key=lambda node: int((getattr(node, "metadata", {}) or {}).get("step_seq", -1)),
    )


def _governance_events(
    conversation_engine: GraphKnowledgeEngine, *, run_id: str
) -> list[Any]:
    nodes = conversation_engine.read.get_nodes(
        where={
            "$and": [
                {"entity_type": "artifact_governance_event"},
                {"run_id": run_id},
            ]
        },
        limit=10_000,
    )
    return sorted(
        list(nodes or []),
        key=lambda node: (
            int((getattr(node, "metadata", {}) or {}).get("step_seq", -1)),
            str(getattr(node, "id", "")),
        ),
    )


def _latest_step_seq(
    conversation_engine: GraphKnowledgeEngine, *, run_id: str
) -> int:
    steps = _step_execs(conversation_engine, run_id=run_id)
    if not steps:
        raise RuntimeError(f"No workflow_step_exec nodes found for run_id={run_id!r}")
    return max(int((node.metadata or {}).get("step_seq", 0)) for node in steps)


def _workflow_run_result(
    *,
    runtime: WorkflowRuntime,
    workflow_id: str,
    conversation_id: str,
    turn_node_id: str,
    run_id: str,
    artifact_blueprint: dict[str, Any],
) -> Any:
    return runtime.run(
        workflow_id=workflow_id,
        conversation_id=conversation_id,
        turn_node_id=turn_node_id,
        initial_state={
            "_deps": {},
            "artifact_blueprint": json.loads(json.dumps(artifact_blueprint)),
        },
        run_id=run_id,
    )


def _persist_governance_workflows(
    workflow_engine: GraphKnowledgeEngine,
    *,
    safe_workflow_id: str,
    unsafe_workflow_id: str,
) -> None:
    safe_pairs = [
        ("start", "start", True, False),
        ("build", "build_artifact", False, False),
        ("emit_built", "emit_artifact_built_event", False, False),
        ("classify", "classify_artifact", False, False),
        ("public_mode", "apply_public_mode", False, False),
        ("emit_filtered", "emit_artifact_filtered_event", False, False),
        ("validate", "validate_artifact", False, False),
        ("emit_validated", "emit_artifact_validated_event", False, False),
        ("before_publish", "before_publish", False, False),
        ("publish", "publish_artifact", False, False),
        ("emit_published", "emit_artifact_published_event", False, False),
        ("end", "end", False, True),
    ]
    safe_nodes = [
        _workflow_node(
            workflow_id=safe_workflow_id,
            node_id=f"wf|demo.artifact.safe|{suffix}",
            op=op,
            start=start,
            terminal=terminal,
        )
        for suffix, op, start, terminal in safe_pairs
    ]
    safe_order = [suffix for suffix, _op, _start, _terminal in safe_pairs]
    safe_edges = [
        _workflow_edge(
            workflow_id=safe_workflow_id,
            edge_id=f"wf|demo.artifact.safe|e|{src}->{dst}",
            src=f"wf|demo.artifact.safe|{src}",
            dst=f"wf|demo.artifact.safe|{dst}",
        )
        for src, dst in zip(safe_order, safe_order[1:])
    ]
    _persist_design(
        workflow_engine,
        workflow_id=safe_workflow_id,
        nodes=safe_nodes,
        edges=safe_edges,
    )

    unsafe_pairs = [
        ("start", "start", True, False),
        ("build", "build_artifact", False, False),
        ("emit_built", "emit_artifact_built_event", False, False),
        ("classify", "classify_artifact", False, False),
        ("validate", "validate_artifact", False, False),
        ("emit_validated", "emit_artifact_validated_event", False, False),
        ("before_publish", "before_publish", False, False),
        ("publish", "publish_artifact", False, False),
        ("emit_published", "emit_artifact_published_event", False, False),
        ("end", "end", False, True),
    ]
    unsafe_nodes = [
        _workflow_node(
            workflow_id=unsafe_workflow_id,
            node_id=f"wf|demo.artifact.unsafe|{suffix}",
            op=op,
            start=start,
            terminal=terminal,
        )
        for suffix, op, start, terminal in unsafe_pairs
    ]
    unsafe_order = [suffix for suffix, _op, _start, _terminal in unsafe_pairs]
    unsafe_edges = [
        _workflow_edge(
            workflow_id=unsafe_workflow_id,
            edge_id=f"wf|demo.artifact.unsafe|e|{src}->{dst}",
            src=f"wf|demo.artifact.unsafe|{src}",
            dst=f"wf|demo.artifact.unsafe|{dst}",
        )
        for src, dst in zip(unsafe_order, unsafe_order[1:])
    ]
    _persist_design(
        workflow_engine,
        workflow_id=unsafe_workflow_id,
        nodes=unsafe_nodes,
        edges=unsafe_edges,
    )


def run_build_artifact_governance_demo(
    *,
    data_dir: str | Path | None = None,
    reset_data: bool = True,
    backend_factory: Any | None = None,
    workflow_engine: GraphKnowledgeEngine | None = None,
    conversation_engine: GraphKnowledgeEngine | None = None,
) -> dict[str, Any]:
    """Golden example for replayable artifact-governance enforcement."""

    if (workflow_engine is None) != (conversation_engine is None):
        raise ValueError(
            "workflow_engine and conversation_engine must be provided together"
        )

    if workflow_engine is None or conversation_engine is None:
        root = (
            Path(data_dir)
            if data_dir is not None
            else Path(".gke-data/build-artifact-governance-demo")
        ).resolve()
        if reset_data and root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        workflow_engine, conversation_engine = _build_engines(
            data_dir=root, backend_factory=backend_factory
        )
    else:
        if data_dir is not None:
            root = Path(data_dir).resolve()
        else:
            workflow_root = Path(str(getattr(workflow_engine, "persist_directory", ".")))
            conversation_root = Path(
                str(getattr(conversation_engine, "persist_directory", "."))
            )
            root = workflow_root.parent
            if conversation_root.parent != root:
                root = workflow_root

    safe_workflow_id = "demo.artifact_governance.safe"
    unsafe_workflow_id = "demo.artifact_governance.unsafe"
    conversation_id = "demo-conv-artifact-governance"
    safe_run_id = "demo-run-artifact-governance-safe"
    unsafe_run_id = "demo-run-artifact-governance-unsafe"

    _persist_governance_workflows(
        workflow_engine,
        safe_workflow_id=safe_workflow_id,
        unsafe_workflow_id=unsafe_workflow_id,
    )

    event_ops = {
        "emit_artifact_built_event": "artifact_built",
        "emit_artifact_filtered_event": "artifact_filtered",
        "emit_artifact_validated_event": "artifact_validated",
        "emit_artifact_published_event": "artifact_published",
    }

    resolver = MappingStepResolver()

    @resolver.register("start")
    def _start(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[
                (
                    "u",
                    {
                        "artifact_blueprint": json.loads(
                            json.dumps(PREDEFINED_ARTIFACT_BLUEPRINT)
                        ),
                        "governance_event_ids": [],
                        "governance_event_types": [],
                        "final_status": "planned",
                    },
                )
            ],
            _route_next=["build_artifact"],
        )

    @resolver.register("build_artifact")
    def _build_artifact(ctx):
        artifact = BuildArtifact.model_validate(ctx.state_view.get("artifact_blueprint"))
        return RunSuccess(
            conversation_node_id=None,
            state_update=[
                (
                    "u",
                    {
                        "artifact_internal": _dump_backend(artifact),
                        "classified_sensitive_components": [],
                        "filter_diff": [],
                        "public_artifact": None,
                        "public_projection_strategy": None,
                        "validated_artifact": None,
                        "validation_passed": False,
                        "validation_errors": [],
                        "published_artifact": None,
                        "final_status": "artifact_built",
                    },
                )
            ],
            _route_next=["emit_artifact_built_event"],
        )

    @resolver.register("classify_artifact")
    def _classify_artifact(ctx):
        artifact = BuildArtifact.model_validate(ctx.state_view.get("artifact_internal"))
        classification = _classify_sensitive_components(artifact)
        next_step = "apply_public_mode"
        if str(ctx.workflow_id).endswith(".unsafe"):
            next_step = "validate_artifact"
        return RunSuccess(
            conversation_node_id=None,
            state_update=[
                (
                    "u",
                    {
                        "classified_sensitive_components": json.loads(
                            json.dumps(classification)
                        ),
                        "final_status": "artifact_classified",
                    },
                )
            ],
            _route_next=[next_step],
        )

    @resolver.register("apply_public_mode")
    def _apply_public_mode(ctx):
        artifact = BuildArtifact.model_validate(ctx.state_view.get("artifact_internal"))
        public_payload, projection_strategy = _project_public_artifact(artifact)
        filter_diff = _diff_payload(_dump_backend(artifact), public_payload)
        return RunSuccess(
            conversation_node_id=None,
            state_update=[
                (
                    "u",
                    {
                        "public_artifact": public_payload,
                        "public_projection_strategy": projection_strategy,
                        "filter_diff": json.loads(json.dumps(filter_diff)),
                        "final_status": "artifact_filtered",
                    },
                )
            ],
            _route_next=["emit_artifact_filtered_event"],
        )

    def _emit_named_event(ctx, event_type: str) -> RunSuccess:
        artifact_payload = _artifact_payload_for_event(
            state=dict(ctx.state_view), event_type=event_type
        )
        event_id = _emit_governance_event(
            conversation_engine,
            run_id=str(ctx.run_id),
            workflow_id=str(ctx.workflow_id),
            conversation_id=ctx.conversation_id,
            step_seq=int(ctx.step_seq),
            event_type=event_type,
            artifact_payload=artifact_payload,
            filter_diff=list(ctx.state_view.get("filter_diff") or []),
            projection_strategy=str(
                ctx.state_view.get("public_projection_strategy") or ""
            )
            or None,
        )
        next_step = {
            "artifact_built": "classify_artifact",
            "artifact_filtered": "validate_artifact",
            "artifact_validated": "before_publish",
            "artifact_published": "end",
        }[event_type]
        return RunSuccess(
            conversation_node_id=event_id,
            state_update=[
                ("a", {"governance_event_ids": event_id}),
                ("a", {"governance_event_types": event_type}),
                (
                    "u",
                    {
                        "last_governance_event": event_type,
                        "final_status": event_type,
                    },
                ),
            ],
            _route_next=[next_step],
        )

    for op_name, event_type in event_ops.items():
        resolver.register(op_name)(
            lambda ctx, _event_type=event_type: _emit_named_event(ctx, _event_type)
        )

    @resolver.register("validate_artifact")
    def _validate_artifact(ctx):
        payload = dict(
            ctx.state_view.get("public_artifact")
            or ctx.state_view.get("artifact_internal")
            or {}
        )
        errors = _public_artifact_violations(payload)
        if errors:
            event_id = _emit_governance_event(
                conversation_engine,
                run_id=str(ctx.run_id),
                workflow_id=str(ctx.workflow_id),
                conversation_id=ctx.conversation_id,
                step_seq=int(ctx.step_seq),
                event_type="artifact_rejected",
                artifact_payload=payload,
                filter_diff=list(ctx.state_view.get("filter_diff") or []),
                projection_strategy=str(
                    ctx.state_view.get("public_projection_strategy") or ""
                )
                or None,
            )
            return RunFailure(
                conversation_node_id=event_id,
                state_update=[
                    (
                        "u",
                        {
                            "validation_passed": False,
                            "validation_errors": list(errors),
                            "final_status": "artifact_rejected",
                        },
                    ),
                    ("a", {"governance_event_ids": event_id}),
                    ("a", {"governance_event_types": "artifact_rejected"}),
                    ("u", {"last_governance_event": "artifact_rejected"}),
                ],
                errors=list(errors),
            )

        return RunSuccess(
            conversation_node_id=None,
            state_update=[
                (
                    "u",
                    {
                        "validation_passed": True,
                        "validation_errors": [],
                        "validated_artifact": payload,
                        "final_status": "artifact_validated",
                    },
                )
            ],
            _route_next=["emit_artifact_validated_event"],
        )

    @resolver.register("before_publish")
    def _before_publish(ctx):
        if not bool(ctx.state_view.get("validation_passed")):
            return RunFailure(
                conversation_node_id=None,
                state_update=[
                    (
                        "u",
                        {
                            "final_status": "publish_blocked",
                            "validation_errors": [
                                "before_publish requires validation_passed == True"
                            ],
                        },
                    )
                ],
                errors=["before_publish requires validation_passed == True"],
            )
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"final_status": "publish_ready"})],
            _route_next=["publish_artifact"],
        )

    @resolver.register("publish_artifact")
    def _publish_artifact(ctx):
        payload = dict(
            ctx.state_view.get("validated_artifact")
            or ctx.state_view.get("public_artifact")
            or {}
        )
        errors = _public_artifact_violations(payload)
        if errors:
            return RunFailure(
                conversation_node_id=None,
                state_update=[
                    (
                        "u",
                        {
                            "final_status": "publish_blocked",
                            "validation_errors": list(errors),
                        },
                    )
                ],
                errors=list(errors),
            )
        return RunSuccess(
            conversation_node_id=None,
            state_update=[
                (
                    "u",
                    {
                        "published_artifact": payload,
                        "final_status": "artifact_published",
                    },
                )
            ],
            _route_next=["emit_artifact_published_event"],
        )

    @resolver.register("end")
    def _end(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"completed": True})],
        )

    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
    )

    safe_result = _workflow_run_result(
        runtime=runtime,
        workflow_id=safe_workflow_id,
        conversation_id=conversation_id,
        turn_node_id="demo-turn-artifact-governance-safe",
        run_id=safe_run_id,
        artifact_blueprint=PREDEFINED_ARTIFACT_BLUEPRINT,
    )
    unsafe_result = _workflow_run_result(
        runtime=runtime,
        workflow_id=unsafe_workflow_id,
        conversation_id=conversation_id,
        turn_node_id="demo-turn-artifact-governance-unsafe",
        run_id=unsafe_run_id,
        artifact_blueprint=PREDEFINED_ARTIFACT_BLUEPRINT,
    )

    safe_replay = replay_to(
        conversation_engine=conversation_engine,
        run_id=safe_run_id,
        target_step_seq=_latest_step_seq(conversation_engine, run_id=safe_run_id),
    )
    unsafe_replay = replay_to(
        conversation_engine=conversation_engine,
        run_id=unsafe_run_id,
        target_step_seq=_latest_step_seq(conversation_engine, run_id=unsafe_run_id),
    )

    safe_events = _governance_events(conversation_engine, run_id=safe_run_id)
    unsafe_events = _governance_events(conversation_engine, run_id=unsafe_run_id)
    safe_steps = _step_execs(conversation_engine, run_id=safe_run_id)
    unsafe_steps = _step_execs(conversation_engine, run_id=unsafe_run_id)

    safe_published = dict(safe_result.final_state.get("published_artifact") or {})
    safe_public_validation = _public_artifact_violations(safe_published)
    unsafe_published = dict(unsafe_result.final_state.get("published_artifact") or {})
    safe_projection_strategy = str(
        safe_result.final_state.get("public_projection_strategy") or ""
    )
    safe_projection_matches_public = (
        safe_published == dict(safe_result.final_state.get("public_artifact") or {})
    )

    return {
        "summary": {
            "data_dir": str(root),
            "safe_status": safe_result.status,
            "unsafe_status": unsafe_result.status,
            "invariant_pass": (
                safe_result.status == "succeeded"
                and unsafe_result.status == "failure"
                and not safe_public_validation
                and not unsafe_published
            ),
            "workflow_graphs_visible": {
                "safe": True,
                "unsafe": True,
            },
        },
        "details": {
            "artifact_blueprint": json.loads(json.dumps(PREDEFINED_ARTIFACT_BLUEPRINT)),
            "workflow_shapes": {
                "safe": _workflow_shape(
                    workflow_engine, workflow_id=safe_workflow_id
                ),
                "unsafe": _workflow_shape(
                    workflow_engine, workflow_id=unsafe_workflow_id
                ),
            },
            "safe": {
                "run_id": safe_run_id,
                "final_state": dict(safe_result.final_state),
                "public_projection_strategy": safe_projection_strategy,
                "public_projection_matches_public_artifact": safe_projection_matches_public,
                "event_types": [
                    str((node.metadata or {}).get("event_type") or "")
                    for node in safe_events
                ],
                "event_ids": [str(getattr(node, "id", "")) for node in safe_events],
                "event_diffs": [
                    json.loads(str((node.metadata or {}).get("diff_json") or "[]"))
                    for node in safe_events
                ],
                "step_ops": [
                    str((node.metadata or {}).get("op") or "") for node in safe_steps
                ],
                "replay_state": safe_replay,
                "public_artifact_violations": safe_public_validation,
            },
            "unsafe": {
                "run_id": unsafe_run_id,
                "final_state": dict(unsafe_result.final_state),
                "public_projection_strategy": str(
                    unsafe_result.final_state.get("public_projection_strategy") or ""
                ),
                "event_types": [
                    str((node.metadata or {}).get("event_type") or "")
                    for node in unsafe_events
                ],
                "event_ids": [str(getattr(node, "id", "")) for node in unsafe_events],
                "event_diffs": [
                    json.loads(str((node.metadata or {}).get("diff_json") or "[]"))
                    for node in unsafe_events
                ],
                "step_ops": [
                    str((node.metadata or {}).get("op") or "") for node in unsafe_steps
                ],
                "replay_state": unsafe_replay,
                "published_artifact": unsafe_published,
            },
        },
    }


if __name__ == "__main__":
    print(
        json.dumps(
            run_build_artifact_governance_demo(),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
