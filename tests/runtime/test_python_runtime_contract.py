from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from kogwistar._rust_bridge import RustExtensionUnavailableError
from kogwistar.engine_core import GraphKnowledgeEngine
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend
from kogwistar.runtime import MappingStepResolver
from kogwistar.runtime.models import RunSuccess
from kogwistar.runtime.runtime import WorkflowRuntime

pytestmark = [pytest.mark.ci, pytest.mark.runtime, pytest.mark.runtime_sync]


@dataclass
class _RuntimeLaneNode:
    id: str
    op: str
    metadata: dict
    terminal: bool = True
    version: str = "v1"
    fanout: bool = False

    def safe_get_id(self) -> str:
        return self.id


@dataclass
class _RuntimeLaneEdge:
    id: str
    source_ids: list[str]
    target_ids: list[str]
    metadata: dict

    def safe_get_id(self) -> str:
        return self.id

    @property
    def priority(self) -> int:
        return int(self.metadata.get("wf_priority", 100))

    @property
    def multiplicity(self) -> str:
        return str(self.metadata.get("wf_multiplicity", "one"))

    @property
    def is_default(self) -> bool:
        return bool(self.metadata.get("wf_is_default", False))

    @property
    def predicate(self) -> str | None:
        value = self.metadata.get("wf_predicate")
        return value if isinstance(value, str) else None


def _lane_runtime_engine(tmp_path: Path) -> GraphKnowledgeEngine:
    return GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        kg_graph_type="conversation",
        backend_factory=build_in_memory_backend,
    )


def test_python_runtime_matches_shared_native_increment_workflow_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the shared native workflow trace through the Python runtime only."""
    contract_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "runtime-native-increment-workflow-v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    workflow_id = contract["workflow_id"]
    nodes = {
        node["id"]: _RuntimeLaneNode(
            id=node["id"],
            op=node["operation"],
            metadata={
                "wf_start": index == 0,
                "wf_terminal": node["terminal"],
                "wf_op": node["operation"],
                "wf_version": node["version"],
            },
            terminal=node["terminal"],
            version=node["version"],
        )
        for index, node in enumerate(contract["nodes"])
    }
    edges = [
        _RuntimeLaneEdge(
            id=f"edge:{route['source_node_id']}:{route['target_node_id']}",
            source_ids=[route["source_node_id"]],
            target_ids=[route["target_node_id"]],
            metadata={"wf_multiplicity": "one", "wf_priority": 100},
        )
        for route in contract["routes"]
    ]
    outgoing = {node_id: [] for node_id in nodes}
    for edge in edges:
        outgoing[edge.source_ids[0]].append(edge)

    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.delenv(selector, raising=False)
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")

    def unavailable_native_loader() -> None:
        raise RustExtensionUnavailableError("Python-only runtime must not load Rust")

    monkeypatch.setattr("kogwistar._rust_bridge._load_extension", unavailable_native_loader)
    monkeypatch.setattr(
        "kogwistar.runtime.runtime.validate_workflow_design",
        lambda **_kwargs: (nodes["first"], nodes, outgoing),
    )
    monkeypatch.setattr(
        "kogwistar.runtime.runtime._compute_may_reach_join_bitsets",
        lambda **kwargs: {node_id: 0 for node_id in kwargs["node_ids"]},
    )

    resolver = MappingStepResolver()
    executed_nodes: list[str] = []

    class _TraceCapture:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def emit(self, evt: dict[str, Any]) -> None:
            self.events.append(evt)

        def flush(self, timeout: float = 1.0) -> bool:
            return True

        def close(self, timeout: float = 1.0) -> None:
            pass

    trace_capture = _TraceCapture()

    @resolver.register("increment-counter", version="v1")
    def increment(ctx):
        executed_nodes.append(ctx.workflow_node_id)
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"counter": int(ctx.state_view["counter"]) + 1})],
        )

    runtime = WorkflowRuntime(
        workflow_engine=cast(GraphKnowledgeEngine, object()),
        conversation_engine=_lane_runtime_engine(tmp_path / "conversation"),
        step_resolver=resolver,
        predicate_registry={},
        sink=trace_capture,
    )
    result = runtime.run(
        workflow_id=workflow_id,
        conversation_id="shared-increment-contract",
        turn_node_id="turn-shared-increment-contract",
        run_id="python-shared-increment-contract",
        initial_state=dict(contract["initial_state"]),
    )

    assert result.status == contract["expected_status"]
    assert result.final_state["counter"] == contract["expected_final_state"]["counter"]
    expected_steps = contract["expected_completed_steps"]
    assert executed_nodes == expected_steps
    completed_steps = [
        event
        for event in trace_capture.events
        if event["type"] == "step_attempt_completed"
    ]
    run_started = next(
        event for event in trace_capture.events if event["type"] == "workflow_run_started"
    )
    assert [event["node_id"] for event in completed_steps] == expected_steps
    assert all(
        event["run_id"] == "python-shared-increment-contract" for event in completed_steps
    )
    assert [event["step_seq"] for event in completed_steps] == contract[
        "expected_step_sequences"
    ]
    assert [json.loads(event["payload_json"])["status"] for event in completed_steps] == [
        "ok",
        "ok",
    ]
    assert run_started["parent_span_id"] == (
        "stp:python-shared-increment-contract:python-shared-increment-contract:0:1:run"
    )
    assert run_started["parent_span_id"] != completed_steps[0]["span_id"]
