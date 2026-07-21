from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from kogwistar.runtime import contract as workflow_contract
from kogwistar.runtime import design as workflow_design
from kogwistar.runtime.runtime import _compute_may_reach_join_bitsets


pytestmark = [pytest.mark.ci, pytest.mark.core]


@dataclass
class _Edge:
    target_ids: list[str | None]
    id: str = "edge"
    metadata: dict[str, object] | None = None


@dataclass
class _Node:
    id: str
    metadata: dict[str, object]


@dataclass
class _ContractEdge:
    dst: str
    edge_id: str = "edge"
    predicate: str | None = None


def _adj(node_ids: list[str], edges: list[tuple[str, str | None]]) -> dict[str, list[_Edge]]:
    result = {node_id: [] for node_id in node_ids}
    for source, target in edges:
        result.setdefault(source, []).append(_Edge([target]))
    return result


def _bfs_oracle(
    node_ids: list[str], edges: list[tuple[str, str | None]], join_ids: list[str]
) -> dict[str, int]:
    successors = {node_id: [] for node_id in node_ids}
    for source, target in edges:
        if source in successors and target in successors:
            successors[source].append(target)
    result: dict[str, int] = {}
    for node_id in node_ids:
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(successors[current])
        result[node_id] = sum(
            1 << index for index, join_id in enumerate(join_ids) if join_id in seen
        )
    return result


@pytest.fixture(scope="module", autouse=True)
def _native_extension():
    return pytest.importorskip("kogwistar._rust")


@pytest.mark.parametrize(
    ("node_ids", "edges", "join_ids"),
    [
        (["start", "middle", "join", "end"], [("start", "middle"), ("middle", "join"), ("join", "end")], ["join"]),
        (["start", "left", "right", "join", "end"], [("start", "left"), ("start", "right"), ("left", "join"), ("right", "join"), ("join", "end")], ["join"]),
        (["start", "join", "island"], [("start", "join")], ["join", "island"]),
        (["start", "cycle_a", "cycle_b", "join"], [("start", "cycle_a"), ("cycle_a", "cycle_b"), ("cycle_b", "cycle_a"), ("cycle_b", "join")], ["join"]),
        (["start", "cycle_a", "cycle_b", "join", "end"], [("start", "cycle_a"), ("cycle_a", "cycle_b"), ("cycle_b", "cycle_a"), ("cycle_b", "join"), ("join", "end")], ["join"]),
        (["start", "join"], [("start", "join"), ("start", "join"), ("unknown", "join"), ("start", "unknown")], ["join", "unknown"]),
    ],
    ids=["dag", "fanout_join", "disconnected_terminal", "cycle", "scc_exit", "duplicate_unknown"],
)
@pytest.mark.parametrize("mode", ["python", "shadow", "rust"])
def test_workflow_lineage_differential(
    monkeypatch,
    mode: str,
    node_ids: list[str],
    edges: list[tuple[str, str | None]],
    join_ids: list[str],
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", mode)
    actual = _compute_may_reach_join_bitsets(
        node_ids=node_ids,
        adj=_adj(node_ids, edges),
        join_ids=join_ids,
    )
    assert actual == _bfs_oracle(node_ids, edges, join_ids)


def test_native_invalid_payload_has_stable_machine_code(_native_extension) -> None:
    with pytest.raises(ValueError) as raised:
        _native_extension.workflow_may_reach_join(
            json.dumps({"node_ids": ["n"], "edges": "bad", "join_ids": []})
        )
    assert getattr(raised.value, "code") == "KOGWISTAR_CONTRACT_WORKFLOW_TOPOLOGY_EDGES_TYPE"


@pytest.mark.parametrize("mode", ["python", "shadow", "rust"])
@pytest.mark.parametrize(
    ("has_exit", "message"),
    [
        (True, None),
        (False, "No terminal reachable from start (ignoring predicates)."),
    ],
)
def test_design_validator_terminal_reachability_modes(
    monkeypatch, mode: str, has_exit: bool, message: str | None
) -> None:
    nodes = {
        "start": _Node("start", {"wf_start": True}),
        "cycle": _Node("cycle", {}),
        "end": _Node("end", {"wf_terminal": True}),
    }
    adj = {
        "start": [_Edge(target_ids=["cycle"], metadata={})],
        "cycle": [_Edge(target_ids=["end" if has_exit else "start"], metadata={})],
        "end": [],
    }
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", mode)
    monkeypatch.setattr(
        workflow_design,
        "load_workflow_design",
        lambda **_: (nodes["start"], nodes, adj, {node_id: [] for node_id in nodes}),
    )
    if message is None:
        assert workflow_design.validate_workflow_design(
            workflow_engine=object(), workflow_id="wf", predicate_registry={}
        ) == (nodes["start"], nodes, adj)
    else:
        with pytest.raises(ValueError) as raised:
            workflow_design.validate_workflow_design(
                workflow_engine=object(), workflow_id="wf", predicate_registry={}
            )
        assert str(raised.value) == message


@pytest.mark.parametrize("mode", ["python", "shadow", "rust"])
@pytest.mark.parametrize(
    ("has_exit", "message"),
    [
        (True, None),
        (
            False,
            "No terminal reachable from start (ignoring predicates). Cyclic graph without exit.",
        ),
    ],
)
def test_contract_validator_terminal_reachability_modes(
    monkeypatch, mode: str, has_exit: bool, message: str | None
) -> None:
    nodes = {
        "start": workflow_contract.WorkflowNodeInfo("start", "op", "v1", True, False, False),
        "cycle": workflow_contract.WorkflowNodeInfo("cycle", "op", "v1", True, False, False),
        "end": workflow_contract.WorkflowNodeInfo("end", "", "v1", True, True, False),
    }
    adj = {
        "start": [_ContractEdge("cycle")],
        "cycle": [_ContractEdge("end" if has_exit else "start")],
        "end": [],
    }
    spec = workflow_contract.WorkflowSpec("wf", "start")
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", mode)
    monkeypatch.setattr(
        workflow_contract,
        "load_workflow_graph",
        lambda **_: (nodes, adj),
    )
    if message is None:
        assert workflow_contract.validate_workflow(
            engine=object(), spec=spec, predicate_registry={}
        ) is None
    else:
        with pytest.raises(ValueError) as raised:
            workflow_contract.validate_workflow(
                engine=object(), spec=spec, predicate_registry={}
            )
        assert str(raised.value) == message
