"""Deterministic optional interop contracts."""

from __future__ import annotations

import pytest

from kogwistar.interop.crewai import (
    crewai_delegated_invocation,
    import_crewai_flow,
    imported_semantic_signature,
    static_semantic_signature,
)


pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.unit]


def test_static_crewai_flow_imports_to_ordinary_graph() -> None:
    design, diagnostics = import_crewai_flow(
        {
            "workflow_id": "crew.demo",
            "tasks": [
                {"id": "plan", "name": "Plan", "role": "planner"},
                {"id": "act", "name": "Act", "depends_on": ["plan"]},
            ],
        }
    )
    assert diagnostics == ()
    assert design.start_node_id == "wf:crew.demo:plan"
    assert len(design.nodes) == 2
    assert design.edges[0].source_ids == ["wf:crew.demo:plan"]


def test_unsupported_crewai_behavior_is_diagnostic_not_authority() -> None:
    design, diagnostics = import_crewai_flow(
        {"workflow_id": "crew.dynamic", "tasks": ["one"], "manager": {"dynamic": True}, "memory": True}
    )
    assert design.nodes[0].metadata["interop_source"] == "crewai"
    assert {item.code for item in diagnostics} == {"dynamic_manager", "opaque_memory"}


def test_crewai_delegation_maps_to_existing_nested_invocation_identity() -> None:
    request = crewai_delegated_invocation(
        {"delegates_to": "research.workflow"}, parent_run_id="parent", step_seq=3
    )
    assert request.workflow_id == "research.workflow"
    assert request.invocation_key == "parent:3:research.workflow"


def test_supported_crewai_subset_has_stable_semantic_signature() -> None:
    description = {
        "workflow_id": "crewai-roundtrip",
        "tasks": [
            {"id": "plan"},
            {"id": "act", "depends_on": ["plan"]},
        ],
    }
    design, _ = import_crewai_flow(description)
    assert imported_semantic_signature(design) == static_semantic_signature(description)


def test_empty_crewai_flow_is_rejected() -> None:
    with pytest.raises(ValueError):
        import_crewai_flow({"workflow_id": "crew.empty", "tasks": []})
