from __future__ import annotations

import pytest

from kogwistar.demo import run_nested_workflow_invocation_demo
from tests._helpers.fake_backend import build_fake_backend

pytestmark = [pytest.mark.workflow]

def _assert_nested_workflow_demo_result(result: dict) -> None:
    assert result["summary"]["status"] == "succeeded"
    assert result["summary"]["workflow_graphs_visible"] == {
        "parent": True,
        "predesigned_child": True,
        "dynamic_child": True,
    }
    assert result["details"]["final_state"]["predesigned_child"]["child_mode"] == (
        "predesigned"
    )
    assert result["details"]["final_state"]["dynamic_child"]["child_mode"] == (
        "dynamic_persisted"
    )
    assert result["details"]["workflow_shapes"]["predesigned_child"]["node_ids"] == [
        "wf|demo.child.predesigned|body",
        "wf|demo.child.predesigned|end",
        "wf|demo.child.predesigned|start",
    ]
    assert result["details"]["workflow_shapes"]["dynamic_child"]["node_ids"] == [
        "wf|demo.child.dynamic|end",
        "wf|demo.child.dynamic|materialize",
        "wf|demo.child.dynamic|start",
    ]
    assert "invoke_predesigned" in result["details"]["conversation_trace"][
        "parent_step_ops"
    ]
    assert "invoke_dynamic" in result["details"]["conversation_trace"][
        "parent_step_ops"
    ]
    assert "predesigned_body" in result["details"]["conversation_trace"][
        "predesigned_child_step_ops"
    ]
    assert "dynamic_materialize" in result["details"]["conversation_trace"][
        "dynamic_child_step_ops"
    ]


@pytest.mark.ci
def test_nested_workflow_invocation_demo_supports_predesigned_and_dynamic_children(
    tmp_path,
):
    result = run_nested_workflow_invocation_demo(
        data_dir=tmp_path / "nested_workflow_demo",
        backend_factory=build_fake_backend,
    )
    _assert_nested_workflow_demo_result(result)


@pytest.mark.ci_full
@pytest.mark.parametrize("backend_kind", ["chroma", "pg"], indirect=True)
def test_nested_workflow_invocation_demo_persistent_backends(
    workflow_engine,
    conversation_engine,
    backend_kind,
):
    _ = backend_kind
    result = run_nested_workflow_invocation_demo(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        reset_data=False,
    )
    _assert_nested_workflow_demo_result(result)
