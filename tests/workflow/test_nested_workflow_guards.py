import pytest

from kogwistar.runtime.base_runtime import BaseRuntime
from kogwistar.runtime.models import WorkflowInvocationRequest

pytestmark = [pytest.mark.workflow, pytest.mark.runtime]


def _runtime(depth=8):
    runtime = BaseRuntime.__new__(BaseRuntime)
    runtime.workflow_id = "wf.parent"
    runtime.max_nested_workflow_depth = depth
    return runtime


def _invocation(workflow_id):
    return WorkflowInvocationRequest(workflow_id=workflow_id, result_state_key="child")


def test_nested_path_rejects_recursive_cycle():
    with pytest.raises(ValueError, match="cycle"):
        _runtime()._child_workflow_initial_state(
            parent_state={"_wf_invocation_path": ["wf.parent", "wf.child"]},
            invocation=_invocation("wf.parent"),
        )


def test_nested_path_enforces_depth_and_preserves_lineage_metadata():
    runtime = _runtime(depth=3)
    child = runtime._child_workflow_initial_state(
        parent_state={"_wf_invocation_path": ["wf.parent"], "_wf_current_run_id": "run-parent"},
        invocation=_invocation("wf.child"),
    )
    assert child["_wf_invocation_path"] == ["wf.parent", "wf.child"]
    assert child["_wf_parent_run_id"] == "run-parent"
    with pytest.raises(ValueError, match="depth limit"):
        runtime._child_workflow_initial_state(
            parent_state={"_wf_invocation_path": ["wf.parent", "wf.child", "wf.grandchild"]},
            invocation=_invocation("wf.greatgrandchild"),
        )
