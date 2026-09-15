from dataclasses import dataclass, field
from typing import Any

import pytest

from kogwistar.runtime.langgraph_converter import (
    LangGraphImportUnsupportedError,
    _route_next,
    from_langgraph,
)
from kogwistar.runtime.models import RunSuccess
from kogwistar.runtime.resolvers import MappingStepResolver


@dataclass
class _Edge:
    id: str
    source_ids: list[str]
    target_ids: list[str]
    predicate: str | None = None
    priority: int = 100
    is_default: bool = False
    multiplicity: str = "one"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Result:
    next_step_names: list[str] = field(default_factory=list)
    status: str = "success"


def test_converter_uses_native_route_authority_for_default_and_invalid_explicit():
    edges = [
        _Edge("conditional", ["start"], ["target"], predicate="ready"),
        _Edge("fallback", ["start"], ["fallback"], is_default=True),
    ]
    assert _route_next(
        edges=edges,
        state={},
        last_result=_Result(next_step_names=["missing"]),
        fanout=False,
        predicate_registry={"ready": lambda *_: False},
    ) == ["fallback"]


def test_converter_preserves_same_priority_fanout_and_feedback_targets():
    edges = [
        _Edge("left", ["router"], ["left"], predicate="go", priority=10, multiplicity="many"),
        _Edge("loop", ["router"], ["router"], predicate="go", priority=10, multiplicity="many"),
    ]
    assert _route_next(
        edges=edges,
        state={},
        last_result=_Result(),
        fanout=True,
        predicate_registry={"go": lambda *_: True},
    ) == ["left", "router"]


def test_compiled_langgraph_reverse_import_is_deterministically_unsupported():
    with pytest.raises(LangGraphImportUnsupportedError, match="WorkflowDesignArtifact"):
        from_langgraph(object(), workflow_id="wf")


@pytest.mark.runtime
def test_converter_invokes_production_mapping_resolver_with_step_context():
    seen = {}
    resolver = MappingStepResolver()

    @resolver.register("record")
    def record(ctx):
        seen.update({"node": ctx.workflow_node_id, "op": ctx.op, "value": ctx.state_view["value"]})
        return RunSuccess(conversation_node_id=None, state_update=[])

    result = resolver.resolve("record")
    from kogwistar.runtime.langgraph_converter import _invoke_step

    out = _invoke_step(
        resolver=resolver,
        fn=result,
        op="record",
        node_id="node-1",
        state={"value": 7},
    )
    assert out.status == "success"
    assert seen == {"node": "node-1", "op": "record", "value": 7}
