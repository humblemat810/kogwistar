from __future__ import annotations
import pytest
pytestmark = [pytest.mark.core, pytest.mark.runtime]

import shutil
import uuid
from pathlib import Path

from kogwistar.conversation.service import ConversationService
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.id_provider import stable_id
from kogwistar.runtime.models import RunSuccess, WorkflowEdge, WorkflowNode
from kogwistar.runtime.replay import replay_to
from kogwistar.runtime.runtime import WorkflowRuntime
from tests._helpers.engine_factories import FakeEmbeddingFunction
from tests._helpers.fake_backend import build_fake_backend
from tests._helpers.workflow_builders import build_workflow_edge, build_workflow_node


def _wf_node(
    *,
    workflow_id: str,
    node_id: str,
    op: str,
    start: bool = False,
    terminal: bool = False,
) -> WorkflowNode:
    return build_workflow_node(
        workflow_id=workflow_id,
        node_id=node_id,
        op=op,
        label=node_id.split("|")[-1],
        start=start,
        terminal=terminal,
    )


def _wf_edge(
    *,
    workflow_id: str,
    edge_id: str,
    src: str,
    dst: str,
    predicate: str | None = None,
    priority: int = 100,
    is_default: bool = True,
) -> WorkflowEdge:
    return build_workflow_edge(
        workflow_id=workflow_id,
        edge_id=edge_id,
        src=src,
        dst=dst,
        predicate=predicate,
        priority=priority,
        is_default=is_default,
    )


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", id="fake", marks=pytest.mark.ci_full),
        pytest.param("chroma", id="chroma", marks=pytest.mark.ci_full),
    ],
)
def test_runtime_event_sourced_cancel_reconciles_and_replay_is_stable(backend_kind):
    """Async mirror: `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_cancellation_drains_inflight`."""
    root = Path(".tmp_runtime_cancel_event_sourced") / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=True)
    try:
        ef = FakeEmbeddingFunction()
        if backend_kind == "fake":
            workflow_engine = GraphKnowledgeEngine(
                persist_directory=str(root / "wf"),
                kg_graph_type="workflow",
                embedding_function=ef,
                backend_factory=build_fake_backend,
            )
            conversation_engine = GraphKnowledgeEngine(
                persist_directory=str(root / "conv"),
                kg_graph_type="conversation",
                embedding_function=ef,
                backend_factory=build_fake_backend,
            )
        else:
            workflow_engine = GraphKnowledgeEngine(
                persist_directory=str(root / "wf"),
                kg_graph_type="workflow",
                embedding_function=ef,
            )
            conversation_engine = GraphKnowledgeEngine(
                persist_directory=str(root / "conv"),
                kg_graph_type="conversation",
                embedding_function=ef,
            )

        workflow_id = "wf_cancel_event_sourced"
        conversation_id = "conv_cancel_evt"
        n_start = _wf_node(
            workflow_id=workflow_id,
            node_id=f"wf|{workflow_id}|start",
            op="trigger_cancel",
            start=True,
        )
        n_after = _wf_node(
            workflow_id=workflow_id,
            node_id=f"wf|{workflow_id}|after",
            op="after_cancel",
        )
        n_end = _wf_node(
            workflow_id=workflow_id,
            node_id=f"wf|{workflow_id}|end",
            op="end",
            terminal=True,
        )
        for node in (n_start, n_after, n_end):
            workflow_engine.write.add_node(node)
        workflow_engine.write.add_edge(
            _wf_edge(
                workflow_id=workflow_id,
                edge_id=f"wf|{workflow_id}|e|start->after",
                src=n_start.id,
                dst=n_after.id,
            )
        )
        workflow_engine.write.add_edge(
            _wf_edge(
                workflow_id=workflow_id,
                edge_id=f"wf|{workflow_id}|e|after->end",
                src=n_after.id,
                dst=n_end.id,
            )
        )

        def resolve_step(op: str):
            def _fn(ctx):
                if op == "trigger_cancel":
                    svc = ConversationService.from_engine(
                        conversation_engine,
                        knowledge_engine=conversation_engine,
                        workflow_engine=workflow_engine,
                    )
                    svc.persist_workflow_cancel_request(
                        conversation_id=conversation_id,
                        run_id=str(ctx.run_id),
                        workflow_id=workflow_id,
                        requested_by="test",
                        reason="event_sourced_cancel_test",
                    )
                return RunSuccess(
                    conversation_node_id=None,
                    state_update=[("a", {"op_log": op})],
                )

            return _fn

        runtime = WorkflowRuntime(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            step_resolver=resolve_step,
            predicate_registry={},
            checkpoint_every_n_steps=1,
            max_workers=1,
            cancel_requested=lambda _rid: False,
        )

        run_result = runtime.run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id="turn-cancel",
            initial_state={},
        )
        run_id = run_result.run_id
        assert run_result.status == "cancelled"
        assert run_result.final_state.get("op_log") == ["trigger_cancel"]

        cancel_req_id = f"wf_cancel_req|{run_id}"
        cancel_req_nodes = conversation_engine.read.get_nodes(ids=[cancel_req_id])
        assert len(cancel_req_nodes) == 1
        assert str(cancel_req_nodes[0].id) == cancel_req_id

        cancel_node_id = f"wf_cancelled|{run_id}"
        cancel_nodes = conversation_engine.read.get_nodes(ids=[cancel_node_id])
        assert len(cancel_nodes) == 1
        cancel_meta = cancel_nodes[0].metadata or {}
        assert int(cancel_meta.get("accepted_step_seq", -1)) == 0
        assert str(cancel_meta.get("cancel_request_node_id") or "") == str(
            cancel_req_nodes[0].id
        )
        expected_last_step_id = f"wf_step|{run_id}|0"
        assert (
            str(cancel_meta.get("last_processed_node_id") or "")
            == expected_last_step_id
        )
        cancelled_at_edge_id = str(
            stable_id(
                "workflow.edge",
                "cancelled_at",
                f"wf_cancelled|{run_id}",
                expected_last_step_id,
            )
        )
        cancelled_at_edge = conversation_engine.backend.edge_get(
            ids=[cancelled_at_edge_id], include=[]
        )
        assert cancelled_at_edge.get("ids") == [cancelled_at_edge_id]

        replayed = replay_to(
            conversation_engine=conversation_engine,
            run_id=run_id,
            target_step_seq=50,
        )
        assert replayed.get("op_log") == ["trigger_cancel"]
    finally:
        shutil.rmtree(root, ignore_errors=True)
