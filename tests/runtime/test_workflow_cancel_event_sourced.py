from __future__ import annotations
import pytest
pytestmark = pytest.mark.ci_full

import shutil
import uuid
from pathlib import Path

from kogwistar.conversation.service import ConversationService
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import (
    Grounding,
    MentionVerification,
    Span,
)
from kogwistar.id_provider import stable_id
from kogwistar.runtime.models import RunSuccess, WorkflowEdge, WorkflowNode
from kogwistar.runtime.replay import replay_to
from kogwistar.runtime.runtime import WorkflowRuntime


class FakeEmbeddingFunction:
    @staticmethod
    def name() -> str:
        return "default"

    def __init__(self, dim: int = 8):
        self._dim = dim
        self.is_legacy = False

    def __call__(self, input):
        return [[0.01] * self._dim for _ in input]


def _span() -> Span:
    return Span(
        collection_page_url="test",
        document_page_url="test",
        doc_id="test",
        insertion_method="test",
        page_number=1,
        start_char=0,
        end_char=4,
        excerpt="test",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="human", is_verified=True, score=1.0, notes="test"
        ),
    )


def _g() -> Grounding:
    return Grounding(spans=[_span()])


def _wf_node(
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
        mentions=[_g()],
        properties={},
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": op,
            "wf_start": start,
            "wf_terminal": terminal,
            "wf_version": "v1",
        },
        domain_id=None,
        canonical_entity_id=None,
        level_from_root=0,
        embedding=None,
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
    return WorkflowEdge(
        id=edge_id,
        source_ids=[src],
        target_ids=[dst],
        relation="wf_next",
        label="wf_next",
        type="relationship",
        summary="next",
        doc_id=workflow_id,
        mentions=[_g()],
        properties={},
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_priority": priority,
            "wf_is_default": is_default,
            "wf_predicate": predicate,
            "wf_multiplicity": "one",
        },
        source_edge_ids=[],
        target_edge_ids=[],
        domain_id=None,
        canonical_entity_id=None,
    )


def test_runtime_event_sourced_cancel_reconciles_and_replay_is_stable():
    root = Path(".tmp_runtime_cancel_event_sourced") / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=True)
    try:
        ef = FakeEmbeddingFunction()
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
            workflow_engine.add_node(node)
        workflow_engine.add_edge(
            _wf_edge(
                workflow_id=workflow_id,
                edge_id=f"wf|{workflow_id}|e|start->after",
                src=n_start.id,
                dst=n_after.id,
            )
        )
        workflow_engine.add_edge(
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

        cancel_req_nodes = conversation_engine.get_nodes(
            where={
                "$and": [{"entity_type": "workflow_cancel_request"}, {"run_id": run_id}]
            },
            limit=10,
        )
        assert len(cancel_req_nodes) == 1

        cancel_nodes = conversation_engine.get_nodes(
            where={"$and": [{"entity_type": "workflow_cancelled"}, {"run_id": run_id}]},
            limit=10,
        )
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
