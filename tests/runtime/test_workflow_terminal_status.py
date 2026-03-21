from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import (
    Grounding,
    MentionVerification,
    Span,
)
from kogwistar.runtime.models import (
    RunSuccess,
    WorkflowCompletedNode,
    WorkflowEdge,
    WorkflowNode,
)
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
        label=node_id,
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


def _wf_edge(*, workflow_id: str, edge_id: str, src: str, dst: str) -> WorkflowEdge:
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
            "wf_priority": 100,
            "wf_is_default": True,
            "wf_predicate": None,
            "wf_multiplicity": "one",
        },
        source_edge_ids=[],
        target_edge_ids=[],
        domain_id=None,
        canonical_entity_id=None,
    )


def test_runtime_persists_completed_terminal_for_leaf_node():
    root = Path(".tmp_runtime_completed_terminal") / str(uuid.uuid4())
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

        workflow_id = "wf_runtime_leaf_terminal"
        conversation_id = "conv_runtime_leaf_terminal"
        start = _wf_node(
            workflow_id=workflow_id, node_id="wf|start", op="start", start=True
        )
        leaf = _wf_node(workflow_id=workflow_id, node_id="wf|leaf", op="leaf")
        workflow_engine.add_node(start)
        workflow_engine.add_node(leaf)
        workflow_engine.add_edge(
            _wf_edge(
                workflow_id=workflow_id,
                edge_id="wf|start->leaf",
                src=start.id,
                dst=leaf.id,
            )
        )

        def resolve_step(op: str):
            def _fn(ctx):
                return RunSuccess(
                    conversation_node_id=None, state_update=[("a", {"last_op": op})]
                )

            return _fn

        runtime = WorkflowRuntime(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            step_resolver=resolve_step,
            predicate_registry={},
            checkpoint_every_n_steps=1,
            max_workers=1,
        )
        result = runtime.run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id="turn-leaf",
            initial_state={},
        )

        assert result.status == "succeeded"
        completed = conversation_engine.get_nodes(
            where={
                "$and": [
                    {"entity_type": "workflow_completed"},
                    {"run_id": result.run_id},
                ]
            },
            limit=10,
        )
        assert len(completed) == 1
        assert isinstance(completed[0], WorkflowCompletedNode)
        meta = completed[0].metadata or {}
        assert meta.get("last_processed_node_id") == f"wf_step|{result.run_id}|1"
    finally:
        shutil.rmtree(root, ignore_errors=True)
