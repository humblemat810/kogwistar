from __future__ import annotations

from dataclasses import dataclass

import pytest

from kogwistar.conversation.models import BaseToolResult, ConversationNode
from kogwistar.conversation.tool_runner import ToolRunner
from kogwistar.engine_core.models import Grounding, MentionVerification, Span
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.runtime import MappingStepResolver, WorkflowRuntime
from kogwistar.runtime.models import (
    RunSuccess,
    WorkflowDesignArtifact,
    WorkflowEdge,
    WorkflowNode,
)
from tests._helpers.embeddings import ConstantEmbeddingFunction
from tests._helpers.fake_backend import build_fake_backend


pytestmark = pytest.mark.conversation


@dataclass(kw_only=True)
class EchoToolResult(BaseToolResult):
    text: str


def _conversation_tail(conversation_id: str, *, turn_index: int = 0) -> ConversationNode:
    return ConversationNode(
        user_id="tester",
        id=f"conv|{conversation_id}|turn|{turn_index}",
        label="Turn",
        type="entity",
        doc_id=f"turn|{turn_index}",
        summary="tail",
        role="user",
        turn_index=turn_index,
        conversation_id=conversation_id,
        mentions=[
            Grounding(
                spans=[
                    Span(
                        collection_page_url=f"conversation/{conversation_id}",
                        document_page_url=f"conversation/{conversation_id}",
                        doc_id=f"conv:{conversation_id}",
                        insertion_method="test",
                        page_number=1,
                        start_char=0,
                        end_char=1,
                        excerpt="tail",
                        context_before="",
                        context_after="",
                        chunk_id=None,
                        source_cluster_id=None,
                        verification=MentionVerification(
                            method="human",
                            is_verified=True,
                            score=1.0,
                            notes="test",
                        ),
                    )
                ]
            )
        ],
        properties={},
        metadata={
            "entity_type": "conversation_turn",
            "conversation_id": conversation_id,
            "in_conversation_chain": True,
            "level_from_root": 0,
        },
        domain_id=None,
        canonical_entity_id=None,
    )


def _make_engine_pair(tmp_path):
    emb = ConstantEmbeddingFunction(dim=8)
    conversation_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "conv"),
        kg_graph_type="conversation",
        embedding_function=emb,
        backend_factory=build_fake_backend,
    )
    workflow_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "wf"),
        kg_graph_type="workflow",
        embedding_function=emb,
        backend_factory=build_fake_backend,
    )
    return conversation_engine, workflow_engine


def _tool_runner(conversation_engine: GraphKnowledgeEngine) -> ToolRunner:
    return ToolRunner(
        tool_call_id_factory=lambda *parts: "tool|" + "|".join(map(str, parts)),
        conversation_engine=conversation_engine,
    )


def test_tool_runner_rejects_sync_only_async_handler(tmp_path):
    conversation_engine, _workflow_engine = _make_engine_pair(tmp_path)
    conversation_engine.write.add_node(_conversation_tail("conv-sync-only"))
    runner = _tool_runner(conversation_engine)

    async def _async_tool(*, text: str):
        return EchoToolResult(node_id_entry=None, text=text)

    with pytest.raises(ValueError, match="sync-only"):
        runner.run_tool(
            conversation_id="conv-sync-only",
            user_id="tester",
            turn_node_id="turn-1",
            turn_index=1,
            tool_name="async_tool",
            args=[],
            kwargs={"text": "x"},
            handler=_async_tool,
            prev_turn_meta_summary=_meta(),
            supports_async=False,
        )


def test_tool_runner_supports_async_handler_and_records_receipt(tmp_path):
    conversation_engine, _workflow_engine = _make_engine_pair(tmp_path)
    conversation_engine.write.add_node(_conversation_tail("conv-async"))
    runner = _tool_runner(conversation_engine)

    async def _async_tool(*, text: str):
        return EchoToolResult(node_id_entry=None, text=f"async:{text}")

    result, call_node_id = runner.run_tool(
        conversation_id="conv-async",
        user_id="tester",
        turn_node_id="turn-1",
        turn_index=1,
        tool_name="async_tool",
        args=[],
        kwargs={"text": "ok"},
        handler=_async_tool,
        prev_turn_meta_summary=_meta(),
        render_result=lambda r: r.text,
        tool_kind="pure/query",
        supports_async=True,
    )

    assert call_node_id
    assert result.text == "async:ok"
    assert runner.last_receipt is not None
    assert runner.last_receipt.kind == "pure/query"
    assert runner.last_receipt.execution_mode == "inline"
    assert runner.last_receipt.status == "completed"
    assert runner.last_receipt.side_effects == []


@pytest.mark.parametrize(
    "tool_kind,node_id_entry,expected_side_effect",
    [
        ("pure/query", None, []),
        ("side-effecting", "node|write", ["node|write"]),
        ("long-running", None, []),
        ("human-approval", None, []),
    ],
)
def test_tool_runner_class_kinds_are_preserved(
    tmp_path, tool_kind, node_id_entry, expected_side_effect
):
    conversation_engine, _workflow_engine = _make_engine_pair(tmp_path)
    conversation_engine.write.add_node(_conversation_tail("conv-kind"))
    runner = _tool_runner(conversation_engine)

    def _tool(*, text: str):
        return EchoToolResult(node_id_entry=node_id_entry, text=text)

    result, _call_node_id = runner.run_tool(
        conversation_id="conv-kind",
        user_id="tester",
        turn_node_id="turn-1",
        turn_index=1,
        tool_name="kind_tool",
        args=[],
        kwargs={"text": "ok"},
        handler=_tool,
        prev_turn_meta_summary=_meta(),
        render_result=lambda r: r.text,
        tool_kind=tool_kind,
    )

    assert result.text == "ok"
    assert runner.last_receipt is not None
    assert runner.last_receipt.kind == tool_kind
    assert runner.last_receipt.side_effects == expected_side_effect


def test_tool_runner_run_subworkflow_records_child_process_receipt(tmp_path):
    conversation_engine, workflow_engine = _make_engine_pair(tmp_path)
    conversation_engine.write.add_node(_conversation_tail("conv-child"))
    runner = _tool_runner(conversation_engine)

    parent_id = "wf_parent_tool"
    child_id = "wf_child_tool"
    child_nodes = [
        _wf_node(workflow_id=child_id, node_id=f"wf|{child_id}|start", op="start", start=True),
        _wf_node(workflow_id=child_id, node_id=f"wf|{child_id}|end", op="end", terminal=True),
    ]
    child_edges = [
        _wf_edge(
            workflow_id=child_id,
            edge_id=f"wf|{child_id}|e|start->end",
            src=f"wf|{child_id}|start",
            dst=f"wf|{child_id}|end",
        )
    ]
    child_design = WorkflowDesignArtifact(
        workflow_id=child_id,
        workflow_version="v_test",
        start_node_id=f"wf|{child_id}|start",
        nodes=child_nodes,
        edges=child_edges,
        source_run_id="parent-run",
        source_workflow_id=parent_id,
        source_step_id=f"wf|{parent_id}|spawn",
        notes="child workflow as tool",
    )

    resolver = MappingStepResolver()

    @resolver.register("start")
    def _start(ctx):
        return RunSuccess(state_update=[("u", {"started": True})])

    @resolver.register("end")
    def _end(ctx):
        return RunSuccess(state_update=[("u", {"ended": True})])

    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
    )

    def _run_child_workflow():
        return runtime.run_subworkflow(
            workflow_id=child_id,
            parent_state={"_deps": {}, "seed": "present"},
            conversation_id="conv-child",
            turn_node_id="turn-1",
            parent_run_id="parent-run",
            result_state_key="child_result",
            workflow_design=child_design,
            initial_state={"seed": "present"},
            run_id="child-run",
        )

    result, _call_node_id = runner.run_subworkflow_tool(
        conversation_id="conv-child",
        user_id="tester",
        turn_node_id="turn-1",
        turn_index=1,
        tool_name="spawn_subworkflow",
        args=[],
        kwargs={},
        subworkflow_runner=lambda **_kw: _run_child_workflow(),
        prev_turn_meta_summary=_meta(),
        render_result=lambda r: str(getattr(r, "status", "")),
    )

    assert getattr(result, "status", None) == "succeeded"
    assert runner.last_receipt is not None
    assert runner.last_receipt.kind == "workflow/subworkflow"
    assert runner.last_receipt.execution_mode == "child-process"


def _meta():
    from kogwistar.conversation.models import MetaFromLastSummary

    return MetaFromLastSummary(
        prev_node_char_distance_from_last_summary=0,
        prev_node_distance_from_last_summary=0,
        tail_turn_index=0,
    )


def _wf_node(*, workflow_id: str, node_id: str, op: str | None, start: bool = False, terminal: bool = False):
    return WorkflowNode(
        id=node_id,
        label=node_id.split("|")[-1],
        type="entity",
        doc_id=node_id,
        summary=op or node_id,
        properties={},
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": op,
            "wf_start": bool(start),
            "wf_terminal": bool(terminal),
            "wf_version": "v_test",
        },
        mentions=[
            Grounding(
                spans=[
                    Span.from_dummy_for_workflow(workflow_id),
                ]
            )
        ],
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )


def _wf_edge(*, workflow_id: str, edge_id: str, src: str, dst: str):
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
            "wf_predicate": None,
            "wf_priority": 100,
            "wf_is_default": True,
            "wf_multiplicity": "one",
            "wf_version": "v_test",
        },
        mentions=[
            Grounding(
                spans=[
                    Span.from_dummy_for_workflow(workflow_id),
                ]
            )
        ],
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )
