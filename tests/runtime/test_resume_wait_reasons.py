from __future__ import annotations

import json

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, MentionVerification, Span
from kogwistar.runtime.models import RunSuccess, RunSuspended, WorkflowEdge, WorkflowNode
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.runtime.runtime import StepContext, WorkflowRuntime
from tests._helpers.embeddings import ConstantEmbeddingFunction
from tests._helpers.fake_backend import build_fake_backend

pytestmark = [pytest.mark.ci, pytest.mark.runtime]


def _dummy_grounding() -> Grounding:
    sp = Span(
        collection_page_url="demo",
        document_page_url="demo",
        doc_id="demo",
        insertion_method="demo",
        page_number=1,
        start_char=0,
        end_char=1,
        excerpt="x",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="system",
            is_verified=True,
            score=1.0,
            notes="demo",
        ),
    )
    return Grounding(spans=[sp])


def _add_node(
    engine: GraphKnowledgeEngine,
    wf_id: str,
    node_id: str,
    op: str,
    *,
    start: bool = False,
    terminal: bool = False,
) -> None:
    engine.write.add_node(
        WorkflowNode(
            id=node_id,
            label=op,
            type="entity",
            doc_id=node_id,
            summary=op,
            properties={},
            mentions=[_dummy_grounding()],
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": wf_id,
                "wf_op": op,
                "wf_start": start,
                "wf_terminal": terminal,
                "wf_version": "v1",
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
            level_from_root=0,
        )
    )


def _add_edge(engine: GraphKnowledgeEngine, wf_id: str, src: str, dst: str) -> None:
    engine.write.add_edge(
        WorkflowEdge(
            id=f"{src}->{dst}",
            label="wf_next",
            type="entity",
            doc_id=f"{src}->{dst}",
            summary="next",
            properties={},
            source_ids=[src],
            target_ids=[dst],
            source_edge_ids=[],
            target_edge_ids=[],
            relation="wf_next",
            mentions=[_dummy_grounding()],
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": wf_id,
                "wf_predicate": None,
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_multiplicity": "one",
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
            level_from_root=0,
        )
    )


def _latest_checkpoint_state(conv_engine: GraphKnowledgeEngine, run_id: str) -> dict:
    ckpts = conv_engine.read.get_nodes(
        where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]}
    )
    latest = max(ckpts, key=lambda c: int(getattr(c, "metadata", {}).get("step_seq", -1)))
    state_json = latest.metadata.get("state_json", {})
    if isinstance(state_json, str):
        state_json = json.loads(state_json)
    return dict(state_json or {})


def _make_engine(tmp_path, graph_type: str) -> GraphKnowledgeEngine:
    return GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        kg_graph_type=graph_type,
        backend_factory=build_fake_backend,
        embedding_function=ConstantEmbeddingFunction(dim=16),
    )


@pytest.mark.parametrize(
    "wait_reason",
    [
        "approval",
        "message",
        "schedule_delay",
        "external_callback",
        "dependency",
    ],
)
def test_wait_reason_can_suspend_and_resume(tmp_path, wait_reason: str) -> None:
    wf_engine = _make_engine(tmp_path / f"wf_{wait_reason}", "workflow")
    conv_engine = _make_engine(tmp_path / f"conv_{wait_reason}", "conversation")

    wf_id = f"wf_wait_{wait_reason}"
    _add_node(wf_engine, wf_id, "start", "start", start=True)
    _add_node(wf_engine, wf_id, "gate", "gate")
    _add_node(wf_engine, wf_id, "end", "end", terminal=True)
    _add_edge(wf_engine, wf_id, "start", "gate")
    _add_edge(wf_engine, wf_id, "gate", "end")

    resolver = MappingStepResolver()

    @resolver.register("start")
    def _start(ctx: StepContext):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"started": True})])

    @resolver.register("gate")
    def _gate(ctx: StepContext):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            wait_reason=wait_reason,
            resume_payload={
                "type": "recoverable_error",
                "category": wait_reason,
                "message": f"wait for {wait_reason}",
                "errors": [wait_reason],
                "repair_payload": {"wait_reason": wait_reason},
            },
        )

    @resolver.register("end")
    def _end(ctx: StepContext):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"ended": True})])

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )

    run_id = f"run_{wait_reason}"
    conv_id = f"conv_{wait_reason}"
    res1 = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": conv_id,
            "user_id": "test",
            "turn_node_id": "turn_1",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "mem_1",
        },  # type: ignore[arg-type]
        run_id=run_id,
    )
    assert res1.status == "suspended"
    state1 = _latest_checkpoint_state(conv_engine, run_id)
    assert state1["wait_reason"] == wait_reason
    assert state1["_rt_join"]["suspended"][0][0] == "gate"
    suspended_token_id = state1["_rt_join"]["suspended"][0][2]

    res2 = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="gate",
        suspended_token_id=suspended_token_id,
        client_result=RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"resumed": True})],
        ),
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
    )
    assert res2.status == "succeeded"
