
from __future__ import annotations

import pytest
from types import SimpleNamespace
from typing import Any, Type, TypeVar, cast

from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from conversation.models import ConversationNode, MetaFromLastSummary, RetrievalResult
from graph_knowledge_engine.id_provider import stable_id
from engine_core.models import Span, Grounding
from graph_knowledge_engine.conversation.conversation_orchestrator import ConversationOrchestrator, get_id_for_conversation_turn
from conversation.agentic_answering import AgentConfig, AgenticAnsweringAgent, AnswerWithCitations, AnswerEvaluation

from tests.conftest import _make_engine_pair  # reuse canonical engine fixture builder

from tests._helpers.conv_view import extract_conv_view, assert_views_equivalent
from tests._helpers.runners import run_v1_scenario, run_v2_scenario

BaseM = TypeVar("BaseM", bound=BaseModel)


class DummyLLM(Runnable):
    model_name = "dummy-llm"

    def invoke(self, input, config=None):
        return SimpleNamespace(content="dummy summary")


dummy_llm = cast(BaseChatModel, DummyLLM())


def _mk_span(doc_id: str) -> Span:
    sp = Span.from_dummy_for_conversation()
    sp.doc_id = doc_id
    return sp


# Avoid touching KG retrieval/materialization
def _retrieve_candidates_stub(q: str):
    return []

def _materialize_evidence_pack_stub(
    agent: "AgenticAnsweringAgent",
    *,
    node_ids: list[str],
    edge_ids: list[str] | None = None,
    depth: str,
    max_chars_per_item: int,
    max_total_chars: int,
) -> dict[str, Any]:
    return {}

def _generate_answer_with_citations_stub(
    agent: "AgenticAnsweringAgent",
    *,
    system_prompt: str,
    question: str,
    evidence_pack: dict[str, Any],
    used_node_ids: list[str],
    out_model_schema: dict[str, Any],
    out_model: Type[BaseM]
):
    # keep signature? ideally match your real one too, but this one may not be inspected by joblib
    return AnswerWithCitations(
        text="ok",
        reasoning="test case reasoning.",
        claims=[],
    ).model_dump()

def _validate_or_repair_citations_stub(
    agent: "AgenticAnsweringAgent",
    *,
    system_prompt: str,
    question: str,
    evidence_pack: dict[str, Any],
    used_node_ids: list[str],
    answer: dict| list,
    answer_in_model : Type[AnswerWithCitations]
):
    return AnswerWithCitations(
        text="ok",
        reasoning="test case reasoning.",
        claims=[],
    ).model_dump()

def _evaluate_answer_stub(
    agent: AgenticAnsweringAgent,
    *,
    system_prompt: str,
    question: str,
    answer_text: str,
    used_node_ids: list[str],
    evidence_pack: dict,
    out_model_schema: dict,
    out_model,
    ):
    return AnswerEvaluation(is_sufficient=True, needs_more_info = False, 
                        missing_aspects = [], 
                        notes = 'this test example is simple and good.').model_dump()
    # return {"score": 1, "issues": []}    

def scenario_summary_snapshot(*, backend_kind: str, tmp_path, sa_engine, pg_schema, monkeypatch) -> tuple[Any, str]:
    kg, conv = _make_engine_pair(backend_kind=backend_kind, tmp_path=tmp_path, sa_engine=sa_engine, pg_schema=pg_schema, dim=3, use_fake=True)
    conv.tool_call_id_factory = stable_id
    orch = ConversationOrchestrator(conversation_engine=conv, ref_knowledge_engine=kg, llm=dummy_llm, tool_call_id_factory=stable_id)

    conversation_id = "phase2d_c1"
    user_id = "u1"

    # Seed 3 turns
    for idx, txt in enumerate(["hi", "tell me X", "ok thanks"], start=0):
        nid = get_id_for_conversation_turn(
            ConversationNode.id_kind,
            user_id,
            conversation_id,
            txt,
            str(idx),
            "user",
            "conversation_turn",
            "True",
        )
        n = ConversationNode(
            id=nid,
            label=f"Turn {idx}",
            type="entity",
            summary=txt,
            role="user",  # type: ignore
            conversation_id=conversation_id,
            turn_index=idx,
            properties={"content": txt},
            metadata={"entity_type": "conversation_turn", "in_conversation_chain": True, "in_ui_chain": True, "level_from_root": 0},
            mentions=[Grounding(spans=[_mk_span(f"test:conv:{conversation_id}")])],
            domain_id=None,
            canonical_entity_id=None,
        )
        conv.add_node(n)

    prev = MetaFromLastSummary(prev_node_char_distance_from_last_summary=0, prev_node_distance_from_last_summary=0, tail_turn_index=2)
    orch._summarize_conversation_batch(conversation_id=conversation_id, current_index=2, batch_size=3, in_conv=True, user_id=user_id, prev_turn_meta_summary=prev)
    return conv, conversation_id


def scenario_answer_flow_snapshots(*, backend_kind: str, tmp_path, sa_engine, pg_schema, monkeypatch) -> tuple[Any, str]:
    kg, conv = _make_engine_pair(backend_kind=backend_kind, tmp_path=tmp_path, sa_engine=sa_engine, pg_schema=pg_schema, dim=3, use_fake=True)
    conv.tool_call_id_factory = stable_id

    conversation_id = "phase2d_c2"
    user_id = "u2"

    nid = get_id_for_conversation_turn(
        ConversationNode.id_kind,
        user_id,
        conversation_id,
        "What is chlorophyll?",
        "0",
        "user",
        "conversation_turn",
        "True",
    )
    conv.add_node(
        ConversationNode(
            id=nid,
            label="Turn 0",
            type="entity",
            summary="What is chlorophyll?",
            role="user",  # type: ignore
            conversation_id=conversation_id,
            turn_index=0,
            properties={"content": "What is chlorophyll?"},
            metadata={"entity_type": "conversation_turn", "in_conversation_chain": True, "in_ui_chain": True, "level_from_root": 0},
            mentions=[Grounding(spans=[_mk_span(f"test:conv:{conversation_id}")])],
            domain_id=None,
            canonical_entity_id=None,
        )
    )

    cfg = AgentConfig(max_iter=1, evidence_selector="bm25", max_used=0)
    agent = AgenticAnsweringAgent(conversation_engine=conv, knowledge_engine=kg, llm=dummy_llm, config=cfg)

    # Stubs to avoid KG retrieval/materialization
    monkeypatch.setattr(agent, "_retrieve_candidates", _retrieve_candidates_stub)
    monkeypatch.setattr(agent, "_materialize_evidence_pack", _materialize_evidence_pack_stub)
    monkeypatch.setattr(agent, "_generate_answer_with_citations", _generate_answer_with_citations_stub)
    monkeypatch.setattr(agent, "_validate_or_repair_citations", _validate_or_repair_citations_stub)
    monkeypatch.setattr(agent, "_evaluate_answer", _evaluate_answer_stub)

    prev = MetaFromLastSummary(prev_node_char_distance_from_last_summary=0, prev_node_distance_from_last_summary=0, tail_turn_index=0)
    agent.answer(conversation_id=conversation_id, user_id=user_id, prev_turn_meta_summary=prev)
    return conv, conversation_id


@pytest.mark.parametrize("backend_kind", ["chroma", "pg"])
def test_phase2d_parity_harness_summary_snapshot(backend_kind, tmp_path, sa_engine, pg_schema, monkeypatch):
    conv_a, cid_a = run_v1_scenario(scenario_summary_snapshot, backend_kind=backend_kind, tmp_path=tmp_path, sa_engine=sa_engine, pg_schema=pg_schema, monkeypatch=monkeypatch)
    conv_b, cid_b = run_v2_scenario(scenario_summary_snapshot, backend_kind=backend_kind, tmp_path=tmp_path, sa_engine=sa_engine, pg_schema=pg_schema, monkeypatch=monkeypatch)

    view_a = extract_conv_view(conv_a, conversation_id=cid_a)
    view_b = extract_conv_view(conv_b, conversation_id=cid_b)
    assert_views_equivalent(view_a, view_b)


@pytest.mark.parametrize("backend_kind", ["chroma", "pg"])
def test_phase2d_parity_harness_answer_flow_snapshots(backend_kind, tmp_path, sa_engine, pg_schema, monkeypatch):
    conv_a, cid_a = run_v1_scenario(scenario_answer_flow_snapshots, backend_kind=backend_kind, tmp_path=tmp_path, sa_engine=sa_engine, pg_schema=pg_schema, monkeypatch=monkeypatch)
    conv_b, cid_b = run_v2_scenario(scenario_answer_flow_snapshots, backend_kind=backend_kind, tmp_path=tmp_path, sa_engine=sa_engine, pg_schema=pg_schema, monkeypatch=monkeypatch)

    view_a = extract_conv_view(conv_a, conversation_id=cid_a)
    view_b = extract_conv_view(conv_b, conversation_id=cid_b)
    assert_views_equivalent(view_a, view_b)

def scenario_backbone_only(*, mode: str, backend_kind: str, tmp_path, sa_engine, pg_schema, monkeypatch) -> tuple[Any, str]:
    kg, conv = _make_engine_pair(backend_kind=backend_kind, tmp_path=tmp_path, sa_engine=sa_engine, pg_schema=pg_schema, dim=3, use_fake=True)
    conv.tool_call_id_factory = stable_id

    conversation_id = "phase2d_backbone_c1"
    user_id = "u1"

    orch = ConversationOrchestrator(
        conversation_engine=conv,
        ref_knowledge_engine=kg,
        workflow_engine=conv,  # exercise v2 workflow path without needing a separate engine in harness
        llm=dummy_llm,
        tool_call_id_factory=stable_id,
    )

    if mode == "v1":
        orch.add_conversation_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            turn_id="t1",
            mem_id="m1",
            role="user",  # type: ignore
            content="hello",
            filtering_callback=lambda **kw: (RetrievalResult(nodes=[], edges=[])),
            add_turn_only=True,
        )
        return conv, "v1"
    else:
        orch.add_conversation_turn_workflow_v2(
            run_id="harnass",
            user_id=user_id,
            conversation_id=conversation_id,
            turn_id="t1",
            mem_id="m1",
            role="user",  # type: ignore
            content="hello",
            filtering_callback=lambda **kw: (RetrievalResult(nodes=[], edges=[])),
            workflow_id="phase2d_backbone",
            add_turn_only=True,
            max_workers=1,
        )
        return conv, "v2"


def test_phase2d_parity_harness_backbone_only(backend_kind, tmp_path, sa_engine, pg_schema, monkeypatch):
    conv_v1, _ = run_v1_scenario(
        scenario_backbone_only,
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        monkeypatch=monkeypatch,
    )
    conv_v2, _ = run_v2_scenario(
        scenario_backbone_only,
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        monkeypatch=monkeypatch,
    )

    view1 = extract_conv_view(conv_v1, conversation_id="phase2d_backbone_c1")
    view2 = extract_conv_view(conv_v2, conversation_id="phase2d_backbone_c1")
    assert_views_equivalent(view1, view2)
