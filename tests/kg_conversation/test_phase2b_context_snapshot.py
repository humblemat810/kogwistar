import pytest
from typing import Any, Type, TypeVar
from pydantic import BaseModel

from graph_knowledge_engine.conversation.conversation_context import (
    ContextItem,
    ContextMessage,
    PromptContext,
)
from graph_knowledge_engine.conversation.models import (
    ConversationNode,
    MetaFromLastSummary,
)
from graph_knowledge_engine.conversation.service import ConversationService
from graph_knowledge_engine.id_provider import stable_id

BaseM = TypeVar("BaseM", bound=BaseModel)
from graph_knowledge_engine.engine_core.models import Span, Grounding
from graph_knowledge_engine.conversation.conversation_orchestrator import (
    ConversationOrchestrator,
    get_id_for_conversation_turn,
)
from graph_knowledge_engine.conversation.agentic_answering import (
    AgentConfig,
    AgenticAnsweringAgent,
    AnswerWithCitations,
    AnswerEvaluation,
)

from tests.conftest import _make_engine_pair


def _mk_span(doc_id: str) -> Span:
    sp = Span.from_dummy_for_conversation()
    sp.doc_id = doc_id
    return sp


@pytest.mark.parametrize("backend_kind", ["chroma", "pg"])
def test_summary_creates_context_snapshot_before_llm_call(
    backend_kind, tmp_path, sa_engine, pg_schema, monkeypatch
):
    kg, conv = _make_engine_pair(
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        dim=3,
        use_fake=True,
    )
    conv.tool_call_id_factory = stable_id
    orch = ConversationOrchestrator(
        conversation_engine=conv,
        ref_knowledge_engine=kg,
        tool_call_id_factory=stable_id,
    )

    conversation_id = "c1"
    user_id = "u1"

    # Create 3 conversation turns so summarization has something to work with
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
            metadata={
                "entity_type": "conversation_turn",
                "in_conversation_chain": True,
                "in_ui_chain": True,
                "level_from_root": 0,
            },
            mentions=[Grounding(spans=[_mk_span(f"test:conv:{conversation_id}")])],
            domain_id=None,
            canonical_entity_id=None,
        )
        conv.add_node(n)

    prev = MetaFromLastSummary(
        prev_node_char_distance_from_last_summary=0,
        prev_node_distance_from_last_summary=0,
        tail_turn_index=2,
    )

    # Run summarization directly (private helper)
    orch._summarize_conversation_batch(
        conversation_id=conversation_id,
        current_index=2,
        batch_size=3,
        in_conv=True,
        user_id=user_id,
        prev_turn_meta_summary=prev,
    )

    snaps = conv.get_nodes(where={"entity_type": "context_snapshot"})
    assert snaps, "expected at least one context_snapshot node for summary"

    snap = snaps[-1]
    assert snap.metadata.get("in_conversation_chain") is False
    assert snap.metadata.get("in_ui_chain") is False
    assert snap.metadata.get("stage") in (
        None,
        "summary",
    )  # stage may be stored in metadata; tolerate older layouts
    assert int(snap.metadata.get("run_step_seq", 0)) >= 0

    # cost should be present on snapshot, not on the turn nodes
    assert "cost.char_count" in snap.metadata
    assert "cost.token_count" in snap.metadata

    turns = conv.get_nodes(where={"entity_type": "conversation_turn"})
    assert all("cost.char_count" not in (t.metadata or {}) for t in turns), (
        "accounting must not be stored on turn nodes"
    )


@pytest.mark.parametrize("backend_kind", ["chroma", "pg"])
def test_answer_flow_creates_context_snapshots_and_edges(
    backend_kind, tmp_path, sa_engine, pg_schema, monkeypatch
):
    kg, conv = _make_engine_pair(
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        dim=3,
        use_fake=True,
    )

    conversation_id = "c2"
    user_id = "u2"

    # Put a single user turn so the agent has a question
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
            metadata={
                "entity_type": "conversation_turn",
                "in_conversation_chain": True,
                "in_ui_chain": True,
                "level_from_root": 0,
            },
            mentions=[Grounding(spans=[_mk_span(f"test:conv:{conversation_id}")])],
            domain_id=None,
            canonical_entity_id=None,
        )
    )

    cfg = AgentConfig(
        max_iter=1,
        evidence_selector="bm25",  # avoid LLM call for selection; we still snapshot for answer/eval stages
        max_used=0,
    )
    agent = AgenticAnsweringAgent(
        conversation_engine=conv,
        knowledge_engine=kg,
        llm_tasks=conv.llm_tasks,
        config=cfg,
    )

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
        out_model: Type[BaseM],
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
        answer: dict | list,
        answer_in_model: Type[AnswerWithCitations],
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
        return AnswerEvaluation(
            is_sufficient=True,
            needs_more_info=False,
            missing_aspects=[],
            notes="this test example is simple and good.",
        ).model_dump()
        # return {"score": 1, "issues": []}

    monkeypatch.setattr(agent, "_retrieve_candidates", _retrieve_candidates_stub)
    monkeypatch.setattr(
        agent, "_materialize_evidence_pack", _materialize_evidence_pack_stub
    )
    monkeypatch.setattr(
        agent, "_generate_answer_with_citations", _generate_answer_with_citations_stub
    )
    monkeypatch.setattr(
        agent, "_validate_or_repair_citations", _validate_or_repair_citations_stub
    )
    monkeypatch.setattr(agent, "_evaluate_answer", _evaluate_answer_stub)

    prev = MetaFromLastSummary(
        prev_node_char_distance_from_last_summary=0,
        prev_node_distance_from_last_summary=0,
        tail_turn_index=0,
    )
    out = agent.answer(
        conversation_id=conversation_id, user_id=user_id, prev_turn_meta_summary=prev
    )

    snaps = conv.get_nodes(where={"entity_type": "context_snapshot"})
    assert snaps, "expected context_snapshot nodes during answer flow"

    # Each snapshot must not be in either chain
    for s in snaps:
        assert s.metadata.get("in_conversation_chain") is False
        assert s.metadata.get("in_ui_chain") is False
        assert "cost.char_count" in s.metadata
        assert "cost.token_count" in s.metadata

    # Edges must exist from each snapshot to at least the user turn (depends_on)
    # (some snapshots may include only conversation items; ensure >=1 depends_on edge total)
    dep_edges = conv.get_edges(where={"relation": "depends_on"})
    assert dep_edges, "expected depends_on edges from snapshots"

    # Accounting must not appear on arbitrary conversation turns
    turns = conv.get_nodes(where={"entity_type": "conversation_turn"})
    assert all("cost.char_count" not in (t.metadata or {}) for t in turns)


@pytest.mark.parametrize("backend_kind", ["chroma", "pg"])
def test_persist_context_snapshot_allows_empty_used_node_ids(
    backend_kind, tmp_path, sa_engine, pg_schema
):
    kg, conv = _make_engine_pair(
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        dim=3,
        use_fake=True,
    )
    svc = ConversationService.from_engine(conversation_engine=conv, knowledge_engine=kg)

    view = PromptContext(
        conversation_id="c-empty-snapshot",
        purpose="answer",
        token_budget=16,
        tokens_used=1,
        items=(
            ContextItem(
                kind="system_prompt",
                role="system",
                text="Only a system prompt.",
                source="system",
                token_cost=1,
            ),
        ),
        messages=(
            ContextMessage(
                role="system",
                content="Only a system prompt.",
                source="system",
            ),
        ),
    )

    snapshot_id = svc.persist_context_snapshot(
        conversation_id="c-empty-snapshot",
        run_id="run-empty",
        run_step_seq=0,
        stage="draft_answer",
        view=view,
        model_name="fake-model",
        budget_tokens=16,
    )

    snaps = conv.backend.node_get(ids=[snapshot_id], include=["metadatas"])
    assert snaps["ids"] == [snapshot_id]
    meta = (snaps.get("metadatas") or [{}])[0] or {}
    assert meta.get("entity_type") == "context_snapshot"
    assert "used_node_ids" not in meta
