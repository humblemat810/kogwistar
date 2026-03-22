import pytest

from kogwistar.conversation.agentic_answering import (
    AgenticAnsweringAgent,
    AgentConfig,
    AnswerEvaluation,
    AnswerWithCitations,
    EvidenceSelection,
)
from kogwistar.id_provider import stable_id
from kogwistar.conversation.models import (
    FilteringResult,
    MetaFromLastSummary,
)
from kogwistar.conversation.service import ConversationService
from kogwistar.llm_tasks import (
    AdjudicateBatchTaskResult,
    AdjudicatePairTaskResult,
    AnswerWithCitationsTaskResult,
    ExtractGraphTaskResult,
    FilterCandidatesTaskResult,
    LLMTaskProviderHints,
    LLMTaskSet,
    RepairCitationsTaskResult,
    SummarizeContextTaskResult,
)

pytestmark = pytest.mark.ci_full


def _fixed_llm_tasks(answer: AnswerWithCitations) -> LLMTaskSet:
    payload = answer.model_dump(mode="python")
    return LLMTaskSet(
        extract_graph=lambda _req: ExtractGraphTaskResult(
            raw=None, parsed_payload=None, parsing_error="unused"
        ),
        adjudicate_pair=lambda _req: AdjudicatePairTaskResult(
            verdict_payload=None, raw=None, parsing_error="unused"
        ),
        adjudicate_batch=lambda _req: AdjudicateBatchTaskResult(
            verdict_payloads=(), raw=None, parsing_error="unused"
        ),
        filter_candidates=lambda _req: FilterCandidatesTaskResult(
            node_ids=(), edge_ids=(), reasoning="", raw=None, parsing_error=None
        ),
        summarize_context=lambda req: SummarizeContextTaskResult(text=req.full_text),
        answer_with_citations=lambda _req: AnswerWithCitationsTaskResult(
            answer_payload=payload, raw=None, parsing_error=None
        ),
        repair_citations=lambda _req: RepairCitationsTaskResult(
            answer_payload=payload, raw=None, parsing_error=None
        ),
        provider_hints=LLMTaskProviderHints(answer_with_citations_provider="custom"),
    )


def _ensure_user_turn(
    conversation_engine, *, user_id: str, conversation_id: str, text: str
):
    """Create a tail user turn in the conversation graph.

    Prefer the public engine facade if available.
    """
    # In your repo, engine exposes `create_conversation` and `add_conversation_turn`.
    # This helper uses them to avoid reaching into collections directly.
    turn_id = "turn_0"
    mem_id = "mem_0"
    mts = MetaFromLastSummary(0, 0)
    svc = ConversationService.from_engine(
        conversation_engine,
        knowledge_engine=conversation_engine,
    )
    svc.add_conversation_turn(
        user_id=user_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        mem_id=mem_id,
        role="user",
        content=text,
        ref_knowledge_engine=conversation_engine,  # not used if add_turn_only=True
        filtering_callback=lambda *_a, **_k: (
            FilteringResult(node_ids=[], edge_ids=[]),
            "noop",
        ),
        prev_turn_meta_summary=mts,
        add_turn_only=True,
    )


@pytest.mark.parametrize("question", ["What is chlorophyll?"])
def test_answer_workflow_v2_runs_end_to_end(
    workflow_engine, conversation_engine, engine, question,
    tmp_path
):
    # Create a conversation with at least one user turn (tail must exist).
    conversation_engine.tool_call_id_factory = stable_id
    engine.tool_call_id_factory = stable_id
    workflow_engine.tool_call_id_factory = stable_id
    user_id = "u_test"
    import time

    conv_svc = ConversationService.from_engine(
        conversation_engine,
        knowledge_engine=engine,
        workflow_engine=workflow_engine,
    )
    conversation_id, _start_id = conv_svc.create_conversation(user_id=user_id)
    time.sleep(0.5)
    _ensure_user_turn(
        conversation_engine,
        user_id=user_id,
        conversation_id=conversation_id,
        text=question,
    )
    time.sleep(0.5)
    # Fixed outputs for each structured step.
    selection = EvidenceSelection(
        used_node_ids=[],
        used_edge_ids=[],
        reasoning="test case is always simple and good.",
    )
    answer = AnswerWithCitations(
        text="dummy", reasoning="testing answer no claim needed", claims=[]
    )
    evaluation = AnswerEvaluation(
        is_sufficient=True,
        needs_more_info=False,
        missing_aspects=[],
        notes="this test example is simple and good.",
    )

    llm_tasks = _fixed_llm_tasks(answer)

    agent = AgenticAnsweringAgent(
        conversation_engine=conversation_engine,
        knowledge_engine=engine,
        llm_tasks=llm_tasks,
        config=AgentConfig(max_iter=1, max_candidates=5),
    )

    assert hasattr(agent, "answer_workflow_v2"), (
        "AgenticAnsweringAgent must expose answer_workflow_v2()"
    )

    mts = MetaFromLastSummary(0, 0)
    out = agent.answer_workflow_v2(
        workflow_engine=workflow_engine,
        conversation_id=conversation_id,
        user_id=user_id,
        prev_turn_meta_summary=mts,
        cache_dir=tmp_path/ "agent_answer_workflow_v2_cache_dir"
    )

    # Minimal end-to-end assertions
    assert isinstance(out, dict)
    assert out.get("assistant_text") == "dummy"
    assert "run_id" in out or "run_node_id" in out or "agent_run" in out
