
import pytest

from graph_knowledge_engine.conversation.agentic_answering import AgenticAnsweringAgent, AgentConfig, AnswerEvaluation, AnswerWithCitations, EvidenceSelection
from graph_knowledge_engine.id_provider import stable_id
from graph_knowledge_engine.conversation.models import MetaFromLastSummary
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable


class _FixedStructuredRunnable(Runnable):
    """Return a fixed structured-output payload in LangChain include_raw format."""

    def __init__(self, parsed):
        self._parsed = parsed

    def invoke(self, input, config=None, **kwargs):
        return {"raw": None, "parsed": self._parsed, "parsing_error": None}

    async def ainvoke(self, input, config=None, **kwargs):
        return self.invoke(input, config=config, **kwargs)


class _FixedLLM(BaseChatModel):
    """Minimal BaseChatModel stub for agentic-answering tests.

    It only needs `.with_structured_output(..., include_raw=True)` for the agentic pipeline.
    """

    model_name: str = "fixed-llm"

    def __init__(self, *, selection, answer, evaluation: AnswerEvaluation):
        super().__init__()
        self._selection = selection
        self._answer = answer
        self._evaluation = evaluation

    @property
    def _llm_type(self) -> str:
        return "fixed"

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        # Route based on schema class name to keep this test robust to import path differences.
        name = getattr(schema, "__name__", str(schema))
        if "EvidenceSelection" in name:
            return _FixedStructuredRunnable(self._selection)
        if "AnswerWithCitations" in name:
            return _FixedStructuredRunnable(self._answer)
        if "AnswerEvaluation" in name:
            return _FixedStructuredRunnable(self._evaluation)
        # Fallback: empty dict
        return _FixedStructuredRunnable({})

    # BaseChatModel abstract methods (not used in this test)
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # pragma: no cover
        raise NotImplementedError("_generate not used; this stub is structured-output only")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # pragma: no cover
        raise NotImplementedError("_agenerate not used; this stub is structured-output only")


def _ensure_user_turn(conversation_engine, *, user_id: str, conversation_id: str, text: str):
    """Create a tail user turn in the conversation graph.

    Prefer the public engine facade if available.
    """
    # In your repo, engine exposes `create_conversation` and `add_conversation_turn`.
    # This helper uses them to avoid reaching into collections directly.
    turn_id = "turn_0"
    mem_id = "mem_0"
    mts = MetaFromLastSummary(0, 0)
    conversation_engine.add_conversation_turn(
        user_id=user_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        mem_id=mem_id,
        role="user",
        content=text,
        ref_knowledge_engine=conversation_engine,  # not used if add_turn_only=True
        prev_turn_meta_summary=mts,
        add_turn_only=True,
    )


@pytest.mark.parametrize("question", ["What is chlorophyll?"])
def test_answer_workflow_v2_runs_end_to_end(workflow_engine, conversation_engine, engine, question):
    # Create a conversation with at least one user turn (tail must exist).
    conversation_engine.tool_call_id_factory=stable_id
    engine.tool_call_id_factory=stable_id
    workflow_engine.tool_call_id_factory=stable_id
    user_id = "u_test"
    import time
    conversation_id, _start_id = conversation_engine.create_conversation(user_id=user_id)
    time.sleep(0.5)
    _ensure_user_turn(conversation_engine, user_id=user_id, conversation_id=conversation_id, text=question)
    time.sleep(0.5)
    # Fixed outputs for each structured step.
    selection = EvidenceSelection(used_node_ids = [], used_edge_ids = [], reasoning="test case is always simple and good.")
    answer = AnswerWithCitations(text = "dummy", reasoning = "testing answer no claim needed", claims= [])
    evaluation = AnswerEvaluation(is_sufficient=True, needs_more_info = False, 
                         missing_aspects = [], 
                         notes = 'this test example is simple and good.')

    llm = _FixedLLM(selection=selection, answer=answer, evaluation=evaluation)

    agent = AgenticAnsweringAgent(
        conversation_engine=conversation_engine,
        knowledge_engine=engine,
        llm=llm,
        config=AgentConfig(max_iter=1, max_candidates=5),
    )

    assert hasattr(agent, "answer_workflow_v2"), "AgenticAnsweringAgent must expose answer_workflow_v2()"

    mts = MetaFromLastSummary(0, 0)
    out = agent.answer_workflow_v2(workflow_engine=workflow_engine, 
                                   conversation_id=conversation_id, 
                                   user_id=user_id, 
                                   prev_turn_meta_summary=mts)

    # Minimal end-to-end assertions
    assert isinstance(out, dict)
    assert out.get("assistant_text") == "dummy"
    assert "run_id" in out or "run_node_id" in out or "agent_run" in out
