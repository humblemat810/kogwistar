
import pytest

# These imports assume the repo layout: graph_knowledge_engine/*.py
from graph_knowledge_engine.conversation.conversation_orchestrator import ConversationOrchestrator
from graph_knowledge_engine.llm_tasks import (
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


def _dummy_task_set() -> LLMTaskSet:
    return LLMTaskSet(
        extract_graph=lambda _req: ExtractGraphTaskResult(raw=None, parsed_payload={"nodes": [], "edges": []}, parsing_error=None),
        adjudicate_pair=lambda _req: AdjudicatePairTaskResult(verdict_payload={"same_entity": False}, raw=None, parsing_error=None),
        adjudicate_batch=lambda _req: AdjudicateBatchTaskResult(verdict_payloads=(), raw=None, parsing_error=None),
        filter_candidates=lambda _req: FilterCandidatesTaskResult(node_ids=(), edge_ids=(), reasoning="", raw=None, parsing_error=None),
        summarize_context=lambda _req: SummarizeContextTaskResult(text=""),
        answer_with_citations=lambda _req: AnswerWithCitationsTaskResult(answer_payload={"text": "", "reasoning": "", "claims": []}, raw=None, parsing_error=None),
        repair_citations=lambda _req: RepairCitationsTaskResult(answer_payload={"text": "", "reasoning": "", "claims": []}, raw=None, parsing_error=None),
        provider_hints=LLMTaskProviderHints(),
    )

class FakeConversationEngine:
    def __init__(self):
        self.kg_graph_type = "conversation"
        self.llm_tasks = _dummy_task_set()
        self.added_nodes = []
        self.added_edges = []
        self._tail = None

    def _iterative_defensive_emb(self, text: str):
        # deterministic dummy embedding
        return [0.0, 0.0, 0.0]

    def add_node(self, node, _):
        self.added_nodes.append(node)
        self._tail = node

    def add_edge(self, edge):
        self.added_edges.append(edge)

    def _get_conversation_tail(self, conversation_id: str):
        return self._tail


class FakeKnowledgeEngine:
    pass


def _noop_filtering_callback(*args, **kwargs):
    # not used when add_turn_only=True
    return (None, "")

# TO-DO stamp seq may need new test

def test_backcompat_missing_run_step_seq_defaults_to_zero():
    eng = FakeConversationEngine()
    kg = FakeKnowledgeEngine()
    orch = ConversationOrchestrator(
        conversation_engine=eng,
        ref_knowledge_engine=kg,
        tool_call_id_factory=lambda *a, **k: "tool_call_id",
    )

    # Seed a tail node without run_step_seq (simulates old graphs)
    class Tail:
        def __init__(self):
            self.id = "tail"
            self.turn_index = 0
            self.summary = "seed"
            self.metadata = {"entity_type": "conversation_turn"}  # no run_step_seq
    eng._tail = Tail()

    orch.add_conversation_turn(
        user_id="u1",
        conversation_id="c1",
        turn_id="t2",
        mem_id="m1",
        role="user",
        content="new",
        filtering_callback=_noop_filtering_callback,
        add_turn_only=True,
    )

    n = eng.added_nodes[-1]
    # prev inferred 0 -> bump -> 1
    assert n.metadata["run_step_seq"] == 1
