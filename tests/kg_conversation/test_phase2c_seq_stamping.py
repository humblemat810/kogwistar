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


def _node_field(node, key: str):
    if hasattr(node, key):
        return getattr(node, key)
    return (getattr(node, "metadata", None) or {}).get(key)


def _matches_where(node, where: dict | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_matches_where(node, clause) for clause in where["$and"])
    for key, expected in where.items():
        actual = _node_field(node, key)
        if isinstance(expected, dict):
            if "$gte" in expected and not (actual is not None and actual >= expected["$gte"]):
                return False
            continue
        if actual != expected:
            return False
    return True


class FakeBackend:
    def __init__(self, engine):
        self._engine = engine

    def node_get(self, where=None, ids=None, include=None):
        nodes = self._engine.nodes
        if ids is not None:
            selected = [node for node in nodes if node.id in ids]
        else:
            selected = [node for node in nodes if _matches_where(node, where)]
        return {
            "ids": [node.id for node in selected],
            "documents": [getattr(node, "summary", None) for node in selected],
            "metadatas": [getattr(node, "metadata", None) for node in selected],
            "embeddings": [getattr(node, "embedding", None) for node in selected],
            "objects": selected,
        }


class FakeRead:
    def nodes_from_single_or_id_query_result(self, got, node_type=None):
        return list(got.get("objects") or [])


class FakeConversationEngine:
    def __init__(self):
        self.kg_graph_type = "conversation"
        self.llm_tasks = _dummy_task_set()
        self.added_nodes = []
        self.added_edges = []
        self.nodes = []
        self._tail = None
        self.backend = FakeBackend(self)
        self.read = FakeRead()

    def _iterative_defensive_emb(self, text: str):
        # deterministic dummy embedding
        return [0.0, 0.0, 0.0]

    def add_node(self, node, _=None):
        self.added_nodes.append(node)
        self.nodes.append(node)
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


def test_seq_stamping_on_turn_node_and_chain_edge():
    eng = FakeConversationEngine()
    kg = FakeKnowledgeEngine()

    orch = ConversationOrchestrator(
        conversation_engine=eng,
        ref_knowledge_engine=kg,
        tool_call_id_factory=lambda *a, **k: "tool_call_id",
    )

    orch.add_conversation_turn(
        user_id="u1",
        conversation_id="c1",
        turn_id="t1",
        mem_id="m1",
        role="user",
        content="hello",
        filtering_callback=_noop_filtering_callback,
        add_turn_only=True,
    )

    assert eng.added_nodes, "expected a turn node to be added"
    first_node = eng.added_nodes[-1]
    assert first_node.metadata["run_id"] == "c1"
    assert first_node.metadata["run_step_seq"] == 1
    assert first_node.metadata["attempt_seq"] == 0

    orch.add_conversation_turn(
        user_id="u1",
        conversation_id="c1",
        turn_id="t2",
        mem_id="m1",
        role="user",
        content="world",
        filtering_callback=_noop_filtering_callback,
        add_turn_only=True,
    )

    second_node = eng.added_nodes[-1]
    assert second_node.metadata["run_id"] == "c1"
    assert second_node.metadata["run_step_seq"] == 2
    assert second_node.metadata["attempt_seq"] == 0

    assert eng.added_edges, "expected a next_turn edge to be added on second turn"
    seq_edge = eng.added_edges[-1]
    assert seq_edge.relation == "next_turn"
    assert seq_edge.metadata["run_id"] == "c1"
    assert seq_edge.metadata["run_step_seq"] == 2
    assert seq_edge.metadata["attempt_seq"] == 0

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
            self.conversation_id = "c1"
            self.summary = "seed"
            self.metadata = {
                "entity_type": "conversation_turn",
                "in_conversation_chain": True,
            }  # no run_step_seq
    eng.add_node(Tail())

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
