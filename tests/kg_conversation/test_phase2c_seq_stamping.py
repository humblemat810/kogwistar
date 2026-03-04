
import pytest

# These imports assume the repo layout: graph_knowledge_engine/*.py
from graph_knowledge_engine.conversation.conversation_orchestrator import ConversationOrchestrator

class FakeConversationEngine:
    def __init__(self):
        self.kg_graph_type = "conversation"
        self.llm = None
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


def test_seq_stamping_on_turn_node_and_chain_edge():
    eng = FakeConversationEngine()
    kg = FakeKnowledgeEngine()

    orch = ConversationOrchestrator(
        conversation_engine=eng,
        ref_knowledge_engine=kg,
        tool_call_id_factory=lambda *a, **k: "tool_call_id",
        llm=None,
    )

    # first turn (no prev tail) -> should stamp node
    r1 = orch.add_conversation_turn(
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
    n1 = eng.added_nodes[-1]
    assert n1.metadata["run_id"] == "c1"
    assert n1.metadata["run_step_seq"] == 1
    assert n1.metadata["attempt_seq"] == 0

    # second turn -> stamps node and creates next_turn edge
    r2 = orch.add_conversation_turn(
        user_id="u1",
        conversation_id="c1",
        turn_id="t2",
        mem_id="m1",
        role="user",
        content="world",
        filtering_callback=_noop_filtering_callback,
        add_turn_only=True,
    )

    n2 = eng.added_nodes[-1]
    assert n2.metadata["run_id"] == "c1"
    assert n2.metadata["run_step_seq"] == 2
    assert n2.metadata["attempt_seq"] == 0

    assert eng.added_edges, "expected a next_turn edge to be added on second turn"
    e = eng.added_edges[-1]
    assert e.relation == "next_turn"
    assert e.metadata["run_id"] == "c1"
    assert e.metadata["run_step_seq"] == 2
    assert e.metadata["attempt_seq"] == 0


def test_backcompat_missing_run_step_seq_defaults_to_zero():
    eng = FakeConversationEngine()
    kg = FakeKnowledgeEngine()
    orch = ConversationOrchestrator(
        conversation_engine=eng,
        ref_knowledge_engine=kg,
        tool_call_id_factory=lambda *a, **k: "tool_call_id",
        llm=None,
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
