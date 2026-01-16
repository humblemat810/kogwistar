from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.conversation_orchestrator import ConversationOrchestrator


def test_orchestrator_has_v2(tmp_path):
    conv = GraphKnowledgeEngine(persist_directory=str(tmp_path / "conv"), kg_graph_type="conversation")
    kg = GraphKnowledgeEngine(persist_directory=str(tmp_path / "kg"), kg_graph_type="knowledge")
    wf = GraphKnowledgeEngine(persist_directory=str(tmp_path / "wf"), kg_graph_type="workflow")

    # NOTE: adapt args to your orchestrator's real __init__ signature
    orch = ConversationOrchestrator(
        conversation_engine=conv,
        ref_knowledge_engine=kg,
        workflow_engine=wf,
        # llm=..., tool_runner=..., etc.
    )

    assert hasattr(orch, "add_conversation_turn_workflow_v2")
