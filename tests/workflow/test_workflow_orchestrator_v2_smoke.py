from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.conversation_orchestrator import ConversationOrchestrator


def test_workflow_runtime_uses_default_resolver(tmp_path):
    """Smoke: a persisted workflow design can be executed using the package default resolver.

    This mirrors the intent of `test_add_turn_like_workflow_dump_bundles`, but keeps
    assertions minimal and avoids HTML bundle dumping.
    """

    from graph_knowledge_engine.models import WorkflowEdge, WorkflowNode, Span, Grounding, MentionVerification
    from graph_knowledge_engine.workflow.runtime import WorkflowRuntime
    from graph_knowledge_engine.workflow.contract import RunResult, State
    from graph_knowledge_engine.workflow.resolvers import default_resolver

    wf_dir = tmp_path / "wf"
    conv_dir = tmp_path / "conv"
    kg_dir = tmp_path / "kg"

    workflow_id = "wf_smoke_default_resolver"

    workflow_engine = GraphKnowledgeEngine(persist_directory=str(wf_dir), kg_graph_type="workflow")
    conversation_engine = GraphKnowledgeEngine(persist_directory=str(conv_dir), kg_graph_type="conversation")
    kg_engine = GraphKnowledgeEngine(persist_directory=str(kg_dir), kg_graph_type="knowledge")
    conversation_id = "test_id::test_workflow_runtime_uses_default_resolver"
    # -----------------------------
    # 1) Persist a tiny add-turn-like workflow design
    # -----------------------------
    def get_self_span(content):
        self_span = Span(
            collection_page_url=f"conversation/{conversation_id}",
            document_page_url=f"conversation/{conversation_id}",
            doc_id=f"conv:{conversation_id}",
            insertion_method="conversation_turn",
            page_number=1,
            start_char=0,
            end_char=len(content),
            excerpt=content,
            context_before="",
            context_after="",
            chunk_id=None,
            source_cluster_id=None,
            verification=MentionVerification(
                method="human",
                is_verified=True,
                score=1.0,
                notes=f"test_workflow_runtime_uses_default_resolver",
            ),
        )
        return self_span
    def n(node_id: str, op: str, *, start: bool = False, terminal: bool = False) -> WorkflowNode:
        return WorkflowNode(
            id=node_id,
            label=node_id.split("|")[-1],
            type="entity",
            doc_id=node_id,
            summary=op,
            properties={},
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": workflow_id,
                "wf_op": op,
                "wf_start": start,
                "wf_terminal": terminal,
                "wf_version": "v1",
            },
            mentions=[Grounding(spans=[get_self_span(op)])],
            level_from_root = 0,
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )

    def e(edge_id: str, src: str, dst: str, *, pred: str | None, priority: int = 100, is_default: bool = True) -> WorkflowEdge:
        return WorkflowEdge(
            id=edge_id,
            source_ids=[src],
            target_ids=[dst],
            relation="wf_next",
            label="wf_next",
            type="relationship",
            summary="next",
            doc_id=workflow_id,
            properties={},
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": workflow_id,
                "wf_priority": priority,
                "wf_is_default": is_default,
                "wf_predicate": pred,
                "wf_multiplicity": "one",
            },
            source_edge_ids=[],
            target_edge_ids=[],
            mentions=[Grounding(spans=[get_self_span("next")])],
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )

    # nodes
    n_start = n(f"wf|{workflow_id}|start", "start", start=True)
    n_mem = n(f"wf|{workflow_id}|mem", "memory_retrieve")
    n_kg = n(f"wf|{workflow_id}|kg", "kg_retrieve")
    n_ans = n(f"wf|{workflow_id}|ans", "answer")
    n_dec = n(f"wf|{workflow_id}|dec", "decide_summarize")
    n_sum = n(f"wf|{workflow_id}|sum", "summarize")
    n_end = n(f"wf|{workflow_id}|end", "end", terminal=True)

    for node in [n_start, n_mem, n_kg, n_ans, n_dec, n_sum, n_end]:
        workflow_engine.add_node(node)

    # edges
    workflow_engine.add_edge(e(f"wf|{workflow_id}|e|start->mem", n_start.id, n_mem.id, pred=None))
    workflow_engine.add_edge(e(f"wf|{workflow_id}|e|mem->kg", n_mem.id, n_kg.id, pred=None))
    workflow_engine.add_edge(e(f"wf|{workflow_id}|e|kg->ans", n_kg.id, n_ans.id, pred=None))
    workflow_engine.add_edge(e(f"wf|{workflow_id}|e|ans->dec", n_ans.id, n_dec.id, pred=None))
    workflow_engine.add_edge(e(f"wf|{workflow_id}|e|dec->sum", n_dec.id, n_sum.id, pred="should_summarize", priority=0, is_default=False))
    workflow_engine.add_edge(e(f"wf|{workflow_id}|e|dec->end", n_dec.id, n_end.id, pred=None, priority=100, is_default=True))
    workflow_engine.add_edge(e(f"wf|{workflow_id}|e|sum->end", n_sum.id, n_end.id, pred=None))

    # -----------------------------
    # 2) Re-open workflow engine to prove “saved graph is runnable”
    # -----------------------------
    workflow_engine2 = GraphKnowledgeEngine(persist_directory=str(wf_dir), kg_graph_type="workflow")

    def predicate_should_summarize(state: State, r: RunResult) -> bool:
        # default_resolver sets ctx.state["decide"] = {"need_summary": bool}
        return bool(state.get("decide", {}).get("need_summary"))

    predicate_registry = {"should_summarize": predicate_should_summarize}

    rt = WorkflowRuntime(
        workflow_engine=workflow_engine2,
        conversation_engine=conversation_engine,
        step_resolver=default_resolver.resolve,
        predicate_registry=predicate_registry,
        checkpoint_every_n_steps=1,
        max_workers=1,
    )

    final_state, run_id = rt.run(
        workflow_id=workflow_id,
        conversation_id="conv_smoke",
        turn_node_id="turn_smoke",
        initial_state={},
    )

    assert run_id
    assert final_state.get("started") is True
    assert final_state.get("op_log")[:2] == ["start", "memory_retrieve"]
    assert "answer" in final_state


def test_orchestrator_has_v2(tmp_path):
    conv = GraphKnowledgeEngine(persist_directory=str(tmp_path / "conv"), kg_graph_type="conversation")
    kg = GraphKnowledgeEngine(persist_directory=str(tmp_path / "kg"), kg_graph_type="knowledge")
    wf = GraphKnowledgeEngine(persist_directory=str(tmp_path / "wf"), kg_graph_type="workflow")

    # NOTE: adapt args to your orchestrator's real __init__ signature
    orch = ConversationOrchestrator(
        conversation_engine=conv,
        ref_knowledge_engine=kg,
        workflow_engine=wf,
        llm = wf.llm,
        # llm=..., tool_runner=..., etc.
    )

    assert hasattr(orch, "add_conversation_turn_workflow_v2")
