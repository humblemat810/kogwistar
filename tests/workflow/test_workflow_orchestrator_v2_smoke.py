
from graph_knowledge_engine.engine import GraphKnowledgeEngine, candiate_filtering_callback
from graph_knowledge_engine.conversation_orchestrator import ConversationOrchestrator
from graph_knowledge_engine.models import MetaFromLastSummary


def test_workflow_runtime_uses_default_resolver(tmp_path):
    """Smoke: a persisted workflow design can be executed using the package default resolver.

    This mirrors the intent of `test_add_turn_like_workflow_dump_bundles`, but keeps
    assertions minimal and avoids HTML bundle dumping.
    """

    from graph_knowledge_engine.models import WorkflowEdge, WorkflowNode, Span, Grounding, MentionVerification
    from graph_knowledge_engine.workflow.runtime import WorkflowRuntime
    from graph_knowledge_engine.conversation_state_contracts import WorkflowStateModel, WorkflowState
    from graph_knowledge_engine.workflow.runtime import RunResult, State
    from graph_knowledge_engine.workflow.resolvers import default_resolver, MappingStepResolver
    from graph_knowledge_engine.tool_runner import ToolRunner
    
    from typing import Callable, TypeVar, ParamSpec, cast
    from joblib import Memory

    P = ParamSpec("P")
    R = TypeVar("R")
    from sys import monitoring
    target_code = default_resolver.resolve.__code__
    target_code2 = MappingStepResolver.resolve.__code__
    TOOL_ID = 0
    hits = 0

    def on_call(code, instruction_offset):
        nonlocal hits
        # 'code' is a code object for the function being called
        if code is target_code or code is target_code2:
            hits += 1

    # monitoring.use_tool_id(TOOL_ID, "call_counter")
    monitoring.register_callback(TOOL_ID, monitoring.events.CALL, on_call)    
    wf_dir = tmp_path / "wf"
    conv_dir = tmp_path / "conv"
    kg_dir = tmp_path / "kg"

    workflow_id = "wf_smoke_default_resolver"

    workflow_engine = GraphKnowledgeEngine(persist_directory=str(wf_dir), kg_graph_type="workflow")
    conversation_engine = GraphKnowledgeEngine(persist_directory=str(conv_dir), kg_graph_type="conversation")
    ref_knowledge_engine = GraphKnowledgeEngine(persist_directory=str(kg_dir), kg_graph_type="knowledge")
    import os
    import numpy as np
    mem = Memory(location = os.path.join('.', 'test_workflow_runtime_uses_default_resolver'))
    content = "where is my car?"
    @mem.cache
    def get_embedding(query):
        test_embedding = ref_knowledge_engine._iterative_defensive_emb(query)
        return test_embedding.tolist() if type(test_embedding) is np.ndarray else test_embedding
    state_embedding = get_embedding(content)
        
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
    workflow_engine.add_edge(e(f"wf|{workflow_id}|e|start->mem", n_start.safe_get_id(), n_mem.safe_get_id(), pred=None))
    workflow_engine.add_edge(e(f"wf|{workflow_id}|e|mem->kg", n_mem.safe_get_id(), n_kg.safe_get_id(), pred=None))
    workflow_engine.add_edge(e(f"wf|{workflow_id}|e|kg->ans", n_kg.safe_get_id(), n_ans.safe_get_id(), pred=None))
    workflow_engine.add_edge(e(f"wf|{workflow_id}|e|ans->dec", n_ans.safe_get_id(), n_dec.safe_get_id(), pred=None))
    workflow_engine.add_edge(e(f"wf|{workflow_id}|e|dec->sum", n_dec.safe_get_id(), n_sum.safe_get_id(), pred="should_summarize", priority=0, is_default=False))
    workflow_engine.add_edge(e(f"wf|{workflow_id}|e|dec->end", n_dec.safe_get_id(), n_end.safe_get_id(), pred=None, priority=100, is_default=True))
    workflow_engine.add_edge(e(f"wf|{workflow_id}|e|sum->end", n_sum.safe_get_id(), n_end.safe_get_id(), pred=None))
    #
    # 1.5) add a user turn.
    user_id="new-test-user"
    turn_id="test-turn-id-123"
    conv_id = conversation_id
    start_node_id = "test-start-turn-id-123"
    conv_id, start_node_id_returned = conversation_engine.create_conversation(user_id, conv_id, start_node_id)
    from graph_knowledge_engine.models import FilteringResult
    from langchain_core.language_models import BaseChatModel
    # def wrapped_cached_callback(llm: BaseChatModel, conversation_content, 
    #                             cand_node_list_str, cand_edge_list_str, 
    #                             candidates_node_ids: list[str], candidate_edge_ids: list[str], context_text):
    #     cached_fn = cached_deco(ignore = ["llm"], fn=cached_inner) 
        
    #     dumped, reasoning = cached_fn(llm, conversation_content, 
    #                             cand_node_list_str, cand_edge_list_str, 
    #                             candidates_node_ids, candidate_edge_ids, context_text)
    #     return FilteringResult.model_validate(dumped), reasoning

    def cached(memory: Memory, fn: Callable[P, R], *args, **kwargs) -> Callable[P, R]:
        return cast(Callable[P, R], memory.cache(fn, *args, **kwargs))
    # candiate_filtering_callback_cached = cached(mem, candiate_filtering_callback)
    from functools import partial
    
    cached_deco = partial(cached, mem)
    def cached_inner(llm: BaseChatModel, conversation_content, 
                                cand_node_list_str, cand_edge_list_str, 
                                candidates_node_ids: list[str], candidate_edge_ids: list[str], context_text):
            filtering_result, filtering_reasining = candiate_filtering_callback(llm, conversation_content, 
                                cand_node_list_str, cand_edge_list_str, 
                                candidates_node_ids, candidate_edge_ids, context_text)
            return filtering_result.model_dump(), filtering_reasining
      
    cached_fn = cached_deco(ignore = ["llm"], fn=cached_inner)     
    # 3. Add Turn 1
    res = conversation_engine.add_conversation_turn(user_id, conv_id, turn_id, "mem0", 
                                                    role="user", content="Hello computer", 
                                                    ref_knowledge_engine=ref_knowledge_engine,
                                                    filtering_callback = cached_fn, add_turn_only = True)    
    # 'c7b0883aefc651a69a69425fb018e5be'
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
    # import uuid
    from graph_knowledge_engine.id_provider import stable_id
    tool_runner=ToolRunner(conversation_engine=conversation_engine,
                                    tool_call_id_factory=stable_id
                                      )
    # max_retrieval_level: int = 2
    # summary_char_threshold: int = 12000
    from graph_knowledge_engine.conversation_orchestrator import ConversationOrchestrator
    orchestrator = ConversationOrchestrator(workflow_engine=workflow_engine, 
                                            ref_knowledge_engine=ref_knowledge_engine,
                                            conversation_engine=conversation_engine,
                                            )
    prev_turn_meta_summary = MetaFromLastSummary(0,0,0)
    deps= {
            "conversation_engine": conversation_engine,
            "ref_knowledge_engine": ref_knowledge_engine,
            "llm": conversation_engine.llm,
            "filtering_callback": cached_fn,
            "tool_runner": tool_runner,
            "max_retrieval_level": 1,
            "summary_char_threshold": 5000,
            "prev_turn_meta_summary": prev_turn_meta_summary,
            "answer_only": lambda *, conversation_id, prev_turn_meta_summary: orchestrator.answer_only(
                conversation_id=conversation_id,
                prev_turn_meta_summary=prev_turn_meta_summary,
            ),
            "summarize_batch": lambda conversation_id, current_index, *, prev_turn_meta_summary: orchestrator._summarize_conversation_batch(
                conversation_id,
                current_index,
                prev_turn_meta_summary=prev_turn_meta_summary,
            ),
            "add_link_to_new_turn": orchestrator.add_link_to_new_turn,
        }
    # init_state = {}
    mem_id = "mem-001"
    role = "user"
    from graph_knowledge_engine.conversation_state_contracts import PrevTurnMetaSummaryModel
    init_state: WorkflowState = WorkflowStateModel(
        conversation_id=conversation_id,
        user_id="new-test-user",
        turn_node_id="test-turn-id-123",
        turn_index=0,
        mem_id=mem_id,
        self_span=get_self_span("selfspan123"),
        role=str(role),
        user_text=content,
        embedding=state_embedding,
        prev_turn_meta_summary=PrevTurnMetaSummaryModel(
            prev_node_char_distance_from_last_summary=prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
            prev_node_distance_from_last_summary=prev_turn_meta_summary.prev_node_distance_from_last_summary,
            tail_turn_index = prev_turn_meta_summary.tail_turn_index
        ),
    ).dump_state()
    init_state["_deps"] = deps
    
    try:
        final_state, run_id = rt.run(
            workflow_id=workflow_id,
            conversation_id="conv_smoke",
            turn_node_id="turn_smoke",
            initial_state=init_state,
        )
    except Exception as _e:
        raise
    finally:
        # Important cleanup so it doesn't affect other tests
        monitoring.register_callback(TOOL_ID, monitoring.events.CALL, None)
        monitoring.free_tool_id(TOOL_ID)
    assert hits >= 1
    assert run_id
    assert final_state.get("started") is True
    # assert final_state.get("op_log")[:2] == ["start", "memory_retrieve"]
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
