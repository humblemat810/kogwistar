import pytest
import uuid
import queue
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.runtime.models import RunSuccess, RunSuspended, WorkflowEdge, WorkflowNode
from graph_knowledge_engine.runtime.runtime import WorkflowRuntime, StepContext
from graph_knowledge_engine.runtime.resolvers import MappingStepResolver
from tests.conftest import FakeEmbeddingFunction

from graph_knowledge_engine.engine_core.models import Span, Grounding

from graph_knowledge_engine.engine_core.models import Span, Grounding, MentionVerification

def _get_dummy_grounding():
    sp = Span(
        collection_page_url="dummy",
        document_page_url="dummy",
        doc_id="dummy",
        insertion_method="dummy",
        page_number=1,
        start_char=0,
        end_char=1,
        excerpt="a",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(method="system", is_verified=True, score=1.0, notes="")
    )
    return Grounding(spans=[sp])


def _create_node(engine: GraphKnowledgeEngine, wf_id: str, node_id: str, op: str, start: bool = False, terminal: bool = False, fanout: bool = False, wf_join: bool = False):
    engine.add_node(
        WorkflowNode(
            id=node_id,
            label=op,
            type="entity",
            doc_id=node_id,
            summary=op,
            properties={},
            mentions=[_get_dummy_grounding()],
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": wf_id,
                "wf_op": op,
                "wf_start": start,
                "wf_terminal": terminal,
                "wf_version": "v1",
                "wf_fanout": fanout,
                "wf_join": wf_join,
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
            level_from_root=0
        )
    )

def _create_edge(engine: GraphKnowledgeEngine, wf_id: str, src: str, dst: str):
    engine.add_edge(
        WorkflowEdge(
            id=f"{src}->{dst}",
            label="wf_next",
            type="entity",
            doc_id=f"{src}->{dst}",
            summary="next",
            properties={},
            source_ids=[src],
            target_ids=[dst],
            source_edge_ids=[],
            target_edge_ids=[],
            relation="wf_next",
            mentions=[_get_dummy_grounding()],
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": wf_id,
                "wf_is_default": True,
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
    )

def test_workflow_suspend_and_resume(tmp_path):
    wf_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "wf"),
        kg_graph_type="workflow",
        embedding_function=FakeEmbeddingFunction(),
    )
    conv_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "conv"),
        kg_graph_type="conversation",
        embedding_function=FakeEmbeddingFunction(),
    )
    
    wf_id = "test_suspend_wf"
    
    # start -> do_suspend -> end
    _create_node(wf_engine, wf_id, "n_start", "start_op", start=True)
    _create_node(wf_engine, wf_id, "n_suspend", "suspend_op")
    _create_node(wf_engine, wf_id, "n_end", "end_op", terminal=True)
    
    _create_edge(wf_engine, wf_id, "n_start", "n_suspend")
    _create_edge(wf_engine, wf_id, "n_suspend", "n_end")

    resolver = MappingStepResolver()

    @resolver.register("start_op")
    def _start(ctx: StepContext):
        return RunSuccess(conversation_node_id=None, state_update=[('u', {'started': True})])

    @resolver.register("suspend_op")
    def _suspend(ctx: StepContext):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={"task": "calculate_pi"}
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(conversation_node_id=None, state_update=[('u', {'ended': True})])

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )

    run_id = f"run_{uuid.uuid4().hex}"
    conv_id = f"conv_{uuid.uuid4().hex}"

    # 1. Run until suspension
    res1 = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state={"conversation_id": "test", "user_id": "test", "turn_node_id": "test", "turn_index": 0, "role": "user", "user_text": "", "mem_id": "test", "self_span": {}}, # type: ignore
        run_id=run_id
    )

    assert res1.status == "suspended"
    assert res1.final_state.get("started") is True
    assert res1.final_state.get("ended") is None

    # Check that it actually persisted a checkpoint with pending tokens
    ckpts = conv_engine.get_nodes(
        where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]}
    )
    assert len(ckpts) > 0
    latest = max(ckpts, key=lambda c: int(getattr(c, "metadata", {}).get("step_seq", -1)))
    state_json = latest.metadata.get("state_json", {})
    import json
    if isinstance(state_json, str):
        state_json = json.loads(state_json)
    rt_join = state_json.get("_rt_join", {})
    suspended = rt_join.get("suspended", [])

    assert len(suspended) == 1
    assert suspended[0][0] == "n_suspend"
    suspended_token_id = suspended[0][2]

    # 2. Emulate client finishing task and providing result
    client_result = RunSuccess(
        conversation_node_id=None,
        state_update=[('u', {'pi': 3.14})]
    )

    # 3. Resume run
    res2 = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="n_suspend",
        suspended_token_id=suspended_token_id,
        client_result=client_result,
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1"
    )

    assert res2.status == "succeeded"
    assert res2.final_state.get("started") is True
    assert res2.final_state.get("pi") == 3.14
    assert res2.final_state.get("ended") is True

def test_workflow_suspend_and_resume_branching(tmp_path):
    wf_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "wf_b"),
        kg_graph_type="workflow",
        embedding_function=FakeEmbeddingFunction(),
    )
    conv_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "conv_b"),
        kg_graph_type="conversation",
        embedding_function=FakeEmbeddingFunction(),
    )
    
    wf_id = "test_suspend_branching_wf"
    
    # start -> fork (fanout) -> a (suspends) -> join -> end
    #                        \> b (normal)  /
    _create_node(wf_engine, wf_id, "start", "start", start=True)
    _create_node(wf_engine, wf_id, "fork", "noop", fanout=True)
    _create_node(wf_engine, wf_id, "a", "suspend_op")
    _create_node(wf_engine, wf_id, "b", "normal_b")
    _create_node(wf_engine, wf_id, "join", "noop", wf_join=True)
    _create_node(wf_engine, wf_id, "end", "end", terminal=True)
    
    _create_edge(wf_engine, wf_id, "start", "fork")
    _create_edge(wf_engine, wf_id, "fork", "a")
    _create_edge(wf_engine, wf_id, "fork", "b")
    _create_edge(wf_engine, wf_id, "a", "join")
    _create_edge(wf_engine, wf_id, "b", "join")
    _create_edge(wf_engine, wf_id, "join", "end")

    resolver = MappingStepResolver()

    @resolver.register("start")
    def _start(ctx: StepContext):
        return RunSuccess(conversation_node_id=None, state_update=[('u', {'started': True})])

    @resolver.register("noop")
    def _noop(ctx: StepContext):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("suspend_op")
    def _suspend(ctx: StepContext):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={"task": "do_something"}
        )

    @resolver.register("normal_b")
    def _normal_b(ctx: StepContext):
        return RunSuccess(conversation_node_id=None, state_update=[('u', {'b_done': True})])

    @resolver.register("end")
    def _end(ctx: StepContext):
        return RunSuccess(conversation_node_id=None, state_update=[('u', {'ended': True})])

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )

    run_id = f"run_{uuid.uuid4().hex}"
    conv_id = f"conv_{uuid.uuid4().hex}"

    # 1. Run until suspension
    # Since branch b completes but branch a suspends, the overall run should end in suspended state
    # waiting for branch a to resume and hit the join.
    res1 = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state={"conversation_id": "test", "user_id": "test", "turn_node_id": "test", "turn_index": 0, "role": "user", "user_text": "", "mem_id": "test", "self_span": {}}, # type: ignore
        run_id=run_id
    )

    assert res1.status == "suspended"
    assert res1.final_state.get("b_done") is True
    assert res1.final_state.get("ended") is None

    # Check pending token
    ckpts = conv_engine.get_nodes(where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]})
    latest = max(ckpts, key=lambda c: int(getattr(c, "metadata", {}).get("step_seq", -1)))
    state_json = latest.metadata.get("state_json", {})
    import json
    if isinstance(state_json, str):
        state_json = json.loads(state_json)
    rt_join = state_json.get("_rt_join", {})
    suspended = rt_join.get("suspended", [])

    assert len(suspended) == 1
    assert suspended[0][0] == "a"
    suspended_token_id = suspended[0][2]

    # 2. Resume
    client_result = RunSuccess(
        conversation_node_id=None,
        state_update=[('u', {'a_done': True})]
    )

    res2 = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="a",
        suspended_token_id=suspended_token_id,
        client_result=client_result,
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1"
    )

    assert res2.status == "succeeded"
    assert res2.final_state.get("a_done") is True
    assert res2.final_state.get("b_done") is True
    assert res2.final_state.get("ended") is True


