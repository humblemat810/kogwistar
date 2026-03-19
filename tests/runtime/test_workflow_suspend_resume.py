import json
import uuid
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.runtime.models import (
    RunFailure,
    RunSuccess,
    RunSuspended,
    WorkflowEdge,
    WorkflowNode,
)
from graph_knowledge_engine.runtime.runtime import WorkflowRuntime, StepContext
from graph_knowledge_engine.runtime.resolvers import MappingStepResolver
from graph_knowledge_engine.runtime.sandbox import SandboxRequest
from tests.conftest import FakeEmbeddingFunction

from graph_knowledge_engine.engine_core.models import Span, Grounding

from graph_knowledge_engine.engine_core.models import (
    MentionVerification,
)


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
        verification=MentionVerification(
            method="system", is_verified=True, score=1.0, notes=""
        ),
    )
    return Grounding(spans=[sp])


def _create_node(
    engine: GraphKnowledgeEngine,
    wf_id: str,
    node_id: str,
    op: str,
    start: bool = False,
    terminal: bool = False,
    fanout: bool = False,
    wf_join: bool = False,
):
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
            level_from_root=0,
        )
    )


def _create_edge(
    engine: GraphKnowledgeEngine,
    wf_id: str,
    src: str,
    dst: str,
    *,
    label: str = "wf_next",
    predicate: str | None = None,
    is_default: bool = True,
):
    engine.add_edge(
        WorkflowEdge(
            id=f"{src}->{dst}",
            label=label,
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
                "wf_predicate": predicate,
                "wf_is_default": is_default,
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
    )


def _latest_checkpoint_state(conv_engine: GraphKnowledgeEngine, run_id: str) -> dict:
    ckpts = conv_engine.get_nodes(
        where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]}
    )
    latest = max(
        ckpts, key=lambda c: int(getattr(c, "metadata", {}).get("step_seq", -1))
    )
    state_json = latest.metadata.get("state_json", {})
    if isinstance(state_json, str):
        state_json = json.loads(state_json)
    return dict(state_json or {})


def _workflow_step_exec_nodes(conv_engine: GraphKnowledgeEngine, run_id: str):
    return conv_engine.get_nodes(
        where={"$and": [{"entity_type": "workflow_step_exec"}, {"run_id": run_id}]}
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
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("suspend_op")
    def _suspend(ctx: StepContext):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={"task": "calculate_pi"},
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

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
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=run_id,
    )

    assert res1.status == "suspended"
    assert res1.final_state.get("started") is True
    assert res1.final_state.get("ended") is None

    # Check that it actually persisted a checkpoint with pending tokens
    ckpts = conv_engine.get_nodes(
        where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]}
    )
    assert len(ckpts) > 0
    latest = max(
        ckpts, key=lambda c: int(getattr(c, "metadata", {}).get("step_seq", -1))
    )
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
        conversation_node_id=None, state_update=[("u", {"pi": 3.14})]
    )

    # 3. Resume run
    res2 = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="n_suspend",
        suspended_token_id=suspended_token_id,
        client_result=client_result,
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
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
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("noop")
    def _noop(ctx: StepContext):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("suspend_op")
    def _suspend(ctx: StepContext):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={"task": "do_something"},
        )

    @resolver.register("normal_b")
    def _normal_b(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"b_done": True})]
        )

    @resolver.register("end")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

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
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=run_id,
    )

    assert res1.status == "suspended"
    assert res1.final_state.get("b_done") is True
    assert res1.final_state.get("ended") is None

    # Check pending token
    ckpts = conv_engine.get_nodes(
        where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]}
    )
    latest = max(
        ckpts, key=lambda c: int(getattr(c, "metadata", {}).get("step_seq", -1))
    )
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
        conversation_node_id=None, state_update=[("u", {"a_done": True})]
    )

    res2 = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="a",
        suspended_token_id=suspended_token_id,
        client_result=client_result,
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
    )

    assert res2.status == "succeeded"
    assert res2.final_state.get("a_done") is True
    assert res2.final_state.get("b_done") is True
    assert res2.final_state.get("ended") is True


def test_workflow_failure_does_not_route_to_terminal(tmp_path):
    wf_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "wf_fail"),
        kg_graph_type="workflow",
        embedding_function=FakeEmbeddingFunction(),
    )
    conv_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "conv_fail"),
        kg_graph_type="conversation",
        embedding_function=FakeEmbeddingFunction(),
    )

    wf_id = "test_failure_stops_routing"
    _create_node(wf_engine, wf_id, "n_start", "start_op", start=True)
    _create_node(wf_engine, wf_id, "n_exec", "python_exec")
    _create_node(wf_engine, wf_id, "n_end", "end_op", terminal=True)
    _create_edge(wf_engine, wf_id, "n_start", "n_exec")
    # No outgoing edge from the failing node: unmatched failure should end the run as failure.

    class _FailingSandbox:
        def run(self, code, state, context):
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=["sandbox failed"],
            )

        def close_run(self, run_id: str) -> None:
            return None

    resolver = MappingStepResolver()
    resolver.set_sandbox(_FailingSandbox())

    @resolver.register("start_op")
    def _start(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("python_exec", is_sandboxed=True)
    def _python_exec(ctx: StepContext):
        return SandboxRequest(
            code="result = {'state_update': [('u', {'sandbox_result': 'ok'})]}"
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )

    res = runtime.run(
        workflow_id=wf_id,
        conversation_id=f"conv_{uuid.uuid4().hex}",
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=f"run_{uuid.uuid4().hex}",
    )

    assert res.status == "failure"
    assert res.final_state.get("started") is True
    assert res.final_state.get("ended") is None


def test_workflow_failure_can_route_to_recovery_branch(tmp_path):
    wf_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "wf_fail_route"),
        kg_graph_type="workflow",
        embedding_function=FakeEmbeddingFunction(),
    )
    conv_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "conv_fail_route"),
        kg_graph_type="conversation",
        embedding_function=FakeEmbeddingFunction(),
    )

    wf_id = "test_failure_routes"
    _create_node(wf_engine, wf_id, "start", "start_op", start=True)
    _create_node(wf_engine, wf_id, "exec", "exec_op")
    _create_node(wf_engine, wf_id, "recover", "recover_op")
    _create_node(wf_engine, wf_id, "end", "end_op", terminal=True)
    _create_edge(wf_engine, wf_id, "start", "exec")
    _create_edge(
        wf_engine,
        wf_id,
        "exec",
        "recover",
        label="recover_on_failure",
        predicate="if_failure",
        is_default=False,
    )
    _create_edge(wf_engine, wf_id, "exec", "end", label="finish", is_default=True)
    _create_edge(wf_engine, wf_id, "recover", "end")

    class _IfFailure:
        def __call__(self, e, state, result):
            return getattr(result, "status", None) == "failure"

    resolver = MappingStepResolver()

    @resolver.register("start_op")
    def _start(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("exec_op")
    def _exec(ctx: StepContext):
        return RunFailure(
            conversation_node_id=None,
            state_update=[("u", {"failed_once": True})],
            errors=["boom"],
        )

    @resolver.register("recover_op")
    def _recover(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"recovered": True})]
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={"if_failure": _IfFailure()},
        checkpoint_every_n_steps=1,
    )

    res = runtime.run(
        workflow_id=wf_id,
        conversation_id=f"conv_{uuid.uuid4().hex}",
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=f"run_{uuid.uuid4().hex}",
    )

    assert res.status == "succeeded"
    assert res.final_state.get("failed_once") is True
    assert res.final_state.get("recovered") is True
    assert res.final_state.get("ended") is True


def test_resume_run_failure_can_route_to_recovery_branch(tmp_path):
    wf_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "wf_resume_fail"),
        kg_graph_type="workflow",
        embedding_function=FakeEmbeddingFunction(),
    )
    conv_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "conv_resume_fail"),
        kg_graph_type="conversation",
        embedding_function=FakeEmbeddingFunction(),
    )

    wf_id = "test_resume_failure_routes"
    _create_node(wf_engine, wf_id, "start", "start_op", start=True)
    _create_node(wf_engine, wf_id, "gate", "suspend_op")
    _create_node(wf_engine, wf_id, "recover", "recover_op")
    _create_node(wf_engine, wf_id, "end", "end_op", terminal=True)
    _create_edge(wf_engine, wf_id, "start", "gate")
    _create_edge(
        wf_engine,
        wf_id,
        "gate",
        "recover",
        label="recover_on_failure",
        predicate="if_failure",
        is_default=False,
    )
    _create_edge(wf_engine, wf_id, "gate", "end", label="finish", is_default=True)
    _create_edge(wf_engine, wf_id, "recover", "end")

    class _IfFailure:
        def __call__(self, e, state, result):
            return getattr(result, "status", None) == "failure"

    resolver = MappingStepResolver()

    @resolver.register("start_op")
    def _start(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("suspend_op")
    def _suspend(ctx: StepContext):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={
                "type": "recoverable_error",
                "op": "suspend_op",
                "category": "missing_input",
                "message": "need fix",
                "errors": ["need fix"],
                "repair_payload": {"prompt": "fix it"},
            },
        )

    @resolver.register("recover_op")
    def _recover(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"failure_routed": True})]
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={"if_failure": _IfFailure()},
        checkpoint_every_n_steps=1,
    )

    run_id = f"run_{uuid.uuid4().hex}"
    conv_id = f"conv_{uuid.uuid4().hex}"
    res1 = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=run_id,
    )
    state1 = _latest_checkpoint_state(conv_engine, run_id)
    suspended_token_id = state1["_rt_join"]["suspended"][0][2]

    res2 = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="gate",
        suspended_token_id=suspended_token_id,
        client_result=RunFailure(
            conversation_node_id=None,
            state_update=[("u", {"resume_failed": True})],
            errors=["still broken"],
        ),
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
    )

    assert res1.status == "suspended"
    assert res2.status == "succeeded"
    assert res2.final_state.get("resume_failed") is True
    assert res2.final_state.get("failure_routed") is True
    assert res2.final_state.get("ended") is True


def test_resume_run_can_resuspend_same_token_with_updated_payload(tmp_path):
    wf_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "wf_resuspend"),
        kg_graph_type="workflow",
        embedding_function=FakeEmbeddingFunction(),
    )
    conv_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "conv_resuspend"),
        kg_graph_type="conversation",
        embedding_function=FakeEmbeddingFunction(),
    )

    wf_id = "test_resuspend"
    _create_node(wf_engine, wf_id, "start", "start_op", start=True)
    _create_node(wf_engine, wf_id, "gate", "suspend_op")
    _create_node(wf_engine, wf_id, "end", "end_op", terminal=True)
    _create_edge(wf_engine, wf_id, "start", "gate")
    _create_edge(wf_engine, wf_id, "gate", "end")

    resolver = MappingStepResolver()

    @resolver.register("start_op")
    def _start(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("suspend_op")
    def _suspend(ctx: StepContext):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={
                "type": "recoverable_error",
                "op": "suspend_op",
                "category": "missing_input",
                "message": "first pause",
                "errors": ["first"],
                "repair_payload": {"attempt": 1},
            },
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )

    run_id = f"run_{uuid.uuid4().hex}"
    conv_id = f"conv_{uuid.uuid4().hex}"
    res1 = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=run_id,
    )
    state1 = _latest_checkpoint_state(conv_engine, run_id)
    suspended_token_id = state1["_rt_join"]["suspended"][0][2]

    res2 = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="gate",
        suspended_token_id=suspended_token_id,
        client_result=RunSuspended(
            conversation_node_id=None,
            state_update=[("u", {"retry_count": 1})],
            resume_payload={
                "type": "recoverable_error",
                "op": "suspend_op",
                "category": "missing_input",
                "message": "second pause",
                "errors": ["second"],
                "repair_payload": {"attempt": 2},
            },
        ),
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
    )

    assert res1.status == "suspended"
    assert res2.status == "suspended"
    assert res2.final_state.get("retry_count") == 1

    state2 = _latest_checkpoint_state(conv_engine, run_id)
    assert len((state2.get("_rt_join", {}) or {}).get("suspended", [])) == 1
    assert (state2.get("_rt_join", {}) or {}).get("suspended", [])[0][0] == "gate"

    step_execs = _workflow_step_exec_nodes(conv_engine, run_id)
    latest_step = max(
        step_execs, key=lambda n: int(getattr(n, "metadata", {}).get("step_seq", -1))
    )
    latest_result = json.loads(
        getattr(latest_step, "metadata", {}).get("result_json", "{}")
    )
    assert getattr(latest_step, "metadata", {}).get("status") == "suspended"
    assert latest_result.get("resume_payload", {}).get("message") == "second pause"


def test_sandbox_recoverable_error_can_suspend_then_resume_success(tmp_path):
    wf_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "wf_sandbox_recoverable"),
        kg_graph_type="workflow",
        embedding_function=FakeEmbeddingFunction(),
    )
    conv_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "conv_sandbox_recoverable"),
        kg_graph_type="conversation",
        embedding_function=FakeEmbeddingFunction(),
    )

    wf_id = "test_sandbox_recoverable"
    _create_node(wf_engine, wf_id, "start", "start_op", start=True)
    _create_node(wf_engine, wf_id, "python_exec", "python_exec")
    _create_node(wf_engine, wf_id, "end", "end_op", terminal=True)
    _create_edge(wf_engine, wf_id, "start", "python_exec")
    _create_edge(wf_engine, wf_id, "python_exec", "end")

    class _RecoverableSandbox:
        def run(self, code, state, context):
            return RunSuspended(
                conversation_node_id=None,
                state_update=[],
                resume_payload={
                    "type": "recoverable_error",
                    "op": str(context.get("op")),
                    "category": "sandbox_code_error",
                    "message": "customer_id is not defined",
                    "errors": ["NameError: customer_id is not defined"],
                    "repair_payload": {"code": code, "state": state},
                },
            )

        def close_run(self, run_id: str) -> None:
            return None

    resolver = MappingStepResolver()
    resolver.set_sandbox(_RecoverableSandbox())

    @resolver.register("start_op")
    def _start(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("python_exec", is_sandboxed=True)
    def _python_exec(ctx: StepContext):
        return SandboxRequest(
            code="result = {'state_update': [('u', {'sandbox_result': 'fixed'})]}"
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )

    run_id = f"run_{uuid.uuid4().hex}"
    conv_id = f"conv_{uuid.uuid4().hex}"
    res1 = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=run_id,
    )
    state1 = _latest_checkpoint_state(conv_engine, run_id)
    suspended_token_id = state1["_rt_join"]["suspended"][0][2]

    res2 = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="python_exec",
        suspended_token_id=suspended_token_id,
        client_result=RunSuccess(
            conversation_node_id=None, state_update=[("u", {"sandbox_result": "fixed"})]
        ),
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
    )

    assert res1.status == "suspended"
    step_execs = _workflow_step_exec_nodes(conv_engine, run_id)
    first_suspend = next(
        n
        for n in step_execs
        if getattr(n, "metadata", {}).get("workflow_node_id") == "python_exec"
    )
    first_result = json.loads(
        getattr(first_suspend, "metadata", {}).get("result_json", "{}")
    )
    assert first_result.get("resume_payload", {}).get("type") == "recoverable_error"
    assert (
        first_result.get("resume_payload", {}).get("category") == "sandbox_code_error"
    )

    assert res2.status == "succeeded"
    assert res2.final_state.get("sandbox_result") == "fixed"
    assert res2.final_state.get("ended") is True
