import json

from graph_knowledge_engine.workflow.executor import WorkflowExecutor
from graph_knowledge_engine.workflow.design import WorkflowSpec, WFNode, WFEdge
from graph_knowledge_engine.workflow.serialize import to_jsonable


def test_executor_emits_checkpoint_and_state_is_serializable():
    workflow_id = "wf_demo"

    n1 = WFNode(node_id="n1", workflow_id=workflow_id, op="a", start=True, terminal=False, fanout=False, version="v1")
    n2 = WFNode(node_id="n2", workflow_id=workflow_id, op="b", start=False, terminal=True, fanout=False, version="v1")

    e = WFEdge(
        edge_id="e1",
        workflow_id=workflow_id,
        src="n1",
        dst="n2",
        predicate=None,
        priority=0,
        is_default=True,
        multiplicity="one",
    )

    spec = WorkflowSpec(
        workflow_id=workflow_id,
        start_node_id="n1",
        nodes={"n1": n1, "n2": n2},
        out_edges={"n1": [e], "n2": []},
    )

    def resolver(op):
        if op == "a":
            def f(ctx):
                ctx.publish({"type": "a_done"})
                return {"x": 1}
            return f
        if op == "b":
            def f(ctx):
                msgs = ctx.drain_messages()
                return {"y": 2, "msgs": msgs}
            return f
        raise KeyError(op)

    preds = {"always": lambda st, r: True}

    executor = WorkflowExecutor(
        workflow=spec,
        step_resolver=resolver,
        predicate_registry=preds,
        max_workers=2,
        checkpoint_every_n_steps=1,
        trace_sink=None,
    )

    events = list(executor.run(run_id="r1", initial_state={"memory": None}))

    # events are dicts: {"type": "...", "payload": {...}}
    assert any(ev.get("type") == "checkpoint" for ev in events), events

    # state must be JSON-serializable
    json.dumps(to_jsonable(executor.state))

    assert "result.a" in executor.state
    assert "result.b" in executor.state
