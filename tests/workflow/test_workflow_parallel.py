import time

from graph_knowledge_engine.workflow.contract import WorkflowSpec
from graph_knowledge_engine.workflow.executor import WorkflowExecutor


class _FakeNode:
    def __init__(self, id, metadata):
        self.id = id
        self.metadata = metadata
        self.embedding = None


class _FakeEdge:
    def __init__(self, id, src, dst, metadata):
        self.id = id
        self.source_ids = [src]
        self.target_ids = [dst]
        self.metadata = metadata


class FakeEngine:
    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges

    def get_nodes(self, where=None, limit=None, ids=None, node_type=None, include=None, resolve_mode=None):
        if where is None:
            return list(self._nodes)
        out = []
        for n in self._nodes:
            if all((n.metadata or {}).get(k) == v for k, v in where.items()):
                out.append(n)
        return out

    def get_edges(self, where=None, limit=None, ids=None, edge_type=None, resolve_mode=None):
        if where is None:
            return list(self._edges)
        out = []
        for e in self._edges:
            if all((e.metadata or {}).get(k) == v for k, v in where.items()):
                out.append(e)
        return out


def test_message_queue_allows_retry_observe_update(tmp_path):
    # Workflow shape (static):
    # think(start, fanout=True) -> crawl (multiplicity=many, default)
    # think -> think if needs_data
    # think -> end if has_data
    # crawl -> think (default)
    #
    # Dynamics:
    # crawl publishes message + sets state["data"]
    # think retries until data exists and can observe the published message during retries.

    nodes = [
        _FakeNode("think", {"entity_type": "workflow_node", "workflow_id": "wf", "wf_op": "think", "wf_start": True, "wf_fanout": True, "wf_cacheable": False}),
        _FakeNode("crawl", {"entity_type": "workflow_node", "workflow_id": "wf", "wf_op": "crawl", "wf_cacheable": True}),
        _FakeNode("end",  {"entity_type": "workflow_node", "workflow_id": "wf", "wf_op": "end", "wf_terminal": True}),
    ]
    edges = [
        _FakeEdge("e1", "think", "crawl", {"entity_type": "workflow_edge", "workflow_id": "wf", "wf_is_default": True, "wf_priority": 1, "wf_multiplicity": "many"}),
        _FakeEdge("e2", "think", "think", {"entity_type": "workflow_edge", "workflow_id": "wf", "wf_predicate": "needs_data", "wf_priority": 2}),
        _FakeEdge("e3", "think", "end",  {"entity_type": "workflow_edge", "workflow_id": "wf", "wf_predicate": "has_data", "wf_priority": 3}),
        _FakeEdge("e4", "crawl", "think", {"entity_type": "workflow_edge", "workflow_id": "wf", "wf_is_default": True, "wf_priority": 1}),
    ]
    engine = FakeEngine(nodes, edges)
    spec = WorkflowSpec(workflow_id="wf", start_node_id="think")

    def needs_data(state, _res): return state.get("data") is None
    def has_data(state, _res): return state.get("data") is not None

    def resolve(op):
        if op == "crawl":
            def _crawl(ctx):
                time.sleep(0.2)
                ctx.state["data"] = "fetched"
                ctx.publish({"type": "crawl_done"})
                return {"data": "fetched"}
            return _crawl

        if op == "think":
            def _think(ctx):
                msgs = ctx.drain_messages()
                if ctx.state.get("data") is None:
                    time.sleep(0.05)
                    return {"answer": None, "msgs": msgs}
                return {"answer": "ok", "msgs": msgs}
            return _think

        if op == "end":
            return lambda ctx: {"done": True}

        raise KeyError(op)

    ex = WorkflowExecutor(
        engine=engine,
        workflow=spec,
        step_resolver=resolve,
        predicate_registry={"needs_data": needs_data, "has_data": has_data},
        cache_root=tmp_path,
        max_workers=2,
    )

    events = list(ex.run(run_id="r1", initial_state={"data": None}))

    assert ex.state.get("data") == "fetched"
    # At least one think result exists
    assert ex.state.get("result.think") is not None
    # Ensure crawl happened
    assert any(ev["type"] == "step_completed" and ev["payload"]["op"] == "crawl" for ev in events)
