import pytest

from graph_knowledge_engine.runtime.contract import WorkflowSpec, validate_workflow


class _FakeNode:
    def safe_get_id(self):
        return self.id

    def __init__(self, id, metadata):
        self.id = id
        self.metadata = metadata
        self.embedding = None


class _FakeEdge:
    def safe_get_id(self):
        return self.id

    def __init__(self, id, src, dst, metadata):
        self.id = id
        self.source_ids = [src]
        self.target_ids = [dst]
        self.metadata = metadata
        self.label = metadata.get("label") or f"{id}:fake_edge_label"


class FakeEngine:
    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges

    def get_nodes(
        self,
        where=None,
        limit=None,
        ids=None,
        node_type=None,
        include=None,
        resolve_mode=None,
    ):
        if where is None:
            return list(self._nodes)
        out = []
        if where.get("$and"):
            where_inner = {}
            for d in where.get("$and"):
                where_inner.update(d)
        else:
            where_inner = where
        for n in self._nodes:
            if all((n.metadata or {}).get(k) == v for k, v in where_inner.items()):
                out.append(n)
        return out

    def get_edges(
        self, where=None, limit=None, ids=None, edge_type=None, resolve_mode=None
    ):
        if where is None:
            return list(self._edges)
        out = []
        if where.get("$and"):
            where_inner = {}
            for d in where.get("$and"):
                where_inner.update(d)
        else:
            where_inner = where
        for e in self._edges:
            if all((e.metadata or {}).get(k) == v for k, v in where_inner.items()):
                out.append(e)
        return out


def test_cycle_allowed_but_exit_required():
    nodes = [
        _FakeNode(
            "A",
            {
                "entity_type": "workflow_node",
                "workflow_id": "wf",
                "wf_op": "x",
                "wf_start": True,
            },
        ),
        _FakeNode(
            "B", {"entity_type": "workflow_node", "workflow_id": "wf", "wf_op": "y"}
        ),
    ]
    edges = [
        _FakeEdge(
            "e1",
            "A",
            "B",
            {
                "entity_type": "workflow_edge",
                "workflow_id": "wf",
                "wf_is_default": True,
            },
        ),
        _FakeEdge(
            "e2",
            "B",
            "A",
            {
                "entity_type": "workflow_edge",
                "workflow_id": "wf",
                "wf_is_default": True,
            },
        ),
    ]
    engine = FakeEngine(nodes, edges)
    spec = WorkflowSpec(workflow_id="wf", start_node_id="A")

    with pytest.raises(ValueError):
        validate_workflow(engine=engine, spec=spec, predicate_registry={})


def test_terminal_satisfies_exit():
    nodes = [
        _FakeNode(
            "A",
            {
                "entity_type": "workflow_node",
                "workflow_id": "wf",
                "wf_op": "x",
                "wf_start": True,
            },
        ),
        _FakeNode(
            "B",
            {
                "entity_type": "workflow_node",
                "workflow_id": "wf",
                "wf_op": "y",
                "wf_terminal": True,
            },
        ),
    ]
    edges = [
        _FakeEdge(
            "e1",
            "A",
            "B",
            {
                "entity_type": "workflow_edge",
                "workflow_id": "wf",
                "wf_is_default": True,
            },
        ),
    ]
    engine = FakeEngine(nodes, edges)
    spec = WorkflowSpec(workflow_id="wf", start_node_id="A")

    validate_workflow(engine=engine, spec=spec, predicate_registry={})
