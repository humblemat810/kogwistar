import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import (
    Edge,
    Grounding,
    MentionVerification,
    Node,
    Span,
)
from kogwistar.runtime.models import WorkflowEdge, WorkflowNode
from kogwistar.utils.kge_debug_dump import dump_d3_bundle
from kogwistar.visualization.graph_viz import (
    _filter_by_insertion_method,
    to_cytoscape,
    to_d3_force,
)
from kogwistar.visualization.basic_visualization import Visualizer
from tests._helpers.fake_backend import build_fake_backend

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.ci
def test_visualization_insertion_method_filter_matches_shared_rust_fixture() -> None:
    fixture_path = ROOT / "contracts/golden/visualization-insertion-filter-v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    class RefCollection:
        def __init__(self, metadatas: list[dict[str, str]]) -> None:
            self.metadatas = metadatas
            self.where: object = None

        def get(self, *, where: object, include: list[str]) -> dict[str, object]:
            self.where = where
            return {"metadatas": self.metadatas}

    for entity_kind in ("node", "edge"):
        ids = [record["id"] for record in fixture[f"{entity_kind}s"]]
        id_field = f"{entity_kind}_id"
        collection = RefCollection(
            [
                {id_field: identifier}
                for identifier in fixture["python_index_matches"][entity_kind]
            ]
        )
        engine = SimpleNamespace(
            **{f"{entity_kind}_refs_collection": collection}
        )
        assert _filter_by_insertion_method(
            engine,
            ids,
            entity_kind,
            fixture["insertion_method"],
            by_doc_id=fixture["doc_id"],
        ) == fixture["expected_matches_by_kind"][entity_kind]
        assert collection.where == fixture["expected_index_query"]

        no_matches = RefCollection([])
        assert _filter_by_insertion_method(
            SimpleNamespace(**{f"{entity_kind}_refs_collection": no_matches}),
            ids,
            entity_kind,
            "missing-method",
            by_doc_id=fixture["doc_id"],
        ) == ids


@pytest.mark.ci
def test_visualization_route_graph_type_selection_matches_shared_rust_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kogwistar.server_mcp_with_admin as server

    fixture_path = ROOT / "contracts/golden/visualization-graph-type-v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    engines = {
        "knowledge": object(),
        "conversation": object(),
        "workflow": object(),
        "wisdom": object(),
    }
    selected: list[object] = []

    monkeypatch.setattr(server, "engine", engines["knowledge"])
    monkeypatch.setattr(server, "conversation_engine", engines["conversation"])
    monkeypatch.setattr(server, "workflow_engine", engines["workflow"])
    monkeypatch.setattr(server, "wisdom_engine", engines["wisdom"])
    monkeypatch.setattr(
        server,
        "to_cytoscape",
        lambda engine, **_kwargs: selected.append(engine) or {"ok": True},
    )

    for case in fixture["cases"]:
        response = server.api_viz_cytoscape(graph_type=case["input"])
        assert response.body == b'{"ok":true}'
        assert selected.pop() is engines[case["expected"]]


@pytest.mark.ci
def test_graph_visualization_payloads_match_shared_python_model_fixture() -> None:
    fixture_path = ROOT / "contracts/golden/python-graph-model-payloads-v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    node_payload = next(case["payload"] for case in fixture["cases"] if case["kind"] == "node")
    edge_payload = next(case["payload"] for case in fixture["cases"] if case["kind"] == "edge")

    source = Node.model_validate(node_payload)
    target_payload = {**node_payload, "id": "parity-target", "label": "Target node"}
    target = Node.model_validate(target_payload)
    edge_payload = {
        **edge_payload,
        "source_ids": [source.id],
        "target_ids": [target.id],
    }
    edge = Edge.model_validate(edge_payload)

    class Backend:
        def node_get(self) -> dict[str, list[str]]:
            return {"ids": [source.id, target.id]}

        def edge_get(self) -> dict[str, list[str]]:
            return {"ids": [edge.id]}

    read = SimpleNamespace(
        load_node_map=lambda ids, **_kwargs: {
            node.id: node for node in (source, target) if node.id in ids
        },
        load_edge_map=lambda ids, **_kwargs: {edge.id: edge} if edge.id in ids else {},
    )
    engine = SimpleNamespace(kg_graph_type="knowledge", backend=Backend(), read=read)

    d3 = to_d3_force(engine)
    expected_nodes = []
    for model, entity_type, label in (
        (source, "entity", source.label),
        (target, "entity", target.label),
        (edge, "edge-node", edge.relation or edge.label or "edge"),
    ):
        value = model.model_dump()
        value.update(
            {
                "id": model.id,
                "label": label,
                "type": entity_type,
                "summary": model.summary,
                "properties": model.properties or {},
            }
        )
        expected_nodes.append(value)
    assert sorted(d3["nodes"], key=lambda item: item["id"]) == sorted(
        expected_nodes, key=lambda item: item["id"]
    )
    assert sorted(
        (link["source"], link["target"], link["role"], link["relation"], link["properties"])
        for link in d3["links"]
    ) == sorted(
        [
            (source.id, edge.id, "src", edge.relation, edge.properties),
            (edge.id, target.id, "tgt", edge.relation, edge.properties),
        ]
    )
    assert d3["mode"] == "reify" and d3["doc_id"] is None

    classic_d3 = to_d3_force(engine, mode="classic")
    expected_classic_links = [
        {
            "source": source.id,
            "target": target.id,
            "relation": edge.relation,
            "properties": edge.properties,
        }
    ]
    assert sorted(classic_d3["nodes"], key=lambda item: item["id"]) == sorted(
        [source.model_dump(), target.model_dump()], key=lambda item: item["id"]
    )
    assert classic_d3["links"] == expected_classic_links
    assert classic_d3["mode"] == "classic" and classic_d3["doc_id"] is None

    cytoscape = to_cytoscape(engine)
    expected_elements = [
        {
            "data": {"id": model.id, "label": label, "type": entity_type},
            "classes": css_class,
        }
        for model, entity_type, label, css_class in (
            (source, "entity", source.label, ""),
            (target, "entity", target.label, ""),
            (edge, "edge-node", edge.relation or edge.label or "edge", "edge-node"),
        )
    ]
    expected_elements.extend(
        [
            {
                "data": {
                    "id": f"{edge.id}::src::{source.id}",
                    "source": source.id,
                    "target": edge.id,
                    "label": edge.relation or edge.label,
                },
                "classes": "src",
            },
            {
                "data": {
                    "id": f"{edge.id}::tgt::{target.id}",
                    "source": edge.id,
                    "target": target.id,
                    "label": edge.relation or edge.label,
                },
                "classes": "tgt",
            },
        ]
    )
    assert sorted(cytoscape["elements"], key=lambda item: item["data"]["id"]) == sorted(
        expected_elements, key=lambda item: item["data"]["id"]
    )
    assert cytoscape["mode"] == "reify" and cytoscape["doc_id"] is None

    classic_cytoscape = to_cytoscape(engine, mode="classic")
    expected_classic_elements = [
        {
            "data": {"id": model.id, "label": model.label, "type": "entity"},
            "classes": "",
        }
        for model in (source, target)
    ]
    expected_classic_elements.append(
        {
            "data": {
                "id": f"{edge.id}::{source.id}->{target.id}",
                "source": source.id,
                "target": target.id,
                "label": edge.relation,
            },
            "classes": "",
        }
    )
    assert sorted(
        classic_cytoscape["elements"], key=lambda item: item["data"]["id"]
    ) == sorted(expected_classic_elements, key=lambda item: item["data"]["id"])
    assert classic_cytoscape["mode"] == "classic"

    outside_target_payload = {
        **node_payload,
        "id": "outside-target",
        "label": "Outside selected document",
        "doc_id": "another-document",
    }
    outside_target = Node.model_validate(outside_target_payload)
    omitted_endpoint_edge = Edge.model_validate(
        {
            **next(case["payload"] for case in fixture["cases"] if case["kind"] == "edge"),
            "id": "doc-scoped-edge",
            "doc_id": "selected-document",
            "source_ids": [source.id],
            "target_ids": [outside_target.id],
        }
    )
    doc_scoped_read = SimpleNamespace(
        node_ids_by_doc=lambda doc_id: [source.id],
        edge_ids_by_doc=lambda doc_id: [omitted_endpoint_edge.id],
        load_node_map=lambda ids, **_kwargs: {source.id: source}
        if source.id in ids
        else {},
        load_edge_map=lambda ids, **_kwargs: (
            {omitted_endpoint_edge.id: omitted_endpoint_edge}
            if omitted_endpoint_edge.id in ids
            else {}
        ),
    )
    doc_scoped_engine = SimpleNamespace(kg_graph_type="knowledge", read=doc_scoped_read)
    incomplete_d3 = to_d3_force(doc_scoped_engine, doc_id="selected-document")
    assert incomplete_d3["doc_id"] == "selected-document"
    assert [node["id"] for node in incomplete_d3["nodes"]] == [source.id]
    assert incomplete_d3["links"] == []

    incomplete_cytoscape = to_cytoscape(
        doc_scoped_engine, doc_id="selected-document"
    )
    assert incomplete_cytoscape["doc_id"] == "selected-document"
    assert [element["data"]["id"] for element in incomplete_cytoscape["elements"]] == [
        source.id
    ]


@pytest.mark.ci
def test_visualization_edge_label_fallback_matches_shared_contract() -> None:
    contract = json.loads(
        (ROOT / "contracts/golden/visualization-label-fallback-v1.json").read_text(
            encoding="utf-8"
        )
    )
    model_fixture = json.loads(
        (ROOT / "contracts/golden/python-graph-model-payloads-v1.json").read_text(
            encoding="utf-8"
        )
    )
    node_payload = next(
        case["payload"] for case in model_fixture["cases"] if case["kind"] == "node"
    )
    edge_template = next(
        case["payload"] for case in model_fixture["cases"] if case["kind"] == "edge"
    )
    source = Node.model_validate({**node_payload, "id": "source"})
    target = Node.model_validate({**node_payload, "id": "target"})
    for case in contract["cases"]:
        edge_payload = {
            **edge_template,
            "id": case["name"],
            "label": case["label"],
            "relation": case["relation"],
            "source_ids": [source.id],
            "target_ids": [target.id],
        }
        edge = Edge.model_validate(edge_payload)
        expected_label = case["expected"]
        assert (edge.relation or edge.label or "edge") == expected_label

        read = SimpleNamespace(
            load_node_map=lambda ids, **_kwargs: {
                node.id: node for node in (source, target) if node.id in ids
            },
            load_edge_map=lambda ids, **_kwargs: {edge.id: edge} if edge.id in ids else {},
        )
        backend = SimpleNamespace(
            node_get=lambda: {"ids": [source.id, target.id]},
            edge_get=lambda: {"ids": [edge.id]},
        )
        engine = SimpleNamespace(kg_graph_type="knowledge", backend=backend, read=read)
        d3 = to_d3_force(engine)
        edge_node = next(node for node in d3["nodes"] if node["id"] == edge.id)
        assert edge_node["label"] == expected_label
        assert all(link["relation"] == expected_label for link in d3["links"])
        classic_d3 = to_d3_force(engine, mode="classic")
        assert all(link["relation"] == expected_label for link in classic_d3["links"])

        reified_cytoscape = to_cytoscape(engine)
        assert all(
            element["data"].get("label") == expected_label
            for element in reified_cytoscape["elements"]
            if element["data"].get("source") == edge.id
            or element["data"].get("target") == edge.id
        )
        classic_cytoscape = to_cytoscape(engine, mode="classic")
        assert all(
            element["data"]["label"] == expected_label
            for element in classic_cytoscape["elements"]
            if element["data"].get("source") == source.id
        )


@pytest.mark.ci
def test_visualization_empty_doc_id_matches_python_route_selection_contract() -> None:
    fixture = json.loads(
        (ROOT / "contracts/golden/visualization-doc-filter-v1.json").read_text(
            encoding="utf-8"
        )
    )
    model_fixture = json.loads(
        (ROOT / "contracts/golden/python-graph-model-payloads-v1.json").read_text(
            encoding="utf-8"
        )
    )
    node_template = next(
        case["payload"] for case in model_fixture["cases"] if case["kind"] == "node"
    )
    edge_template = next(
        case["payload"] for case in model_fixture["cases"] if case["kind"] == "edge"
    )
    nodes = [
        Node.model_validate(
            {**node_template, "id": "node-a", "label": "A", "doc_id": "doc-a"}
        ),
        Node.model_validate(
            {**node_template, "id": "node-b", "label": "B", "doc_id": "doc-b"}
        ),
    ]
    edges = [
        Edge.model_validate(
            {
                **edge_template,
                "id": "edge-a",
                "doc_id": "doc-a",
                "source_ids": ["node-a"],
                "target_ids": ["node-a"],
            }
        ),
        Edge.model_validate(
            {
                **edge_template,
                "id": "edge-b",
                "doc_id": "doc-b",
                "source_ids": ["node-b"],
                "target_ids": ["node-b"],
            }
        ),
    ]
    read = SimpleNamespace(
        node_ids_by_doc=lambda doc_id: [
            node.id for node in nodes if node.doc_id == doc_id
        ],
        edge_ids_by_doc=lambda doc_id: [
            edge.id for edge in edges if edge.doc_id == doc_id
        ],
        load_node_map=lambda ids, **_kwargs: {
            node.id: node for node in nodes if node.id in ids
        },
        load_edge_map=lambda ids, **_kwargs: {
            edge.id: edge for edge in edges if edge.id in ids
        },
    )
    backend = SimpleNamespace(
        node_get=lambda: {"ids": [node.id for node in nodes]},
        edge_get=lambda: {"ids": [edge.id for edge in edges]},
    )
    engine = SimpleNamespace(kg_graph_type="knowledge", backend=backend, read=read)

    for case in fixture["cases"]:
        payload = to_d3_force(engine, doc_id=case["doc_id"])
        assert payload["doc_id"] == case["response_doc_id"]
        selected_doc_ids = sorted(
            node["doc_id"]
            for node in payload["nodes"]
            if node["type"] == "entity"
        )
        assert selected_doc_ids == case["selected_doc_ids"]
        selected_edge_doc_ids = sorted(
            node["doc_id"]
            for node in payload["nodes"]
            if node["type"] == "edge-node"
        )
        assert selected_edge_doc_ids == case["selected_edge_doc_ids"]


@pytest.mark.ci
def test_pretty_print_graph():
    doc_id = "doc::test::pretty_print_graph"
    engine = GraphKnowledgeEngine(
        persist_directory=str(Path("test_pretty_print_graph") / "kg"),
        backend_factory=build_fake_backend,
        embedding_function=lambda texts: [[0.1] * 384 for _ in texts],
    )

    def _span(excerpt: str) -> Span:
        return Span(
            collection_page_url=f"document_collection/{doc_id}",
            document_page_url=f"document/{doc_id}",
            doc_id=doc_id,
            insertion_method="test",
            page_number=1,
            start_char=0,
            end_char=min(len(excerpt), 4),
            excerpt=excerpt,
            context_before="",
            context_after="",
            chunk_id=None,
            source_cluster_id=None,
            verification=MentionVerification(
                method="human", is_verified=True, score=1.0, notes="test"
            ),
        )

    def _g(excerpt: str) -> Grounding:
        return Grounding(spans=[_span(excerpt)])

    node_a = Node(
        id="A",
        label="Smoking",
        type="entity",
        summary="Smoking",
        mentions=[_g("smok")],
        properties={},
        metadata={"entity_type": "test_node", "level_from_root": 0},
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        doc_id=doc_id,
        level_from_root=0,
    )
    node_b = Node(
        id="B",
        label="Lung Cancer",
        type="entity",
        summary="Lung Cancer",
        mentions=[_g("lung")],
        properties={},
        metadata={"entity_type": "test_node", "level_from_root": 0},
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        doc_id=doc_id,
        level_from_root=0,
    )
    edge = Edge(
        id="E1",
        label="Smoking causes Lung Cancer",
        type="relationship",
        summary="causal",
        relation="causes",
        source_ids=["A"],
        target_ids=["B"],
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[_g("smok")],
        doc_id=doc_id,
        properties={},
        metadata={"entity_type": "test_edge", "level_from_root": 0},
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )

    engine.write.add_node(node_a, doc_id=doc_id)
    engine.write.add_node(node_b, doc_id=doc_id)
    engine.write.add_edge(edge, doc_id=doc_id)
    visualiser = Visualizer(engine=engine)

    txt = visualiser.pretty_print_graph(by_doc_id=doc_id, include_refs=True)

    assert txt.startswith("Nodes:")
    assert "Smoking" in txt
    assert "Lung Cancer" in txt
    assert "Edges:" in txt
    assert "causes" in txt
    assert "(empty)" not in txt




def _span() -> Span:
    return Span(
        collection_page_url="test",
        document_page_url="test",
        doc_id="test",
        insertion_method="test",
        page_number=1,
        start_char=0,
        end_char=4,
        excerpt="test",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="human", is_verified=True, score=1.0, notes="test"
        ),
    )


def _g() -> Grounding:
    return Grounding(spans=[_span()])


def _wf_node(
    *, workflow_id: str, node_id: str, op: str, start=False, terminal=False
) -> WorkflowNode:
    return WorkflowNode(
        id=node_id,
        label=node_id.split("|")[-1],
        type="entity",
        doc_id=workflow_id,
        summary=op,
        mentions=[_g()],
        properties={},
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": op,
            "wf_start": start,
            "wf_terminal": terminal,
            "wf_version": "v1",
        },
        domain_id=None,
        canonical_entity_id=None,
    )


def _wf_edge(
    *,
    workflow_id: str,
    edge_id: str,
    src: str,
    dst: str,
    predicate: str | None,
    priority: int,
    is_default: bool,
) -> WorkflowEdge:
    return WorkflowEdge(
        id=edge_id,
        source_ids=[src],
        target_ids=[dst],
        relation="wf_next",
        label="wf_next",
        type="relationship",
        summary="next",
        doc_id=workflow_id,
        mentions=[_g()],
        properties={},
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_priority": priority,
            "wf_is_default": is_default,
            "wf_predicate": predicate,
            "wf_multiplicity": "one",
        },
        source_edge_ids=[],
        target_edge_ids=[],
        domain_id=None,
        canonical_entity_id=None,
    )


@pytest.mark.ci_full
def test_d3_litmus_workflow_self_and_parallel():

    wf_id = "wf_d3_litmus"
    wf_dir = Path("test_d3_litmus_workflow_self_and_parallel") / "wf"
    workflow_engine = GraphKnowledgeEngine(
        persist_directory=str(wf_dir), kg_graph_type="workflow"
    )

    # nodes
    workflow_engine.write.add_node(
        _wf_node(
            workflow_id=wf_id,
            node_id="A",
            op="noop",
        )
    )
    workflow_engine.write.add_node(
        _wf_node(
            workflow_id=wf_id,
            node_id="B",
            op="noop",
        )
    )

    # edges
    workflow_engine.write.add_edge(
        _wf_edge(
            workflow_id=wf_id,
            edge_id="A->B-1",
            src="A",
            dst="B",
            priority=100,
            is_default=True,
            predicate=None,
        )
    )

    workflow_engine.write.add_edge(
        _wf_edge(
            workflow_id=wf_id,
            edge_id="A->B-2",
            src="A",
            dst="B",
            priority=100,
            is_default=True,
            predicate=None,
        )
    )

    workflow_engine.write.add_edge(
        _wf_edge(
            workflow_id=wf_id,
            edge_id="A->A",
            src="A",
            dst="A",
            priority=100,
            is_default=True,
            predicate=None,
        )
    )

    # dump
    out_dir = Path("test_d3_litmus_workflow_self_and_parallel") / "d3"
    out_dir.mkdir(parents=True, exist_ok=True)
    template_html = Path("kogwistar") / "templates" / "d3.html"
    template_html = template_html.read_text(encoding="utf-8")
    dump_d3_bundle(
        template_html=template_html,
        engine=workflow_engine,
        # doc_id=wf_id,
        out_html=out_dir / "wf_self_and_multiple.html",
    )
    assert (out_dir / "wf_self_and_multiple.html").exists()
