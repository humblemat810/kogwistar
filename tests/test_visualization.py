import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from starlette.responses import JSONResponse

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import (
    Edge,
    Grounding,
    MentionVerification,
    Node,
    Span,
)
from kogwistar.runtime.models import WorkflowEdge, WorkflowNode
from kogwistar.server.auth_middleware import claims_ctx
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
        assert _filter_by_insertion_method(
            SimpleNamespace(**{f"{entity_kind}_refs_collection": collection}),
            ids,
            entity_kind,
            "",
            by_doc_id=fixture["doc_id"],
        ) == fixture["expected_empty_method_filter_by_kind"][entity_kind]

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
def test_cytoscape_shell_route_matches_shared_rust_contract() -> None:
    from fastapi.testclient import TestClient

    import kogwistar.server_mcp_with_admin as server

    contract = json.loads(
        (ROOT / "contracts/golden/visualization-cytoscape-shell-http-v1.json").read_text(
            encoding="utf-8"
        )
    )
    with TestClient(server.app) as client:
        response = client.get(contract["path_and_query"])
    assert response.status_code == contract["success_status"]
    assert response.headers["content-type"] == contract["content_type"]
    for fragment in contract["required_fragments"]:
        assert fragment in response.text
    for fragment in contract["forbidden_fragments"]:
        assert fragment not in response.text


@pytest.mark.ci
def test_d3_shell_route_matches_shared_rust_contract() -> None:
    from fastapi.testclient import TestClient

    import kogwistar.server_mcp_with_admin as server

    contract = json.loads(
        (ROOT / "contracts/golden/visualization-d3-shell-http-v1.json").read_text(
            encoding="utf-8"
        )
    )
    with TestClient(server.app) as client:
        response = client.get(contract["path_and_query"])
    assert response.status_code == contract["success_status"]
    assert response.headers["content-type"] == contract["content_type"]
    for fragment in contract["required_fragments"]:
        assert fragment in response.text
    for fragment in contract["forbidden_fragments"]:
        assert fragment not in response.text


@pytest.mark.ci
def test_go_shell_route_matches_shared_rust_contract() -> None:
    from fastapi.testclient import TestClient

    import kogwistar.server_mcp_with_admin as server

    contract = json.loads(
        (ROOT / "contracts/golden/visualization-go-shell-http-v1.json").read_text(
            encoding="utf-8"
        )
    )
    with TestClient(server.app) as client:
        response = client.get(contract["path_and_query"])
    assert response.status_code == contract["success_status"]
    assert response.headers["content-type"] == contract["content_type"]
    for fragment in contract["required_fragments"]:
        assert fragment in response.text
    for fragment in contract["forbidden_fragments"]:
        assert fragment not in response.text


@pytest.mark.ci
def test_visualization_uses_acl_policy_for_selected_python_graph_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kogwistar.server_mcp_with_admin as server

    graph_engines = {
        graph_type: GraphKnowledgeEngine(
            persist_directory=str(tmp_path / graph_type),
            backend_factory=build_fake_backend,
            kg_graph_type=graph_type,
            acl_enabled=True,
        )
        for graph_type in ("knowledge", "conversation", "workflow", "wisdom")
    }
    acl_fixture = json.loads(
        (ROOT / "contracts/golden/graph-acl-v1.json").read_text(encoding="utf-8")
    )
    isolation = acl_fixture["graph_type_isolation"]
    owner_policy = next(
        target for target in acl_fixture["targets"] if target["id"] == "node-owner"
    )
    private_policy = next(
        target for target in acl_fixture["targets"] if target["id"] == "node-private"
    )
    scope_policy = next(
        target for target in acl_fixture["targets"] if target["id"] == "node-scope"
    )
    no_acl_node_policy = next(
        target for target in acl_fixture["targets"] if target["id"] == "node-no-acl"
    )
    share_policies = [
        target
        for target in acl_fixture["targets"]
        if target.get("grain") == "node"
        and target.get("mode") in {"shared", "group"}
    ]
    edge_share_policies = [
        target
        for target in acl_fixture["targets"]
        if target.get("grain") == "edge"
        and target.get("mode") in {"shared", "group"}
    ]
    edge_policy = next(
        target for target in acl_fixture["targets"] if target["id"] == "edge-public"
    )
    private_edge_policy = next(
        target for target in acl_fixture["targets"] if target["id"] == "edge-private"
    )
    scope_edge_policy = next(
        target for target in acl_fixture["targets"] if target["id"] == "edge-scope"
    )
    no_acl_edge_policy = next(
        target for target in acl_fixture["targets"] if target["id"] == "edge-no-acl"
    )
    policies = {
        case["truth_graph"]: case
        for case in isolation["cases"]
    }
    try:
        monkeypatch.setattr(server, "engine", graph_engines["knowledge"])
        monkeypatch.setattr(server, "conversation_engine", graph_engines["conversation"])
        monkeypatch.setattr(server, "workflow_engine", graph_engines["workflow"])
        monkeypatch.setattr(server, "wisdom_engine", graph_engines["wisdom"])

        for graph_type, engine in graph_engines.items():
            node = Node(
                id=isolation["entity_id"],
                label="Graph-scoped node",
                type="entity",
                summary="Graph-scoped ACL fixture",
                mentions=[_g()],
                metadata={"entity_type": "fixture"},
                embedding=None,
                doc_id=None,
                level_from_root=0,
            )
            engine.raw_write.add_node(node)
            policy = policies.get(
                graph_type,
                {"mode": "public", "expected_visible": True},
            )
            engine.acl_graph.add_record(
                truth_graph=graph_type,
                entity_id=node.id,
                version=1,
                mode=policy["mode"],
                owner_id=policy.get("owner_id"),
            )
            grounding_ids, span_ids = engine.acl.usage_ids_for_item(node)
            for grain, target_ids in (
                ("grounding", grounding_ids),
                ("span", span_ids),
            ):
                for target_id in target_ids:
                    engine.acl_graph.add_record(
                        grain=grain,
                        truth_graph=graph_type,
                        entity_id=node.id,
                        target_item_id=target_id,
                        version=1,
                        mode="public",
                    )

        conversation_edge_id = "conversation-private-edge"
        conversation_edge = Edge(
            id=conversation_edge_id,
            label="Conversation private edge",
            type="relationship",
            summary="Private edge with no endpoint projection",
            relation="references",
            source_ids=[],
            target_ids=[],
            source_edge_ids=[],
            target_edge_ids=[],
            mentions=[_g()],
            doc_id=None,
            properties={},
            metadata={"entity_type": "fixture"},
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
        conversation_engine = graph_engines["conversation"]
        conversation_engine.raw_write.add_edge(conversation_edge)
        conversation_edge_acl = {
            "truth_graph": "conversation",
            "entity_id": conversation_edge_id,
            "version": 1,
            "mode": policies["conversation"]["mode"],
            "owner_id": policies["conversation"]["owner_id"],
        }
        conversation_engine.acl_graph.add_record(
            grain="edge", **conversation_edge_acl
        )
        grounding_ids, span_ids = conversation_engine.acl.usage_ids_for_item(
            conversation_edge
        )
        for grain, target_ids in (("grounding", grounding_ids), ("span", span_ids)):
            for target_id in target_ids:
                conversation_engine.acl_graph.add_record(
                    grain=grain,
                    target_item_id=target_id,
                    **conversation_edge_acl,
                )

        workflow_edge_id = "workflow-private-edge"
        workflow_edge = Edge(
            id=workflow_edge_id,
            label="Workflow private edge",
            type="relationship",
            summary="Private edge with no endpoint projection",
            relation="references",
            source_ids=[],
            target_ids=[],
            source_edge_ids=[],
            target_edge_ids=[],
            mentions=[_g()],
            doc_id=None,
            properties={},
            metadata={"entity_type": "fixture"},
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
        workflow_engine = graph_engines["workflow"]
        workflow_engine.raw_write.add_edge(workflow_edge)
        workflow_edge_acl = {
            "truth_graph": "workflow",
            "entity_id": workflow_edge_id,
            "version": 1,
            "mode": "private",
            "owner_id": "workflow-owner",
        }
        workflow_engine.acl_graph.add_record(grain="edge", **workflow_edge_acl)
        grounding_ids, span_ids = workflow_engine.acl.usage_ids_for_item(workflow_edge)
        for grain, target_ids in (("grounding", grounding_ids), ("span", span_ids)):
            for target_id in target_ids:
                workflow_engine.acl_graph.add_record(
                    grain=grain,
                    target_item_id=target_id,
                    **workflow_edge_acl,
                )

        wisdom_edge_id = "wisdom-private-edge"
        wisdom_edge = Edge(
            id=wisdom_edge_id,
            label="Wisdom private edge",
            type="relationship",
            summary="Private wisdom edge with no endpoint projection",
            relation="references",
            source_ids=[],
            target_ids=[],
            source_edge_ids=[],
            target_edge_ids=[],
            mentions=[_g()],
            doc_id=None,
            properties={},
            metadata={"entity_type": "fixture"},
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
        wisdom_engine = graph_engines["wisdom"]
        wisdom_engine.raw_write.add_edge(wisdom_edge)
        wisdom_edge_acl = {
            "truth_graph": "wisdom",
            "entity_id": wisdom_edge_id,
            "version": 1,
            "mode": "private",
            "owner_id": "wisdom-owner",
        }
        wisdom_engine.acl_graph.add_record(grain="edge", **wisdom_edge_acl)
        grounding_ids, span_ids = wisdom_engine.acl.usage_ids_for_item(wisdom_edge)
        for grain, target_ids in (("grounding", grounding_ids), ("span", span_ids)):
            for target_id in target_ids:
                wisdom_engine.acl_graph.add_record(
                    grain=grain,
                    target_item_id=target_id,
                    **wisdom_edge_acl,
                )

        def add_wisdom_shared_node(
            node_id: str,
            *,
            mode: str,
            shared_with_principals: list[str] | None = None,
            shared_with_groups: list[str] | None = None,
        ) -> None:
            node = Node(
                id=node_id,
                label=node_id,
                type="entity",
                summary="Wisdom graph role sharing fixture",
                mentions=[_g()],
                metadata={"entity_type": "fixture"},
                embedding=None,
                doc_id=None,
                level_from_root=0,
            )
            wisdom_engine.raw_write.add_node(node)
            acl_record: dict[str, Any] = {
                "truth_graph": "wisdom",
                "entity_id": node_id,
                "version": 1,
                "mode": mode,
            }
            if shared_with_principals is not None:
                acl_record["shared_with_principals"] = shared_with_principals
            if shared_with_groups is not None:
                acl_record["shared_with_groups"] = shared_with_groups
            wisdom_engine.acl_graph.add_record(**acl_record)
            usage_grounding_ids, usage_span_ids = wisdom_engine.acl.usage_ids_for_item(
                node
            )
            for grain, target_ids in (
                ("grounding", usage_grounding_ids),
                ("span", usage_span_ids),
            ):
                for target_id in target_ids:
                    wisdom_engine.acl_graph.add_record(
                        grain=grain,
                        target_item_id=target_id,
                        **acl_record,
                    )

        wisdom_direct_id = "wisdom-direct-share-node"
        wisdom_group_id = "wisdom-group-share-node"
        add_wisdom_shared_node(
            wisdom_direct_id,
            mode="shared",
            shared_with_principals=["wisdom-direct-reader"],
        )
        add_wisdom_shared_node(
            wisdom_group_id,
            mode="group",
            shared_with_groups=["wisdom-reviewers"],
        )

        owner_engine = graph_engines[acl_fixture["truth_graph"]]
        no_acl_node = Node(
            id=no_acl_node_policy["id"],
            label="Node without an ACL record",
            type="entity",
            summary="Must fail closed when no ACL exists",
            mentions=[_g()],
            metadata={"entity_type": "fixture"},
            embedding=None,
            doc_id=None,
            level_from_root=0,
        )
        owner_engine.raw_write.add_node(no_acl_node)
        owner_node = Node(
            id=owner_policy["id"],
            label="Owner-only graph node",
            type="entity",
            summary="Private node visible to its owner",
            mentions=[_g()],
            metadata={"entity_type": "fixture"},
            embedding=None,
            doc_id=None,
            level_from_root=0,
        )
        owner_engine.raw_write.add_node(owner_node)
        owner_engine.acl_graph.add_record(
            truth_graph=acl_fixture["truth_graph"],
            entity_id=owner_node.id,
            version=1,
            mode=owner_policy["mode"],
            owner_id=owner_policy["owner_id"],
        )
        grounding_ids, span_ids = owner_engine.acl.usage_ids_for_item(owner_node)
        for grain, target_ids in (("grounding", grounding_ids), ("span", span_ids)):
            for target_id in target_ids:
                owner_engine.acl_graph.add_record(
                    grain=grain,
                    truth_graph=acl_fixture["truth_graph"],
                    entity_id=owner_node.id,
                    target_item_id=target_id,
                    version=1,
                    mode=owner_policy["mode"],
                    owner_id=owner_policy["owner_id"],
                )
        private_node = Node(
            id=private_policy["id"],
            label="Other owner's graph node",
            type="entity",
            summary="Private node owned by another principal",
            mentions=[_g()],
            metadata={"entity_type": "fixture"},
            embedding=None,
            doc_id=None,
            level_from_root=0,
        )
        owner_engine.raw_write.add_node(private_node)
        owner_engine.acl_graph.add_record(
            truth_graph=acl_fixture["truth_graph"],
            entity_id=private_node.id,
            version=1,
            mode=private_policy["mode"],
            owner_id=private_policy["owner_id"],
        )
        grounding_ids, span_ids = owner_engine.acl.usage_ids_for_item(private_node)
        for grain, target_ids in (("grounding", grounding_ids), ("span", span_ids)):
            for target_id in target_ids:
                owner_engine.acl_graph.add_record(
                    grain=grain,
                    truth_graph=acl_fixture["truth_graph"],
                    entity_id=private_node.id,
                    target_item_id=target_id,
                    version=1,
                    mode=private_policy["mode"],
                    owner_id=private_policy["owner_id"],
                )
        scope_node = Node(
            id=scope_policy["id"],
            label="Team-scoped graph node",
            type="entity",
            summary="Visible only within its security scope",
            mentions=[_g()],
            metadata={"entity_type": "fixture"},
            embedding=None,
            doc_id=None,
            level_from_root=0,
        )
        owner_engine.raw_write.add_node(scope_node)
        scope_acl = {
            "truth_graph": acl_fixture["truth_graph"],
            "entity_id": scope_node.id,
            "version": 1,
            "mode": scope_policy["mode"],
            "security_scope": scope_policy["security_scope"],
        }
        owner_engine.acl_graph.add_record(**scope_acl)
        grounding_ids, span_ids = owner_engine.acl.usage_ids_for_item(scope_node)
        for grain, target_ids in (("grounding", grounding_ids), ("span", span_ids)):
            for target_id in target_ids:
                owner_engine.acl_graph.add_record(
                    grain=grain,
                    target_item_id=target_id,
                    **scope_acl,
                )
        for share_policy in share_policies:
            shared_node = Node(
                id=share_policy["id"],
                label=f"{share_policy['mode']} graph node",
                type="entity",
                summary="Visible only to an explicitly shared principal or group",
                mentions=[_g()],
                metadata={"entity_type": "fixture"},
                embedding=None,
                doc_id=None,
                level_from_root=0,
            )
            owner_engine.raw_write.add_node(shared_node)
            share_acl: dict[str, Any] = {
                "truth_graph": acl_fixture["truth_graph"],
                "entity_id": shared_node.id,
                "version": 1,
                "mode": share_policy["mode"],
            }
            if share_policy["mode"] == "shared":
                share_acl["shared_with_principals"] = share_policy[
                    "shared_with_principals"
                ]
            else:
                share_acl["shared_with_groups"] = share_policy["shared_with_groups"]
            owner_engine.acl_graph.add_record(**share_acl)
            grounding_ids, span_ids = owner_engine.acl.usage_ids_for_item(shared_node)
            for grain, target_ids in (("grounding", grounding_ids), ("span", span_ids)):
                for target_id in target_ids:
                    owner_engine.acl_graph.add_record(
                        grain=grain,
                        target_item_id=target_id,
                        **share_acl,
                    )
        token = claims_ctx.set(
            {"agent_id": "reader-b", "ns": "knowledge"}
        )
        try:
            route_results: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
            for graph_type in graph_engines:
                d3_response = cast(
                    JSONResponse,
                    server.api_viz_d3(mode="classic", graph_type=graph_type),
                )
                cytoscape_response = cast(
                    JSONResponse,
                    server.api_viz_cytoscape(mode="classic", graph_type=graph_type),
                )
                route_results[graph_type] = (
                    json.loads(bytes(d3_response.body)),
                    json.loads(bytes(cytoscape_response.body)),
                )
        finally:
            claims_ctx.reset(token)

        for graph_type in graph_engines:
            d3_payload, cytoscape_payload = route_results[graph_type]
            expected_ids = (
                [isolation["entity_id"]]
                if policies.get(
                    graph_type, {"expected_visible": True}
                )["expected_visible"]
                else []
            )
            assert [node["id"] for node in d3_payload["nodes"]] == expected_ids
            assert [
                item["data"]["id"] for item in cytoscape_payload["elements"]
            ] == expected_ids

        token = claims_ctx.set(
            {"agent_id": "reader-b", "ns": "conversation"}
        )
        try:
            conversation_non_owner_response = cast(
                JSONResponse,
                server.api_viz_d3(mode="reify", graph_type="conversation"),
            )
            conversation_non_owner_payload = json.loads(
                bytes(conversation_non_owner_response.body)
            )
        finally:
            claims_ctx.reset(token)
        assert conversation_edge_id not in [
            node["id"] for node in conversation_non_owner_payload["nodes"]
        ]

        conversation_owner_id = policies["conversation"]["owner_id"]
        token = claims_ctx.set(
            {"agent_id": conversation_owner_id, "ns": "conversation"}
        )
        try:
            conversation_owner_response = cast(
                JSONResponse,
                server.api_viz_d3(mode="reify", graph_type="conversation"),
            )
            conversation_owner_payload = json.loads(
                bytes(conversation_owner_response.body)
            )
        finally:
            claims_ctx.reset(token)
        assert isolation["entity_id"] in [
            node["id"] for node in conversation_owner_payload["nodes"]
        ]
        assert conversation_edge_id in [
            node["id"] for node in conversation_owner_payload["nodes"]
        ]

        for principal_id, expected_visible in (
            ("workflow-reader", False),
            ("workflow-owner", True),
        ):
            token = claims_ctx.set(
                {"agent_id": principal_id, "ns": "workflow"}
            )
            try:
                workflow_response = cast(
                    JSONResponse,
                    server.api_viz_d3(mode="reify", graph_type="workflow"),
                )
                workflow_payload = json.loads(bytes(workflow_response.body))
            finally:
                claims_ctx.reset(token)
            assert (workflow_edge_id in [
                node["id"] for node in workflow_payload["nodes"]
            ]) is expected_visible

        for principal_id, expected_visible in (
            ("wisdom-reader", False),
            ("wisdom-owner", True),
        ):
            token = claims_ctx.set({"agent_id": principal_id, "ns": "wisdom"})
            try:
                wisdom_response = cast(
                    JSONResponse,
                    server.api_viz_d3(mode="reify", graph_type="wisdom"),
                )
                wisdom_payload = json.loads(bytes(wisdom_response.body))
            finally:
                claims_ctx.reset(token)
            assert (wisdom_edge_id in [
                node["id"] for node in wisdom_payload["nodes"]
            ]) is expected_visible

        for principal_id, groups, expected_id in (
            ("wisdom-direct-reader", [], wisdom_direct_id),
            ("wisdom-group-reader", ["wisdom-reviewers"], wisdom_group_id),
        ):
            token = claims_ctx.set(
                {"agent_id": principal_id, "ns": "wisdom", "groups": groups}
            )
            try:
                sharing_response = cast(
                    JSONResponse,
                    server.api_viz_d3(mode="reify", graph_type="wisdom"),
                )
                sharing_payload = json.loads(bytes(sharing_response.body))
            finally:
                claims_ctx.reset(token)
            visible_ids = [node["id"] for node in sharing_payload["nodes"]]
            assert expected_id in visible_ids
            assert (wisdom_direct_id in visible_ids) is (
                principal_id == "wisdom-direct-reader"
            )
            assert (wisdom_group_id in visible_ids) is (
                principal_id == "wisdom-group-reader"
            )

        for edge_policy_item in (
            edge_policy,
            private_edge_policy,
            scope_edge_policy,
            *edge_share_policies,
        ):
            edge = Edge(
                id=edge_policy_item["id"],
                label=f"{edge_policy_item['mode']} fixture edge",
                type="relationship",
                summary="Edge ACL fixture",
                relation="references",
                source_ids=[isolation["entity_id"]],
                target_ids=[isolation["entity_id"]],
                source_edge_ids=[],
                target_edge_ids=[],
                mentions=[_g()],
                doc_id=None,
                properties={},
                metadata={"entity_type": "fixture"},
                domain_id=None,
                canonical_entity_id=None,
                embedding=None,
            )
            owner_engine.raw_write.add_edge(edge)
            edge_acl: dict[str, Any] = {
                "truth_graph": acl_fixture["truth_graph"],
                "entity_id": edge.id,
                "version": 1,
                "mode": edge_policy_item["mode"],
            }
            if owner_id := edge_policy_item.get("owner_id"):
                edge_acl["owner_id"] = owner_id
            if security_scope := edge_policy_item.get("security_scope"):
                edge_acl["security_scope"] = security_scope
            if shared_with_principals := edge_policy_item.get("shared_with_principals"):
                edge_acl["shared_with_principals"] = shared_with_principals
            if shared_with_groups := edge_policy_item.get("shared_with_groups"):
                edge_acl["shared_with_groups"] = shared_with_groups
            owner_engine.acl_graph.add_record(grain="edge", **edge_acl)
            grounding_ids, span_ids = owner_engine.acl.usage_ids_for_item(edge)
            for grain, target_ids in (("grounding", grounding_ids), ("span", span_ids)):
                for target_id in target_ids:
                    owner_engine.acl_graph.add_record(
                        grain=grain,
                        target_item_id=target_id,
                        **edge_acl,
                    )
        no_acl_edge = Edge(
            id=no_acl_edge_policy["id"],
            label="Edge without an ACL record",
            type="relationship",
            summary="Must fail closed when no ACL exists",
            relation="references",
            source_ids=[isolation["entity_id"]],
            target_ids=[isolation["entity_id"]],
            source_edge_ids=[],
            target_edge_ids=[],
            mentions=[_g()],
            doc_id=None,
            properties={},
            metadata={"entity_type": "fixture"},
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
        owner_engine.raw_write.add_edge(no_acl_edge)
        token = claims_ctx.set({"agent_id": "reader-b", "ns": "knowledge"})
        try:
            public_edge_d3 = cast(
                JSONResponse,
                server.api_viz_d3(
                    mode="reify", graph_type=acl_fixture["truth_graph"]
                ),
            )
            public_edge_cytoscape = cast(
                JSONResponse,
                server.api_viz_cytoscape(
                    mode="reify", graph_type=acl_fixture["truth_graph"]
                ),
            )
            public_edge_d3_payload = json.loads(bytes(public_edge_d3.body))
            public_edge_cytoscape_payload = json.loads(bytes(public_edge_cytoscape.body))
        finally:
            claims_ctx.reset(token)
        assert edge_policy["id"] in [
            node["id"] for node in public_edge_d3_payload["nodes"]
        ]
        assert edge_policy["id"] in [
            item["data"]["id"] for item in public_edge_cytoscape_payload["elements"]
        ]
        assert private_edge_policy["id"] not in [
            node["id"] for node in public_edge_d3_payload["nodes"]
        ]
        assert scope_edge_policy["id"] not in [
            node["id"] for node in public_edge_d3_payload["nodes"]
        ]
        for share_policy in edge_share_policies:
            assert share_policy["id"] not in [
                node["id"] for node in public_edge_d3_payload["nodes"]
            ]
        assert no_acl_node_policy["id"] not in [
            node["id"] for node in public_edge_d3_payload["nodes"]
        ]
        assert no_acl_edge_policy["id"] not in [
            node["id"] for node in public_edge_d3_payload["nodes"]
        ]
        assert owner_engine.acl_graph.latest_record(
            grain="node",
            truth_graph=acl_fixture["truth_graph"],
            entity_id=no_acl_node_policy["id"],
        ) is None

        for principal_id, expected_visible in (
            (owner_policy["owner_id"], False),
            (private_edge_policy["owner_id"], True),
        ):
            token = claims_ctx.set(
                {
                    "agent_id": principal_id,
                    "ns": acl_fixture["truth_graph"],
                    "security_scope": acl_fixture["principal"]["security_scope"],
                }
            )
            try:
                private_edge_response = cast(
                    JSONResponse,
                    server.api_viz_d3(
                        mode="reify", graph_type=acl_fixture["truth_graph"]
                    ),
                )
                private_edge_payload = json.loads(bytes(private_edge_response.body))
            finally:
                claims_ctx.reset(token)
            assert (
                private_edge_policy["id"]
                in [node["id"] for node in private_edge_payload["nodes"]]
            ) is expected_visible

        token = claims_ctx.set(
            {
                "agent_id": owner_policy["owner_id"],
                "ns": acl_fixture["truth_graph"],
                "security_scope": acl_fixture["principal"]["security_scope"],
                "groups": acl_fixture["principal"]["groups"],
            }
        )
        try:
            owner_d3 = cast(
                JSONResponse,
                server.api_viz_d3(mode="classic", graph_type=acl_fixture["truth_graph"]),
            )
            owner_cytoscape = cast(
                JSONResponse,
                server.api_viz_cytoscape(
                    mode="classic", graph_type=acl_fixture["truth_graph"]
                ),
            )
            owner_d3_payload = json.loads(bytes(owner_d3.body))
            owner_cytoscape_payload = json.loads(bytes(owner_cytoscape.body))
        finally:
            claims_ctx.reset(token)
        assert owner_policy["id"] in [
            node["id"] for node in owner_d3_payload["nodes"]
        ]
        assert owner_policy["id"] in [
            item["data"]["id"] for item in owner_cytoscape_payload["elements"]
        ]
        assert private_policy["id"] not in [
            node["id"] for node in owner_d3_payload["nodes"]
        ]
        assert private_policy["id"] not in [
            item["data"]["id"] for item in owner_cytoscape_payload["elements"]
        ]
        assert scope_policy["id"] in [
            node["id"] for node in owner_d3_payload["nodes"]
        ]
        assert scope_policy["id"] in [
            item["data"]["id"] for item in owner_cytoscape_payload["elements"]
        ]
        token = claims_ctx.set(
            {
                "agent_id": owner_policy["owner_id"],
                "ns": acl_fixture["truth_graph"],
                "security_scope": acl_fixture["principal"]["security_scope"],
            }
        )
        try:
            scoped_edge_response = cast(
                JSONResponse,
                server.api_viz_d3(
                    mode="reify", graph_type=acl_fixture["truth_graph"]
                ),
            )
            scoped_edge_payload = json.loads(bytes(scoped_edge_response.body))
        finally:
            claims_ctx.reset(token)
        assert scope_edge_policy["id"] in [
            node["id"] for node in scoped_edge_payload["nodes"]
        ]
        for share_policy in share_policies:
            assert share_policy["id"] in [
                node["id"] for node in owner_d3_payload["nodes"]
            ]

        token = claims_ctx.set(
            {
                "agent_id": owner_policy["owner_id"],
                "ns": acl_fixture["truth_graph"],
                "security_scope": acl_fixture["principal"]["security_scope"],
            }
        )
        try:
            directly_shared = cast(
                JSONResponse,
                server.api_viz_d3(
                    mode="reify", graph_type=acl_fixture["truth_graph"]
                ),
            )
            directly_shared_payload = json.loads(bytes(directly_shared.body))
        finally:
            claims_ctx.reset(token)
        assert share_policies[0]["id"] in [
            node["id"] for node in directly_shared_payload["nodes"]
        ]
        assert all(
            policy["id"] not in [
                node["id"] for node in directly_shared_payload["nodes"]
            ]
            for policy in share_policies
            if policy["mode"] == "group"
        )
        assert edge_share_policies[0]["id"] in [
            node["id"] for node in directly_shared_payload["nodes"]
        ]
        assert edge_share_policies[1]["id"] not in [
            node["id"] for node in directly_shared_payload["nodes"]
        ]

        token = claims_ctx.set(
            {
                "agent_id": owner_policy["owner_id"],
                "ns": acl_fixture["truth_graph"],
                "security_scope": acl_fixture["principal"]["security_scope"],
                "groups": acl_fixture["principal"]["groups"],
            }
        )
        try:
            group_shared = cast(
                JSONResponse,
                server.api_viz_d3(
                    mode="reify", graph_type=acl_fixture["truth_graph"]
                ),
            )
            group_shared_payload = json.loads(bytes(group_shared.body))
        finally:
            claims_ctx.reset(token)
        assert edge_share_policies[1]["id"] in [
            node["id"] for node in group_shared_payload["nodes"]
        ]

        token = claims_ctx.set(
            {
                "agent_id": owner_policy["owner_id"],
                "ns": acl_fixture["truth_graph"],
                "security_scope": "team-b",
            }
        )
        try:
            mismatched_scope = cast(
                JSONResponse,
                server.api_viz_d3(
                    mode="classic", graph_type=acl_fixture["truth_graph"]
                ),
            )
            mismatched_scope_payload = json.loads(bytes(mismatched_scope.body))
        finally:
            claims_ctx.reset(token)
        assert scope_policy["id"] not in [
            node["id"] for node in mismatched_scope_payload["nodes"]
        ]
        assert scope_edge_policy["id"] not in [
            node["id"] for node in mismatched_scope_payload["nodes"]
        ]
    finally:
        for engine in graph_engines.values():
            engine.close()


@pytest.mark.ci
def test_visualization_public_routes_forward_shared_fixture_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kogwistar.server_mcp_with_admin as server

    fixture_path = ROOT / "contracts/golden/visualization-insertion-filter-v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    engine = object()
    calls: list[tuple[str, object, dict[str, object]]] = []
    d3_payload = {"nodes": [{"id": "n1"}, {"id": "e1"}], "links": []}
    cytoscape_payload = {"elements": [{"data": {"id": "n1"}}]}

    def record_d3(selected_engine: object, **kwargs: object) -> dict[str, object]:
        calls.append(("d3", selected_engine, kwargs))
        return d3_payload

    def record_cytoscape(
        selected_engine: object, **kwargs: object
    ) -> dict[str, object]:
        calls.append(("cytoscape", selected_engine, kwargs))
        return cytoscape_payload

    monkeypatch.setattr(server, "engine", engine)
    monkeypatch.setattr(server, "to_d3_force", record_d3)
    monkeypatch.setattr(server, "to_cytoscape", record_cytoscape)

    options = {
        "doc_id": fixture["doc_id"],
        "mode": "reify",
        "insertion_method": fixture["insertion_method"],
    }
    d3_response = server.api_viz_d3(**options, graph_type="knowledge")
    cytoscape_response = server.api_viz_cytoscape(**options, graph_type="knowledge")

    assert json.loads(d3_response.body) == d3_payload
    assert json.loads(cytoscape_response.body) == cytoscape_payload
    assert calls == [
        ("d3", engine, options),
        ("cytoscape", engine, options),
    ]


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
def test_doc_visualization_fallback_matches_metadata_doc_membership() -> None:
    from kogwistar.visualization.graph_viz import _ids_by_doc

    class Backend:
        def node_docs_get(self, **_kwargs: object) -> dict[str, object]:
            return {"metadatas": []}

        def edge_endpoints_get(self, **_kwargs: object) -> dict[str, object]:
            return {"metadatas": []}

        def node_get(self, **_kwargs: object) -> dict[str, object]:
            return {
                "ids": ["node-doc-ids", "node-payload-metadata", "node-other-doc"],
                "documents": [
                    '{"id":"node-doc-ids"}',
                    '{"id":"node-payload-metadata","metadata":{"doc_id":"doc-a"}}',
                    '{"id":"node-other-doc"}',
                ],
                "metadatas": [
                    {"doc_ids": ["doc-a"]},
                    {},
                    {"doc_ids": ["doc-b"]},
                ],
            }

        def edge_get(self, **_kwargs: object) -> dict[str, object]:
            return {
                "ids": ["edge-doc-id", "edge-other-doc"],
                "documents": [
                    '{"id":"edge-doc-id"}',
                    '{"id":"edge-other-doc"}',
                ],
                "metadatas": [
                    {"doc_id": "doc-a"},
                    {"doc_id": "doc-b"},
                ],
            }

    empty_index = SimpleNamespace(
        node_ids_by_doc=lambda _doc_id: [],
        edge_ids_by_doc=lambda _doc_id: [],
    )
    nodes, edges = _ids_by_doc(SimpleNamespace(backend=Backend(), read=empty_index), "doc-a")
    assert nodes == ["node-doc-ids", "node-payload-metadata"]
    assert edges == ["edge-doc-id"]


@pytest.mark.ci
def test_hyperedge_visualization_matches_shared_python_contract() -> None:
    fixture = json.loads(
        (ROOT / "contracts/golden/visualization-hyperedge-v1.json").read_text(
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
        Node.model_validate({**node_template, **node}) for node in fixture["nodes"]
    ]
    edges = [
        Edge.model_validate(
            {
                **edge_template,
                **edge,
                "label": edge["relation"],
            }
        )
        for edge in fixture["edges"]
    ]
    node_map = {node.id: node for node in nodes}
    edge_map = {edge.id: edge for edge in edges}
    backend = SimpleNamespace(
        node_get=lambda: {"ids": list(node_map)},
        edge_get=lambda: {"ids": list(edge_map)},
    )
    read = SimpleNamespace(
        load_node_map=lambda ids, **_kwargs: {
            identifier: node_map[identifier]
            for identifier in ids
            if identifier in node_map
        },
        load_edge_map=lambda ids, **_kwargs: {
            identifier: edge_map[identifier]
            for identifier in ids
            if identifier in edge_map
        },
    )
    engine = SimpleNamespace(
        kg_graph_type="knowledge", backend=backend, read=read
    )

    d3 = to_d3_force(engine, mode="reify")
    assert [node["id"] for node in d3["nodes"]] == fixture["expected"][
        "reified_d3_node_order"
    ]
    actual_reified_links = {
        (link["source"], link["target"], link["role"], link["relation"])
        for link in d3["links"]
    }
    expected_reified_links = {
        (link["source"], link["target"], link["role"], link["relation"])
        for link in fixture["expected"]["reified_d3_links"]
    }
    assert actual_reified_links == expected_reified_links

    classic_d3 = to_d3_force(engine, mode="classic")
    assert [node["id"] for node in classic_d3["nodes"]] == fixture["expected"][
        "classic_d3_node_order"
    ]
    assert {
        (link["source"], link["target"], link["relation"])
        for link in classic_d3["links"]
    } == {
        (link["source"], link["target"], link["relation"])
        for link in fixture["expected"]["classic_d3_links"]
    }

    reified_cytoscape = to_cytoscape(engine, mode="reify")
    assert [element["data"]["id"] for element in reified_cytoscape["elements"]] == (
        fixture["expected"]["reified_cytoscape_element_order"]
    )
    actual_reified_cytoscape_links = {
        (
            element["data"]["id"],
            element["data"].get("source"),
            element["data"].get("target"),
            element["data"].get("label"),
            element["classes"],
        )
        for element in reified_cytoscape["elements"]
        if element["classes"] in {"src", "tgt"}
    }
    assert actual_reified_cytoscape_links == {
        (
            element["id"],
            element["source"],
            element["target"],
            element["label"],
            element["class"],
        )
        for element in fixture["expected"]["reified_cytoscape_links"]
    }

    classic_cytoscape = to_cytoscape(engine, mode="classic")
    assert [element["data"]["id"] for element in classic_cytoscape["elements"]] == (
        fixture["expected"]["classic_cytoscape_element_order"]
    )
    assert {
        (
            element["data"]["id"],
            element["data"]["source"],
            element["data"]["target"],
            element["data"]["label"],
            element["classes"],
        )
        for element in classic_cytoscape["elements"]
        if "source" in element["data"]
    } == {
        (
            element["id"],
            element["source"],
            element["target"],
            element["label"],
            element["class"],
        )
        for element in fixture["expected"]["classic_cytoscape_links"]
    }


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
