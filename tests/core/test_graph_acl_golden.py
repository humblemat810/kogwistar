from __future__ import annotations

import json
from pathlib import Path

from kogwistar.acl.graph import ACLGraph


def test_python_acl_visibility_matches_shared_golden_contract() -> None:
    fixture = json.loads(
        (Path(__file__).resolve().parents[2] / "contracts/golden/graph-acl-v1.json")
        .read_text(encoding="utf-8")
    )
    assert fixture["schema_version"] == 1
    principal = fixture["principal"]
    acl = ACLGraph(cache_enabled=True)
    for target in fixture["targets"]:
        if "mode" not in target:
            continue
        acl.add_record(
            grain=target["grain"],
            truth_graph=fixture["truth_graph"],
            entity_id=target["id"],
            version=1,
            mode=target["mode"],
            owner_id=target.get("owner_id"),
            security_scope=target.get("security_scope"),
            shared_with_principals=target.get("shared_with_principals", ()),
            shared_with_groups=target.get("shared_with_groups", ()),
            tombstoned=target.get("tombstoned", False),
        )

    visible = {"nodes": [], "edges": []}
    for target in fixture["targets"]:
        grain = target["grain"]
        if acl.decide(
            grain=grain,
            truth_graph=fixture["truth_graph"],
            entity_id=target["id"],
            principal_id=principal["id"],
            principal_groups=principal["groups"],
            security_scope=principal["security_scope"],
        ).visible:
            visible["nodes" if grain == "node" else "edges"].append(target["id"])

    assert visible == fixture["expected_visible"]


def test_python_node_item_usage_matches_shared_golden_contract() -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/graph-acl-item-usage-v1.json"
        ).read_text(encoding="utf-8")
    )
    principal = fixture["principal"]
    for case in fixture["cases"]:
        acl = ACLGraph(cache_enabled=True)
        for record in case["acl_records"]:
            acl.add_record(
                grain=record["grain"],
                truth_graph=fixture["truth_graph"],
                entity_id=case["node_id"],
                target_item_id=record.get("target_item_id"),
                version=1,
                mode=record["mode"],
                owner_id=record.get("owner_id"),
                shared_with_principals=record.get("shared_with_principals", ()),
                shared_with_groups=record.get("shared_with_groups", ()),
            )
        grounding_ids = [
            f"{case['node_id']}::mention::{mention_index}"
            for mention_index, _span_count in enumerate(case["mention_span_counts"])
        ]
        span_ids = [
            f"{case['node_id']}::mention::{mention_index}::span::{span_index}"
            for mention_index, span_count in enumerate(case["mention_span_counts"])
            for span_index in range(span_count)
        ]
        decision = acl.decide_node_read(
            item_grain="span",
            truth_graph=fixture["truth_graph"],
            entity_id=case["node_id"],
            target_item_ids=span_ids,
            grounding_item_ids=grounding_ids,
            principal_id=principal["id"],
            principal_groups=principal["groups"],
            security_scope=principal["security_scope"],
        )
        assert decision.visible is case["expected_visible"], case["node_id"]


def test_python_edge_usage_matches_shared_golden_contract() -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/graph-acl-edge-usage-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert fixture["schema_version"] == 1
    principal = fixture["principal"]
    for case in fixture["cases"]:
        acl = ACLGraph(cache_enabled=True)
        acl.add_record(
            grain="edge",
            truth_graph=fixture["truth_graph"],
            entity_id=case["edge_id"],
            version=1,
            mode="shared",
            owner_id=fixture["owner_id"],
            shared_with_principals=fixture["shared_with_principals"],
        )
        if case.get("source_edge_id") and case.get("source_edge_exists", True):
            acl.add_record(
                grain="edge",
                truth_graph=fixture["truth_graph"],
                entity_id=case["source_edge_id"],
                version=1,
                mode="private",
                owner_id=fixture["owner_id"],
            )
        for endpoint, mode_key in (
            (case["source_node_id"], "source_node_mode"),
            (case["target_node_id"], "target_node_mode"),
        ):
            acl.add_record(
                grain="node",
                truth_graph=fixture["truth_graph"],
                entity_id=endpoint,
                version=1,
                mode=case[mode_key],
                owner_id=fixture["owner_id"],
                shared_with_principals=fixture["shared_with_principals"],
            )

        mention_ids = [
            f"{case['edge_id']}::mention::{mention_index}"
            for mention_index, _span_count in enumerate(case["mention_span_counts"])
        ]
        for item_id in mention_ids:
            acl.add_record(
                grain="grounding",
                truth_graph=fixture["truth_graph"],
                entity_id=case["edge_id"],
                target_item_id=item_id,
                version=1,
                mode="public",
            )
        span_ids: list[str] = []
        for mention_index, span_count in enumerate(case["mention_span_counts"]):
            for span_index in range(span_count):
                item_id = (
                    f"{case['edge_id']}::mention::{mention_index}::span::{span_index}"
                )
                span_ids.append(item_id)
                private_span = case.get("private_span") == [mention_index, span_index]
                acl.add_record(
                    grain="span",
                    truth_graph=fixture["truth_graph"],
                    entity_id=case["edge_id"],
                    target_item_id=item_id,
                    version=1,
                    mode="private" if private_span else "public",
                    owner_id=fixture["owner_id"] if private_span else None,
                )

        visible = acl.decide(
            grain="edge",
            truth_graph=fixture["truth_graph"],
            entity_id=case["edge_id"],
            principal_id=principal["id"],
            principal_groups=principal["groups"],
            security_scope=principal["security_scope"],
        ).visible
        visible = visible and all(
            acl.decide(
                grain="grounding",
                truth_graph=fixture["truth_graph"],
                entity_id=case["edge_id"],
                target_item_id=item_id,
                principal_id=principal["id"],
                principal_groups=principal["groups"],
                security_scope=principal["security_scope"],
            ).visible
            for item_id in mention_ids
        )
        visible = visible and all(
            acl.decide(
                grain="span",
                truth_graph=fixture["truth_graph"],
                entity_id=case["edge_id"],
                target_item_id=item_id,
                principal_id=principal["id"],
                principal_groups=principal["groups"],
                security_scope=principal["security_scope"],
            ).visible
            for item_id in span_ids
        )
        visible = visible and all(
            acl.decide(
                grain="node",
                truth_graph=fixture["truth_graph"],
                entity_id=endpoint,
                principal_id=principal["id"],
                principal_groups=principal["groups"],
                security_scope=principal["security_scope"],
            ).visible
            for endpoint in (case["source_node_id"], case["target_node_id"])
        )
        if case.get("source_edge_id"):
            visible = visible and acl.decide(
                grain="edge",
                truth_graph=fixture["truth_graph"],
                entity_id=case["source_edge_id"],
                principal_id=principal["id"],
                principal_groups=principal["groups"],
                security_scope=principal["security_scope"],
            ).visible
        assert visible is case["expected_visible"], case["edge_id"]
