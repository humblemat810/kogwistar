from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from kogwistar.acl.graph import ACLGraph


def _fixture() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/graph-acl-history-v1.json"
            ).read_text(encoding="utf-8")
        ),
    )


def test_python_acl_history_matches_shared_golden_contract() -> None:
    fixture = _fixture()
    principal = fixture["principal"]
    acl = ACLGraph(cache_enabled=True)
    for target in fixture["targets"]:
        for record in target["acl_records"]:
            acl.add_record(
                grain=target["grain"],
                truth_graph=fixture["truth_graph"],
                entity_id=target["id"],
                version=record["version"],
                mode=record["mode"],
                owner_id=record.get("owner_id"),
                security_scope=record.get("security_scope"),
                shared_with_principals=record.get("shared_with_principals", ()),
                shared_with_groups=record.get("shared_with_groups", ()),
                tombstoned=record.get("tombstoned", False),
            )

    visible = {"nodes": [], "edges": []}
    for target in fixture["targets"]:
        decision = acl.decide(
            grain=target["grain"],
            truth_graph=fixture["truth_graph"],
            entity_id=target["id"],
            principal_id=principal["id"],
            principal_groups=principal["groups"],
            security_scope=principal["security_scope"],
        )
        if decision.visible:
            visible["nodes" if target["grain"] == "node" else "edges"].append(
                target["id"]
            )

    assert visible == fixture["expected_visible"]
