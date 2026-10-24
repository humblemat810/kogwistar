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
