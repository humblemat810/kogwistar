from __future__ import annotations

import json
from pathlib import Path

from kogwistar.graph_kinds import normalize_graph_kind


def test_python_graph_kind_normalization_matches_shared_fixture() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts/golden/graph-kind-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert fixture["contract_version"] == 1
    for case in fixture["cases"]:
        assert normalize_graph_kind(case["input"]) == case["expected"]
