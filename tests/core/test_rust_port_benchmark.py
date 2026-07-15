from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.rust_port_benchmark import DATASET_VERSION, run_benchmarks


pytestmark = [pytest.mark.ci, pytest.mark.core]
ROOT = Path(__file__).resolve().parents[2]


def test_rust_port_baseline_covers_required_workloads(tmp_path: Path) -> None:
    report = run_benchmarks(tmp_path / "benchmark", items=4, workflow_iterations=1)

    assert report["dataset"] == DATASET_VERSION
    assert set(report["metrics"]) == {
        "graph_ingest",
        "graph_retrieval",
        "event_replay",
        "queue_claim",
        "workflow_fanout_join",
    }
    assert all(metric["output_count"] > 0 for metric in report["metrics"].values())
    assert all(metric["wall_ms"] >= 0 for metric in report["metrics"].values())
    assert all(metric["peak_rss_bytes"] > 0 for metric in report["metrics"].values())
    assert report["database_load"] == {"kind": "in-memory", "external_queries": 0}


def test_committed_python_baseline_records_all_required_metrics() -> None:
    report = json.loads(
        (ROOT / "contracts" / "benchmarks" / "python-memory-windows.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["dataset"] == DATASET_VERSION
    assert report["items"] == 20
    assert set(report["metrics"]) == {
        "graph_ingest",
        "graph_retrieval",
        "event_replay",
        "queue_claim",
        "workflow_fanout_join",
    }
    assert all(metric["peak_rss_bytes"] > 0 for metric in report["metrics"].values())
