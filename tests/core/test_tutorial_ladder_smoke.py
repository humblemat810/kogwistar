from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


pytest.importorskip("chromadb")

ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )


def _extract_last_json(stdout: str) -> dict:
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip().startswith("{"):
            candidate = "\n".join(lines[idx:])
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    raise AssertionError(f"No JSON object found in stdout:\n{stdout}")


def test_tutorial_ladder_levels_0_to_2_smoke(tmp_path: Path):
    data_dir = tmp_path / "tutorial-ladder"

    _run(["scripts/rag_tutorial_ladder.py", "reset", "--data-dir", str(data_dir)])
    seed = _extract_last_json(_run(["scripts/rag_tutorial_ladder.py", "seed", "--data-dir", str(data_dir)]).stdout)
    assert seed.get("ok") is True

    level0 = _extract_last_json(
        _run(
            [
                "scripts/rag_tutorial_ladder.py",
                "level0",
                "--data-dir",
                str(data_dir),
                "--question",
                "How does this repo implement simple RAG?",
            ]
        ).stdout
    )
    assert level0.get("checkpoint_pass") is True
    assert level0.get("evidence")

    level1 = _extract_last_json(
        _run(
            [
                "scripts/rag_tutorial_ladder.py",
                "level1",
                "--data-dir",
                str(data_dir),
                "--question",
                "How does architecture reinforce retrieval?",
                "--max-retrieval-level",
                "2",
            ]
        ).stdout
    )
    assert level1.get("checkpoint_pass") is True
    assert level1.get("seed_kg_node_ids")
    assert level1.get("added_by_seed")

    level2 = _extract_last_json(
        _run(
            [
                "scripts/rag_tutorial_ladder.py",
                "level2",
                "--data-dir",
                str(data_dir),
                "--question",
                "Show evidence and provenance for retrieval decisions.",
                "--max-retrieval-level",
                "2",
            ]
        ).stdout
    )
    assert level2.get("checkpoint_pass") is True
    assert level2.get("pinned_kg_pointer_node_ids")
    assert level2.get("pinned_kg_edge_ids")


def test_level3_claw_command_path_smoke(tmp_path: Path):
    data_dir = tmp_path / "claw-loop"

    _run(["scripts/claw_runtime_loop.py", "init", "--data-dir", str(data_dir)])
    _run(
        [
            "scripts/claw_runtime_loop.py",
            "enqueue",
            "--data-dir",
            str(data_dir),
            "--conversation-id",
            "conv-test",
            "--event-type",
            "user.message",
            "--payload",
            '{"text":"hello claw","ttl":1}',
        ]
    )
    _run(["scripts/claw_runtime_loop.py", "run-once", "--data-dir", str(data_dir)])
    in_rows = _run(
        ["scripts/claw_runtime_loop.py", "list-events", "--data-dir", str(data_dir), "--direction", "in", "--limit", "10"]
    ).stdout
    out_rows = _run(
        ["scripts/claw_runtime_loop.py", "list-events", "--data-dir", str(data_dir), "--direction", "out", "--limit", "10"]
    ).stdout
    assert "in|" in in_rows
    assert ("out|" in out_rows) or ("claw.gate.output" in out_rows)


def test_runtime_tutorial_ladder_levels_0_to_3_smoke(tmp_path: Path):
    data_dir = tmp_path / "runtime-tutorial-ladder"

    reset = _extract_last_json(_run(["scripts/runtime_tutorial_ladder.py", "reset", "--data-dir", str(data_dir)]).stdout)
    assert reset.get("ok") is True

    level0 = _extract_last_json(_run(["scripts/runtime_tutorial_ladder.py", "level0", "--data-dir", str(data_dir)]).stdout)
    assert level0.get("checkpoint_pass") is True
    assert level0.get("status") == "suspended"
    assert level0.get("step_exec_count", 0) > 0
    assert level0.get("checkpoint_count", 0) > 0

    level1 = _extract_last_json(_run(["scripts/runtime_tutorial_ladder.py", "level1", "--data-dir", str(data_dir)]).stdout)
    assert level1.get("checkpoint_pass") is True
    assert level1.get("dep_echo") == "runtime-tutorial"
    assert "tutorial_resolver_note" in (level1.get("custom_event_types") or [])

    level2 = _extract_last_json(_run(["scripts/runtime_tutorial_ladder.py", "level2", "--data-dir", str(data_dir)]).stdout)
    assert level2.get("checkpoint_pass") is True
    assert level2.get("initial_status") == "suspended"
    assert level2.get("resumed_status") == "succeeded"
    assert level2.get("final_state", {}).get("ended") is True

    level3 = _extract_last_json(_run(["scripts/runtime_tutorial_ladder.py", "level3", "--data-dir", str(data_dir)]).stdout)
    assert level3.get("checkpoint_pass") is True
    assert level3.get("viewer_asset_exists") is True
    assert "workflow_run_completed" in (level3.get("trace_event_types") or [])
    assert "/api/workflow/runs/" in str(level3.get("runtime_event_endpoint"))

