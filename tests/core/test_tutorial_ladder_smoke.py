from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

import scripts.runtime_tutorial_ladder as runtime_tutorial_ladder
import scripts.tutorial_ladder as tutorial_ladder
from graph_knowledge_engine.runtime.design import validate_workflow_design

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


def _run_allow_fail(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT),
        check=False,
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


@contextlib.contextmanager
def _workspace_temp_dir(prefix: str):
    root = ROOT / ".tmp_pytest"
    root.mkdir(parents=True, exist_ok=True)
    temp_dir = root / f"{prefix}{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_tutorial_ladder_levels_0_to_2b_smoke():
    with _workspace_temp_dir("test_tutorial_ladder_") as temp_dir:
        data_dir = temp_dir / "tutorial-ladder"

        _run(["scripts/rag_tutorial_ladder.py", "reset", "--data-dir", str(data_dir)])
        seed = _extract_last_json(
            _run(
                ["scripts/rag_tutorial_ladder.py", "seed", "--data-dir", str(data_dir)]
            ).stdout
        )
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

        level2b = _extract_last_json(
            _run(
                [
                    "scripts/rag_tutorial_ladder.py",
                    "level2b",
                    "--data-dir",
                    str(data_dir),
                    "--question",
                    "Show the equivalent provenance flow through add_turn_workflow_v2.",
                    "--max-retrieval-level",
                    "2",
                ]
            ).stdout
        )
        assert level2b.get("checkpoint_pass") is True
        assert level2b.get("assistant_turn_node_id")
        assert level2b.get("pinned_kg_pointer_node_ids")
        assert level2b.get("pinned_kg_edge_ids")
        assert level2b.get("transcript_roles") == ["user", "assistant"]


def test_level3_claw_command_path_smoke():
    with _workspace_temp_dir("test_claw_loop_") as temp_dir:
        data_dir = temp_dir / "claw-loop"

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
            [
                "scripts/claw_runtime_loop.py",
                "list-events",
                "--data-dir",
                str(data_dir),
                "--direction",
                "in",
                "--limit",
                "10",
            ]
        ).stdout
        out_rows = _run(
            [
                "scripts/claw_runtime_loop.py",
                "list-events",
                "--data-dir",
                str(data_dir),
                "--direction",
                "out",
                "--limit",
                "10",
            ]
        ).stdout
        assert "in|" in in_rows
        assert ("out|" in out_rows) or ("claw.gate.output" in out_rows)


def test_tutorial_ladder_level2b_missing_ollama_model_fails_clearly():
    with _workspace_temp_dir("test_tutorial_ladder_missing_ollama_") as temp_dir:
        data_dir = temp_dir / "tutorial-ladder"
        _run(["scripts/rag_tutorial_ladder.py", "reset", "--data-dir", str(data_dir)])
        _run(["scripts/rag_tutorial_ladder.py", "seed", "--data-dir", str(data_dir)])

        missing_model = "definitely-missing-ollama-model-for-tests:1b"
        result = _run_allow_fail(
            [
                "scripts/rag_tutorial_ladder.py",
                "level2b",
                "--data-dir",
                str(data_dir),
                "--question",
                "Answer from the collected evidence pack.",
                "--max-retrieval-level",
                "2",
                "--llm-provider",
                "ollama",
                "--llm-model",
                missing_model,
            ]
        )

        assert result.returncode != 0
        assert f"Run `ollama pull {missing_model}` first" in result.stderr


def test_extract_ollama_model_names_supports_typed_response_objects():
    model_a = type("Model", (), {"model": "qwen3:4b"})()
    model_b = type("Model", (), {"name": "phi4-mini"})()
    response = type("ListResponse", (), {"models": [model_a, model_b]})()

    assert tutorial_ladder._extract_ollama_model_names(response) == [
        "qwen3:4b",
        "phi4-mini",
    ]


def test_runtime_tutorial_ladder_levels_0_to_4_smoke():
    with _workspace_temp_dir("test_runtime_tutorial_ladder_") as temp_dir:
        data_dir = temp_dir / "runtime-tutorial-ladder"
        try:
            reset = _extract_last_json(
                _run(
                    [
                        "scripts/runtime_tutorial_ladder.py",
                        "reset",
                        "--data-dir",
                        str(data_dir),
                    ]
                ).stdout
            )
        except Exception as _e:
            raise
        assert reset.get("ok") is True

        level0 = _extract_last_json(
            _run(
                [
                    "scripts/runtime_tutorial_ladder.py",
                    "level0",
                    "--data-dir",
                    str(data_dir),
                ]
            ).stdout
        )
        assert level0.get("checkpoint_pass") is True
        assert level0.get("status") == "suspended"
        assert level0.get("step_exec_count", 0) > 0
        assert level0.get("checkpoint_count", 0) > 0

        level1 = _extract_last_json(
            _run(
                [
                    "scripts/runtime_tutorial_ladder.py",
                    "level1",
                    "--data-dir",
                    str(data_dir),
                ]
            ).stdout
        )
        assert level1.get("checkpoint_pass") is True
        assert level1.get("dep_echo") == "runtime-tutorial"
        assert "tutorial_resolver_note" in (level1.get("custom_event_types") or [])

        level2 = _extract_last_json(
            _run(
                [
                    "scripts/runtime_tutorial_ladder.py",
                    "level2",
                    "--data-dir",
                    str(data_dir),
                ]
            ).stdout
        )
        assert level2.get("checkpoint_pass") is True
        assert level2.get("initial_status") == "suspended"
        assert level2.get("resumed_status") == "succeeded"
        assert level2.get("final_state", {}).get("ended") is True

        level3 = _extract_last_json(
            _run(
                [
                    "scripts/runtime_tutorial_ladder.py",
                    "level3",
                    "--data-dir",
                    str(data_dir),
                ]
            ).stdout
        )
        assert level3.get("checkpoint_pass") is True
        assert level3.get("viewer_asset_exists") is True
        assert "workflow_run_completed" in (level3.get("trace_event_types") or [])
        assert "/api/workflow/runs/" in str(level3.get("runtime_event_endpoint"))

        level4 = _extract_last_json(
            _run(
                [
                    "scripts/runtime_tutorial_ladder.py",
                    "level4",
                    "--data-dir",
                    str(data_dir),
                ]
            ).stdout
        )
        assert level4.get("sandbox_type") == "docker"
        assert (
            "python_exec" in (level4.get("sandboxed_ops") or [])
            or level4.get("sandbox_available") is False
        )
        if level4.get("sandbox_available") is False:
            assert level4.get("sandbox_executed") is False
            assert level4.get("status") == "sandbox_unavailable"
        else:
            assert level4.get("checkpoint_pass") is True
            assert level4.get("sandbox_executed") is True
            assert level4.get("sandbox_mode") == "per_op"
            assert level4.get("sandbox_result") == "HELLO FROM LLM SANDBOX"


def test_runtime_tutorial_ladder_keeps_level4_workflow_separate():
    with _workspace_temp_dir("test_runtime_tutorial_ladder_separate_") as temp_dir:
        data_dir = temp_dir / "runtime-tutorial-ladder-separate"

        workflow_engine, _conversation_engine = (
            runtime_tutorial_ladder.ensure_workflow_seed(data_dir)
        )
        runtime_tutorial_ladder.ensure_sandbox_workflow_seed(data_dir)

        start_a, nodes_a, _adj_a = validate_workflow_design(
            workflow_engine=workflow_engine,
            workflow_id=runtime_tutorial_ladder.WORKFLOW_ID,
            predicate_registry={"always": runtime_tutorial_ladder.PredAlwaysTrue()},
            resolver=None,
        )
        start_b, nodes_b, _adj_b = validate_workflow_design(
            workflow_engine=workflow_engine,
            workflow_id=runtime_tutorial_ladder.SANDBOX_WORKFLOW_ID,
            predicate_registry={},
            resolver=None,
        )

        assert start_a.id == runtime_tutorial_ladder.RT_START_NODE_ID
        assert start_b.id == "sb4:start"
        assert set(nodes_a) == {
            runtime_tutorial_ladder.RT_START_NODE_ID,
            runtime_tutorial_ladder.RT_FORK_NODE_ID,
            runtime_tutorial_ladder.RT_BRANCH_A_NODE_ID,
            runtime_tutorial_ladder.RT_BRANCH_B_NODE_ID,
            runtime_tutorial_ladder.RT_JOIN_NODE_ID,
            runtime_tutorial_ladder.RT_END_NODE_ID,
        }
        assert set(nodes_b) == {"sb4:start", "sb4:python_exec", "sb4:end"}


def test_tutorial_section_15_historical_smoke():
    out = _extract_last_json(
        _run(
            ["scripts/tutorial_sections/15_historical_search_tombstone_redirect.py"]
        ).stdout
    )
    assert out.get("checkpoint_pass") is True
    assert "N_SUGAR_OLD" in (out.get("then_ids") or [])
    assert "N_SUGAR_NEW" in (out.get("now_ids") or [])
