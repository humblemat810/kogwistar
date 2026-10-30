from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ci, pytest.mark.core]

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
SQLITE_TESTS = ROOT / "tests" / "core" / "test_sqlite_context_invariant.py"


def _between(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_dependency_install_steps_allow_transient_package_index_latency() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    install_sections = (
        _between(
            workflow,
            "      - name: Install CPython dependencies",
            "      - name: Install PyPy 3.11 Python-authority dependencies",
        ),
        _between(
            workflow,
            "      - name: Install test dependencies and native extension",
            "      - name: Enforce SQLite architectural invariants",
        ),
        _between(
            workflow,
            "      - name: Install dependencies",
            "      - name: Run full CI tests",
        ),
    )

    for section in install_sections:
        assert 'PIP_DEFAULT_TIMEOUT: "60"' in section
        assert 'PIP_RETRIES: "10"' in section


def test_sqlite_differential_gate_selects_only_sqlite_cases() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    gate = _between(
        workflow,
        "  sqlite-architectural-invariants:",
        "  pypy-beta-best-effort:",
    )

    assert "-k sqlite" in gate
    assert 'python -m pytest tests/core/test_sqlite_context_invariant.py -q' in gate
    assert "-m \"ci and core and not requires_pgvector\"" in gate


def test_pypy_ci_excludes_native_extension_tests() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    pypy_tests = _between(
        workflow,
        "      - name: Run PyPy 3.11 CI tests",
        "      - name: Post Set up Python",
    )
    runner = (WORKFLOW.parents[2] / "scripts" / "run_pypy311_ci.py").read_text(
        encoding="utf-8"
    )

    assert "not requires_rust" in pypy_tests
    for marker in ("requires_rust", "requires_langgraph", "requires_pgvector"):
        assert f"not {marker}" in pypy_tests
        assert f"not {marker}" in runner


def test_ci_declares_the_supported_interpreter_matrix() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    matrix = _between(workflow, "      matrix:\n        include:", "    steps:")

    expected_cells = {
        'label: cpython312\n            python-version: "3.12"',
        'label: cpython313\n            python-version: "3.13"',
        'label: cpython314\n            python-version: "3.14"',
        'label: pypy311\n            python-version: pypy-3.11-v7.3.20',
    }

    for cell in expected_cells:
        assert cell in matrix, f"missing required CI interpreter cell: {cell!r}"


def test_rust_backend_interop_ci_node_ids_exist() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    handoff_step = _between(
        workflow,
        "      - name: Verify sequential Python-Rust backend handoffs",
        '          python - "$report" <<\'PY\'',
    )
    selectors = re.findall(r"(tests/[^\s]+)::(test_[A-Za-z0-9_]+)", handoff_step)

    assert len(selectors) == 3
    for relative_path, test_name in selectors:
        test_path = ROOT / relative_path
        tree = ast.parse(test_path.read_text(encoding="utf-8"))
        test_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert test_name in test_names, (
            f"Rust backend interop CI selects missing test {relative_path}::{test_name}"
        )


def test_sqlite_native_extension_cases_are_marked_requires_rust() -> None:
    tree = ast.parse(SQLITE_TESTS.read_text(encoding="utf-8"))
    missing_markers: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        uses_native = any(
            isinstance(child, ast.Name)
            and child.id in {"RustEngineSQLite", "_rust"}
            for child in ast.walk(node)
        )
        if uses_native and not any(
            ast.unparse(decorator) == "pytest.mark.requires_rust"
            for decorator in node.decorator_list
        ):
            missing_markers.append(node.name)

    assert not missing_markers, (
        "SQLite tests using the Rust extension must carry requires_rust: "
        f"{sorted(missing_markers)}"
    )
