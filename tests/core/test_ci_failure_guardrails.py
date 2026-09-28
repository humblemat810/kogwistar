from __future__ import annotations

import ast
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

    assert "not requires_rust" in pypy_tests


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
