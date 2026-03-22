from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "kogwistar"
ALLOWED_FILES = {
    PACKAGE_DIR / "__init__.py",
    PACKAGE_DIR / "__main__.py",
}


def _python_files() -> list[Path]:
    return sorted(
        path
        for path in PACKAGE_DIR.rglob("*.py")
        if path.is_file() and path not in ALLOWED_FILES
    )


def test_internal_code_does_not_depend_on_root_facade() -> None:
    offenders: list[str] = []

    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "kogwistar":
                    names = ", ".join(alias.name for alias in node.names)
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} -> from kogwistar import {names}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "kogwistar":
                        offenders.append(
                            f"{path.relative_to(ROOT)}:{node.lineno} -> import kogwistar"
                        )

    assert not offenders, (
        "internal code must not depend on the root kogwistar facade:\n"
        + "\n".join(offenders)
    )
