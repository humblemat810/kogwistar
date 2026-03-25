from __future__ import annotations

import ast
from pathlib import Path
import pytest

pytestmark = pytest.mark.ci

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / "kogwistar" / "runtime"


def _runtime_python_files() -> list[Path]:
    return sorted(path for path in RUNTIME_DIR.rglob("*.py") if path.is_file())


def test_runtime_package_does_not_import_conversation_models() -> None:
    offenders: list[str] = []
    for path in _runtime_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "conversation.models" in module:
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} -> from {module} import ..."
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "conversation.models" in alias.name:
                        offenders.append(
                            f"{path.relative_to(ROOT)}:{node.lineno} -> import {alias.name}"
                        )
    assert not offenders, (
        "runtime package must not import conversation.models:\n" + "\n".join(offenders)
    )
