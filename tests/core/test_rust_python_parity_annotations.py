from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUST_ROOT = ROOT / "rust" / "crates"
ANNOTATION_PREFIX = "// PYTHON_PARITY_V1: "
pytestmark = [pytest.mark.ci, pytest.mark.core]
ANNOTATION_FIELDS = {
    "rust",
    "python",
    "fixture",
    "rust_test",
    "python_test",
}


def _parse_target(target: str) -> tuple[Path, str]:
    path_text, symbol = target.split("#", maxsplit=1)
    return ROOT / path_text, symbol


def _python_symbol_exists(source: str, symbol: str) -> bool:
    tree = ast.parse(source)
    parts = symbol.split(".")
    if len(parts) == 1:
        return any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == parts[0]
            for node in tree.body
        )
    if len(parts) == 2:
        owner, member = parts
        return any(
            isinstance(node, ast.ClassDef)
            and node.name == owner
            and any(
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == member
                for item in node.body
            )
            for node in tree.body
        )
    return False


def test_rust_python_parity_annotations_resolve_to_shared_executable_contracts() -> None:
    annotations: list[tuple[Path, dict[str, str]]] = []
    for rust_file in sorted(RUST_ROOT.rglob("*.rs")):
        source = rust_file.read_text(encoding="utf-8")
        for line in source.splitlines():
            if ANNOTATION_PREFIX not in line:
                continue
            payload = line.split(ANNOTATION_PREFIX, maxsplit=1)[1]
            fields = dict(
                part.strip().split("=", maxsplit=1)
                for part in payload.split(";")
            )
            assert set(fields) == ANNOTATION_FIELDS, rust_file
            annotations.append((rust_file, fields))

    assert annotations, "Rust/Python parity annotations must be verified"
    seen_rust_targets: set[str] = set()
    for rust_file, fields in annotations:
        rust_target_path, rust_symbol = _parse_target(fields["rust"])
        assert rust_target_path == rust_file
        assert rust_symbol not in seen_rust_targets
        seen_rust_targets.add(rust_symbol)
        rust_source = rust_file.read_text(encoding="utf-8")
        assert re.search(rf"\b(?:pub\s+)?fn {re.escape(rust_symbol)}\s*\(", rust_source)

        python_path, python_symbol = _parse_target(fields["python"])
        python_source = python_path.read_text(encoding="utf-8")
        assert _python_symbol_exists(python_source, python_symbol)

        rust_test_path, rust_test_symbol = _parse_target(fields["rust_test"])
        rust_test_source = rust_test_path.read_text(encoding="utf-8")
        assert re.search(
            rf"\bfn {re.escape(rust_test_symbol)}\s*\(", rust_test_source
        )

        python_test_path, python_test_symbol = _parse_target(fields["python_test"])
        python_test_source = python_test_path.read_text(encoding="utf-8")
        assert _python_symbol_exists(python_test_source, python_test_symbol)

        fixture_path = ROOT / fields["fixture"]
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        assert fixture["schema_version"] == 1
        assert fixture_path.name in rust_test_source
        assert fixture_path.name in python_test_source
