from __future__ import annotations

import ast
import json
import re
import textwrap
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
            (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == parts[0]
            )
            or (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == parts[0]
            )
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == parts[0]
                    for target in node.targets
                )
            )
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


def _python_test_consumes_fixture(source: str, symbol: str, fixture_name: str) -> bool:
    tree = ast.parse(source)
    fixture_variables = {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and fixture_name in (ast.get_source_segment(source, node.value) or "")
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == symbol
        ),
        None,
    )
    if function is None:
        return False
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    ]
    fixture_read_calls = {
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "read_text"
        and (
            fixture_name in (ast.get_source_segment(source, call) or "")
            or (
                isinstance(call.func.value, ast.Name)
                and call.func.value.id in fixture_variables
            )
        )
    }
    return any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "loads"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "json"
        and any(child in fixture_read_calls for child in ast.walk(call))
        for call in calls
    )


def _rust_test_consumes_fixture(source: str, symbol: str, fixture_name: str) -> bool:
    match = re.search(rf"\bfn {re.escape(symbol)}\s*\(", source)
    if match is None:
        return False
    next_attribute = re.search(r"^\s*#\[", source[match.end() :], re.MULTILINE)
    body_source = source[
        match.start() : match.end() + next_attribute.start()
        if next_attribute is not None
        else len(source)
    ]
    for include_match in re.finditer(r"include_str!\s*\(", body_source):
        depth = 0
        in_string = False
        escaped = False
        end = None
        for index in range(include_match.end() - 1, len(body_source)):
            char = body_source[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            continue
        macro_source = body_source[include_match.start() : end]
        string_literals = re.findall(r'"(?:\\.|[^"\\])*"', macro_source)
        if any(fixture_name in literal for literal in string_literals):
            return True
    return False


def _fixture_versions(value: object) -> list[object]:
    if isinstance(value, dict):
        versions = [
            item
            for key, item in value.items()
            if key in {"schema_version", "contract_version"}
        ]
        for item in value.values():
            versions.extend(_fixture_versions(item))
        return versions
    if isinstance(value, list):
        return [version for item in value for version in _fixture_versions(item)]
    return []


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
        versions = _fixture_versions(fixture)
        if "schema_version" in fixture:
            assert fixture["schema_version"] == 1
        else:
            assert fixture_path.stem.endswith("-v1")
            assert "v1" in versions
        assert _rust_test_consumes_fixture(
            rust_test_source, rust_test_symbol, fixture_path.name
        ), rust_test_path
        assert _python_test_consumes_fixture(
            python_test_source, python_test_symbol, fixture_path.name
        ), python_test_path


def test_parity_fixture_verifier_requires_test_scoped_loading() -> None:
    python_type_alias = textwrap.dedent("""
    from typing import TypeAlias
    ProviderEvent: TypeAlias = Started | Completed
    """)
    assert _python_symbol_exists(python_type_alias, "ProviderEvent")

    rust_body = textwrap.dedent("""
    #[test]
    fn parity_test() {
        let data = include_str!(\"contracts/golden/shared.json\");
        assert!(!data.is_empty());
    }
    """)
    python_body = textwrap.dedent("""
    def parity_test():
        fixture = json.loads((ROOT / \"shared.json\").read_text())
        assert fixture
    """)
    assert _rust_test_consumes_fixture(rust_body, "parity_test", "shared.json")
    assert _python_test_consumes_fixture(python_body, "parity_test", "shared.json")

    rust_comment_only = textwrap.dedent("""
    #[test]
    fn parity_test() {
        // contracts/golden/shared.json
        assert!(true);
    }
    """)
    python_comment_only = textwrap.dedent("""
    def parity_test():
        # shared.json
        json.loads((ROOT / \"other.json\").read_text())
    """)
    rust_other_fixture = textwrap.dedent("""
    #[test]
    fn parity_test() {
        // \"shared.json\"
        let _data = include_str!(\"other.json\");
    }
    """)
    assert not _rust_test_consumes_fixture(
        rust_comment_only, "parity_test", "shared.json"
    )
    assert not _python_test_consumes_fixture(
        python_comment_only, "parity_test", "shared.json"
    )
    assert not _rust_test_consumes_fixture(
        rust_other_fixture, "parity_test", "shared.json"
    )
