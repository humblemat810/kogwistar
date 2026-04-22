from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import re

import pytest

pytestmark = [pytest.mark.ci]


SYNC_MODULE = "tests.runtime.test_sync_runtime_bijection_contract"
ASYNC_MODULE = "tests.runtime.test_async_runtime_bijection_contract"
SYNC_FILE = "test_sync_runtime_bijection_contract"
ASYNC_FILE = "test_async_runtime_bijection_contract"
SYNC_PREFIX = "test_sync_runtime_"
ASYNC_PREFIX = "test_async_runtime_"
REPO_ROOT = Path(__file__).resolve().parents[2]
MOVED_FROM_RE = re.compile(r"Moved from `(?P<path>[^`]+)::(?P<name>test_[A-Za-z0-9_]+)`")
REFACTORED_FROM_RE = re.compile(r"Refactored from `(?P<path>[^`]+)::(?P<name>test_[A-Za-z0-9_]+)`")
PROVENANCE_RE = re.compile(r"(?:Moved|Refactored) from `(?P<path>[^`]+)(?:::[^`]+)?`")


def _collect_tests(module_name: str) -> dict[str, object]:
    module = importlib.import_module(module_name)
    return {
        name: obj
        for name, obj in vars(module).items()
        if name.startswith("test_") and callable(obj)
    }


def test_runtime_bijection_naming_and_docstrings():
    sync_mod = importlib.import_module(SYNC_MODULE)
    async_mod = importlib.import_module(ASYNC_MODULE)

    sync_tests = _collect_tests(SYNC_MODULE)
    async_tests = _collect_tests(ASYNC_MODULE)

    assert sync_mod.__name__.rsplit(".", 1)[-1] == SYNC_FILE
    assert async_mod.__name__.rsplit(".", 1)[-1] == ASYNC_FILE
    assert sync_tests
    assert async_tests

    sync_suffixes = set()
    async_suffixes = set()

    for name in sync_tests:
        assert name.startswith(SYNC_PREFIX)
        sync_suffixes.add(name.removeprefix(SYNC_PREFIX))

    for name in async_tests:
        assert name.startswith(ASYNC_PREFIX)
        async_suffixes.add(name.removeprefix(ASYNC_PREFIX))

    assert sync_suffixes == async_suffixes

    for suffix in sorted(sync_suffixes):
        sync_name = f"{SYNC_PREFIX}{suffix}"
        async_name = f"{ASYNC_PREFIX}{suffix}"

        sync_fn = sync_tests[sync_name]
        async_fn = async_tests[async_name]

        sync_doc = inspect.getdoc(sync_fn)
        async_doc = inspect.getdoc(async_fn)
        assert sync_doc and async_doc
        assert f"{ASYNC_FILE}.py" in sync_doc.splitlines()[0]
        assert f"{SYNC_FILE}.py" in async_doc.splitlines()[0]
        assert (
            ("Moved from" in sync_doc)
            or ("Refactored from" in sync_doc)
            or ("New sync mirror" in sync_doc)
        )
        assert (
            ("Moved from" in async_doc)
            or ("Refactored from" in async_doc)
            or ("New async mirror" in async_doc)
        )


def test_runtime_bijection_moved_from_sources_are_removed():
    for module_name in (SYNC_MODULE, ASYNC_MODULE):
        for test_name, test_fn in _collect_tests(module_name).items():
            doc = inspect.getdoc(test_fn) or ""
            for match in MOVED_FROM_RE.finditer(doc):
                source_path = REPO_ROOT / match.group("path")
                source_name = match.group("name")
                if not source_path.exists():
                    continue
                source_text = source_path.read_text(encoding="utf-8")
                assert not re.search(rf"^def {re.escape(source_name)}\b", source_text, re.MULTILINE), (
                    f"{test_name} says it moved from {source_path}::{source_name}, "
                    "but source test still exists"
                )


def test_runtime_bijection_refactored_from_sources_remain():
    for module_name in (SYNC_MODULE, ASYNC_MODULE):
        for test_name, test_fn in _collect_tests(module_name).items():
            doc = inspect.getdoc(test_fn) or ""
            for match in REFACTORED_FROM_RE.finditer(doc):
                assert "Source retained:" in doc, (
                    f"{test_name} says it was refactored from a source test, "
                    "but does not explain why the source remains"
                )
                source_path = REPO_ROOT / match.group("path")
                source_name = match.group("name")
                assert source_path.exists(), (
                    f"{test_name} says it was refactored from {source_path}::{source_name}, "
                    "but source file does not exist"
                )
                source_text = source_path.read_text(encoding="utf-8")
                assert re.search(rf"^def {re.escape(source_name)}\b", source_text, re.MULTILINE), (
                    f"{test_name} says it was refactored from {source_path}::{source_name}, "
                    "but source test does not exist"
                )


def test_runtime_bijection_sync_provenance_is_not_async_source():
    for test_name, test_fn in _collect_tests(SYNC_MODULE).items():
        doc = inspect.getdoc(test_fn) or ""
        for match in PROVENANCE_RE.finditer(doc):
            source_path = match.group("path").replace("\\", "/")
            assert "test_async_runtime" not in source_path, (
                f"{test_name} is a sync bijection test but its provenance points at "
                f"async source {source_path}"
            )
