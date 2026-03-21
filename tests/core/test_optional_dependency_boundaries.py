from __future__ import annotations

import pytest

pytestmark = pytest.mark.ci

import subprocess
import sys
import textwrap


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )


def _script_with_blocked_imports(*, blocked_roots: tuple[str, ...], body: str) -> str:
    preamble = textwrap.dedent(
        f"""
import builtins

_real_import = builtins.__import__
_blocked_roots = {blocked_roots!r}

def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root in _blocked_roots:
        raise ModuleNotFoundError(name)
    return _real_import(name, globals, locals, fromlist, level)

builtins.__import__ = _blocked_import
"""
    )
    return f"{preamble}\n{textwrap.dedent(body).strip()}\n"


def test_base_import_is_safe_without_optional_dependencies() -> None:
    code = _script_with_blocked_imports(
        blocked_roots=(
            "chromadb",
            "sqlalchemy",
            "pgvector",
            "langchain_openai",
            "langchain_google_genai",
            "langgraph",
        ),
        body="""
import graph_knowledge_engine.engine_core as engine_core
assert hasattr(engine_core, "GraphKnowledgeEngine")
print("ok")
""",
    )
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr


def test_optional_modules_import_safely_without_optional_dependencies() -> None:
    code = _script_with_blocked_imports(
        blocked_roots=(
            "chromadb",
            "sqlalchemy",
            "pgvector",
            "langchain_core",
            "langchain_openai",
            "langchain_google_genai",
            "langgraph",
        ),
        body="""
import graph_knowledge_engine.ingester as ingester
import graph_knowledge_engine.ocr as ocr
import graph_knowledge_engine.runtime as runtime_pkg
import graph_knowledge_engine.runtime.langgraph_converter as langgraph_converter
import graph_knowledge_engine.utils.langchain as langchain_utils

assert hasattr(ingester, "PagewiseSummaryIngestor")
assert hasattr(ocr, "RawOCRResponse")
assert hasattr(runtime_pkg, "WorkflowRuntime")
assert hasattr(langgraph_converter, "to_langgraph")
assert hasattr(langchain_utils, "GeminiCostCallbackHandler")
print("ok")
""",
    )
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr


def test_server_entrypoint_requires_server_dependency() -> None:
    code = _script_with_blocked_imports(
        blocked_roots=(
            "chromadb",
            "sqlalchemy",
            "pgvector",
            "langchain_core",
            "langchain_openai",
            "langchain_google_genai",
            "langgraph",
            "fastmcp",
        ),
        body="""
try:
    import graph_knowledge_engine.server_mcp_with_admin as server
except RuntimeError as e:
    msg = str(e)
    assert "kogwistar[server]" in msg
    print("ok")
else:
    raise AssertionError("expected missing server dependency error")
""",
    )
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr


def test_missing_chroma_dependency_error_is_actionable() -> None:
    code = _script_with_blocked_imports(
        blocked_roots=("chromadb",),
        body="""
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
try:
    GraphKnowledgeEngine(persist_directory=".")
except RuntimeError as e:
    msg = str(e)
    assert "kogwistar[chroma]" in msg
    print("ok")
else:
    raise AssertionError("expected missing chroma dependency error")
""",
    )
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr


def test_missing_langgraph_dependency_error_is_actionable() -> None:
    code = _script_with_blocked_imports(
        blocked_roots=("langgraph",),
        body="""
from graph_knowledge_engine.runtime.langgraph_converter import to_langgraph

try:
    to_langgraph(
        workflow_engine=None,
        workflow_id="wf",
        step_resolver=None,
        predicate_registry={},
    )
except RuntimeError as e:
    msg = str(e)
    assert "kogwistar[langgraph]" in msg
    print("ok")
else:
    raise AssertionError("expected missing langgraph dependency error")
""",
    )
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr


def test_missing_gemini_dependency_error_is_actionable() -> None:
    code = _script_with_blocked_imports(
        blocked_roots=("langchain_google_genai",),
        body="""
from graph_knowledge_engine.llm_tasks import (
    DefaultTaskProviderConfig,
    SummarizeContextTaskRequest,
    build_default_llm_tasks,
)

tasks = build_default_llm_tasks(
    DefaultTaskProviderConfig(summarize_context_provider="gemini")
)
try:
    tasks.summarize_context(SummarizeContextTaskRequest(full_text="hello"))
except Exception as e:
    msg = str(e)
    assert "kogwistar[gemini]" in msg
    print("ok")
else:
    raise AssertionError("expected missing gemini dependency error")
""",
    )
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr


def test_missing_ocr_dependency_error_is_actionable() -> None:
    code = _script_with_blocked_imports(
        blocked_roots=("langchain_core", "langchain_google_genai"),
        body="""
import graph_knowledge_engine.ocr as ocr

try:
    ocr.SystemMessage("hello")
except RuntimeError as e:
    msg = str(e)
    assert "kogwistar[ingestion-gemini]" in msg
    print("ok")
else:
    raise AssertionError("expected missing OCR dependency error")
""",
    )
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr


def test_missing_openai_dependency_error_is_actionable() -> None:
    code = _script_with_blocked_imports(
        blocked_roots=("langchain_openai",),
        body="""
from graph_knowledge_engine.llm_tasks import (
    DefaultTaskProviderConfig,
    SummarizeContextTaskRequest,
    build_default_llm_tasks,
)

tasks = build_default_llm_tasks(
    DefaultTaskProviderConfig(summarize_context_provider="openai")
)
try:
    tasks.summarize_context(SummarizeContextTaskRequest(full_text="hello"))
except Exception as e:
    msg = str(e)
    assert "kogwistar[openai]" in msg
    print("ok")
else:
    raise AssertionError("expected missing openai dependency error")
""",
    )
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr
