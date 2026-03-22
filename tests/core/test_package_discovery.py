from __future__ import annotations

import subprocess
import sys

import pytest

import kogwistar

pytestmark = pytest.mark.ci


def test_list_submodules_includes_core_packages():
    submodules = kogwistar.list_submodules()

    assert "kogwistar.engine_core" in submodules
    assert "kogwistar.runtime" in submodules
    assert "kogwistar.conversation" in submodules


def test_dir_exposes_curated_root_api():
    package_dir = dir(kogwistar)

    assert "GraphKnowledgeEngine" in package_dir
    assert "WorkflowRuntime" in package_dir
    assert "Node" in package_dir
    assert "shortids" in package_dir
    assert "kogwistar.engine_core" not in package_dir


def test_root_api_exports_are_real_attributes():
    public_names = [
        "GraphKnowledgeEngine",
        "WorkflowRuntime",
        "ConversationOrchestrator",
        "ConversationService",
        "LLMTaskSet",
        "DefaultTaskProviderConfig",
        "build_default_llm_tasks",
        "Node",
        "Edge",
        "Document",
        "Span",
        "Grounding",
        "shortids",
        "list_submodules",
    ]

    for name in public_names:
        assert hasattr(kogwistar, name), name


def test_recursive_submodule_discovery_has_no_import_side_effects():
    sys.modules.pop("kogwistar.runtime", None)
    sys.modules.pop("kogwistar.conversation", None)

    submodules = kogwistar.list_submodules(recursive=True)

    assert "kogwistar.runtime" in submodules
    assert "kogwistar.conversation" in submodules
    assert "kogwistar.runtime" not in sys.modules
    assert "kogwistar.conversation" not in sys.modules


def test_root_does_not_expose_server_launch_objects():
    assert not hasattr(kogwistar, "mcp")
    assert not hasattr(kogwistar, "app")
    assert not hasattr(kogwistar, "main")


def test_root_imports_work_from_python_entrypoint():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from kogwistar import "
                "GraphKnowledgeEngine, WorkflowRuntime, ConversationService, "
                "ConversationOrchestrator, LLMTaskSet, DefaultTaskProviderConfig, "
                "build_default_llm_tasks, Node, Edge, Document, Span, Grounding, shortids; "
                "print('ok')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
