"""Curated top-level package API for Kogwistar."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

if TYPE_CHECKING:
    from kogwistar.conversation.conversation_orchestrator import (
        ConversationOrchestrator,
    )
    from kogwistar.conversation.service import ConversationService
    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Document, Edge, Grounding, Node, Span
    from kogwistar.llm_tasks import (
        DefaultTaskProviderConfig,
        LLMTaskSet,
        build_default_llm_tasks,
    )
    from kogwistar.runtime import WorkflowRuntime
    from kogwistar import shortids

__all__ = [
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

_PACKAGE_DIR = Path(__file__).resolve().parent
_DIR_DUNDERS = {
    "__all__",
    "__annotations__",
    "__builtins__",
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__path__",
    "__spec__",
}
_EXPORTS: dict[str, tuple[str, str | None]] = {
    "GraphKnowledgeEngine": ("kogwistar.engine_core.engine", "GraphKnowledgeEngine"),
    "WorkflowRuntime": ("kogwistar.runtime", "WorkflowRuntime"),
    "ConversationOrchestrator": (
        "kogwistar.conversation.conversation_orchestrator",
        "ConversationOrchestrator",
    ),
    "ConversationService": (
        "kogwistar.conversation.service",
        "ConversationService",
    ),
    "LLMTaskSet": ("kogwistar.llm_tasks", "LLMTaskSet"),
    "DefaultTaskProviderConfig": (
        "kogwistar.llm_tasks",
        "DefaultTaskProviderConfig",
    ),
    "build_default_llm_tasks": (
        "kogwistar.llm_tasks",
        "build_default_llm_tasks",
    ),
    "Node": ("kogwistar.engine_core.models", "Node"),
    "Edge": ("kogwistar.engine_core.models", "Edge"),
    "Document": ("kogwistar.engine_core.models", "Document"),
    "Span": ("kogwistar.engine_core.models", "Span"),
    "Grounding": ("kogwistar.engine_core.models", "Grounding"),
    "shortids": ("kogwistar.shortids", None),
}


def _is_public_module_name(name: str) -> bool:
    return not name.startswith("_")


def _iter_package_modules(package_dir: Path, package_name: str, recursive: bool):
    for child in package_dir.iterdir():
        if child.name == "__pycache__" or not _is_public_module_name(child.name):
            continue

        if child.is_file():
            if child.suffix != ".py" or child.stem == "__init__":
                continue
            yield f"{package_name}.{child.stem}"
            continue

        if child.is_dir() and (child / "__init__.py").is_file():
            child_package = f"{package_name}.{child.name}"
            yield child_package
            if recursive:
                yield from _iter_package_modules(child, child_package, recursive=True)


def list_submodules(recursive: bool = False) -> list[str]:
    """Return importable submodules under :mod:`kogwistar` without importing them."""

    return sorted(set(_iter_package_modules(_PACKAGE_DIR, __name__, recursive)))


def __getattr__(name: str) -> Any:
    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = export
    module = import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose only real root-visible names for interactive discovery."""

    return sorted(_DIR_DUNDERS | set(__all__))
