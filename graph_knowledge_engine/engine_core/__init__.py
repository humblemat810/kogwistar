"""Legacy compatibility alias for `graph_knowledge_engine.engine_core.*`."""

from __future__ import annotations

from importlib import import_module
import sys

_engine_core = import_module("kogwistar.engine_core")

from kogwistar.engine_core import *  # noqa: F401,F403

sys.modules.setdefault("graph_knowledge_engine.engine_core", _engine_core)
sys.modules.setdefault(
    "graph_knowledge_engine.engine_core.engine",
    import_module("kogwistar.engine_core.engine"),
)
sys.modules.setdefault(
    "graph_knowledge_engine.engine_core.models",
    import_module("kogwistar.engine_core.models"),
)
sys.modules.setdefault(
    "graph_knowledge_engine.engine_core.chroma_backend",
    import_module("kogwistar.engine_core.chroma_backend"),
)
sys.modules.setdefault(
    "graph_knowledge_engine.engine_core.engine_sqlite",
    import_module("kogwistar.engine_core.engine_sqlite"),
)
sys.modules.setdefault(
    "graph_knowledge_engine.engine_core.storage_backend",
    import_module("kogwistar.engine_core.storage_backend"),
)
sys.modules.setdefault(
    "graph_knowledge_engine.engine_core.types",
    import_module("kogwistar.engine_core.types"),
)
sys.modules.setdefault(
    "graph_knowledge_engine.engine_core.async_compat",
    import_module("kogwistar.engine_core.async_compat"),
)
