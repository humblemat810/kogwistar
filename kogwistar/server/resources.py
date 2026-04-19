from __future__ import annotations

import importlib
import os
import pathlib
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from fastapi.templating import Jinja2Templates

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.graph_query import GraphQuery
from kogwistar.server.bootstrap import build_graph_engine, build_sqlalchemy_engine, load_server_storage_settings
from kogwistar.server.chat_service import ChatRunService
from kogwistar.server.run_registry import RunRegistry

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

T = TypeVar("T")


storage_settings = load_server_storage_settings()
persist_directory = storage_settings.knowledge_dir
conversation_persist_directory = storage_settings.conversation_dir
workflow_persist_directory = storage_settings.workflow_dir
wisdom_persist_directory = storage_settings.wisdom_dir


class _LazyResource(Generic[T]):
    """Thread-safe lazy wrapper.

    `get()` is the typed access path. Attribute proxying remains only for
    compatibility with older call sites and should not be relied on for typing.
    """

    def __init__(self, factory: Callable[[], T], name: str) -> None:
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_value", None)
        object.__setattr__(self, "_lock", Lock())

    def get(self) -> T:
        value = object.__getattribute__(self, "_value")
        if value is None:
            lock = object.__getattribute__(self, "_lock")
            with lock:
                value = object.__getattribute__(self, "_value")
                if value is None:
                    value = object.__getattribute__(self, "_factory")()
                    object.__setattr__(self, "_value", value)
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self.get(), name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"_factory", "_name", "_value", "_lock"}:
            object.__setattr__(self, name, value)
            return
        setattr(self.get(), name, value)

    def __repr__(self) -> str:
        value = object.__getattribute__(self, "_value")
        state = "initialized" if value is not None else "lazy"
        return f"<_LazyResource {object.__getattribute__(self, '_name')} ({state})>"


def _build_pg_sqlalchemy_engine() -> Engine:
    return build_sqlalchemy_engine(storage_settings)


def _shared_sqlalchemy_engine() -> Engine | None:
    if storage_settings.backend != "pg":
        return None
    return pg_sqlalchemy_engine.get()


def _build_engine() -> GraphKnowledgeEngine:
    return build_graph_engine(
        settings=storage_settings,
        graph_type="knowledge",
        sa_engine=_shared_sqlalchemy_engine(),
    )


def _build_conversation_engine() -> GraphKnowledgeEngine:
    eng = build_graph_engine(
        settings=storage_settings,
        graph_type="conversation",
        sa_engine=_shared_sqlalchemy_engine(),
    )
    return eng


def _build_workflow_engine() -> GraphKnowledgeEngine:
    return build_graph_engine(
        settings=storage_settings,
        graph_type="workflow",
        sa_engine=_shared_sqlalchemy_engine(),
    )


def _build_wisdom_engine() -> GraphKnowledgeEngine:
    return build_graph_engine(
        settings=storage_settings,
        graph_type="wisdom",
        sa_engine=_shared_sqlalchemy_engine(),
    )


pg_sqlalchemy_engine: _LazyResource[Engine] = _LazyResource(
    _build_pg_sqlalchemy_engine, "pg_sqlalchemy_engine"
)


def _init_auth() -> Engine:
    from kogwistar.server.auth.db import create_auth_engine, init_auth_db

    auth_engine = _shared_sqlalchemy_engine()
    if auth_engine is None:
        auth_db_url = os.getenv("AUTH_DB_URL", "sqlite:///auth.sqlite")
        auth_engine = create_auth_engine(auth_db_url)
    init_auth_db(auth_engine)
    return auth_engine


auth_engine_resource: _LazyResource[Engine] = _LazyResource(_init_auth, "auth_engine")

engine: _LazyResource[GraphKnowledgeEngine] = _LazyResource(
    _build_engine, "knowledge_engine"
)
conversation_engine: _LazyResource[GraphKnowledgeEngine] = _LazyResource(
    _build_conversation_engine, "conversation_engine"
)
workflow_engine: _LazyResource[GraphKnowledgeEngine] = _LazyResource(
    _build_workflow_engine, "workflow_engine"
)
wisdom_engine: _LazyResource[GraphKnowledgeEngine] = _LazyResource(
    _build_wisdom_engine, "wisdom_engine"
)
gq: _LazyResource[GraphQuery] = _LazyResource(
    lambda: GraphQuery(engine.get()), "knowledge_graph_query"
)
conversation_gq: _LazyResource[GraphQuery] = _LazyResource(
    lambda: GraphQuery(conversation_engine.get()), "conversation_graph_query"
)
wisdom_gq: _LazyResource[GraphQuery] = _LazyResource(
    lambda: GraphQuery(wisdom_engine.get()), "wisdom_graph_query"
)
run_registry: _LazyResource[RunRegistry] = _LazyResource(
    lambda: RunRegistry(conversation_engine.get().meta_sqlite),
    "chat_run_registry",
)


def _import_override_from_env(env_name: str) -> Callable[..., Any] | None:
    raw = str(os.getenv(env_name) or "").strip()
    if not raw:
        return None
    module_name, sep, attr_name = raw.partition(":")
    if not sep or not module_name or not attr_name:
        raise RuntimeError(
            f"{env_name} must be in 'module.path:attribute_name' format"
        )
    module = importlib.import_module(module_name)
    target = getattr(module, attr_name, None)
    if not callable(target):
        raise RuntimeError(f"{env_name} target is not callable: {raw}")
    return target


def _build_chat_service() -> ChatRunService:
    answer_runner_factory = _import_override_from_env(
        "KOGWISTAR_TEST_ANSWER_RUNNER_IMPORT"
    )
    runtime_runner_factory = _import_override_from_env(
        "KOGWISTAR_TEST_RUNTIME_RUNNER_IMPORT"
    )
    answer_runner = answer_runner_factory() if answer_runner_factory else None
    runtime_runner = runtime_runner_factory() if runtime_runner_factory else None
    return ChatRunService(
        get_knowledge_engine=lambda: engine.get(),
        get_conversation_engine=lambda: conversation_engine.get(),
        get_workflow_engine=lambda: workflow_engine.get(),
        run_registry=run_registry.get(),
        answer_runner=answer_runner,
        runtime_runner=runtime_runner,
    )


chat_service: _LazyResource[ChatRunService] = _LazyResource(
    _build_chat_service,
    "chat_run_service",
)

templates = Jinja2Templates(
    directory=os.path.join(
        str(pathlib.Path(__file__).resolve().parents[1]), "templates"
    )
)
