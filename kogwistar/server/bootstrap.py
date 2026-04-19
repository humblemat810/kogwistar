from __future__ import annotations

import os
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.async_compat import (
    run_awaitable_blocking,
    run_sync_or_awaitable,
)
from kogwistar.engine_core.chroma_backend import ChromaBackend

if TYPE_CHECKING:
    pass

GraphType = Literal["knowledge", "conversation", "workflow", "wisdom"]
BackendKind = Literal["chroma", "pg"]


def _normalize_backend_name(raw_backend: str | None) -> BackendKind:
    value = str(raw_backend or "chroma").strip().lower()
    aliases = {
        "sqlite": "chroma",
        "chroma": "chroma",
        "pg": "pg",
        "postgres": "pg",
        "pgvector": "pg",
    }
    normalized = aliases.get(value)
    if normalized is None:
        raise RuntimeError(
            f"Unsupported GKE_BACKEND={value!r}; expected 'chroma' or 'pg'."
        )
    return normalized


def _derive_index_dir_from_knowledge_dir(knowledge_dir: str) -> str:
    path = Path(knowledge_dir)
    if path.parent == path:
        return str(path / "index")
    leaf = path.name or "knowledge"
    return str(path.parent / "index" / leaf)


def _import_callable_from_env(env_name: str):
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


@dataclass(frozen=True)
class ServerStorageSettings:
    backend: BackendKind
    knowledge_dir: str
    conversation_dir: str
    workflow_dir: str
    wisdom_dir: str
    index_dir: str
    chroma_async: bool = False
    chroma_host: str | None = None
    chroma_port: int | None = None
    pg_url: str | None = None
    pg_schema_base: str = "gke"
    embedding_dim: int = 384
    pg_async: bool = False

    def persist_directory_for(self, graph_type: GraphType) -> str:
        mapping = {
            "knowledge": self.knowledge_dir,
            "conversation": self.conversation_dir,
            "workflow": self.workflow_dir,
            "wisdom": self.wisdom_dir,
        }
        return mapping[graph_type]

    def schema_for(self, graph_type: GraphType) -> str:
        return f"{self.pg_schema_base}_{graph_type}"


def load_server_storage_settings(
    env: dict[str, str] | None = None,
) -> ServerStorageSettings:
    values = env or os.environ
    backend = _normalize_backend_name(values.get("GKE_BACKEND"))
    base_persist = str(values.get("GKE_PERSIST_DIRECTORY") or "./.gke-data")
    has_shared_persist_root = bool(values.get("GKE_PERSIST_DIRECTORY"))

    if backend == "chroma":
        chroma_async = str(values.get("GKE_CHROMA_ASYNC") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        chroma_host = values.get("GKE_CHROMA_HOST") or values.get("MCP_CHROMA_HOST")
        chroma_port_raw = values.get("GKE_CHROMA_PORT") or values.get("MCP_CHROMA_PORT")
        chroma_port = int(chroma_port_raw) if chroma_port_raw else None
        knowledge_dir = str(
            values.get("MCP_CHROMA_DIR")
            or values.get("GKE_KNOWLEDGE_PERSIST_DIRECTORY")
            or (
                os.path.join(base_persist, "knowledge")
                if has_shared_persist_root
                else "./.chroma-mcp"
            )
        )
        conversation_dir = str(
            values.get("MCP_CHROMA_DIR_CONVERSATION")
            or values.get("GKE_CONVERSATION_PERSIST_DIRECTORY")
            or (
                os.path.join(base_persist, "conversation")
                if has_shared_persist_root
                else f"{knowledge_dir}-conversation"
            )
        )
        workflow_dir = str(
            values.get("MCP_CHROMA_DIR_WORKFLOW")
            or values.get("GKE_WORKFLOW_PERSIST_DIRECTORY")
            or (
                os.path.join(base_persist, "workflow")
                if has_shared_persist_root
                else f"{knowledge_dir}-workflow"
            )
        )
        wisdom_dir = str(
            values.get("MCP_CHROMA_DIR_WISDOM")
            or values.get("GKE_WISDOM_PERSIST_DIRECTORY")
            or (
                os.path.join(base_persist, "wisdom")
                if has_shared_persist_root
                else f"{knowledge_dir}-wisdom"
            )
        )
        index_dir = str(
            values.get("GKE_INDEX_DIR")
            or (
                os.path.join(base_persist, "index")
                if has_shared_persist_root
                else _derive_index_dir_from_knowledge_dir(knowledge_dir)
            )
        )
        return ServerStorageSettings(
            backend=backend,
            chroma_async=chroma_async,
            chroma_host=None if chroma_host is None else str(chroma_host),
            chroma_port=chroma_port,
            knowledge_dir=knowledge_dir,
            conversation_dir=conversation_dir,
            workflow_dir=workflow_dir,
            wisdom_dir=wisdom_dir,
            index_dir=index_dir,
        )

    pg_url = values.get("GKE_PG_URL") or values.get("DATABASE_URL")
    if not pg_url:
        raise RuntimeError("GKE_BACKEND=pg requires GKE_PG_URL (or DATABASE_URL).")
    pg_async = str(values.get("GKE_PG_ASYNC") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    index_dir = str(values.get("GKE_INDEX_DIR") or os.path.join(base_persist, "index"))
    return ServerStorageSettings(
        backend=backend,
        pg_async=pg_async,
        knowledge_dir=str(
            values.get("GKE_KNOWLEDGE_PERSIST_DIRECTORY")
            or os.path.join(base_persist, "knowledge")
        ),
        conversation_dir=str(
            values.get("GKE_CONVERSATION_PERSIST_DIRECTORY")
            or os.path.join(base_persist, "conversation")
        ),
        workflow_dir=str(
            values.get("GKE_WORKFLOW_PERSIST_DIRECTORY")
            or os.path.join(base_persist, "workflow")
        ),
        wisdom_dir=str(
            values.get("GKE_WISDOM_PERSIST_DIRECTORY")
            or os.path.join(base_persist, "wisdom")
        ),
        index_dir=index_dir,
        pg_url=str(pg_url),
        pg_schema_base=str(values.get("GKE_PG_SCHEMA") or "gke"),
        embedding_dim=int(values.get("GKE_EMBEDDING_DIM") or "384"),
    )


def build_sqlalchemy_engine(settings: ServerStorageSettings) -> Any:
    if settings.backend != "pg" or not settings.pg_url:
        raise RuntimeError(
            "SQLAlchemy engine is only available for pg storage settings."
        )
    import sqlalchemy as sa

    return sa.create_engine(settings.pg_url, future=True)


def build_graph_engine(
    *,
    settings: ServerStorageSettings,
    graph_type: GraphType,
    sa_engine: Any | None = None,
) -> GraphKnowledgeEngine:
    persist_directory = settings.persist_directory_for(graph_type)
    embedding_factory = _import_callable_from_env(
        "KOGWISTAR_TEST_EMBEDDING_FUNCTION_IMPORT"
    )
    embedding_function = embedding_factory() if embedding_factory else None
    if settings.backend == "chroma":
        if settings.chroma_async:
            if not settings.chroma_host or settings.chroma_port is None:
                raise RuntimeError(
                    "GKE_BACKEND=chroma with GKE_CHROMA_ASYNC=1 requires "
                    "GKE_CHROMA_HOST and GKE_CHROMA_PORT."
                )
            import chromadb

            client = run_awaitable_blocking(
                chromadb.AsyncHttpClient(
                    host=str(settings.chroma_host), port=int(settings.chroma_port)
                )
            )

            def _make_backend(_engine: GraphKnowledgeEngine) -> ChromaBackend:
                collections = {
                    "node_index": run_awaitable_blocking(
                        client.get_or_create_collection(
                            name="nodes_index",
                            metadata={"hnsw:space": "cosine"},
                        )
                    ),
                    "node": run_awaitable_blocking(
                        client.get_or_create_collection(
                            name="nodes",
                            metadata={"hnsw:space": "cosine"},
                        )
                    ),
                    "edge": run_awaitable_blocking(
                        client.get_or_create_collection(
                            name="edges",
                            metadata={"hnsw:space": "cosine"},
                        )
                    ),
                    "edge_endpoints": run_awaitable_blocking(
                        client.get_or_create_collection(
                            name="edge_endpoints",
                            metadata={"hnsw:space": "cosine"},
                        )
                    ),
                    "document": run_awaitable_blocking(
                        client.get_or_create_collection(
                            name="documents",
                            metadata={"hnsw:space": "cosine"},
                        )
                    ),
                    "domain": run_awaitable_blocking(
                        client.get_or_create_collection(
                            name="domains",
                            metadata={"hnsw:space": "cosine"},
                        )
                    ),
                    "node_docs": run_awaitable_blocking(
                        client.get_or_create_collection(
                            name="node_docs",
                            metadata={"hnsw:space": "cosine"},
                        )
                    ),
                    "node_refs": run_awaitable_blocking(
                        client.get_or_create_collection(name="node_refs")
                    ),
                    "edge_refs": run_awaitable_blocking(
                        client.get_or_create_collection(name="edge_refs")
                    ),
                }
                return ChromaBackend(
                    node_index_collection=collections["node_index"],
                    node_collection=collections["node"],
                    edge_collection=collections["edge"],
                    edge_endpoints_collection=collections["edge_endpoints"],
                    document_collection=collections["document"],
                    domain_collection=collections["domain"],
                    node_docs_collection=collections["node_docs"],
                    node_refs_collection=collections["node_refs"],
                    edge_refs_collection=collections["edge_refs"],
                )

            return GraphKnowledgeEngine(
                persist_directory=persist_directory,
                kg_graph_type=graph_type,
                embedding_function=embedding_function,
                backend_factory=_make_backend,
            )
        return GraphKnowledgeEngine(
            persist_directory=persist_directory,
            kg_graph_type=graph_type,
            embedding_function=embedding_function,
        )

    if settings.pg_async:
        from kogwistar.engine_core.engine_postgres import (
            EnginePostgresConfig,
            build_async_postgres_backend,
        )

        if not settings.pg_url:
            raise RuntimeError("pg backend requires GKE_PG_URL (or DATABASE_URL).")
        backend, _uow = build_async_postgres_backend(
            EnginePostgresConfig(
                dsn=settings.pg_url,
                embedding_dim=settings.embedding_dim,
                schema=settings.schema_for(graph_type),
            )
        )
        return GraphKnowledgeEngine(
            persist_directory=persist_directory,
            kg_graph_type=graph_type,
            embedding_function=embedding_function,
            backend=backend,
        )

    if sa_engine is None:
        raise RuntimeError("pg backend requires a shared SQLAlchemy engine.")
    from kogwistar.engine_core.postgres_backend import PgVectorBackend

    backend = PgVectorBackend(
        engine=sa_engine,
        embedding_dim=settings.embedding_dim,
        schema=settings.schema_for(graph_type),
    )
    return GraphKnowledgeEngine(
        persist_directory=persist_directory,
        kg_graph_type=graph_type,
        embedding_function=embedding_function,
        backend=backend,
    )
