from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine

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


@dataclass(frozen=True)
class ServerStorageSettings:
    backend: BackendKind
    knowledge_dir: str
    conversation_dir: str
    workflow_dir: str
    wisdom_dir: str
    index_dir: str
    pg_url: str | None = None
    pg_schema_base: str = "gke"
    embedding_dim: int = 1536

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
            knowledge_dir=knowledge_dir,
            conversation_dir=conversation_dir,
            workflow_dir=workflow_dir,
            wisdom_dir=wisdom_dir,
            index_dir=index_dir,
        )

    pg_url = values.get("GKE_PG_URL") or values.get("DATABASE_URL")
    if not pg_url:
        raise RuntimeError("GKE_BACKEND=pg requires GKE_PG_URL (or DATABASE_URL).")
    index_dir = str(values.get("GKE_INDEX_DIR") or os.path.join(base_persist, "index"))
    return ServerStorageSettings(
        backend=backend,
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
        embedding_dim=int(values.get("GKE_EMBEDDING_DIM") or "1536"),
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
    if settings.backend == "chroma":
        return GraphKnowledgeEngine(
            persist_directory=persist_directory,
            kg_graph_type=graph_type,
        )

    if sa_engine is None:
        raise RuntimeError("pg backend requires a shared SQLAlchemy engine.")
    from graph_knowledge_engine.engine_core.postgres_backend import PgVectorBackend

    backend = PgVectorBackend(
        engine=sa_engine,
        embedding_dim=settings.embedding_dim,
        schema=settings.schema_for(graph_type),
    )
    return GraphKnowledgeEngine(
        persist_directory=persist_directory,
        kg_graph_type=graph_type,
        backend=backend,
    )
