from __future__ import annotations

import os

import pytest

from kogwistar.server.bootstrap import load_server_storage_settings
from kogwistar.server.bootstrap import build_graph_engine

pytestmark = pytest.mark.ci


def _norm(path: str) -> str:
    return path.replace("\\", "/")


def test_load_server_storage_settings_chroma_legacy_defaults():
    settings = load_server_storage_settings({})

    assert settings.backend == "chroma"
    assert settings.knowledge_dir == "./.chroma-mcp"
    assert settings.conversation_dir == "./.chroma-mcp-conversation"
    assert settings.workflow_dir == "./.chroma-mcp-workflow"
    assert settings.wisdom_dir == "./.chroma-mcp-wisdom"
    assert _norm(settings.index_dir).endswith("index/.chroma-mcp")


def test_load_server_storage_settings_chroma_shared_root():
    root = os.path.join("var", "lib", "gke")
    settings = load_server_storage_settings({"GKE_PERSIST_DIRECTORY": root})

    assert settings.backend == "chroma"
    assert settings.knowledge_dir == os.path.join(root, "knowledge")
    assert settings.conversation_dir == os.path.join(root, "conversation")
    assert settings.workflow_dir == os.path.join(root, "workflow")
    assert settings.wisdom_dir == os.path.join(root, "wisdom")
    assert settings.index_dir == os.path.join(root, "index")


def test_load_server_storage_settings_pg_requires_url():
    with pytest.raises(RuntimeError, match="GKE_PG_URL"):
        load_server_storage_settings({"GKE_BACKEND": "pg"})


def test_load_server_storage_settings_pg_uses_schema_suffixes():
    root = os.path.join("srv", "gke")
    settings = load_server_storage_settings(
        {
            "GKE_BACKEND": "postgres",
            "GKE_PG_URL": "postgresql+psycopg://user:pass@db:5432/app",
            "GKE_PERSIST_DIRECTORY": root,
            "GKE_PG_SCHEMA": "tenant",
            "GKE_EMBEDDING_DIM": "3072",
        }
    )

    assert settings.backend == "pg"
    assert settings.pg_url == "postgresql+psycopg://user:pass@db:5432/app"
    assert settings.embedding_dim == 3072
    assert settings.knowledge_dir == os.path.join(root, "knowledge")
    assert settings.index_dir == os.path.join(root, "index")
    assert settings.schema_for("knowledge") == "tenant_knowledge"
    assert settings.schema_for("workflow") == "tenant_workflow"


def test_load_server_storage_settings_reads_server_acl_opt_in():
    assert load_server_storage_settings({}).acl_enabled is False
    settings = load_server_storage_settings({"KOGWISTAR_SERVER_ACL_ENABLED": "1"})

    assert settings.acl_enabled is True


def test_build_graph_engine_propagates_server_acl_opt_in(monkeypatch, tmp_path):
    from kogwistar.server.bootstrap import ServerStorageSettings

    captured: dict[str, object] = {}

    class FakeGraphKnowledgeEngine:
        def __init__(self, **kwargs: object):
            captured.update(kwargs)

    monkeypatch.setattr(
        "kogwistar.server.bootstrap.GraphKnowledgeEngine",
        FakeGraphKnowledgeEngine,
    )
    settings = ServerStorageSettings(
        backend="chroma",
        knowledge_dir=str(tmp_path / "knowledge"),
        conversation_dir=str(tmp_path / "conversation"),
        workflow_dir=str(tmp_path / "workflow"),
        wisdom_dir=str(tmp_path / "wisdom"),
        index_dir=str(tmp_path / "index"),
        acl_enabled=True,
    )

    build_graph_engine(settings=settings, graph_type="conversation")

    assert captured["acl_enabled"] is True


def test_pg_resource_factory_resolves_engine_type_at_runtime(monkeypatch):
    from kogwistar.server import resources

    sentinel = object()
    monkeypatch.setattr(resources, "build_sqlalchemy_engine", lambda _settings: sentinel)

    assert resources._build_pg_sqlalchemy_engine() is sentinel
