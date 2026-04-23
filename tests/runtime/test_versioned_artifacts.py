from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine, scoped_namespace
from kogwistar.engine_core.models import Grounding, Node, Span
from kogwistar.maintenance.artifacts import write_versioned_artifact
from tests._helpers.fake_backend import build_fake_backend


pytestmark = [pytest.mark.ci, pytest.mark.runtime]


def _make_engine() -> tuple[GraphKnowledgeEngine, Path]:
    test_db_dir = Path.cwd() / ".tmp_runtime_artifacts" / str(uuid.uuid4())
    test_db_dir.mkdir(parents=True, exist_ok=True)
    engine = GraphKnowledgeEngine(
        persist_directory=str(test_db_dir),
        backend_factory=build_fake_backend,
    )
    return engine, test_db_dir


def _mentions() -> list[Grounding]:
    return [Grounding(spans=[Span(
        collection_page_url="conversation/demo",
        document_page_url="conversation/demo",
        doc_id="conv:demo",
        insertion_method="test",
        page_number=1,
        start_char=0,
        end_char=1,
        excerpt="x",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
    )])]


def test_write_versioned_artifact_redirects_old_ids_to_new_active_version():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:kg:derived"

    try:
        with scoped_namespace(engine, namespace):
            engine.write.add_node(
                Node(
                    id="alice-v1",
                    label="Alice",
                    type="entity",
                    summary="old",
                        mentions=_mentions(),
                    metadata={
                        "workspace_id": "demo",
                        "artifact_kind": "derived_knowledge",
                        "label": "Alice",
                    },
                )
            )

        result = write_versioned_artifact(
            engine,
            namespace=namespace,
            match_where={
                "workspace_id": "demo",
                "artifact_kind": "derived_knowledge",
                "label": "Alice",
            },
            build_node=lambda existing, created_at_ms: Node(
                id="alice-v2",
                label="Alice",
                type="entity",
                summary="new",
                mentions=_mentions(),
                metadata={
                    "workspace_id": "demo",
                    "artifact_kind": "derived_knowledge",
                    "label": "Alice",
                    "created_at_ms": created_at_ms,
                    "replaces_ids": [str(node.id) for node in existing],
                },
            ),
            replace_existing=True,
        )

        with scoped_namespace(engine, namespace):
            active = engine.read.get_nodes(
                where={"artifact_kind": "derived_knowledge", "label": "Alice"}
            )
            redirected = engine.read.get_nodes(
                ids=["alice-v1"],
                resolve_mode="redirect",
            )
            old_all = engine.read.get_nodes(
                ids=["alice-v1"],
                resolve_mode="include_tombstones",
            )

        assert result.artifact_id == "alice-v2"
        assert result.created_at_ms == active[0].metadata["created_at_ms"]
        assert result.replaced_ids == ("alice-v1",)
        assert [node.id for node in active] == ["alice-v2"]
        assert [node.id for node in redirected] == ["alice-v2"]
        assert old_all[0].metadata["lifecycle_status"] == "tombstoned"
        assert old_all[0].metadata["redirect_to_id"] == "alice-v2"
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)


def test_write_versioned_artifact_can_append_without_replacing_existing():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:kg:derived"

    try:
        with scoped_namespace(engine, namespace):
            engine.write.add_node(
                Node(
                    id="alice-v1",
                    label="Alice",
                    type="entity",
                    summary="old",
                        mentions=_mentions(),
                    metadata={
                        "workspace_id": "demo",
                        "artifact_kind": "derived_knowledge",
                        "label": "Alice",
                    },
                )
            )

        write_versioned_artifact(
            engine,
            namespace=namespace,
            match_where={
                "workspace_id": "demo",
                "artifact_kind": "derived_knowledge",
                "label": "Alice",
            },
            build_node=lambda existing, created_at_ms: Node(
                id="alice-v2",
                label="Alice",
                type="entity",
                summary="new",
                mentions=_mentions(),
                metadata={
                    "workspace_id": "demo",
                    "artifact_kind": "derived_knowledge",
                    "label": "Alice",
                    "created_at_ms": created_at_ms,
                    "replaces_ids": [str(node.id) for node in existing],
                },
            ),
            replace_existing=False,
        )

        with scoped_namespace(engine, namespace):
            active = engine.read.get_nodes(
                where={"artifact_kind": "derived_knowledge", "label": "Alice"},
                resolve_mode="include_tombstones",
            )

        assert {node.id for node in active} == {"alice-v1", "alice-v2"}
        by_id = {node.id: node for node in active}
        assert by_id["alice-v1"].metadata.get("lifecycle_status", "active") == "active"
        assert by_id["alice-v1"].metadata.get("redirect_to_id") is None
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)
