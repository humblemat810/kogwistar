from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine, scoped_namespace
from kogwistar.engine_core.models import Grounding, Node, Span
from kogwistar.maintenance.grouped_artifacts import write_grouped_versioned_artifacts
from tests._helpers.fake_backend import build_fake_backend


pytestmark = pytest.mark.ci


def _make_engine() -> tuple[GraphKnowledgeEngine, Path]:
    test_db_dir = Path.cwd() / ".tmp_runtime_grouped_artifacts" / str(uuid.uuid4())
    test_db_dir.mkdir(parents=True, exist_ok=True)
    engine = GraphKnowledgeEngine(
        persist_directory=str(test_db_dir),
        backend_factory=build_fake_backend,
    )
    return engine, test_db_dir


def _mentions() -> list[Grounding]:
    return [
        Grounding(
            spans=[
                Span(
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
                )
            ]
        )
    ]


def test_write_grouped_versioned_artifacts_groups_candidates_and_writes_replacements():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:kg:derived"

    try:
        with scoped_namespace(engine, namespace):
            for node_id, label in (
                ("alice-v1", "Alice"),
                ("alice-v2", "Alice"),
                ("bob-v1", "Bob"),
            ):
                engine.write.add_node(
                    Node(
                        id=node_id,
                        label=label,
                        type="entity",
                        summary="source",
                        mentions=_mentions(),
                        metadata={
                            "workspace_id": "demo",
                            "artifact_kind": "promoted_knowledge",
                            "label": label,
                        },
                    )
                )

        results = write_grouped_versioned_artifacts(
            engine,
            target_engine=engine,
            source_namespace=namespace,
            target_namespace=namespace,
            source_where={"artifact_kind": "promoted_knowledge", "workspace_id": "demo"},
            group_key_for_node=lambda node: str(node.metadata.get("label")),
            match_where_for_group=lambda label: {
                "workspace_id": "demo",
                "artifact_kind": "derived_knowledge",
                "label": label,
            },
            build_node_for_group=lambda label, nodes, existing, created_at_ms: Node(
                id=f"{label.lower()}-{created_at_ms}",
                label=label,
                type="entity",
                summary=f"Derived {label}",
                mentions=_mentions(),
                metadata={
                    "workspace_id": "demo",
                    "artifact_kind": "derived_knowledge",
                    "label": label,
                    "created_at_ms": created_at_ms,
                    "source_node_ids": [str(node.id) for node in nodes],
                    "replaces_ids": [str(node.id) for node in existing],
                },
            ),
        )

        with scoped_namespace(engine, namespace):
            derived = engine.read.get_nodes(
                where={"workspace_id": "demo", "artifact_kind": "derived_knowledge"}
            )

        assert {result.group_key for result in results} == {"Alice", "Bob"}
        assert len(derived) == 2
        alice = next(node for node in derived if node.label == "Alice")
        bob = next(node for node in derived if node.label == "Bob")
        assert set(alice.metadata["source_node_ids"]) == {"alice-v1", "alice-v2"}
        assert set(bob.metadata["source_node_ids"]) == {"bob-v1"}
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)
