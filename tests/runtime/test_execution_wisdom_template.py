from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine, scoped_namespace
from kogwistar.engine_core.models import Grounding, Node, Span
from kogwistar.wisdom.template import write_execution_wisdom_artifacts
from tests._helpers.fake_backend import build_fake_backend


pytestmark = pytest.mark.core


def _make_engine() -> tuple[GraphKnowledgeEngine, Path]:
    test_db_dir = Path.cwd() / ".tmp_runtime_execution_wisdom_template" / str(uuid.uuid4())
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


def _step_exec(node_id: str, *, step_op: str, status: str, run_id: str) -> Node:
    return Node(
        id=node_id,
        label=node_id,
        type="entity",
        summary="step exec",
        mentions=_mentions(),
        metadata={
            "workspace_id": "demo",
            "entity_type": "workflow_step_exec",
            "step_op": step_op,
            "status": status,
            "run_id": run_id,
        },
    )


def test_write_execution_wisdom_artifacts_groups_repeated_failures():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"

    try:
        with scoped_namespace(engine, namespace):
            for node in (
                _step_exec("run-a-1", step_op="distill", status="failure", run_id="run-a"),
                _step_exec("run-a-2", step_op="distill", status="error", run_id="run-b"),
                _step_exec("run-b-1", step_op="other", status="failure", run_id="run-c"),
            ):
                engine.write.add_node(node)

        result = write_execution_wisdom_artifacts(
            engine,
            target_engine=engine,
            source_namespace=namespace,
            target_namespace="ws:demo:wisdom",
            source_where={"entity_type": "workflow_step_exec"},
            min_failure_signals=2,
            match_where_for_pattern=lambda pattern: {
                "workspace_id": "demo",
                "artifact_kind": "execution_wisdom",
                "step_op": pattern.step_op,
            },
            build_node_for_pattern=lambda pattern, existing, created_at_ms: Node(
                id=f"{pattern.step_op}-{created_at_ms}",
                label=f"execution_failure_pattern:{pattern.step_op}",
                type="entity",
                summary="Repeated failure pattern",
                mentions=_mentions(),
                metadata={
                    "workspace_id": "demo",
                    "artifact_kind": "execution_wisdom",
                    "step_op": pattern.step_op,
                    "failure_count": len(pattern.failure_nodes),
                    "evidence_run_ids": list(pattern.run_ids),
                    "created_at_ms": created_at_ms,
                    "replaces_ids": [str(node.id) for node in existing],
                },
            ),
        )

        assert len(result) == 1
        assert result[0].step_op == "distill"
        assert result[0].failure_count == 2
        with scoped_namespace(engine, "ws:demo:wisdom"):
            wisdom_nodes = engine.read.get_nodes(
                where={"workspace_id": "demo", "artifact_kind": "execution_wisdom"}
            )
        assert len(wisdom_nodes) == 1
        assert wisdom_nodes[0].metadata["failure_count"] == 2
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)


def test_write_execution_wisdom_artifacts_reads_source_and_writes_target():
    source_engine, source_db_dir = _make_engine()
    target_engine, target_db_dir = _make_engine()
    source_namespace = "ws:demo:conv:bg"
    target_namespace = "ws:demo:wisdom"

    try:
        with scoped_namespace(source_engine, source_namespace):
            source_engine.write.add_node(
                _step_exec("src-1", step_op="route", status="failure", run_id="run-a")
            )
            source_engine.write.add_node(
                _step_exec("src-2", step_op="route", status="error", run_id="run-b")
            )

        result = write_execution_wisdom_artifacts(
            source_engine,
            target_engine=target_engine,
            source_namespace=source_namespace,
            target_namespace=target_namespace,
            source_where={"entity_type": "workflow_step_exec"},
            min_failure_signals=2,
            match_where_for_pattern=lambda pattern: {
                "workspace_id": "demo",
                "artifact_kind": "execution_wisdom",
                "step_op": pattern.step_op,
            },
            build_node_for_pattern=lambda pattern, existing, created_at_ms: Node(
                id=f"{pattern.step_op}-{created_at_ms}",
                label=f"execution_failure_pattern:{pattern.step_op}",
                type="entity",
                summary="Repeated failure pattern",
                mentions=_mentions(),
                metadata={
                    "workspace_id": "demo",
                    "artifact_kind": "execution_wisdom",
                    "step_op": pattern.step_op,
                    "failure_count": len(pattern.failure_nodes),
                    "evidence_run_ids": list(pattern.run_ids),
                    "created_at_ms": created_at_ms,
                    "replaces_ids": [str(node.id) for node in existing],
                },
            ),
        )

        assert len(result) == 1
        assert result[0].step_op == "route"
        assert result[0].failure_count == 2

        with scoped_namespace(source_engine, source_namespace):
            source_nodes = source_engine.read.get_nodes(
                where={"entity_type": "workflow_step_exec"}
            )
            assert len(source_nodes) == 2
            assert source_engine.read.get_nodes(
                where={"artifact_kind": "execution_wisdom"}
            ) == []

        with scoped_namespace(target_engine, target_namespace):
            wisdom_nodes = target_engine.read.get_nodes(
                where={"workspace_id": "demo", "artifact_kind": "execution_wisdom"}
            )
        assert len(wisdom_nodes) == 1
        assert wisdom_nodes[0].metadata["failure_count"] == 2
    finally:
        shutil.rmtree(source_db_dir, ignore_errors=True)
        shutil.rmtree(target_db_dir, ignore_errors=True)
