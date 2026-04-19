from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine, scoped_namespace
from tests._helpers.fake_backend import build_fake_backend


pytestmark = pytest.mark.core


def _make_engine() -> tuple[GraphKnowledgeEngine, Path]:
    test_db_dir = Path.cwd() / ".tmp_lane_rebuild" / str(uuid.uuid4())
    test_db_dir.mkdir(parents=True, exist_ok=True)
    engine = GraphKnowledgeEngine(
        persist_directory=str(test_db_dir),
        backend_factory=build_fake_backend,
        kg_graph_type="conversation",
    )
    return engine, test_db_dir


def _rebuild_lane_projection_from_truth(engine: GraphKnowledgeEngine, *, namespace: str) -> None:
    meta = engine.meta_sqlite
    if hasattr(meta, "clear_projection_namespace"):
        meta.clear_projection_namespace(namespace)
    nodes = engine.read.get_nodes(where={"artifact_kind": "lane_message"}, limit=10_000)
    for node in nodes:
        md = dict(getattr(node, "metadata", {}) or {})
        if str(md.get("namespace") or "") != str(namespace):
            continue
        meta.project_lane_message(
            message_id=str(getattr(node, "id", "") or ""),
            namespace=namespace,
            inbox_id=str(md.get("inbox_id") or ""),
            conversation_id=str(md.get("conversation_id") or ""),
            recipient_id=str(md.get("recipient_id") or ""),
            sender_id=str(md.get("sender_id") or ""),
            msg_type=str(md.get("msg_type") or ""),
            status=str(md.get("status") or "pending"),
            created_at=0,
            available_at=0,
            run_id=md.get("run_id"),
            step_id=md.get("step_id"),
            correlation_id=md.get("correlation_id"),
            payload_json=None,
            error_json=None,
        )


def test_lane_message_projection_can_be_rebuilt_from_authoritative_truth():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"

    try:
        with scoped_namespace(engine, namespace):
            first = engine.send_lane_message(
                conversation_id="conv-rebuild",
                inbox_id="inbox:worker:rebuild",
                sender_id="lane:foreground",
                recipient_id="lane:worker:rebuild",
                msg_type="request.rebuild",
                payload={"kind": "rebuild"},
            )
            second = engine.send_lane_message(
                conversation_id="conv-rebuild",
                inbox_id="inbox:worker:rebuild",
                sender_id="lane:foreground",
                recipient_id="lane:worker:rebuild",
                msg_type="request.rebuild",
                payload={"kind": "rebuild-2"},
            )
            meta = engine.meta_sqlite
            assert meta.list_projected_lane_messages(
                namespace=namespace, inbox_id="inbox:worker:rebuild"
            )

            _rebuild_lane_projection_from_truth(engine, namespace=namespace)

            rows = meta.list_projected_lane_messages(
                namespace=namespace, inbox_id="inbox:worker:rebuild"
            )
            assert len(rows) == 2
            by_id = {row.message_id: row for row in rows}
            assert by_id[first.message_id].status == "pending"
            assert by_id[second.message_id].status == "pending"
            assert by_id[first.message_id].seq == 1
            assert by_id[second.message_id].seq == 2
            assert by_id[first.message_id].inbox_tail_message_id == first.message_id
            assert by_id[second.message_id].prev_message_id == first.message_id
            assert by_id[second.message_id].inbox_tail_message_id == second.message_id
            assert by_id[second.message_id].conversation_tail_message_id == second.message_id
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)
