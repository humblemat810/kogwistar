from __future__ import annotations

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend
from kogwistar.engine_core.models import Grounding, Node, Span
from kogwistar.engine_core.search_index.models import IndexingItem
from kogwistar.server.auth_middleware import claims_ctx
from tests._helpers.embeddings import build_test_embedding_function


def _node(node_id: str, label: str) -> Node:
    span = Span.from_dummy_for_conversation()
    return Node(
        id=node_id,
        label=label,
        type="entity",
        summary=label,
        mentions=[Grounding(spans=[span])],
        metadata={"entity_type": "test"},
        embedding=None,
        doc_id=span.doc_id,
        level_from_root=0,
    )


def _add_acl(engine: GraphKnowledgeEngine, node: Node, mode: str) -> None:
    engine.acl_graph.add_record(
        truth_graph="knowledge",
        entity_id=node.id,
        version=1,
        mode=mode,
        owner_id="owner-a",
    )
    if mode != "public":
        return

    grounding_ids, span_ids = engine.acl.usage_ids_for_item(node)
    for grain, target_ids in (
        ("grounding", grounding_ids),
        ("span", span_ids),
    ):
        for target_id in target_ids:
            engine.acl_graph.add_record(
                grain=grain,
                truth_graph="knowledge",
                entity_id=node.id,
                target_item_id=target_id,
                version=1,
                mode="public",
            )


@pytest.mark.ci
@pytest.mark.parametrize("resolve_node", [False, True])
def test_hybrid_search_filters_private_index_entries_by_graph_acl(
    tmp_path, resolve_node: bool
) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_in_memory_backend,
        embedding_function=build_test_embedding_function("constant", dim=8),
        kg_graph_type="knowledge",
        acl_enabled=True,
    )
    try:
        private_node = _node("private-law", "Hidden Regulation")
        public_node = _node("public-law", "Visible Regulation")
        engine.raw_write.add_node(private_node)
        engine.raw_write.add_node(public_node)
        _add_acl(engine, private_node, "private")
        _add_acl(engine, public_node, "public")
        engine.search_index.upsert_entries(
            [
                IndexingItem(
                    node_id=private_node.id,
                    canonical_title="Hidden Regulation",
                    keywords=["classified", "permit"],
                    aliases=[],
                    provision="confidential regulatory text",
                ),
                IndexingItem(
                    node_id=public_node.id,
                    canonical_title="Visible Regulation",
                    keywords=["classified", "permit"],
                    aliases=[],
                    provision="public regulatory text",
                ),
            ]
        )

        token = claims_ctx.set({"agent_id": "reader-b", "ns": "docs"})
        try:
            result = engine.search_index.search_hybrid(
                "classified permit", limit=10, resolve_node=resolve_node
            )
        finally:
            claims_ctx.reset(token)

        rows = result["results"]
        assert [row["node_id"] for row in rows] == [public_node.id]
        assert all("Hidden Regulation" not in str(row) for row in rows)
        assert all("confidential regulatory text" not in str(row) for row in rows)
    finally:
        engine.close()


@pytest.mark.ci
def test_hybrid_search_keeps_legacy_index_only_results_without_acl(
    tmp_path,
) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_in_memory_backend,
        embedding_function=build_test_embedding_function("constant", dim=8),
    )
    try:
        engine.search_index.upsert_entries(
            [
                IndexingItem(
                    node_id="not-yet-materialized",
                    canonical_title="Deferred Entry",
                    keywords=["legacy", "search"],
                    aliases=[],
                    provision="index-only compatibility entry",
                )
            ]
        )
        result = engine.search_index.search_hybrid("legacy search", limit=5)
        assert result["results"][0]["node_id"] == "not-yet-materialized"
    finally:
        engine.close()
