from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import List

import pytest


from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core import models

pytestmark = pytest.mark.ci_full


def _dummy_span(doc_id: str = "_dummy_doc") -> models.Span:
    """
    Construct a Span without calling Document.from_dummy().
    Span requires URLs, doc_id, insertion_method, page_number, start/end chars, excerpt, contexts.
    """
    excerpt = "dummy excerpt"
    return models.Span(
        collection_page_url=f"document_collection/{doc_id}",
        document_page_url=f"document/{doc_id}",
        doc_id=doc_id,
        insertion_method="test",
        page_number=1,
        start_char=0,
        end_char=len(excerpt),
        excerpt=excerpt,
        context_before="",
        context_after="",
    )


def _dummy_grounding(doc_id: str = "_dummy_doc") -> models.Grounding:
    return models.Grounding(spans=[_dummy_span(doc_id)])


def _make_node(
    *, summary: str, label: str = "N", doc_id: str = "_dummy_doc"
) -> models.Node:
    return models.Node(
        label=label,
        type="entity",
        summary=summary,
        mentions=[_dummy_grounding(doc_id)],
        metadata={"level_from_root": 0},
        doc_id=doc_id,
        embedding=None,
        # Optional but keep stable:
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )


def _make_edge(
    *, src: str, tgt: str, relation: str = "related_to", doc_id: str = "_dummy_doc"
) -> models.Edge:
    return models.Edge(
        label="E",
        type="relationship",
        summary=f"{src}->{tgt}",
        mentions=[_dummy_grounding(doc_id)],
        metadata={},
        doc_id=doc_id,
        relation=relation,
        source_ids=[src],
        target_ids=[tgt],
        # These are required in your model (Field(...)) even if empty:
        source_edge_ids=[],
        target_edge_ids=[],
        embedding=None,
        # Optional:
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )


def _engine(tmp_path: Path) -> GraphKnowledgeEngine:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return GraphKnowledgeEngine(persist_directory=str(tmp_path), kg_graph_type="kg")


def test_phase1_uow_nesting_joins_single_transaction(tmp_path: Path) -> None:
    eng = _engine(tmp_path)

    with eng.uow() as conn1:
        with eng.uow() as conn2:
            assert conn2 is conn1


def test_phase1_edge_endpoints_materialized_for_hypergraph(tmp_path: Path) -> None:
    eng = _engine(tmp_path)

    n1 = _make_node(summary="A", label="A")
    n2 = _make_node(summary="B", label="B")
    eng.add_node(n1)
    eng.add_node(n2)

    e = _make_edge(src=n1.safe_get_id(), tgt=n2.safe_get_id(), relation="links_to")
    # Hyperedge: multiple sources
    e.source_ids = [n1.safe_get_id(), n2.safe_get_id()]
    e.target_ids = [n2.safe_get_id()]
    eng.add_edge(e)

    got = eng.backend.edge_endpoints_get(
        where={"endpoint_id": n1.safe_get_id()},
        include=["metadatas", "documents"],
        limit=200,
    )

    ids = got.get("ids")
    metas: list[dict] = got.get("metadatas")

    assert len(ids) > 0, "Expected edge_endpoints rows for endpoint_id=n1"
    assert any(m.get("edge_id") == e.safe_get_id() for m in metas), (
        "Expected rows to reference created edge_id"
    )
    assert any(m.get("role") in ("src", "tgt") for m in metas), (
        "Expected role to be materialized"
    )


def test_phase1_no_direct_collection_method_calls_in_engine() -> None:
    """
    Phase-1 invariant: engine should not call chroma collections directly.
    It should always go through `self.backend.*`.
    """
    src = inspect.getsource(GraphKnowledgeEngine)

    direct_patterns = [
        r"\.node_collection\.(add|get|query|update|upsert|delete)\(",
        r"\.edge_collection\.(add|get|query|update|upsert|delete)\(",
        r"\.edge_endpoints_collection\.(add|get|query|update|upsert|delete)\(",
        r"\.node_docs_collection\.(add|get|query|update|upsert|delete)\(",
        r"\.node_refs_collection\.(add|get|query|update|upsert|delete)\(",
        r"\.edge_refs_collection\.(add|get|query|update|upsert|delete)\(",
        r"\.document_collection\.(add|get|query|update|upsert|delete)\(",
        r"\.domain_collection\.(add|get|query|update|upsert|delete)\(",
        r"\.nodes_index_collection\.(add|get|query|update|upsert|delete)\(",
    ]

    offenders: List[str] = [pat for pat in direct_patterns if re.search(pat, src)]
    assert not offenders, (
        "Direct collection calls still exist in GraphKnowledgeEngine. "
        "Route them through `self.backend.*`.\n"
        f"Matched patterns: {offenders}"
    )
