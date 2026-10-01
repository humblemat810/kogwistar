from __future__ import annotations

import pytest

pytestmark = [pytest.mark.core, pytest.mark.ci_full]

from kogwistar.engine_core.models import Node
from kogwistar.engine_core.vector_search import similarity_from_distance
from tests._kg_factories import kg_document, kg_grounding
from tests.conftest import _make_engine_pair


def test_vector_similarity_normalization_is_metric_explicit():
    assert similarity_from_distance(0.0, metric="cosine") == 1.0
    assert similarity_from_distance(0.25, metric="cosine") == 0.75
    assert similarity_from_distance(3.0, metric="l2") == 0.25
    assert similarity_from_distance(-0.8, metric="ip", distance_kind="negative_inner_product") == 0.8
    assert similarity_from_distance(0.2, metric="ip", distance_kind="distance") == 0.8


@pytest.mark.ci
def test_search_nodes_as_of_scored_exposes_normalized_similarity_and_threshold(
    tmp_path,
):
    eng, _ = _make_engine_pair(
        backend_kind="fake",
        tmp_path=tmp_path,
        sa_engine=None,
        pg_schema=None,
        dim=3,
    )
    try:
        doc = kg_document(
            doc_id="doc::scored-search",
            content="A scored vector search fixture.",
            source="test_search_nodes_as_of_scored_exposes_normalized_similarity_and_threshold",
        )
        eng.write.add_document(doc)
        node = _mk_claim_node(
            node_id="N_SCORED",
            label="Scored vector fixture",
            summary="A scored vector search fixture.",
            doc_id=doc.id,
            effective_from="2020-01-01T00:00:00+00:00",
        )
        eng.write.add_node(node)

        hits = eng.search_nodes_as_of_scored(
            query="scored vector fixture",
            as_of_ts="2021-01-01T00:00:00+00:00",
            n_results=5,
            similarity_threshold=0.99,
        )
        assert [hit.node.id for hit in hits] == ["N_SCORED"]
        assert hits[0].metric == "cosine"
        assert hits[0].distance_kind == "distance"
        assert hits[0].raw_distance == 0.0
        assert hits[0].similarity == 1.0

        assert eng.search_nodes_as_of_scored(
            query="scored vector fixture",
            as_of_ts="2021-01-01T00:00:00+00:00",
            n_results=5,
            similarity_threshold=1.01,
        ) == []
    finally:
        eng.close()


def _mk_claim_node(
    *, node_id: str, label: str, summary: str, doc_id: str, effective_from: str
) -> Node:
    return Node(
        id=node_id,
        label=label,
        type="entity",
        summary=summary,
        mentions=[kg_grounding(doc_id, excerpt=label)],
        properties={},
        metadata={"effective_from": effective_from},
        domain_id=None,
        canonical_entity_id=None,
        level_from_root=0,
        embedding=None,
    )


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
        pytest.param("pg", marks=pytest.mark.ci_full),
    ],
)
def test_search_nodes_as_of_redirect_cutoff(
    request, backend_kind, tmp_path
):
    sa_engine = pg_schema = None
    if backend_kind == "pg":
        sa_engine = request.getfixturevalue("sa_engine")
        pg_schema = request.getfixturevalue("pg_schema")
    eng, _ = _make_engine_pair(
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        dim=3,
        use_fake=True,
    )

    doc = kg_document(
        doc_id=f"doc::{backend_kind}::sugar",
        content="Historical nutrition narrative example for audit semantics.",
        source="test_search_nodes_as_of_redirect_cutoff",
    )
    eng.write.add_document(doc)

    old_node = _mk_claim_node(
        node_id="N_SUGAR_OLD",
        label="Sugar-Fat Claim (Old)",
        summary="Historic framing overemphasized fat and underemphasized added sugar.",
        doc_id=doc.id,
        effective_from="1967-01-01T00:00:00+00:00",
    )
    new_node = _mk_claim_node(
        node_id="N_SUGAR_NEW",
        label="Sugar-Fat Claim (Revised)",
        summary="Revised framing emphasizes added sugar risk and balanced dietary context.",
        doc_id=doc.id,
        effective_from="2016-01-01T00:00:00+00:00",
    )
    eng.write.add_node(old_node)
    eng.write.add_node(new_node)
    assert eng.redirect_node(
        "N_SUGAR_OLD",
        "N_SUGAR_NEW",
        deleted_at="2016-01-01T00:00:00+00:00",
        reason="historical_revision",
    )

    hits_then = eng.search_nodes_as_of(
        query="sugar fat claim",
        as_of_ts="2010-01-01T00:00:00+00:00",
        n_results=100,
    )
    ids_then = {n.id for n in hits_then}
    assert "N_SUGAR_OLD" in ids_then
    assert "N_SUGAR_NEW" not in ids_then

    hits_now = eng.search_nodes_as_of(
        query="sugar fat claim",
        as_of_ts="2020-01-01T00:00:00+00:00",
        n_results=100,
    )
    ids_now = {n.id for n in hits_now}
    assert "N_SUGAR_NEW" in ids_now
    assert "N_SUGAR_OLD" not in ids_now


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
        pytest.param("pg", marks=pytest.mark.ci_full),
    ],
)
def test_search_nodes_as_of_tombstone_and_future_effective(
    request, backend_kind, tmp_path
):
    sa_engine = pg_schema = None
    if backend_kind == "pg":
        sa_engine = request.getfixturevalue("sa_engine")
        pg_schema = request.getfixturevalue("pg_schema")
    eng, _ = _make_engine_pair(
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        dim=3,
        use_fake=True,
    )
    doc = kg_document(
        doc_id=f"doc::{backend_kind}::eggs",
        content="Historical egg and cholesterol narrative example for audit semantics.",
        source="test_search_nodes_as_of_tombstone_and_future_effective",
    )
    eng.write.add_document(doc)

    old_egg = _mk_claim_node(
        node_id="N_EGG_OLD",
        label="Egg Claim (Old)",
        summary="Historic framing strongly discouraged eggs for most diets.",
        doc_id=doc.id,
        effective_from="1970-01-01T00:00:00+00:00",
    )
    new_egg = _mk_claim_node(
        node_id="N_EGG_NEW",
        label="Egg Claim (Revised)",
        summary="Revised framing allows eggs in broader dietary context.",
        doc_id=doc.id,
        effective_from="2015-01-01T00:00:00+00:00",
    )
    dead = _mk_claim_node(
        node_id="N_DEAD",
        label="Deprecated Claim",
        summary="A deprecated claim with no replacement target.",
        doc_id=doc.id,
        effective_from="1990-01-01T00:00:00+00:00",
    )
    eng.write.add_node(old_egg)
    eng.write.add_node(new_egg)
    eng.write.add_node(dead)

    assert eng.redirect_node(
        "N_EGG_OLD",
        "N_EGG_NEW",
        deleted_at="2015-01-01T00:00:00+00:00",
        reason="historical_revision",
    )
    assert eng.tombstone_node(
        "N_DEAD",
        deleted_at="2010-01-01T00:00:00+00:00",
        reason="retired_without_replacement",
    )

    hits_pre = eng.search_nodes_as_of(
        query="egg cholesterol claim",
        as_of_ts="2005-01-01T00:00:00+00:00",
        n_results=100,
    )
    ids_pre = {n.id for n in hits_pre}
    assert "N_EGG_OLD" in ids_pre
    assert "N_EGG_NEW" not in ids_pre

    hits_after = eng.search_nodes_as_of(
        query="deprecated claim",
        as_of_ts="2012-01-01T00:00:00+00:00",
        n_results=100,
    )
    ids_after = {n.id for n in hits_after}
    assert "N_DEAD" not in ids_after


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
        pytest.param("pg", marks=pytest.mark.ci_full),
    ],
)
def test_search_nodes_as_of_redirect_cycle_and_invalid_target_safety(
    request, backend_kind, tmp_path
):
    sa_engine = pg_schema = None
    if backend_kind == "pg":
        sa_engine = request.getfixturevalue("sa_engine")
        pg_schema = request.getfixturevalue("pg_schema")
    eng, _ = _make_engine_pair(
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        dim=3,
        use_fake=True,
    )
    doc = kg_document(
        doc_id=f"doc::{backend_kind}::cycle",
        content="Lifecycle cycle safety case.",
        source="test_search_nodes_as_of_redirect_cycle_and_invalid_target_safety",
    )
    eng.write.add_document(doc)

    cycle_a = _mk_claim_node(
        node_id="N_CYCLE_A",
        label="Cycle A",
        summary="Cycle source A.",
        doc_id=doc.id,
        effective_from="1990-01-01T00:00:00+00:00",
    )
    cycle_b = _mk_claim_node(
        node_id="N_CYCLE_B",
        label="Cycle B",
        summary="Cycle source B.",
        doc_id=doc.id,
        effective_from="1991-01-01T00:00:00+00:00",
    )
    invalid_src = _mk_claim_node(
        node_id="N_INVALID_SRC",
        label="Invalid Redirect Source",
        summary="Redirects to a missing node id.",
        doc_id=doc.id,
        effective_from="1992-01-01T00:00:00+00:00",
    )
    eng.write.add_node(cycle_a)
    eng.write.add_node(cycle_b)
    eng.write.add_node(invalid_src)

    assert eng.redirect_node(
        "N_CYCLE_A", "N_CYCLE_B", deleted_at="2000-01-01T00:00:00+00:00"
    )
    assert eng.redirect_node(
        "N_CYCLE_B", "N_CYCLE_A", deleted_at="2001-01-01T00:00:00+00:00"
    )
    assert eng.redirect_node(
        "N_INVALID_SRC", "N_MISSING_TARGET", deleted_at="2000-01-01T00:00:00+00:00"
    )

    hits = eng.search_nodes_as_of(
        query="cycle redirect source",
        as_of_ts="2020-01-01T00:00:00+00:00",
        n_results=100,
    )
    ids = {n.id for n in hits}
    assert "N_CYCLE_A" not in ids
    assert "N_CYCLE_B" not in ids
    assert "N_INVALID_SRC" not in ids
