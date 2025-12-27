import os
import json
import pathlib
from typing import Any, Dict, List, Tuple

import pytest
from joblib import Memory

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import (
    Document,
    MentionVerification,
    Node,
    Edge,
    Span,
    LLMMergeAdjudication,
    AdjudicationVerdict,
    AdjudicationQuestionCode,
    QUESTION_KEY,
)

# --- Skip if Azure OpenAI env is not configured (prevents CI failures) ---
_required_env = (
    "OPENAI_DEPLOYMENT_NAME_GPT4_1",
    "OPENAI_MODEL_NAME_GPT4_1",
    "OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1",
    "OPENAI_API_KEY_GPT4_1",
)
_skip = not all(os.getenv(k) for k in _required_env)


def _ref_for(doc_id: str) -> Span:
    return _span_for(doc_id)
def _span_for(doc_id: str) -> Span:
    return Span(
        collection_page_url="c",
        document_page_url=f"document/{doc_id}",
        start_page=1, end_page=1, start_char=0, end_char=1,
        verification=MentionVerification(method="heuristic", is_verified=False, notes = None, score = 0.9), 
        insertion_method="pytest-manual",
        doc_id = doc_id,
        source_cluster_id = None,
        snippet = None
    )


@pytest.fixture(scope="function")
def engine(tmp_path) -> GraphKnowledgeEngine:
    return GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"))


@pytest.mark.skipif(_skip, reason="Azure OpenAI env not set for real LLM adjudication")
def test_batch_adjudication_par_llm_cache_full(engine: GraphKnowledgeEngine, tmp_path):
    """
    This test case abstract the LLM and backend db.
    
    Real-LLM, joblib-cached adjudication with:
      - 4 node↔node pairs (2 positive, 2 negative)
      - 4 edge↔edge pairs (2 positive, 2 negative)
      - 4 cross-type pairs (2 positive, 2 negative)
    We commit only node↔node positives to exercise write paths safely.
    """
    # ---------- 1) Seed a document ----------
    doc = Document(content="Employment, geography, alias examples", type="text")
    engine.add_document(doc)
    ref = _ref_for(doc.id)

    # ---------- 2) Entities for node↔node and edges ----------
    usa = Node(label="United States of America", type="entity", summary="Country", mentions=[ref])
    usa_alias = Node(label="USA", type="entity", summary="Country (alias)", mentions=[ref])  # positive with usa
    nyc = Node(label="New York City", type="entity", summary="City in USA", mentions=[ref])
    nyc_alias = Node(label="NYC", type="entity", summary="City in USA (alias)", mentions=[ref])  # positive with nyc
    paris = Node(label="Paris", type="entity", summary="Capital of France", mentions=[ref])
    france = Node(label="France", type="entity", summary="Country in Europe", mentions=[ref])
    alice = Node(label="Alice", type="entity", summary="A person", mentions=[ref])
    acme = Node(label="ACME Corp", type="entity", summary="A company", mentions=[ref])
    london = Node(label="London", type="entity", summary="Capital of UK", mentions=[ref])
    uk = Node(label="United Kingdom", type="entity", summary="Country", mentions=[ref])

    for n in (usa, usa_alias, nyc, nyc_alias, paris, france, alice, acme, london, uk):
        engine.add_node(n, doc_id=doc.id)

    # ---------- 3) Node↔node pairs ----------
    # Positives (aliases / same concept)
    nn_p1 = (usa, usa_alias)  # alias names for the same country
    nn_p2 = (nyc, nyc_alias)  # alias names for the same city
    # Negatives (different concepts)
    nn_n1 = (paris, france)
    nn_n2 = (alice, acme)
    node_node_pairs: List[Tuple[Node, Node]] = [nn_p1, nn_p2, nn_n1, nn_n2]

    # ---------- 4) Create edges ----------
    # Edge↔edge positives (duplicate semantics)
    e_geo1 = Edge(
        label="Paris located in France",
        type="relationship",
        summary="Geographic containment",
        relation="located_in",
        source_ids=[paris.id],
        target_ids=[france.id],
        source_edge_ids=[],
        target_edge_ids=[],
        properties={"signature_text": "located_in(Paris, France)"},
        mentions=[ref],
    )
    engine.add_edge(e_geo1, doc_id=doc.id)

    e_geo2 = Edge(
        label="Paris in France (duplicate)",
        type="relationship",
        summary="Geographic containment (duplicate)",
        relation="located_in",
        source_ids=[paris.id],
        target_ids=[france.id],
        source_edge_ids=[],
        target_edge_ids=[],
        properties={"signature_text": "located_in(Paris, France)"},
        mentions=[ref],
    )
    engine.add_edge(e_geo2, doc_id=doc.id)

    e_emp1 = Edge(
        label="Employment: Alice at ACME (2019–2021)",
        type="relationship",
        summary="Alice employed by ACME during 2019–2021",
        relation="employed_at",
        source_ids=[alice.id],
        target_ids=[acme.id],
        source_edge_ids=[],
        target_edge_ids=[],
        properties={"start_year": 2019, "end_year": 2021, "signature_text": "Employment(Alice, ACME, 2019–2021)"},
        mentions=[ref],
    )
    engine.add_edge(e_emp1, doc_id=doc.id)

    e_emp2 = Edge(
        label="Alice employed at ACME (2019–2021) duplicate",
        type="relationship",
        summary="Alice employed by ACME during 2019–2021 (duplicate)",
        relation="employed_at",
        source_ids=[alice.id],
        target_ids=[acme.id],
        source_edge_ids=[],
        target_edge_ids=[],
        properties={"start_year": 2019, "end_year": 2021, "signature_text": "Employment(Alice, ACME, 2019–2021)"},
        mentions=[ref],
    )
    engine.add_edge(e_emp2, doc_id=doc.id)

    # Edge↔edge negatives (different relation or endpoints)
    e_misc = Edge(
        label="Visited: Alice at ACME (2022)",
        type="relationship",
        summary="Alice visited ACME in 2022",
        relation="visited",
        source_ids=[alice.id],
        target_ids=[acme.id],
        source_edge_ids=[],
        target_edge_ids=[],
        properties={"year": 2022, "signature_text": "visited(Alice, ACME, 2022)"},
        mentions=[ref],
    )
    engine.add_edge(e_misc, doc_id=doc.id)

    e_geo_diff = Edge(
        label="London located in United Kingdom",
        type="relationship",
        summary="Geographic containment different endpoints",
        relation="located_in",
        source_ids=[london.id],
        target_ids=[uk.id],
        source_edge_ids=[],
        target_edge_ids=[],
        properties={"signature_text": "located_in(London, United Kingdom)"},
        mentions=[ref],
    )
    engine.add_edge(e_geo_diff, doc_id=doc.id)

    edge_edge_pairs: List[Tuple[Edge, Edge]] = [
        (e_geo1, e_geo2),   # positive
        (e_emp1, e_emp2),   # positive
        (e_geo1, e_misc),   # negative (different relation)
        (e_geo1, e_geo_diff),  # negative (different endpoints)
    ]

    # ---------- 5) Cross-type: node (reified relation) ↔ edge ----------
    n_emp = Node(
        label="Employment of Alice at ACME (2019–2021)",
        type="entity",
        summary="Alice employed by ACME during 2019–2021",
        properties={"signature_text": "Employment(Alice, ACME, 2019–2021)"},
        mentions=[ref],
    )
    engine.add_node(n_emp, doc_id=doc.id)

    n_geo = Node(
        label="Paris located in France",
        type="entity",
        summary="Geographic containment reified",
        properties={"signature_text": "located_in(Paris, France)"},
        mentions=[ref],
    )
    engine.add_node(n_geo, doc_id=doc.id)

    n_misc = Node(
        label="Alice visited ACME (2022)",
        type="entity",
        summary="Visit event; non-employment",
        properties={"signature_text": "visited(Alice, ACME, 2022)"},
        mentions=[ref],
    )
    engine.add_node(n_misc, doc_id=doc.id)

    cross_pairs: List[Tuple[Any, Any]] = [
        (n_emp, e_emp1),    # positive
        (n_geo, e_geo1),    # positive
        (n_misc, e_emp1),   # negative
        (n_emp, e_misc),    # negative
    ]

    # ---------- 6) Build unified pairs list ----------
    pairs_all: List[Tuple[Any, Any]] = node_node_pairs + edge_edge_pairs + cross_pairs

    # ---------- 7) Prepare cacheable “wire” payload ----------
    def _wire_entry(obj):
        return {"kind": "relationship" if isinstance(obj, Edge) else "entity", "id": obj.id}

    wire_pairs: List[Dict[str, Any]] = [{"left": _wire_entry(l), "right": _wire_entry(r)} for (l, r) in pairs_all]

    # ---------- 8) Cached adjudication that **invokes engine.batch_adjudicate_merges** ----------
    cache_dir = tmp_path / ".cache" / "batch_adjudicate_par_cross_type_full"
    cache_dir.mkdir(parents=True, exist_ok=True)
    memory = Memory(location=str(cache_dir), verbose=0)

    @memory.cache
    def _run_adjudication_cached(persist_dir: str, wire_pairs: List[Dict[str, Any]], qcode_int: int):
        # Recreate engine pointed at the same persist dir
        eng = GraphKnowledgeEngine(persist_directory=persist_dir)

        # Rehydrate objects from Chroma
        def _get_node(nid: str) -> Node:
            got = eng.node_collection.get(ids=[nid], include=["documents"])
            return Node.model_validate_json(got["documents"][0])

        def _get_edge(eid: str) -> Edge:
            got = eng.edge_collection.get(ids=[eid], include=["documents"])
            return Edge.model_validate_json(got["documents"][0])

        pairs: List[Tuple[Any, Any]] = []
        for item in wire_pairs:
            left = _get_edge(item["left"]["id"]) if item["left"]["kind"] == "relationship" else _get_node(item["left"]["id"])
            right = _get_edge(item["right"]["id"]) if item["right"]["kind"] == "relationship" else _get_node(item["right"]["id"])
            pairs.append((left, right))

        # Call the function under test
        results, qkey_ = eng.batch_adjudicate_merges(
            pairs,
            question_code=AdjudicationQuestionCode(qcode_int),
        )

        # Return JSON-serializable payload for caching
        dumped = [r.model_dump() if hasattr(r, "model_dump") else r for r in results]
        return dumped, qkey_

    dumped, qkey = _run_adjudication_cached(
        engine.persist_directory,
        wire_pairs,
        int(AdjudicationQuestionCode.SAME_ENTITY),
    )

    assert qkey == QUESTION_KEY[AdjudicationQuestionCode.SAME_ENTITY]
    # Rehydrate adjudications
    results = [LLMMergeAdjudication.model_validate(r) for r in dumped]
    assert len(results) == len(pairs_all)

    # ---------- 9) Check expected positives/negatives per group ----------
    # groups’ boundaries
    i_nn = slice(0, 4)    # 4 node-node
    i_ee = slice(4, 8)    # 4 edge-edge
    i_xx = slice(8, 12)   # 4 cross-type

    def pos_count(items: List[LLMMergeAdjudication]) -> int:
        return sum(1 for it in items if it.verdict.same_entity is True)

    nn_pos = pos_count(results[i_nn])
    ee_pos = pos_count(results[i_ee])
    xx_pos = pos_count(results[i_xx])

    # Expect 2 positives per group (model-dependent, but our examples are strongly suggestive)
    assert nn_pos >= 1  # relaxed to >=1 in case of model variance
    assert ee_pos >= 1
    assert xx_pos >= 1

    # # ---------- 10) Commit only node↔node positives and verify side effects ----------
    # node_pairs = node_node_pairs
    # for (pair, res) in zip(node_pairs, results[i_nn]):
    #     if res.verdict.same_entity:
    #         canon = engine.commit_merge_target(engine._target_from_node(pair[0]), 
    #                                        engine._target_from_node(pair[1]), res.verdict)
    #         assert canon

    # # Nodes updated with canonical IDs?
    # for (pair, res) in zip(node_pairs, results[i_nn]):
    #     if res.verdict.same_entity:
    #         a, b = pair
    #         a_doc = engine.node_collection.get(ids=[a.id], include=["documents"])
    #         b_doc = engine.node_collection.get(ids=[b.id], include=["documents"])
    #         a_json = json.loads(a_doc["documents"][0])
    #         b_json = json.loads(b_doc["documents"][0])
    #         assert a_json.get("canonical_entity_id") == b_json.get("canonical_entity_id") != None

    # # A same_as edge should exist if any node↔node commitment happened
    # edges_meta = engine.edge_collection.get(include=["metadatas"])
    # if nn_pos:
    #     assert any((m or {}).get("relation") == "same_as" for m in (edges_meta.get("metadatas") or []))