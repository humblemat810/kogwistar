import os
from typing import Any, Dict, List, Tuple

import pytest
from joblib import Memory

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.models import (
    AdjudicationQuestionCode,
    Edge,
    LLMMergeAdjudication,
    Node,
    QUESTION_KEY,
)
from tests._kg_factories import kg_document, kg_grounding


_required_env = (
    "OPENAI_DEPLOYMENT_NAME_GPT4_1",
    "OPENAI_MODEL_NAME_GPT4_1",
    "OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1",
    "OPENAI_API_KEY_GPT4_1",
)
_skip = not all(os.getenv(k) for k in _required_env)


@pytest.fixture(scope="function")
def engine(tmp_path) -> GraphKnowledgeEngine:
    return GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"))


@pytest.mark.skipif(_skip, reason="Azure OpenAI env not set for real LLM adjudication")
def test_batch_adjudication_par_llm_cache_full(engine: GraphKnowledgeEngine, tmp_path):
    """
    Real-LLM, joblib-cached adjudication with:
      - 4 node-node pairs
      - 4 edge-edge pairs
      - 4 cross-type pairs
    """
    doc = kg_document(
        doc_id="doc::test_batch_adjudication_par_llm_cache_full",
        content="Employment, geography, alias examples",
        source="test_batch_adjudication_par_llm_cache_full",
    )
    engine.write.add_document(doc)
    ref = kg_grounding(doc.id)

    usa = Node(label="United States of America", type="entity", summary="Country", mentions=[ref])
    usa_alias = Node(label="USA", type="entity", summary="Country (alias)", mentions=[ref])
    nyc = Node(label="New York City", type="entity", summary="City in USA", mentions=[ref])
    nyc_alias = Node(label="NYC", type="entity", summary="City in USA (alias)", mentions=[ref])
    paris = Node(label="Paris", type="entity", summary="Capital of France", mentions=[ref])
    france = Node(label="France", type="entity", summary="Country in Europe", mentions=[ref])
    alice = Node(label="Alice", type="entity", summary="A person", mentions=[ref])
    acme = Node(label="ACME Corp", type="entity", summary="A company", mentions=[ref])
    london = Node(label="London", type="entity", summary="Capital of UK", mentions=[ref])
    uk = Node(label="United Kingdom", type="entity", summary="Country", mentions=[ref])

    for node in (usa, usa_alias, nyc, nyc_alias, paris, france, alice, acme, london, uk):
        engine.write.add_node(node, doc_id=doc.id)

    node_node_pairs: List[Tuple[Node, Node]] = [
        (usa, usa_alias),
        (nyc, nyc_alias),
        (paris, france),
        (alice, acme),
    ]

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
    engine.write.add_edge(e_geo1, doc_id=doc.id)

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
    engine.write.add_edge(e_geo2, doc_id=doc.id)

    e_emp1 = Edge(
        label="Employment: Alice at ACME (2019-2021)",
        type="relationship",
        summary="Alice employed by ACME during 2019-2021",
        relation="employed_at",
        source_ids=[alice.id],
        target_ids=[acme.id],
        source_edge_ids=[],
        target_edge_ids=[],
        properties={"start_year": 2019, "end_year": 2021, "signature_text": "Employment(Alice, ACME, 2019-2021)"},
        mentions=[ref],
    )
    engine.write.add_edge(e_emp1, doc_id=doc.id)

    e_emp2 = Edge(
        label="Alice employed at ACME (2019-2021) duplicate",
        type="relationship",
        summary="Alice employed by ACME during 2019-2021 (duplicate)",
        relation="employed_at",
        source_ids=[alice.id],
        target_ids=[acme.id],
        source_edge_ids=[],
        target_edge_ids=[],
        properties={"start_year": 2019, "end_year": 2021, "signature_text": "Employment(Alice, ACME, 2019-2021)"},
        mentions=[ref],
    )
    engine.write.add_edge(e_emp2, doc_id=doc.id)

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
    engine.write.add_edge(e_misc, doc_id=doc.id)

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
    engine.write.add_edge(e_geo_diff, doc_id=doc.id)

    edge_edge_pairs: List[Tuple[Edge, Edge]] = [
        (e_geo1, e_geo2),
        (e_emp1, e_emp2),
        (e_geo1, e_misc),
        (e_geo1, e_geo_diff),
    ]

    n_emp = Node(
        label="Employment of Alice at ACME (2019-2021)",
        type="entity",
        summary="Alice employed by ACME during 2019-2021",
        properties={"signature_text": "Employment(Alice, ACME, 2019-2021)"},
        mentions=[ref],
    )
    engine.write.add_node(n_emp, doc_id=doc.id)

    n_geo = Node(
        label="Paris located in France",
        type="entity",
        summary="Geographic containment reified",
        properties={"signature_text": "located_in(Paris, France)"},
        mentions=[ref],
    )
    engine.write.add_node(n_geo, doc_id=doc.id)

    n_misc = Node(
        label="Alice visited ACME (2022)",
        type="entity",
        summary="Visit event; non-employment",
        properties={"signature_text": "visited(Alice, ACME, 2022)"},
        mentions=[ref],
    )
    engine.write.add_node(n_misc, doc_id=doc.id)

    cross_pairs: List[Tuple[Any, Any]] = [
        (n_emp, e_emp1),
        (n_geo, e_geo1),
        (n_misc, e_emp1),
        (n_emp, e_misc),
    ]

    pairs_all: List[Tuple[Any, Any]] = node_node_pairs + edge_edge_pairs + cross_pairs

    def _wire_entry(obj):
        return {"kind": "relationship" if isinstance(obj, Edge) else "entity", "id": obj.id}

    wire_pairs: List[Dict[str, Any]] = [{"left": _wire_entry(left), "right": _wire_entry(right)} for left, right in pairs_all]

    cache_dir = tmp_path / ".cache" / "batch_adjudicate_par_cross_type_full"
    cache_dir.mkdir(parents=True, exist_ok=True)
    memory = Memory(location=str(cache_dir), verbose=0)

    @memory.cache
    def _run_adjudication_cached(persist_dir: str, cached_wire_pairs: List[Dict[str, Any]], qcode_int: int):
        eng = GraphKnowledgeEngine(persist_directory=persist_dir)

        def _get_node(node_id: str) -> Node:
            got = eng.backend.node_get(ids=[node_id], include=["documents"])
            return Node.model_validate_json(got["documents"][0])

        def _get_edge(edge_id: str) -> Edge:
            got = eng.backend.edge_get(ids=[edge_id], include=["documents"])
            return Edge.model_validate_json(got["documents"][0])

        pairs: List[Tuple[Any, Any]] = []
        for item in cached_wire_pairs:
            left = _get_edge(item["left"]["id"]) if item["left"]["kind"] == "relationship" else _get_node(item["left"]["id"])
            right = _get_edge(item["right"]["id"]) if item["right"]["kind"] == "relationship" else _get_node(item["right"]["id"])
            pairs.append((left, right))

        results, qkey_ = eng.batch_adjudicate_merges(
            pairs,
            question_code=AdjudicationQuestionCode(qcode_int),
        )
        dumped = [result.model_dump() if hasattr(result, "model_dump") else result for result in results]
        return dumped, qkey_

    dumped, qkey = _run_adjudication_cached(
        engine.persist_directory,
        wire_pairs,
        int(AdjudicationQuestionCode.SAME_ENTITY),
    )

    assert qkey == QUESTION_KEY[AdjudicationQuestionCode.SAME_ENTITY]
    results = [LLMMergeAdjudication.model_validate(result) for result in dumped]
    assert len(results) == len(pairs_all)

    i_nn = slice(0, 4)
    i_ee = slice(4, 8)
    i_xx = slice(8, 12)

    def pos_count(items: List[LLMMergeAdjudication]) -> int:
        return sum(1 for item in items if item.verdict.same_entity is True)

    assert pos_count(results[i_nn]) >= 1
    assert pos_count(results[i_ee]) >= 1
    assert pos_count(results[i_xx]) >= 1
