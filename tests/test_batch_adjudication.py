import os
import json
import shutil
import uuid
import pytest

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import (
    Node,
    LLMMergeAdjudication,
    AdjudicationVerdict,
    AdjudicationQuestionCode,
    QUESTION_KEY,
)

# --------- Fake LLM that handles many=True for adjudication ---------
class _FakeBatchAdjudicator:
    """Mocks .with_structured_output(LLMMergeAdjudication, many=True)."""

    def with_structured_output(self, schema, include_raw=False, many=False):
        # We only care that schema is LLMMergeAdjudication and many=True
        self._schema = schema
        self._many = many
        self._include_raw = include_raw
        return self

    def invoke(self, variables):
        # variables contains:
        #   - "mapping": mapping_table (we ignore for the fake)
        #   - "pairs": list of { left, right, question_code }
        pairs = variables["pairs"]

        results = []
        for item in pairs:
            left = item["left"]
            right = item["right"]
            # simple rule: if first token of labels match (case-insensitive), say same_entity=True
            ltok = (left.get("label") or "").split()[:1]
            rtok = (right.get("label") or "").split()[:1]
            same = bool(ltok and rtok and ltok[0].lower() == rtok[0].lower())

            ver = AdjudicationVerdict(
                same_entity=same,
                confidence=0.95 if same else 0.20,
                reason="Rule-based fake adjudicator: first-token match." if same else "Different first tokens.",
                canonical_entity_id=str(uuid.uuid4()) if same else None,
            )
            results.append(LLMMergeAdjudication(verdict=ver))
        return results


# --------- Fixtures ---------
@pytest.fixture(scope="function")
def tmp_chroma_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("chroma_db_batch")
    yield str(d)
    shutil.rmtree(d, ignore_errors=True)

@pytest.fixture(scope="function")
def engine(tmp_chroma_dir):
    eng = GraphKnowledgeEngine(persist_directory=tmp_chroma_dir)
    # Patch in our fake batch adjudicator
    eng.llm = _FakeBatchAdjudicator()
    return eng


# --------- Tests ---------
import json
import uuid
import pytest

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import (
    Node,
    Edge,
    Document,
    ReferenceSession,
    AdjudicationQuestionCode,
    QUESTION_KEY,
    AdjudicationVerdict,
    LLMMergeAdjudication,
)

@pytest.fixture(scope="function")
def engine():
    # Fresh engine per test (optionally point persist_directory to a tmp path)
    return GraphKnowledgeEngine()

def _ref_for(doc_id: str) -> ReferenceSession:
    # Minimal required span fields
    return ReferenceSession(
        collection_page_url="c",
        document_page_url=f"document/{doc_id}",
        insertion_method="pytest-manual",
        start_page=1, end_page=1, start_char=0, end_char=1, doc_id = doc_id,
    )

def test_batch_adjudication_and_commit(engine, monkeypatch):
    """ test batch_adjudication with the adjudication decision maker being fake LLM response, not cached
    This test only test the engines own internal processing pipeline before and after llm call.

    Args:
        engine (_type_): _description_
        monkeypatch (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # Create a document so nodes/edges can carry doc_id
    doc = Document(content="dummy", type="text")
    engine.add_document(doc)

    # Create three nodes; A & B should be considered the same by our fake rule, A & C not.
    ref = _ref_for(doc.id)
    a = Node(label="Chlorophyll a", type="entity", summary="Pigment in plants", references=[ref])
    b = Node(label="Chlorophyll b", type="entity", summary="Another chlorophyll pigment", references=[ref])
    c = Node(label="Hemoglobin",   type="entity", summary="Protein in red blood cells", references=[ref])

    # Insert them so commit_merge can update Chroma later
    engine.add_node(a, doc_id=doc.id)
    engine.add_node(b, doc_id=doc.id)
    engine.add_node(c, doc_id=doc.id)

    pairs = [(a, b), (a, c)]

    # --- Monkeypatch adjudication to avoid LLM and be deterministic ---
    def fake_batch_adjudicate_merges(pairs, question_code=AdjudicationQuestionCode.SAME_ENTITY):
        # (a,b) -> same_entity True; (a,c) -> False
        outs = []
        for left, right in pairs:
            if "Chlorophyll a" in left.label and "Chlorophyll b" in right.label:
                v = AdjudicationVerdict(same_entity=True, confidence=0.9, reason="similar pigments", canonical_entity_id=None)
            else:
                v = AdjudicationVerdict(same_entity=False, confidence=0.2, reason="different concepts", canonical_entity_id=None)
            outs.append(LLMMergeAdjudication(verdict=v))
        qkey = QUESTION_KEY[AdjudicationQuestionCode(question_code)]
        return outs, qkey

    monkeypatch.setattr(engine, "batch_adjudicate_merges", fake_batch_adjudicate_merges)

    # Run batch adjudication
    results, qkey = engine.batch_adjudicate_merges(
        pairs,
        question_code=AdjudicationQuestionCode.SAME_ENTITY,
    )

    # Check mapping key returned
    assert qkey == QUESTION_KEY[AdjudicationQuestionCode.SAME_ENTITY]

    # Should get one result per pair
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(hasattr(r, "verdict") for r in results)

    v1 = results[0].verdict
    v2 = results[1].verdict

    # By fake rule, (a,b) same_entity True; (a,c) False
    assert v1.same_entity is True and v1.confidence > 0.5
    assert v2.same_entity is False and v2.confidence <= 0.5

    # Commit the positive merge and verify canonical IDs + same_as edge
    canonical = engine.commit_merge(a, b, v1)
    assert canonical

    # Fetch updated nodes and verify canonical_entity_id persisted
    a_got = engine.node_collection.get(ids=[a.id])
    b_got = engine.node_collection.get(ids=[b.id])
    a_doc = json.loads(a_got["documents"][0])
    b_doc = json.loads(b_got["documents"][0])
    assert a_doc.get("canonical_entity_id") == canonical
    assert b_doc.get("canonical_entity_id") == canonical

    # Verify same_as edge exists (check metadata relation)
    edges = engine.edge_collection.get(include=["metadatas"])
    assert any((m or {}).get("relation") == "same_as" for m in edges.get("metadatas") or [])