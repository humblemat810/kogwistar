# tests/conftest.py
import os, shutil, uuid, json
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parents))
import pytest
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import (
    Edge, LLMGraphExtraction, LLMNode, LLMEdge,
    LLMMergeAdjudication, AdjudicationVerdict, Node, Span
)
from typing import Any, Dict, List, Optional
from langchain_core.runnables import Runnable
from graph_knowledge_engine.models import LLMMergeAdjudication, AdjudicationVerdict

class FakeStructuredRunnable(Runnable):
    """A minimal Runnable that returns a fixed structured result."""
    def __init__(self, parsed: Any, include_raw: bool = False):
        self._parsed = parsed
        self._include_raw = include_raw

    # sync single
    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        if self._include_raw:
            return {"raw": None, "parsed": self._parsed, "parsing_error": None}
        return self._parsed

    # async single
    async def ainvoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        return self.invoke(input, config=config, **kwargs)

    # sync batch
    def batch(self, inputs: List[Any], config: Optional[dict] = None, **kwargs: Any) -> List[Any]:
        return [self.invoke(i, config=config, **kwargs) for i in inputs]

    # async batch
    async def abatch(self, inputs: List[Any], config: Optional[dict] = None, **kwargs: Any) -> List[Any]:
        return [self.invoke(i, config=config, **kwargs) for i in inputs]


class FakeLLMForAdjudication:
    """
    Test double for your LLM. Mimics `.with_structured_output(...)` by returning
    a Runnable that yields a fixed LLMMergeAdjudication.
    """
    def __init__(self, verdict: AdjudicationVerdict, include_raw: bool = False):
        self._verdict = verdict
        self._include_raw = include_raw

    def with_structured_output(self, schema, include_raw: bool = False, many: bool = False):
        # Build a deterministic structured reply; ignore schema/many in this simple fake
        parsed = LLMMergeAdjudication(verdict=self._verdict)
        return FakeStructuredRunnable(parsed, include_raw=include_raw or self._include_raw)
class _FakeLLMForExtraction:
    """Mocks .with_structured_output(..., include_raw=True) → .invoke(...) for extraction."""
    def with_structured_output(self, schema, include_raw=False, many=False):
        self._include_raw = include_raw
        self._schema = schema
        self._many = many
        return self

    def invoke(self, variables):
        # Deterministic graph from any document
        parsed = LLMGraphExtraction(
            nodes=[
                LLMNode(label="Photosynthesis", type="entity", summary="Process converting light to chemical energy"),
                LLMNode(label="Chlorophyll", type="entity", summary="Molecule absorbing sunlight"),
            ],
            edges=[
                LLMEdge(
                    label="causes",
                    type="relationship",
                    summary="Chlorophyll absorption enables photosynthesis",
                    source_ids=["Chlorophyll"],  # will be mapped later in your pipeline
                    target_ids=["Photosynthesis"],
                    relation="enables"
                )
            ],
        )
        if self._include_raw:
            return {"raw": "fake_raw", "parsed": parsed, "parsing_error": None}
        return parsed

class _FakeLLMForAdjudication:
    """Mocks .with_structured_output(LLMMergeAdjudication) for adjudication."""
    def with_structured_output(self, schema, include_raw=False, many=False):
        self._schema = schema
        self._include_raw = include_raw
        self._many = many
        return self

    def invoke(self, variables):
        # Always say "same entity" with high confidence for test simplicity
        ver = AdjudicationVerdict(
            same_entity=True,
            confidence=0.97,
            reason="Labels and summaries strongly match.",
            canonical_entity_id=str(uuid.uuid4()),
        )
        return LLMMergeAdjudication(verdict=ver)

class _CompositeFakeLLM:
    """Single fake that behaves for both extraction and adjudication chains."""
    def with_structured_output(self, schema, include_raw=False, many=False):
        # route by schema class name
        if getattr(schema, "__name__", "") == "LLMGraphExtraction":
            self._impl = _FakeLLMForExtraction()
        else:
            self._impl = _FakeLLMForAdjudication()
        return self._impl

@pytest.fixture(scope="module")
def tmp_chroma_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("chroma_db")
    yield str(d)
    shutil.rmtree(d, ignore_errors=True)

@pytest.fixture(scope="function")
def engine(tmp_chroma_dir, monkeypatch):
    eng = GraphKnowledgeEngine(persist_directory=tmp_chroma_dir)
    # Patch the real LLM with a deterministic fake
    #eng.llm = _CompositeFakeLLM()
    return eng


@pytest.fixture()
def real_small_graph():
    e = GraphKnowledgeEngine(persist_directory = "small_graph")
    doc_id = "D1"
    # nodes
    def add_node(nid, label):
        n = Node(id=nid, label=label, type="entity", summary=label, mentions=[Span(
            collection_page_url=f"document_collection/{doc_id}", document_page_url=f"document/{doc_id}", doc_id=doc_id,
            insertion_method = 'pytest-conftext-fixture',
            start_page=1, end_page=1, start_char=0, end_char=1
        )], doc_id=doc_id)
        e.node_collection.add(ids=[nid], documents=[n.model_dump_json(field_mode = 'backend')], metadatas=[{"doc_id": doc_id, "label": n.label, "type": n.type}])
        # node_docs link
        ndid = f"{nid}::{doc_id}"
        row = {"id": ndid, "node_id": nid, "doc_id": doc_id}
        e.node_docs_collection.add(ids=[ndid], documents=[json.dumps(row)], metadatas=[row])
        return n

    A = add_node("A", "Smoking")
    B = add_node("B", "Lung Cancer")
    C = add_node("C", "Cough")

    # edge A -[causes]-> B
    e_id = "E1"
    edge = Edge(id=e_id, label="Smoking causes Lung Cancer", type="relationship", summary="causal", relation="causes",
                source_ids=["A"], target_ids=["B"], source_edge_ids=[], target_edge_ids=[],
                mentions=A.mentions, doc_id=doc_id)
    e.edge_collection.add(ids=[e_id], documents=[edge.model_dump_json(field_mode = 'backend')], metadatas=[{"doc_id": doc_id, "relation": "causes"}])
    # endpoints fan-out
    rows = [
        {"id": f"{e_id}::src::node::A", "edge_id": e_id, "endpoint_id": "A", "endpoint_type": "node", "role": "src", "relation": "causes", "doc_id": doc_id},
        {"id": f"{e_id}::tgt::node::B", "edge_id": e_id, "endpoint_id": "B", "endpoint_type": "node", "role": "tgt", "relation": "causes", "doc_id": doc_id},
    ]
    e.edge_endpoints_collection.add(ids=[r["id"] for r in rows], documents=[json.dumps(r) for r in rows], metadatas=rows)

    # final summary link S -> docnode:D1
    S = add_node("S", "Final Summary")
    e_id2 = "E2"
    e.edge_collection.add(ids=[e_id2], documents=[Edge(
        id=e_id2, label="summarizes_document", type="relationship", summary="S summarizes document", relation="summarizes_document",
        source_ids=["S"], target_ids=[f"docnode:{doc_id}"], source_edge_ids=[], target_edge_ids=[], mentions=S.mentions, doc_id=doc_id
    ).model_dump_json(field_mode = 'backend')], metadatas=[{"doc_id": doc_id, "relation": "summarizes_document"}])
    rows2 = [
        {"id": f"{e_id2}::src::node::S", "edge_id": e_id2, "endpoint_id": "S", "endpoint_type": "node", "role": "src", "relation": "summarizes_document", "doc_id": doc_id},
        {"id": f"{e_id2}::tgt::node::docnode:{doc_id}", "edge_id": e_id2, "endpoint_id": f"docnode:{doc_id}", "endpoint_type": "node", "role": "tgt", "relation": "summarizes_document", "doc_id": doc_id},
    ]
    e.edge_endpoints_collection.add(ids=[r["id"] for r in rows2], documents=[json.dumps(r) for r in rows2], metadatas=rows2)

    return e, doc_id
@pytest.fixture()
def small_test_docs_nodes_edge_adjudcate():
    # ---------- 0) Documents (exact strings you provided) ----------
    docs = {
        "DOC_A": (
            "In plant biology, many processes sustain life. "
            "Photosynthesis is a process used by plants to convert light energy into chemical energy. "
            "Chlorophyll is the molecule that absorbs sunlight. "
            "Within the leaves, specialized cells organize the chloroplasts. "
            "Plants perform photosynthesis in their leaves. "
            "Inside each chloroplast, stacks of thylakoid membranes called grana host the light-dependent reactions, "
            "where photons drive the formation of ATP and NADPH. "
            "These energy carriers then power the Calvin cycle in the stroma, fixing atmospheric CO2 into sugars using the enzyme Rubisco. "
            "Leaf anatomy supports this workflow: palisade mesophyll concentrates chloroplasts for maximal light capture, "
            "while spongy mesophyll and intercellular spaces facilitate gas diffusion. "
            "Stomata on the epidermis balance CO2 uptake with water conservation, opening and closing in response to light, humidity, and internal signals. "
            "Environmental factors—light intensity, temperature, CO2 concentration, and water availability—modulate overall photosynthetic rate. "
            "Different strategies evolved to mitigate photorespiration and arid stress: C3 plants fix carbon directly via Rubisco, "
            "C4 plants use a spatial separation with PEP carboxylase in bundle sheath cells, and CAM plants temporally separate uptake and fixation. "
            "Through these coordinated structures and pathways, photosynthesis underpins plant growth and, ultimately, most food webs."
        ),
        "DOC_B": (
            "Botanists distinguish several pigments in leaves. "
            "In most plants, chlorophyll a and chlorophyll b are present. "
            "The pigment chlorophyll gives leaves their green color. "
            "Both variants aid in harvesting light across different wavelengths. "
            "Chlorophyll a typically absorbs strongly in the blue-violet and red regions, while chlorophyll b extends coverage further into blue and a slightly different red band, "
            "broadening the effective spectrum plants can use. "
            "Accessory pigments such as carotenoids protect photosystems and capture light that chlorophylls miss, funneling energy into reaction centers by resonance transfer. "
            "The relative abundance of these pigments varies with species, developmental stage, and light environment—shade leaves often adjust pigment ratios to maximize efficiency. "
            "Seasonal changes alter pigment visibility: as chlorophyll degrades in autumn, carotenoids and, in some species, anthocyanins become more apparent, shifting leaf color. "
            "At the microscopic level, pigments are organized within protein complexes (photosystems I and II) embedded in thylakoid membranes, "
            "where the arrangement optimizes energy capture, minimizes photodamage, and supports electron transport."
        ),
        "DOC_C": (
            "Across the literature, the pigment chlorophyll is essential for photosynthesis, especially in terrestrial plants and algae. "
            "This role has been confirmed by numerous experiments. "
            "Classic action spectra align the rate of photosynthesis with wavelengths that chlorophyll absorbs, and historic demonstrations—such as Engelmann’s—linked oxygen evolution to those bands. "
            "Mutational studies that reduce chlorophyll content depress photosynthetic performance, while stressors that damage chlorophyll or its binding proteins impair growth. "
            "Comparative work shows parallel solutions in other phototrophs: bacteriochlorophylls in anoxygenic bacteria illustrate how pigment chemistry adapts to distinct ecological niches. "
            "Modern techniques, including chlorophyll fluorescence measurements and satellite indices like NDVI, exploit chlorophyll’s optical signatures to assess plant health and productivity at scales from leaves to landscapes. "
            "Biotechnological approaches aim to tune pigment composition and antenna size to minimize energy losses under high light, increase carbon gain, and enhance crop yields. "
            "Taken together, diverse lines of evidence establish chlorophyll as the core photochemical hub that initiates and regulates the flow of solar energy into biosynthetic pathways."
        ),
        "DOC_D": (
            "In human physiology, hemoglobin plays a central role. "
            "Hemoglobin absorbs oxygen in red blood cells and transports it to tissues. "
            "Binding is mediated by iron within the heme groups. "
            "Structurally, adult hemoglobin (HbA) is a tetramer whose subunits exhibit cooperative binding, creating a sigmoidal oxygen dissociation curve that suits loading in the lungs and unloading in tissues. "
            "Physiological modulators—including pH and CO2 (the Bohr effect), temperature, and 2,3-bisphosphoglycerate—shift hemoglobin’s affinity to match metabolic demand. "
            "Fetal hemoglobin (HbF) binds oxygen more tightly, facilitating transfer across the placenta, while variants such as sickle hemoglobin alter red-cell properties and clinical outcomes. "
            "Heme iron (Fe2+) reversibly binds O2 but can be blocked by carbon monoxide, which competes strongly at the same site; methemoglobin formation (Fe3+) reduces oxygen-carrying capacity. "
            "Beyond oxygen transport, hemoglobin contributes to CO2 carriage and acid–base buffering. "
            "When erythrocytes are recycled, heme is catabolized to bilirubin and iron is conserved—a tightly regulated process reflecting hemoglobin’s systemic importance."
        ),
    }

    # ---- helpers: find spans & build ReferenceSession-like dicts ----
    def _find_span(doc_id: str, excerpt: str) -> tuple[int, int]:
        text = docs[doc_id]
        idx = text.find(excerpt)
        if idx < 0:
            # try case-insensitive
            idx = text.lower().find(excerpt.lower())
        if idx < 0:
            raise AssertionError(f"Excerpt not found in {doc_id}: {excerpt!r}")
        return idx, idx + len(excerpt)

    def _ref(doc_id: str, excerpt: str, method: str = "llm"):
        s, e = _find_span(doc_id, excerpt)
        return {
            "doc_id": doc_id,
            "collection_page_url": "N/A",
            "document_page_url": "N/A",
            "start_page": 1,
            "end_page": 1,
            "start_char": s,
            "end_char": e,
            "excerpt": docs[doc_id][s:e],
            "verification": {"method": method, "is_verified": True, "score": 1.0, "notes": "fixture"},
            "insertion_method": "fixture_sample",
        }
    # ---------- Nodes ----------
    nodes = [
        {
            "id": "N_CHLORO",
            "label": "Chlorophyll",
            "type": "entity",
            "summary": "A green pigment found in plants.",
            "references": [
                _ref("DOC_A", "Chlorophyll is the molecule that absorbs sunlight."),
                _ref("DOC_C", "chlorophyll is essential for photosynthesis"),
            ],
        },
        {
            "id": "N_CHLORO_ALIAS",
            "label": "Chlorophyll (pigment)",
            "type": "entity",
            "summary": "Alias name for the same pigment.",
            "references": [
                _ref("DOC_C", "the pigment chlorophyll is essential"),
            ],
        },
        {
            "id": "N_CHLORO_A",
            "label": "Chlorophyll a",
            "type": "entity",
            "summary": "One variant of chlorophyll.",
            "references": [
                _ref("DOC_B", "chlorophyll a and chlorophyll b are present."),
            ],
        },
        {
            "id": "N_PHOTOSYN",
            "label": "Photosynthesis",
            "type": "entity",
            "summary": "Converts light energy to chemical energy.",
            "references": [
                _ref("DOC_A", "Photosynthesis is a process used by plants to convert light energy"),
                _ref("DOC_C", "rate of photosynthesis with wavelengths that chlorophyll absorbs"),
            ],
        },
        {
            "id": "N_LEAVES",
            "label": "Leaves",
            "type": "entity",
            "summary": "Plant organs that host photosynthesis.",
            "references": [
                _ref("DOC_A", "Plants perform photosynthesis in their leaves."),
                _ref("DOC_B", "The pigment chlorophyll gives leaves their green color."),
            ],
        },
        {
            "id": "N_SUN",
            "label": "Sunlight",
            "type": "entity",
            "summary": "Incoming solar radiation.",
            "references": [
                _ref("DOC_A", "light energy into chemical energy"),
            ],
        },
        {
            "id": "N_HEMO",
            "label": "Hemoglobin",
            "type": "entity",
            "summary": "Oxygen transport protein in blood.",
            "references": [
                _ref("DOC_D", "Hemoglobin absorbs oxygen in red blood cells and transports it to tissues."),
            ],
        },
        {
            "id": "N_OXY",
            "label": "Oxygen",
            "type": "entity",
            "summary": "O₂ molecule transported in blood.",
            "references": [
                _ref("DOC_D", "Hemoglobin absorbs oxygen in red blood cells and transports it to tissues."),
            ],
        },
        # Reified relation as a node (for cross-type positive)
        {
            "id": "N_PHOTO_REIFIED",
            "label": "Photosynthesis occurs in Leaves",
            "type": "entity",
            "summary": "Photosynthesis occurs in Leaves", # Reified relation concept.
            "properties": {"signature_text": "occurs_in(Photosynthesis, Leaves)"},
            "references": [
                _ref("DOC_A", "Plants perform photosynthesis in their leaves."),
            ],
        },
    ]

    # ---------- Edges ----------
    edges = [
        {
            "id": "E_CHLORO_ABSORB",
            "label": "Chlorophyll absorbs sunlight",
            "type": "relationship",
            "summary": "Chlorophyll absorbs sunlight.",
            "relation": "absorbs",
            "source_ids": ["N_CHLORO"],
            "target_ids": ["N_SUN"],
            "source_edge_ids": [],
            "target_edge_ids": [],
            "references": [
                _ref("DOC_A", "Chlorophyll is the molecule that absorbs sunlight."),
            ],
            "properties": {"signature_text": "absorbs(Chlorophyll, Sunlight)"},
        },
        {
            "id": "E_PHOTO_LEAVES",
            "label": "Photosynthesis occurs in leaves",
            "type": "relationship",
            "summary": "Photosynthesis happens in leaves.",
            "relation": "occurs_in",
            "source_ids": ["N_PHOTOSYN"],
            "target_ids": ["N_LEAVES"],
            "source_edge_ids": [],
            "target_edge_ids": [],
            "references": [
                _ref("DOC_A", "Plants perform photosynthesis in their leaves."),
            ],
            "properties": {"signature_text": "occurs_in(Photosynthesis, Leaves)"},
        },
        {
            "id": "E_PHOTO_LEAVES_DUP",
            "label": "Photosynthesis in leaves (duplicate)",
            "type": "relationship",
            "summary": "Duplicate of Photosynthesis→Leaves relation.",
            "relation": "occurs_in",
            "source_ids": ["N_PHOTOSYN"],
            "target_ids": ["N_LEAVES"],
            "source_edge_ids": [],
            "target_edge_ids": [],
            "references": [
                _ref("DOC_A", "Plants perform photosynthesis in their leaves."),
            ],
            "properties": {"signature_text": "occurs_in(Photosynthesis, Leaves)"},
        },
        {
            "id": "E_HEMO_TRANSPORT",
            "label": "Hemoglobin transports oxygen",
            "type": "relationship",
            "summary": "Hemoglobin transports oxygen in blood.",
            "relation": "transports",
            "source_ids": ["N_HEMO"],
            "target_ids": ["N_OXY"],
            "source_edge_ids": [],
            "target_edge_ids": [],
            "references": [
                _ref("DOC_D", "Hemoglobin absorbs oxygen in red blood cells and transports it to tissues."),
            ],
            "properties": {"signature_text": "transports(Hemoglobin, Oxygen)"},
        },
    ]

    # ---------- Ground-truth pairs for tests ----------
    adjudication_pairs = {
        # Node ↔ Node (2 positive, 2 negative typical)
        "positive": [
            ["N_CHLORO", "N_CHLORO_ALIAS"],   # alias
            ["N_PHOTO_REIFIED", "E_PHOTO_LEAVES"],  # cross-type positive (see below, also listed under cross)
        ],
        "negative": [
            ["N_CHLORO", "N_HEMO"],           # different domains
            ["N_CHLORO_A", "N_CHLORO_ALIAS"], # related but not same
        ],
        # Edge ↔ Edge
        "edge_positive": [
            ["E_PHOTO_LEAVES", "E_PHOTO_LEAVES_DUP"],  # duplicate semantics
        ],
        "edge_negative": [
            ["E_CHLORO_ABSORB", "E_HEMO_TRANSPORT"],   # different relation/entities
        ],
        # Cross-type (node ↔ edge)
        "cross_positive": [
            ["N_PHOTO_REIFIED", "E_PHOTO_LEAVES"],     # reified node vs relation
        ],
        "cross_negative": [
            ["N_HEMO", "E_CHLORO_ABSORB"],             # unrelated
        ],
    }

    sample_dataset = {"docs": docs, "nodes": nodes, "edges": edges, "adjudication_pairs": adjudication_pairs}

    return sample_dataset