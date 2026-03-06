# tests/conftest.py
import shutil, uuid, json
import os

import sitecustomize
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
import sqlalchemy as sa
from testcontainers.postgres import PostgresContainer

import os
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parents))
import pytest
from graph_knowledge_engine.conversation.models import ConversationEdge, ConversationNode
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.models import (
    Edge, LLMGraphExtraction, LLMNode, LLMEdge,
    LLMMergeAdjudication, AdjudicationVerdict, Node, Span, Grounding,
    MentionVerification
)
from graph_knowledge_engine.engine_core.postgres_backend import PgVectorBackend
from typing import Any, List, Optional, Sequence, Iterator
from langchain_core.runnables import Runnable
from graph_knowledge_engine.engine_core.models import LLMMergeAdjudication, AdjudicationVerdict


_TEST_NS = uuid.UUID("00000000-0000-0000-0000-000000000000")
@pytest.fixture(scope="session")
def stable_uuid(*parts: object) -> str:
    return str(uuid.uuid5(_TEST_NS, "|".join(str(p) for p in parts)))
from pathlib import Path

import logging
logging.captureWarnings(True)
from pathlib import Path
from graph_knowledge_engine.utils.log import EngineLogManager, EngineLogConfig

def pytest_addoption(parser):
    parser.addoption(
        "--run-manual",
        action="store_true",
        default=False,
        help="Run tests marked as manual.",
    )


def _normalize_pytest_arg(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


def _is_specific_test_function_target(arg: str) -> bool:
    """
    True only for explicit function/method nodeids, e.g.:
      - tests/x.py::test_case
      - tests/x.py::TestClass::test_case
      - tests/x.py::test_case[param]
    False for file/class/folder targets.
    """
    if "::" not in arg:
        return False
    leaf = arg.rsplit("::", 1)[-1]
    leaf_base = leaf.split("[", 1)[0]
    return leaf_base.startswith("test_")


def _is_manual_test_explicitly_selected(config: pytest.Config, item: pytest.Item) -> bool:
    nodeid = _normalize_pytest_arg(item.nodeid)
    cli_args = getattr(config.invocation_params, "args", ()) or ()

    for raw_arg in cli_args:
        if not isinstance(raw_arg, str):
            continue
        arg = _normalize_pytest_arg(raw_arg)
        if not arg or arg.startswith("-"):
            continue

        # Explicit function/method nodeid selection only.
        if _is_specific_test_function_target(arg):
            if (nodeid == arg or nodeid.startswith(arg + "[") or nodeid.rsplit('/',1)[-1] in arg 
                or pathlib.Path(arg).parts[-1] == nodeid.rsplit('/',1)[-1]):
                return True

    return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-manual"):
        return

    skip_manual = pytest.mark.skip(
        reason="manual test skipped by default; run with --run-manual or target a specific test function."
    )
    for item in items:
        if "manual" in item.keywords and not _is_manual_test_explicitly_selected(config, item):
            item.add_marker(skip_manual)


def pytest_configure(config):
    EngineLogManager.configure(
        # EngineLogConfig(
            base_dir=Path(".logs/test"),
            app_name="gke_test",
            level=logging.DEBUG,
            enable_files=True,        # <-- ENABLE FILE LOGGING
            enable_sqlite=False,
            # mode="prod",              # <-- NOT pytest
            enable_jsonl=True
        # )
    )
from chromadb.utils.embedding_functions import EmbeddingFunction
from chromadb.api.types import Embeddings
class FakeEmbeddingFunction(EmbeddingFunction):
    @staticmethod
    def name() -> str:
        return "default"

    def __init__(self, dim: int = 384):
        self._dim = dim

    def __call__(self, documents_or_texts: Sequence[str]) -> Embeddings:
        return [[0.01] * self._dim for _ in documents_or_texts]
    
def _make_engine_pair(*, backend_kind: str, tmp_path, sa_engine, pg_schema, dim: int = 3, use_fake = False):
    """
    Build (kg_engine, conv_engine) for either chroma or pgvector.
    """
    # ef = _fake_ef_dim(dim)
    _ef = None
    if use_fake:
        _ef = FakeEmbeddingFunction(dim=dim)
    if backend_kind == "chroma":
        kg_engine = GraphKnowledgeEngine(persist_directory=str(tmp_path / "kg"), kg_graph_type="knowledge", embedding_function=FakeEmbeddingFunction(dim=dim)
                                         )
        conv_engine = GraphKnowledgeEngine(persist_directory=str(tmp_path / "conv"), kg_graph_type="conversation", embedding_function=FakeEmbeddingFunction(dim=dim)
                                           )
        return kg_engine, conv_engine

    if backend_kind == "pg":
        if sa_engine is None or pg_schema is None:
            pytest.skip("pg backend requested but sa_engine/pg_schema fixtures not available")
        kg_schema = f"{pg_schema}_kg"
        conv_schema = f"{pg_schema}_conv"
        kg_backend = PgVectorBackend(engine=sa_engine, embedding_dim=dim, schema=kg_schema)
        conv_backend = PgVectorBackend(engine=sa_engine, embedding_dim=dim, schema=conv_schema)
        kg_engine = GraphKnowledgeEngine(persist_directory=str(tmp_path / "kg_meta"), 
                                         kg_graph_type="knowledge", embedding_function=FakeEmbeddingFunction(dim=dim), backend=kg_backend)
        conv_engine = GraphKnowledgeEngine(persist_directory=str(tmp_path / "conv_meta"),
                                           kg_graph_type="conversation", embedding_function=FakeEmbeddingFunction(dim=dim), backend=conv_backend)
        return kg_engine, conv_engine

    raise ValueError(f"unknown backend_kind: {backend_kind!r}")
def _mk_span_from_excerpt(*, doc_id: str, content: str, excerpt: str, insertion_method: str, page_number: int = 1):
    idx = content.index(excerpt)  # will raise if excerpt not present -> good early failure
    start = idx
    end = idx + len(excerpt)
    return {
        "collection_page_url": "N/A",
        "document_page_url": "N/A",
        "doc_id": doc_id,
        "insertion_method": insertion_method,
        "page_number": page_number,
        "start_char": start,
        "end_char": end,
        "excerpt": excerpt,
        "context_before": content[max(0, start - 40):start],
        "context_after": content[end:end + 40],
        # optional fields
        "chunk_id": None,
        "source_cluster_id": None,
        "verification": {
            "method": "heuristic",
            "is_verified": False,
            "score": None,
            "notes": "no explicit verification from LLM",
        },
    }
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


@pytest.fixture(scope="session")
def pg_container() -> Iterator[PostgresContainer]:
    """
    Spin up a disposable Postgres for the whole test session.

    Requirements:
      - Docker daemon running (Docker Desktop on Windows/macOS)
      - Python deps: testcontainers[postgresql], psycopg[binary], sqlalchemy
    """
    image = os.getenv("GKE_TEST_PG_IMAGE", "postgres:16")
    os.environ['TESTCONTAINERS_RYUK_DISABLED'] = "true"
    with PostgresContainer(image) as pg:
        yield pg


@pytest.fixture(scope="session")
def pg_dsn(pg_container: PostgresContainer) -> str:
    """
    SQLAlchemy DSN for the running test container.
    """
    url = pg_container.get_connection_url()
    # Normalize to psycopg (optional). Comment out if you rely on psycopg2.
    url = url.replace("postgresql://", "postgresql+psycopg://")
    return url


@pytest.fixture(scope="session")
def sa_engine(pg_dsn: str) -> sa.Engine:
    return sa.create_engine(pg_dsn, future=True)


@pytest.fixture()
def pg_schema(sa_engine: sa.Engine) -> Iterator[str]:
    """
    Unique schema per test, dropped afterwards.

    Tests should pass this schema into PgVectorBackend(schema=...).
    """
    schema = f"gke_test_{uuid.uuid4().hex}"
    with sa_engine.begin() as conn:
        conn.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
    try:
        yield schema
    finally:
        with sa_engine.begin() as conn:
            conn.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))

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
    eng = GraphKnowledgeEngine(persist_directory=os.path.join(tmp_chroma_dir, "kg"), 
                               embedding_cache_path=os.path.join(os.getcwd(), '.embedding_cache'))
    # Patch the real LLM with a deterministic fake
    #eng.llm = _CompositeFakeLLM()
    return eng

@pytest.fixture(scope="module")
def tmp_conv_chroma_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("chroma_db")
    yield str(d)
    shutil.rmtree(d, ignore_errors=True)
@pytest.fixture(scope="function")
def conversation_engine(tmp_conv_chroma_dir, monkeypatch):
    eng = GraphKnowledgeEngine(persist_directory=os.path.join(tmp_conv_chroma_dir, "conversation"), kg_graph_type = "conversation")
    # Patch the real LLM with a deterministic fake
    #eng.llm = _CompositeFakeLLM()
    return eng
@pytest.fixture(scope="function")
def workflow_engine(tmp_conv_chroma_dir, monkeypatch):
    eng = GraphKnowledgeEngine(persist_directory=os.path.join(tmp_conv_chroma_dir, "workflow"), kg_graph_type = "workflow")
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
    """
    sample llm extracted tripples emulating data from llm graph extraction with insertion_method = "llm_graph_extraction"
    """
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
            raise AssertionError(
                f"Exact excerpt not found in {doc_id}: {excerpt!r}"
            )
        return idx, idx + len(excerpt)

    def _ref(doc_id: str, excerpt: str, method: str = "llm"):
        s, e = _find_span(doc_id, excerpt)
        # Span-like dict compatible with graph_knowledge_engine.models.Span
        # (kept as a dict because this fixture emulates a JSON payload).
        return {
            "doc_id": doc_id,
            "collection_page_url": "N/A",
            "document_page_url": "N/A",
            "page_number": 1,
            "start_char": s,
            "end_char": e,
            "excerpt": docs[doc_id][s:e],
            "context_before": docs[doc_id][max(0, s - 40) : s],
            "context_after": docs[doc_id][e : min(len(docs[doc_id]), e + 40)],
            # Optional verification payload
            "verification": {"method": method, "is_verified": True, "score": 1.0, "notes": "fixture"},
            "insertion_method": "llm_graph_extraction",
        }
    # ---------- Nodes ----------
    nodes = [
        {
            "id": "N_CHLORO",
            "label": "Chlorophyll",
            "type": "entity",
            "summary": "A green pigment found in plants.",
            "mentions": [{"spans": [
                _ref("DOC_A", "Chlorophyll is the molecule that absorbs sunlight."),
                _ref("DOC_C", "chlorophyll is essential for photosynthesis"),
            ]}],
        },
        {
            "id": "N_CHLORO_ALIAS",
            "label": "Chlorophyll (pigment)",
            "type": "entity",
            "summary": "Alias name for the same pigment.",
            "mentions": [{"spans": [
                _ref("DOC_C", "the pigment chlorophyll is essential"),
            ]}],
        },
        {
            "id": "N_CHLORO_A",
            "label": "Chlorophyll a",
            "type": "entity",
            "summary": "One variant of chlorophyll.",
            "mentions": [{"spans": [
                _ref("DOC_B", "chlorophyll a and chlorophyll b are present."),
            ]}],
        },
        {
            "id": "N_PHOTOSYN",
            "label": "Photosynthesis",
            "type": "entity",
            "summary": "Converts light energy to chemical energy.",
            "mentions": [{"spans": [
                _ref("DOC_A", "Photosynthesis is a process used by plants to convert light energy"),
                _ref("DOC_C", "rate of photosynthesis with wavelengths that chlorophyll absorbs"),
            ]}],
        },
        {
            "id": "N_LEAVES",
            "label": "Leaves",
            "type": "entity",
            "summary": "Plant organs that host photosynthesis.",
            "mentions": [{"spans": [
                _ref("DOC_A", "Plants perform photosynthesis in their leaves."),
                _ref("DOC_B", "The pigment chlorophyll gives leaves their green color."),
            ]}],
        },
        {
            "id": "N_SUN",
            "label": "Sunlight",
            "type": "entity",
            "summary": "Incoming solar radiation.",
            "mentions": [{"spans": [
                _ref("DOC_A", "light energy into chemical energy"),
            ]}],
        },
        {
            "id": "N_HEMO",
            "label": "Hemoglobin",
            "type": "entity",
            "summary": "Oxygen transport protein in blood.",
            "mentions": [{"spans": [
                _ref("DOC_D", "Hemoglobin absorbs oxygen in red blood cells and transports it to tissues."),
            ]}],
        },
        {
            "id": "N_OXY",
            "label": "Oxygen",
            "type": "entity",
            "summary": "O₂ molecule transported in blood.",
            "mentions": [{"spans": [
                _ref("DOC_D", "Hemoglobin absorbs oxygen in red blood cells and transports it to tissues."),
            ]}],
        },
        # Reified relation as a node (for cross-type positive)
        {
            "id": "N_PHOTO_REIFIED",
            "label": "Photosynthesis occurs in Leaves",
            "type": "entity",
            "summary": "Photosynthesis occurs in Leaves", # Reified relation concept.
            "properties": {"signature_text": "occurs_in(Photosynthesis, Leaves)"},
            "mentions": [{"spans": [
                _ref("DOC_A", "Plants perform photosynthesis in their leaves."),
            ]}],
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
            "mentions": [{"spans": [
                _ref("DOC_A", "Chlorophyll is the molecule that absorbs sunlight."),
            ]}],
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
            "mentions": [{"spans": [
                _ref("DOC_A", "Plants perform photosynthesis in their leaves."),
            ]}],
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
            "mentions": [{"spans": [
                _ref("DOC_A", "Plants perform photosynthesis in their leaves."),
            ]}],
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
            "mentions": [{"spans": [
                _ref("DOC_D", "Hemoglobin absorbs oxygen in red blood cells and transports it to tissues."),
            ]}],
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

    # Pre-flight validate the JSON-ish payload against the current models.
    # This keeps the fixture from silently drifting out-of-date with models.py.
    try:
        _ = LLMGraphExtraction.FromLLMSlice({"nodes": nodes, "edges": edges}, insertion_method="fixture_sample")
    except Exception as e:
        raise AssertionError(f"Fixture small_test_docs_nodes_edge_adjudcate is not model-compatible: {e}")

    sample_dataset = {"docs": docs, "nodes": nodes, "edges": edges, "adjudication_pairs": adjudication_pairs}

    return sample_dataset

def mk_verification(
    *,
    method: str = "human",
    is_verified: bool = True,
    score: float = 1.0,
    notes: str = "seed",
) -> MentionVerification:
    return MentionVerification(
        method=method,
        is_verified=is_verified,
        score=score,
        notes=notes,
    )


def mk_span(
    *,
    doc_id: str,
    full_text: str,
    start_char: int = 0,
    end_char: Optional[int] = None,
    page_number: int = 1,
    insertion_method: str = "seed",
    collection_page_url: str = "url",
    document_page_url: str = "url",
    context_before: str = "",
    context_after: str = "",
    chunk_id: Optional[str] = None,
    source_cluster_id: Optional[str] = None,
    verification: Optional[MentionVerification] = None,
) -> Span:
    if end_char is None:
        end_char = len(full_text)
    excerpt = full_text[start_char:end_char]
    return Span(
        collection_page_url=collection_page_url,
        document_page_url=document_page_url,
        doc_id=doc_id,
        insertion_method=insertion_method,
        page_number=page_number,
        start_char=start_char,
        end_char=end_char,
        excerpt=excerpt,
        context_before=context_before,
        context_after=context_after,
        chunk_id=chunk_id,
        source_cluster_id=source_cluster_id,
        verification=verification or mk_verification(notes=f"seed:{insertion_method}"),
    )


def mk_grounding(*spans: Span) -> Grounding:
    return Grounding(spans=list(spans))


def add_node_raw(
    engine: GraphKnowledgeEngine,
    node: Node | ConversationNode,
    *,
    embedding_dim: int = 384,
    embedding: Optional[Sequence[float]] = None,
) -> None:
    """
    Adds a node by directly writing to the underlying Chroma collection,
    using the engine's own serialization helper.

    This mirrors what your existing test already does:
      doc, meta = engine._node_doc_and_meta(n)
      engine.node_collection.add(...)
    """
    doc, meta = engine._node_doc_and_meta(node)
    if embedding is None and getattr(node, "embedding", None) is None:
        embedding = [0.1] * embedding_dim
    if getattr(node, "embedding", None) is None:
        node.embedding = embedding  # type: ignore

    engine.node_collection.add(
        ids=[node.id],
        documents=[doc],
        embeddings=[list(node.embedding)],  # type: ignore[arg-type]
        metadatas=[meta],
    )


def add_edge_raw(
    engine: Any,
    edge: Edge | ConversationEdge,
    *,
    embedding_dim: int = 384,
    embedding: Optional[Sequence[float]] = None,
) -> None:
    """
    Same idea for edges.
    """
    doc, meta = engine._edge_doc_and_meta(edge)
    if embedding is None and getattr(edge, "embedding", None) is None:
        embedding = [0.1] * embedding_dim
    if getattr(edge, "embedding", None) is None:
        edge.embedding = embedding  # type: ignore

    engine.edge_collection.add(
        ids=[edge.id],
        documents=[doc],
        embeddings=[list(edge.embedding)],  # type: ignore[arg-type]
        metadatas=[meta],
    )


# ---------------------------------------------------------------------
# Seed KG graph (real Node/Edge objects with proper mentions/spans)
# ---------------------------------------------------------------------

def seed_kg_graph(*, kg_engine: GraphKnowledgeEngine, kg_doc_id: str = "D_KG_001") -> dict[str, Any]:
    """
    Seeds a minimal KG doc with:
      - N1, N2 nodes
      - E1 edge N1 -> N2
    Returns ids for later linking from conversation graph.
    """
    text1 = "Project KGE stores entities and relations with provenance spans."
    text2 = "Conversation graph nodes can reference KG nodes/edges for grounding."

    n1 = Node(
        id="KG_N1",
        label="KGE provenance",
        type="entity",
        summary="KGE stores entities/relations with spans for provenance.",
        mentions=[
            mk_grounding(
                mk_span(
                    doc_id=kg_doc_id,
                    full_text=text1,
                    insertion_method="seed_kg_node",
                    document_page_url=f"doc/{kg_doc_id}#KG_N1",
                    collection_page_url=f"collection/{kg_doc_id}",
                )
            )
        ],
        metadata={"level_from_root": 0, "entity_type": "kg_entity"},
        domain_id=None,
        canonical_entity_id=None,
        properties={"kind": "kg_node"},
        embedding=None,
        doc_id=kg_doc_id,
        level_from_root=0,
    )

    n2 = Node(
        id="KG_N2",
        label="Conversation grounding",
        type="entity",
        summary="Conversation graph nodes can reference KG items for grounding.",
        mentions=[
            mk_grounding(
                mk_span(
                    doc_id=kg_doc_id,
                    full_text=text2,
                    insertion_method="seed_kg_node",
                    document_page_url=f"doc/{kg_doc_id}#KG_N2",
                    collection_page_url=f"collection/{kg_doc_id}",
                )
            )
        ],
        metadata={"level_from_root": 0, "entity_type": "kg_entity"},
        domain_id=None,
        canonical_entity_id=None,
        properties={"kind": "kg_node"},
        embedding=None,
        doc_id=kg_doc_id,
        level_from_root=0,
    )

    e1 = Edge(
        id="KG_E1",
        label="supports",
        type="relationship",
        summary="Provenance spans support conversation grounding.",
        source_ids=[n1.id],
        target_ids=[n2.id],
        relation="supports",
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[
            mk_grounding(
                mk_span(
                    doc_id=kg_doc_id,
                    full_text="supports",
                    insertion_method="seed_kg_edge",
                    document_page_url=f"doc/{kg_doc_id}#KG_E1",
                    collection_page_url=f"collection/{kg_doc_id}",
                    start_char=0,
                    end_char=8,
                )
            )
        ],
        metadata={"level_from_root": 0, "entity_type": "kg_edge"},
        domain_id=None,
        canonical_entity_id=None,
        properties={"kind": "kg_edge"},
        embedding=None,
        doc_id=kg_doc_id,
    )

    # Insert (bypass ingestion)
    add_node_raw(kg_engine, n1)
    add_node_raw(kg_engine, n2)
    add_edge_raw(kg_engine, e1)

    return {
        "doc_id": kg_doc_id,
        "node_ids": (n1.id, n2.id),
        "edge_ids": (e1.id,),
        "n1": n1,
        "n2": n2,
        "e1": e1,
    }


# ---------------------------------------------------------------------
# Seed Conversation graph with refs to KG
# ---------------------------------------------------------------------

def seed_conversation_graph(
    *,
    conversation_engine: GraphKnowledgeEngine,
    user_id: str = "U_TEST",
    conversation_id: str = "CONV_TEST_001",
    start_node_id: str = "CONV_START_001",
    kg_seed: dict[str, Any],
) -> dict[str, Any]:
    """
    Seeds:
      - conversation start (via engine.create_conversation)
      - two turns (user + assistant)
      - next_turn edge between turns
      - memory_context node
      - summary node
      - kg_ref node referencing kg_seed[n1/e1]
    """
    # Create conversation (your engine already builds the start node)
    conv_id, start_id = conversation_engine.create_conversation(
        user_id,
        conversation_id,
        start_node_id,
    )
    assert conv_id == conversation_id
    assert start_id == start_node_id

    # Turn 0 (user)
    t0_text = "Show me what happened in the graph engine."
    t0_id = "TURN_000"
    t0_span = mk_span(
        doc_id=f"conv:{conv_id}",
        full_text=t0_text,
        insertion_method="conversation_turn",
        collection_page_url=f"conversation/{conv_id}",
        document_page_url=f"conversation/{conv_id}#{t0_id}",
        page_number=1,
    )
    turn0 = ConversationNode(
        user_id=user_id,
        id=t0_id,
        label="Turn 0 (user)",
        type="entity",
        doc_id=t0_id,
        summary=t0_text,
        role="user",  # type: ignore
        turn_index=0,
        conversation_id=conv_id,
        mentions=[mk_grounding(t0_span)],
        properties={},
        metadata={
            "entity_type": "conversation_turn",
            "level_from_root": 0,
            "in_conversation_chain": True,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=0,
    )
    conversation_engine.add_node(turn0, None)

    # Turn 1 (assistant)
    t1_text = "Here are the relevant KG nodes and the conversation timeline."
    t1_id = "TURN_001"
    t1_span = mk_span(
        doc_id=f"conv:{conv_id}",
        full_text=t1_text,
        insertion_method="conversation_turn",
        collection_page_url=f"conversation/{conv_id}",
        document_page_url=f"conversation/{conv_id}#{t1_id}",
        page_number=1,
    )
    turn1 = ConversationNode(
        user_id=user_id,
        id=t1_id,
        label="Turn 1 (assistant)",
        type="entity",
        doc_id=t1_id,
        summary=t1_text,
        role="assistant",  # type: ignore
        turn_index=1,
        conversation_id=conv_id,
        mentions=[mk_grounding(t1_span)],
        properties={},
        metadata={
            "entity_type": "conversation_turn",
            "level_from_root": 0,
            "in_conversation_chain": True,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=0,
    )
    conversation_engine.add_node(turn1, None)

    # next_turn edge (TURN_000 -> TURN_001)
    next_edge = ConversationEdge(
        id="EDGE_NEXT_000_001",
        source_ids=[turn0.safe_get_id()],
        target_ids=[turn1.safe_get_id()],
        relation="next_turn",
        label="next_turn",
        type="relationship",
        summary="Sequential flow",
        doc_id=f"conv:{conv_id}",
        mentions=[mk_grounding(t1_span)],
        metadata={"causal_type": "chain"},
        domain_id=None,
        canonical_entity_id=None,
        properties={"entity_type": "conversation_edge"},
        embedding=None,
        source_edge_ids=[],
        target_edge_ids=[],
    )
    conversation_engine.add_edge(next_edge)

    # memory_context node (references memory nodes/edges if you want; keep empty here but schema-valid)
    memctx_id = "MEMCTX_001"
    memctx_text = "Active memory context: user wants graph debugging view."
    memctx_span = mk_span(
        doc_id=f"conv:{conv_id}",
        full_text=memctx_text,
        insertion_method="memory_context",
        collection_page_url=f"conversation/{conv_id}",
        document_page_url=f"conversation/{conv_id}#{memctx_id}",
    )
    memctx = ConversationNode(
        user_id=user_id,
        id=memctx_id,
        label="Memory context (turn 1)",
        type="entity",
        doc_id=memctx_id,
        summary=memctx_text,
        role="system",  # type: ignore
        turn_index=1,
        conversation_id=conv_id,
        mentions=[mk_grounding(memctx_span)],
        properties={
            "user_id": user_id,
            "source_memory_nodes_ids": [],
            "source_memory_edges_ids": [],
        },
        metadata={
            "entity_type": "memory_context",
            "level_from_root": 0,
            "in_conversation_chain": False,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=0,
    )
    conversation_engine.add_node(memctx, None)

    # summary node (system)
    summ_id = "SUMM_001"
    summ_text = "Summary: user asks to inspect graph flow; assistant will show KG + conversation links."
    summ_span = mk_span(
        doc_id=f"conv:{conv_id}",
        full_text=summ_text,
        insertion_method="conversation_summary",
        collection_page_url=f"conversation/{conv_id}",
        document_page_url=f"conversation/{conv_id}#{summ_id}",
    )
    summ = ConversationNode(
        user_id=user_id,
        id=summ_id,
        label="Summary 0-1",
        type="entity",
        doc_id=summ_id,
        summary=summ_text,  # type: ignore
        role="system",  # type: ignore
        turn_index=1,
        conversation_id=conv_id,
        mentions=[mk_grounding(summ_span)],
        properties={"content": summ_text},
        metadata={
            "entity_type": "conversation_summary",
            "level_from_root": 1,
            "in_conversation_chain": True,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=1,
    )
    conversation_engine.add_node(summ, None)

    # summarizes edge (summary -> turns)
    summ_edge = ConversationEdge(
        id="EDGE_SUMM_001",
        source_ids=[summ.safe_get_id()],
        target_ids=[turn0.safe_get_id(), turn1.safe_get_id()],
        relation="summarizes",
        label="summarizes",
        type="relationship",
        summary="Memory summarization",
        doc_id=f"conv:{conv_id}",
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        mentions=[mk_grounding(summ_span)],
        metadata={"causal_type": "summary"},
        source_edge_ids=[],
        target_edge_ids=[],
    )
    conversation_engine.add_edge(summ_edge)

    # kg_ref node: points to KG node/edge (this is what your dump tool should bridge)
    kg_ref_id = "KGREF_001"
    kg_ref_text = f"KG ref: node={kg_seed['node_ids'][0]} edge={kg_seed['edge_ids'][0]}"
    kg_ref_span = mk_span(
        doc_id=f"conv:{conv_id}",
        full_text=kg_ref_text,
        insertion_method="kg_ref",
        collection_page_url=f"conversation/{conv_id}",
        document_page_url=f"conversation/{conv_id}#{kg_ref_id}",
    )
    kg_ref_node = ConversationNode(
        user_id=user_id,
        id=kg_ref_id,
        label="KG reference",
        type="entity",
        doc_id=kg_ref_id,
        summary=kg_ref_text,
        role="system",  # type: ignore
        turn_index=1,
        conversation_id=conv_id,
        mentions=[mk_grounding(kg_ref_span)],
        properties={
            # critical: dump uses these to build hyperlinks / cross-bundle paths
            "ref_kind": "kg",
            "ref_doc_id": kg_seed["doc_id"],
            "ref_node_ids": list(kg_seed["node_ids"]),
            "ref_edge_ids": list(kg_seed["edge_ids"]),
        },
        metadata={
            "entity_type": "kg_ref",
            "level_from_root": 0,
            "in_conversation_chain": False,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=0,
    )
    conversation_engine.add_node(kg_ref_node, None)

    # edge: turn1 -> kg_ref (optional but useful for viz)
    ref_edge = ConversationEdge(
        id="EDGE_TURN1_KGREF",
        source_ids=[turn1.safe_get_id()],
        target_ids=[kg_ref_node.safe_get_id()],
        relation="mentions_kg",
        label="mentions_kg",
        type="relationship",
        summary="Assistant mentions KG refs",
        doc_id=f"conv:{conv_id}",
        domain_id=None,
        canonical_entity_id=None,
        properties={"ref_kind": "kg"},
        embedding=None,
        mentions=[mk_grounding(kg_ref_span)],
        metadata={"causal_type": "reference"},
        source_edge_ids=[],
        target_edge_ids=[],
    )
    conversation_engine.add_edge(ref_edge)
    conv_id = "conv_test_1"
    user_id = "user_test_1"

    kg_target_id = "KG_N1"  # must exist in the KG bundle if you want openRef to succeed later

    kg_ref_node = ConversationNode(
        id="CONV_REF_KG_N1",
        label="KG ref → KG_N1",
        type="reference_pointer",  # allowed: 'entity' | 'relationship' | 'reference_pointer'
        summary="Conversation-side pointer to a KG node (for testing focus/openRef).",
        doc_id="CONV_REF_KG_N1",
        mentions=[
            Grounding(
                spans=[Span.from_dummy_for_conversation()]  # ensures spans>=1, mentions>=1
            )
        ],
        properties={
            # keep these JSON-primitive only (validator expects primitives / lists / nested mappings)
            "ref_target_kind": "kg_node",
            "ref_target_id": kg_target_id,
        },
        metadata={
            # REQUIRED by ConversationNodeMetadata
            "level_from_root": 0,
            "entity_type": "kg_ref",
            "in_conversation_chain": False,

            # OPTIONAL but useful (ConversationRoleMixin syncs these too)
            "role": "system",
            "turn_index": 1,
            "conversation_id": conv_id,
            "user_id": user_id,
        },
        role="system",
        turn_index=1,
        conversation_id=conv_id,
        user_id=user_id,        
        embedding=None,        
        domain_id=None,
        canonical_entity_id=None,        
        level_from_root=0,
    )

    # then upsert it into the conversation engine alongside your other seeded nodes
    conversation_engine.add_node(kg_ref_node)
    return {
        "conversation_id": conv_id,
        "start_node_id": start_id,
        "turn_ids": (turn0.id, turn1.id),
        "edge_ids": (next_edge.id, summ_edge.id, ref_edge.id),
        "memctx_id": memctx.id,
        "summary_id": summ.id,
        "kg_ref_id": kg_ref_node.id,
    }


# def seed_both_graphs(
#     *,
#     kg_engine: Any,
#     conversation_engine: Any,
#     user_id: str = "U_TEST",
# ) -> dict[str, Any]:
#     kg_seed = seed_kg_graph(kg_engine=kg_engine, kg_doc_id="D_KG_001")
#     conv_seed = seed_conversation_graph(
#         conversation_engine=conversation_engine,
#         user_id=user_id,
#         conversation_id="CONV_TEST_001",
#         start_node_id="CONV_START_001",
#         kg_seed=kg_seed,
#     )
#     return {"kg": kg_seed, "conversation": conv_seed}



@pytest.fixture
def seeded_kg_and_conversation(tmp_path: Path):
    """
    Returns (kg_engine, conversation_engine, kg_seed, conv_seed, kg_dir, conv_dir)

    - Real persisted engines in tmp_path
    - KG is seeded with real Node/Edge objects (schema-correct Span/Grounding)
    - Conversation is seeded with conversation nodes/edges + memory ctx + summary + kg_ref
    - Conversation kg_ref points to KG ids via properties.ref_node_ids/ref_edge_ids
    """
    kg_dir = tmp_path / "chroma_kg"
    conv_dir = tmp_path / "chroma_conversation"

    kg_engine = GraphKnowledgeEngine(persist_directory=str(kg_dir), kg_graph_type="knowledge")
    conversation_engine = GraphKnowledgeEngine(persist_directory=str(conv_dir), kg_graph_type="conversation")

    kg_seed = seed_kg_graph(kg_engine=kg_engine, kg_doc_id="D_KG_001")
    conv_seed = seed_conversation_graph(
        conversation_engine=conversation_engine,
        user_id="U_TEST",
        conversation_id="CONV_TEST_001",
        start_node_id="CONV_START_001",
        kg_seed=kg_seed,
    )

    return kg_engine, conversation_engine, kg_seed, conv_seed, kg_dir, conv_dir
