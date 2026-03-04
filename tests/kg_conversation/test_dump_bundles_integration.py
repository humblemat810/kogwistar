from __future__ import annotations

import json
from pathlib import Path
import pathlib
import importlib

import pytest

import re

# import pytest
# from graph_knowledge_engine.engine import GraphKnowledgeEngine
# from graph_knowledge_engine.postgres_backend import PgVectorBackend
from conversation.models import ConversationNode
from tests.conftest import _make_engine_pair

def _minimal_bundle_template() -> str:
    """
    Minimal HTML template used for dump tests.

    This template intentionally excludes D3 and rendering logic.
    It exists solely to provide stable injection points for:
      - window.__EMBEDDED_DATA__
      - window.__BUNDLE_META__
    """
    # Minimal template to make injection + parsing reliable for tests.
    # Your real template works too, but this makes tests robust.
    return """<!doctype html><html><body><script>
window.__EMBEDDED_DATA__ = null;
window.__BUNDLE_META__ = null;
</script></body></html>
"""


# def _fake_ef_dim(dim: int):
#     def _ef(texts):
#         return [[0.0] * dim for _ in texts]
#     _ef.name = "_fake_ef_dim"
#     return _ef


# def _make_engine_pair(*, backend_kind: str, tmp_path, sa_engine, pg_schema: str, dim: int = 3):
#     """
#     Build (kg_engine, conv_engine) for either chroma or pgvector.
#     """
#     # ef = _fake_ef_dim(dim)

#     if backend_kind == "chroma":
#         kg_engine = GraphKnowledgeEngine(persist_directory=str(tmp_path / "kg"), kg_graph_type="knowledge"
#                                         #  , embedding_function=ef
#                                          )
#         conv_engine = GraphKnowledgeEngine(persist_directory=str(tmp_path / "conv"), kg_graph_type="conversation"
#                                         #    , embedding_function=ef
#                                            )
#         return kg_engine, conv_engine

#     if backend_kind == "pg":
#         if sa_engine is None or pg_schema is None:
#             pytest.skip("pg backend requested but sa_engine/pg_schema fixtures not available")
#         kg_schema = f"{pg_schema}_kg"
#         conv_schema = f"{pg_schema}_conv"
#         kg_backend = PgVectorBackend(engine=sa_engine, embedding_dim=dim, schema=kg_schema)
#         conv_backend = PgVectorBackend(engine=sa_engine, embedding_dim=dim, schema=conv_schema)
#         kg_engine = GraphKnowledgeEngine(persist_directory=str(tmp_path / "kg_meta"), kg_graph_type="knowledge", 
#                                         #  embedding_function=ef, 
#                                          backend=kg_backend)
#         conv_engine = GraphKnowledgeEngine(persist_directory=str(tmp_path / "conv_meta"), kg_graph_type="conversation", 
#                                         #    embedding_function=ef, 
#                                            backend=conv_backend)
#         return kg_engine, conv_engine

#     raise ValueError(f"unknown backend_kind: {backend_kind!r}")


@pytest.fixture(params=["chroma", "pg"])
def seeded_kg_and_conversation(request, tmp_path, sa_engine, pg_schema):
    """
    Override the global fixture with a parametrized version so the same tests
    run against both chroma and pg backends.
    """
    backend_kind: str = request.param
    kg_engine, conv_engine = _make_engine_pair(
        backend_kind=backend_kind, tmp_path=tmp_path, sa_engine=sa_engine, pg_schema=pg_schema, dim=3
    )

    from engine_core.models import Node, Grounding, Span, MentionVerification
    conv_id = "test-conv-seeded_kg_and_conversation"
    t0_text = "show me what happened in the graph engine"
    def _span():
        return Span(
            collection_page_url="test",
            document_page_url="test",
            doc_id="seed",
            insertion_method="pytest-seed",
            page_number=1,
            start_char=0,
            end_char=1,
            excerpt=t0_text,
            context_before="",
            context_after="",
            chunk_id=None,
            source_cluster_id=None,
            verification=MentionVerification(method="human", is_verified=True, score=1.0, notes="seed"),
        )

    g = Grounding(spans=[_span()])

    def _node(i: str, emb):
        n = Node(
            id=i,
            label=i,
            type="entity",
            doc_id="KG_DOC",
            summary=i,
            mentions=[g],
            properties={},
            metadata={"name": i},
            domain_id=None,
            canonical_entity_id=None,
            embedding=emb,
            level_from_root=0,
        )
        return n

    kg_engine.add_node(_node("A", [1.0, 0.0, 0.0]))
    kg_engine.add_node(_node("B", [0.9, 0.1, 0.0]))
    kg_engine.add_node(_node("C", [0.0, 1.0, 0.0]))

    conv_n = ConversationNode(
        id="conv|turn|1",
        label="turn1",
        type="entity",
        doc_id="CONV_DOC",
        summary="turn1",
        mentions=[g],
        properties={"refers_to_id": "A"},
        domain_id=None,
        canonical_entity_id=None,
        metadata={
                "entity_type": "conversation_turn",
                "level_from_root": 0,
                "in_conversation_chain": True,
            },
        role="user",  # type: ignore
        turn_index=0,
        conversation_id=conv_id,
        embedding=[0.0, 2.1, 0.18],
        level_from_root=0,
    )
    conv_engine.add_node(conv_n)

    kg_dir = pathlib.Path(getattr(kg_engine, "persist_directory", str(tmp_path / "kg")))
    conv_dir = pathlib.Path(getattr(conv_engine, "persist_directory", str(tmp_path / "conv")))
    kg_seed = {"node_ids": ["A", "B", "C"]}
    conv_seed = {"node_ids": ["conv|turn|1"]}
    return kg_engine, conv_engine, kg_seed, conv_seed, kg_dir, conv_dir

# Regex-based extraction is used instead of string slicing because:
# - HTML formatting may change
# - JSON may contain semicolons in string values
# - We want a robust, template-agnostic extractor
_EMBEDDED_RE = re.compile(
    r"window\.__EMBEDDED_DATA__\s*=\s*(\{.*?\})\s*;",
    re.DOTALL,
)

def _extract_embedded_payload(html: str) -> dict:
    """
    Extracts the embedded graph payload from a bundle HTML file.

    This parses the JSON assigned to window.__EMBEDDED_DATA__.
    Rendering logic is intentionally ignored.
    """
    m = _EMBEDDED_RE.search(html)
    assert m, "Cannot find window.__EMBEDDED_DATA__ assignment"
    return json.loads(m.group(1))


def test_dump_paired_bundles_embeds_graph_data_and_links(tmp_path: Path, seeded_kg_and_conversation):
    kg_engine, conv_engine, kg_seed, conv_seed, kg_dir, conv_dir = seeded_kg_and_conversation
    mod = importlib.import_module("graph_knowledge_engine.utils.kge_debug_dump")

    out_dir = tmp_path / "dump_run"
    meta = mod.dump_paired_bundles(
        kg_engine=kg_engine,
        conversation_engine=conv_engine,
        template_html=_minimal_bundle_template(),
        out_dir=out_dir,
        kg_out="kg.bundle.html",
        conversation_out="conversation.bundle.html",
        kg_doc_id="KG_DOC",
        conversation_doc_id="CONV_DOC",
        mode="reify",
        insertion_method="pytest-seed",
    )

    # Files exist
    kg_html_path = out_dir / "kg.bundle.html"
    conv_html_path = out_dir / "conversation.bundle.html"
    meta_path = out_dir / "bundle.meta.json"
    assert kg_html_path.exists()
    assert conv_html_path.exists()
    assert meta_path.exists()

    # Meta has relative linking so conversation can open KG offline
    assert meta["kg_bundle_href"] == "./kg.bundle.html"
    assert meta["conversation_bundle_href"] == "./conversation.bundle.html"

    # Parse embedded payloads
    kg_payload = _extract_embedded_payload(kg_html_path.read_text(encoding="utf-8"))
    conv_payload = _extract_embedded_payload(conv_html_path.read_text(encoding="utf-8"))

    kg_node_ids = {n["id"] for n in kg_payload.get("nodes", [])}
    assert {"A", "B", "C"} <= kg_node_ids  # seeded KG nodes

    # Find conversation node that points at KG
    conv_nodes = conv_payload.get("nodes", [])
    ref_nodes = [n for n in conv_nodes if (n.get("properties") or {}).get("refers_to_id")]
    assert ref_nodes, "Expected at least one conversation node with properties.refers_to_id"

    # Ensure the referenced ID exists in KG payload
    ref_id = (ref_nodes[0]["properties"]["refers_to_id"])
    assert ref_id in kg_node_ids
    
    
import subprocess
import sys

def test_cli_pair_real_persisted(tmp_path: Path, seeded_kg_and_conversation):
    kg_engine, conv_engine, kg_seed, conv_seed, kg_dir, conv_dir = seeded_kg_and_conversation

    # The CLI creates engines from persist dirs, so we pass the same dirs.
    kg_dir = Path(kg_engine.persist_directory) if hasattr(kg_engine, "persist_directory") else (tmp_path / "chroma_kg")
    conv_dir = Path(conv_engine.persist_directory) if hasattr(conv_engine, "persist_directory") else (tmp_path / "chroma_conv")
    workflow_dir = Path(conv_engine.persist_directory) if hasattr(conv_engine, "persist_directory") else (tmp_path / "workflow")
    
    template = tmp_path / "d3.html"
    template.write_text("""<script>
window.__EMBEDDED_DATA__ = null;
window.__BUNDLE_META__ = null;
</script>""", encoding="utf-8")

    out_dir = tmp_path / "cli_dump"

    cmd = [
        sys.executable, "-m", "graph_knowledge_engine.utils.kge_debug_dump",
        "bundle",
        "--kg-persist-dir", str(kg_dir),
        "--conversation-persist-dir", str(conv_dir),
        "--workflow-persist-dir", str(workflow_dir),
        "--template", str(template),
        "--out-dir", str(out_dir),
        "--kg-doc-id", "KG_DOC",
        "--conversation-doc-id", "CONV_DOC",
        "--mode", "reify",
        "--insertion-method", "pytest-seed",
    ]
    subprocess.check_call(cmd)

    assert (out_dir / "kg.bundle.html").exists()
    assert (out_dir / "conversation.bundle.html").exists()
    assert (out_dir / "bundle.meta.json").exists()
    
    
