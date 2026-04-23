from __future__ import annotations

import json
from pathlib import Path
import pathlib
import importlib

import pytest
pytestmark = pytest.mark.core

import re

# import pytest
# from kogwistar.engine_core.engine import GraphKnowledgeEngine
# from kogwistar.engine_core.postgres_backend import PgVectorBackend
from tests.graph_seed_helpers import seed_kg_and_conversation_bundle_for_backend


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
window.__EMBEDDED_DATA__ = {{ embedded_data | safe if embedded_data is defined else "null" }};
window.__BUNDLE_META__ = {{ bundle_meta | safe if bundle_meta is defined else "null" }};
</script></body></html>
"""
# def _fake_ef_dim(dim: int):
#     def _ef(texts):
#         return [[0.0] * dim for _ in texts]
#     _ef.name = "_fake_ef_dim"
#     return _ef


@pytest.fixture(
    params=[
        pytest.param("fake", id="fake", marks=pytest.mark.ci),
        pytest.param("chroma", id="chroma", marks=pytest.mark.ci_full),
        pytest.param("pg", id="pg", marks=pytest.mark.ci_full),
    ],
)
def seeded_kg_and_conversation(request, tmp_path):
    """
    Override the global fixture with a parametrized version so the same tests
    run against fake, chroma, and pg backends.
    """
    data = seed_kg_and_conversation_bundle_for_backend(
        backend_kind=request.param,
        tmp_path=tmp_path,
        request=request,
        kg_doc_id="KG_DOC",
        user_id="user_test_1",
        conversation_id="test-conv-seeded_kg_and_conversation",
        start_node_id="CONV_START_001",
        dim=3,
        legacy_bundle=True,
    )
    return data


@pytest.fixture(
    params=[
        pytest.param("fake", id="fake", marks=pytest.mark.ci),
        pytest.param("chroma", id="chroma", marks=pytest.mark.ci_full),
        pytest.param("pg", id="pg", marks=pytest.mark.ci_full),
    ],
)
def seeded_kg_and_conversation_real(request, tmp_path):
    """
    Real persisted backends only.

    The CLI subprocess reopens persist dirs, so the in-memory fake backend is
    not suitable for the subprocess branch of this test. We still keep the fake
    case here by exercising the bundle writer in-process.
    """
    data = seed_kg_and_conversation_bundle_for_backend(
        backend_kind=request.param,
        tmp_path=tmp_path,
        request=request,
        kg_doc_id="KG_DOC",
        user_id="user_test_1",
        conversation_id="test-conv-seeded_kg_and_conversation",
        start_node_id="CONV_START_001",
        dim=3,
        legacy_bundle=True,
    )
    return data


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


def test_dump_paired_bundles_embeds_graph_data_and_links(
    tmp_path: Path, seeded_kg_and_conversation
):
    kg_engine, conv_engine, kg_seed, conv_seed, kg_dir, conv_dir = (
        seeded_kg_and_conversation
    )
    mod = importlib.import_module("kogwistar.utils.kge_debug_dump")
    is_fake_backend = getattr(kg_engine, "backend_kind", None) in {
        "fake",
        "memory",
    }


    out_dir = tmp_path / "dump_run"
    bundle_kwargs = {
        "kg_engine": kg_engine,
        "conversation_engine": conv_engine,
        "template_html": _minimal_bundle_template(),
        "out_dir": out_dir,
        "kg_out": "kg.bundle.html",
        "conversation_out": "conversation.bundle.html",
        "mode": "reify",
    }
    if is_fake_backend:
        meta = mod.dump_paired_bundles(**bundle_kwargs)
    else:
        meta = mod.dump_paired_bundles(
            **bundle_kwargs,
            kg_doc_id="KG_DOC",
            conversation_doc_id="CONV_DOC",
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
    ref_nodes = [
        n for n in conv_nodes if (n.get("properties") or {}).get("refers_to_id")
    ]
    assert ref_nodes, (
        "Expected at least one conversation node with properties.refers_to_id"
    )

    # Ensure the referenced ID exists in KG payload
    ref_id = ref_nodes[0]["properties"]["refers_to_id"]
    assert ref_id in kg_node_ids


import subprocess
import sys


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", id="fake", marks=pytest.mark.ci),
        pytest.param("chroma", id="chroma", marks=pytest.mark.ci_full),
    ],
)
def test_cli_pair_real_persisted(
    tmp_path: Path,
    backend_kind: str,
):
    kg_engine, conv_engine, kg_seed, conv_seed, kg_dir, conv_dir = (
        seed_kg_and_conversation_bundle_for_backend(
            backend_kind=backend_kind,
            tmp_path=tmp_path,
            kg_doc_id="KG_DOC",
            user_id="user_test_1",
            conversation_id="test-conv-seeded_kg_and_conversation",
            start_node_id="CONV_START_001",
            dim=3,
            legacy_bundle=True,
        )
    )

    # The CLI creates engines from persist dirs, so we pass the same dirs.
    kg_dir = (
        Path(kg_engine.persist_directory)
        if hasattr(kg_engine, "persist_directory")
        else (tmp_path / "chroma_kg")
    )
    conv_dir = (
        Path(conv_engine.persist_directory)
        if hasattr(conv_engine, "persist_directory")
        else (tmp_path / "chroma_conv")
    )
    workflow_dir = (
        tmp_path / "workflow"
    )

    template = tmp_path / "d3.html"
    template.write_text(
        """<script>
window.__EMBEDDED_DATA__ = {{ embedded_data | safe if embedded_data is defined else "null" }};
window.__BUNDLE_META__ = {{ bundle_meta | safe if bundle_meta is defined else "null" }};
</script>""",
        encoding="utf-8",
    )

    out_dir = tmp_path / "cli_dump"

    if backend_kind == "fake":
        mod = importlib.import_module("kogwistar.utils.kge_debug_dump")
        mod.dump_paired_bundles(
            kg_engine=kg_engine,
            conversation_engine=conv_engine,
            workflow_engine=None,
            template_html=template.read_text(encoding="utf-8"),
            out_dir=out_dir,
            kg_out="kg.bundle.html",
            conversation_out="conversation.bundle.html",
            kg_doc_id="KG_DOC",
            conversation_doc_id="CONV_DOC",
            mode="reify",
            insertion_method="pytest-seed",
        )
    else:
        cmd = [
            sys.executable,
            "-m",
            "kogwistar.utils.kge_debug_dump",
            "bundle",
            "--kg-persist-dir",
            str(kg_dir),
            "--conversation-persist-dir",
            str(conv_dir),
            "--workflow-persist-dir",
            str(workflow_dir),
            "--template",
            str(template),
            "--out-dir",
            str(out_dir),
            "--kg-doc-id",
            "KG_DOC",
            "--conversation-doc-id",
            "CONV_DOC",
            "--mode",
            "reify",
            "--insertion-method",
            "pytest-seed",
        ]
        subprocess.check_call(cmd)

    assert (out_dir / "kg.bundle.html").exists()
    assert (out_dir / "conversation.bundle.html").exists()
    assert (out_dir / "bundle.meta.json").exists()
