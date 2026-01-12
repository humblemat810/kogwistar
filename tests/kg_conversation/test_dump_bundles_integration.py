from __future__ import annotations

import json
from pathlib import Path
import importlib


def _template_html() -> str:
    # Minimal template to make injection + parsing reliable for tests.
    # Your real template works too, but this makes tests robust.
    return """<!doctype html><html><body><script>
window.__EMBEDDED_DATA__ = null;
window.__BUNDLE_META__ = null;
</script></body></html>
"""


def _extract_embedded_payload(html: str) -> dict:
    marker = "window.__EMBEDDED_DATA__ ="
    start = html.index(marker) + len(marker)
    end = html.index(";", start)
    return json.loads(html[start:end].strip())


def test_dump_paired_bundles_end_to_end(tmp_path: Path, seeded_kg_and_conversation):
    kg_engine, conv_engine = seeded_kg_and_conversation
    mod = importlib.import_module("graph_knowledge_engine.utils.kge_debug_dump")

    out_dir = tmp_path / "dump_run"
    meta = mod.dump_paired_bundles(
        kg_engine=kg_engine,
        conversation_engine=conv_engine,
        template_html=_template_html(),
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
    kg_engine, conv_engine = seeded_kg_and_conversation

    # The CLI creates engines from persist dirs, so we pass the same dirs.
    kg_dir = Path(kg_engine.persist_directory) if hasattr(kg_engine, "persist_directory") else (tmp_path / "chroma_kg")
    conv_dir = Path(conv_engine.persist_directory) if hasattr(conv_engine, "persist_directory") else (tmp_path / "chroma_conv")

    template = tmp_path / "d3.html"
    template.write_text("""<script>
window.__EMBEDDED_DATA__ = null;
window.__BUNDLE_META__ = null;
</script>""", encoding="utf-8")

    out_dir = tmp_path / "cli_dump"

    cmd = [
        sys.executable, "-m", "graph_knowledge_engine.utils.kge_debug_dump",
        "pair",
        "--kg-persist-dir", str(kg_dir),
        "--conversation-persist-dir", str(conv_dir),
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