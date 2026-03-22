from __future__ import annotations
import pytest
pytestmark = pytest.mark.ci_full

import importlib
import json
from typing import Any



def _template_html() -> str:
    # Minimal template that matches the injection points used by the new dump util
    return """<!doctype html>
<html>
  <head><meta charset="utf-8"></head>
  <body>
    <script>
      window.__EMBEDDED_DATA__ = null;
      window.__BUNDLE_META__ = null;
    </script>
    <div id="app">ok</div>
  </body>
</html>
"""


def _fake_payload() -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": "n1",
                "label": "Node 1",
                "type": "entity",
                "properties": {"refers_to_id": "kg:n1"},
            },
            {"id": "e1", "label": "Edge 1", "type": "edge-node"},
        ],
        "links": [
            {"source": "n1", "target": "e1", "relation": "rel_a"},
            {"source": "e1", "target": "n1", "relation": "rel_b"},
        ],
    }


def test_dump_d3_bundle_injects_embedded_data_and_meta(tmp_path, monkeypatch):
    mod = importlib.import_module("kogwistar.utils.kge_debug_dump")

    payload = _fake_payload()

    # Stub out to_d3_force so we don't need real engines / chroma
    monkeypatch.setattr(mod, "to_d3_force", lambda *args, **kwargs: payload)

    out_html = tmp_path / "one.bundle.html"
    bundle_meta = {
        "kg_bundle_href": "./kg.bundle.html",
        "conversation_bundle_href": "./conversation.bundle.html",
    }

    # Call "debugger path"
    mod.dump_d3_bundle(
        engine=object(),  # opaque stub
        template_html=_template_html(),
        out_html=out_html,
        doc_id="DOC1",
        mode="reify",
        insertion_method="document_ingestion",
        bundle_meta=bundle_meta,
    )

    text = out_html.read_text(encoding="utf-8")

    # Embedded data is injected as a JS object literal
    assert "window.__EMBEDDED_DATA__ = " in text
    assert '"nodes"' in text and '"links"' in text
    # Meta is injected too
    assert "window.__BUNDLE_META__ = " in text
    assert "kg.bundle.html" in text


def test_dump_paired_bundles_writes_both_and_meta_json(tmp_path, monkeypatch):
    mod = importlib.import_module("kogwistar.utils.kge_debug_dump")

    payload = _fake_payload()
    monkeypatch.setattr(mod, "to_d3_force", lambda *args, **kwargs: payload)

    out_dir = tmp_path / "bundle"
    meta = mod.dump_paired_bundles(
        kg_engine=object(),
        conversation_engine=object(),
        template_html=_template_html(),
        out_dir=out_dir,
        kg_out="kg.bundle.html",
        conversation_out="conversation.bundle.html",
        kg_doc_id="KG_DOC",
        conversation_doc_id="CONV_DOC",
        mode="reify",
        insertion_method="api_upsert",
    )

    # Files exist
    kg_file = out_dir / "kg.bundle.html"
    conv_file = out_dir / "conversation.bundle.html"
    meta_file = out_dir / "bundle.meta.json"

    assert kg_file.exists()
    assert conv_file.exists()
    assert meta_file.exists()

    # Meta has the relative cross-links needed for "Open ref"
    assert meta["kg_bundle_href"] == "./kg.bundle.html"
    assert meta["conversation_bundle_href"] == "./conversation.bundle.html"

    # Meta JSON matches returned meta
    meta_json = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta_json["kg_bundle_href"] == meta["kg_bundle_href"]

    # Both HTML files contain bundle meta injection
    assert "window.__BUNDLE_META__" in kg_file.read_text(encoding="utf-8")
    assert "window.__BUNDLE_META__" in conv_file.read_text(encoding="utf-8")


def test_cli_one_writes_output_html(tmp_path, monkeypatch, capsys):
    """
    Tests CLI entrypoint without subprocess:
    - patches build_engine + to_d3_force
    - runs main() with monkeypatched sys.argv
    """
    mod = importlib.import_module("kogwistar.utils.kge_debug_dump")

    payload = _fake_payload()
    monkeypatch.setattr(mod, "to_d3_force", lambda *args, **kwargs: payload)

    # Avoid constructing a real GraphKnowledgeEngine
    monkeypatch.setattr(mod, "build_engine", lambda **kwargs: object())

    template_path = tmp_path / "d3.html"
    template_path.write_text(_template_html(), encoding="utf-8")

    out_path = tmp_path / "cli.bundle.html"

    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kge_debug_dump",
            "one",
            "--persist-dir",
            str(tmp_path / "fake_chroma"),
            "--graph-type",
            "knowledge",
            "--template",
            str(template_path),
            "--out",
            str(out_path),
            "--doc-id",
            "DOC_CLI",
            "--mode",
            "reify",
        ],
    )

    mod.main()

    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    assert "window.__EMBEDDED_DATA__" in html
    assert '"Node 1"' in html  # sanity check payload was embedded

    stdout = capsys.readouterr().out
    assert "[OK]" in stdout
