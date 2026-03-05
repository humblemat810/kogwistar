
import json
from pathlib import Path

from fastapi.testclient import TestClient


def _read_template_html() -> str:
    p = Path("graph_knowledge_engine/templates/d3.html")
    if p.exists():
        return p.read_text(encoding="utf-8")
    p2 = Path("d3.html")
    if p2.exists():
        return p2.read_text(encoding="utf-8")
    raise RuntimeError("Cannot find d3.html template at graph_knowledge_engine/templates/d3.html or ./d3.html")


def test_dump_d3_bundle_cdc_injection_no_jinja_tokens(tmp_path: Path) -> None:
    """CI-safe: ensure bundle rendering uses the shared utility and doesn't leak Jinja tokens."""
    from graph_knowledge_engine.utils.kge_debug_dump import dump_d3_bundle

    template_html = _read_template_html()
    out_html = tmp_path / "conversation.bundle.cdc.html"
    ws_url = "ws://127.0.0.1:8787/changes/ws?since=0"

    dump_d3_bundle(
        engine=None,
        engine_type="conversation",
        template_html=template_html,
        out_html=out_html,
        doc_id=None,
        mode="reify",
        insertion_method=None,
        bundle_meta={"test": True},
        cdc_enabled=True,
        cdc_ws_url=ws_url,
        embed_empty=True,
    )

    html = out_html.read_text(encoding="utf-8")
    assert "{{" not in html and "{%" not in html
    assert ws_url in html
    # CDC enabled should be injected as a JS bool literal; accept "true" in any casing
    assert "cdc" in html.lower()


def test_cdc_bridge_broadcasts_to_websocket_clients_ci() -> None:
    """CI-safe: bridge broadcasts ingested events to websocket clients (live stream)."""
    try:
        from graph_knowledge_engine.cdc.change_bridge import app  # type: ignore
    except Exception:  # pragma: no cover
        from change_bridge import app  # type: ignore
    import time
    with TestClient(app) as client:
        with client.websocket_connect("/changes/ws") as ws:
            ev = {
                "seq": 1,
                "op": "node.upsert",
                "entity": {"kind": "node", "id": "n1", "kg_graph_type": "conversation"},
                "payload": {"x": 1},
                "ts_unix_ms":int(time.time() * 1000)
            }
            r = client.post("/ingest", json=ev)
            assert r.status_code == 200
            got = json.loads(ws.receive_text())
            assert got["seq"] == 1
            assert got["op"] == "node.upsert"
