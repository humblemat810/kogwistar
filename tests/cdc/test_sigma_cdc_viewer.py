from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from kogwistar.cdc.change_bridge import create_app
from kogwistar.utils.kge_debug_dump import dump_sigma_bundle
from kogwistar.visualization import graph_viz


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SIGMA_TEMPLATE = _REPO_ROOT / "kogwistar" / "templates" / "sigma.html"
_FORGE_TEMPLATE = _REPO_ROOT / "kogwistar" / "templates" / "cdc_event_forge.html"


def _read(path: Path) -> str:
    assert path.is_file(), f"missing debug template: {path}"
    return path.read_text(encoding="utf-8")


def _event(*, seq: int, op: str, kind: str, entity_id: str, payload: dict) -> dict:
    return {
        "seq": seq,
        "op": op,
        "ts_unix_ms": 1_700_000_000_000 + seq,
        "entity": {
            "kind": kind,
            "id": entity_id,
            "kg_graph_type": "knowledge",
            "url": None,
        },
        "payload": payload,
        "run_id": "sigma-unit-run",
        "step_id": "canonical-hyperedge",
    }


def test_sigma_snapshot_preserves_all_hyperedge_endpoint_kinds(monkeypatch) -> None:
    class Model:
        def __init__(self, **data):
            self.__dict__.update(data)

        def model_dump(self, **kwargs):
            excluded = set(kwargs.get("exclude") or ())
            return {key: value for key, value in self.__dict__.items() if key not in excluded}

    node = Model(id="shared-id", label="Entity with colliding ID", properties={})
    edge = Model(
        id="shared-id",
        label="Meta relationship",
        relation="qualifies",
        source_ids=["node-a", "node-b"],
        target_ids=["node-c"],
        source_edge_ids=["edge-a"],
        target_edge_ids=["edge-b"],
        properties={"confidence": 0.8},
        embedding=[1.0, 2.0],
    )
    monkeypatch.setattr(graph_viz, "_collect_ids", lambda *_args: ([node.id], [edge.id]))
    monkeypatch.setattr(graph_viz, "_load_node_map", lambda *_args: {node.id: node})
    monkeypatch.setattr(graph_viz, "_load_edge_map", lambda *_args: {edge.id: edge})

    payload = graph_viz.to_sigma_hypergraph(object())

    assert payload["mode"] == "raw-hypergraph"
    assert payload["raw_nodes"][0]["id"] == "shared-id"
    rendered_edge = payload["raw_edges"][0]
    assert rendered_edge["id"] == "shared-id"
    assert rendered_edge["source_ids"] == ["node-a", "node-b"]
    assert rendered_edge["target_ids"] == ["node-c"]
    assert rendered_edge["source_edge_ids"] == ["edge-a"]
    assert rendered_edge["target_edge_ids"] == ["edge-b"]
    assert "embedding" not in rendered_edge


def test_dump_sigma_bundle_injects_mode_stream_and_ws_without_jinja(
    tmp_path: Path,
) -> None:
    out_html = tmp_path / "workflow.sigma.html"
    ws_url = "ws://127.0.0.1:8787/changes/ws?since=7"

    dump_sigma_bundle(
        engine=None,
        engine_type="workflow",
        template_html=_read(_SIGMA_TEMPLATE),
        out_html=out_html,
        mode="compact",
        bundle_meta={"purpose": "unit-test"},
        cdc_enabled=True,
        cdc_ws_url=ws_url,
        embed_empty=True,
    )

    html = out_html.read_text(encoding="utf-8")
    assert "{{" not in html
    assert "{%" not in html
    assert 'window.__INITIAL_MODE__ = "compact";' in html
    assert 'window.__BUNDLE_GRAPH_TYPE__ = "workflow";' in html
    assert "window.__CDC_ENABLED__ = true;" in html
    assert f'window.__CDC_WS_URL__ = "{ws_url}";' in html
    assert "https://cdnjs.cloudflare.com/ajax/libs/graphology/0.25.4/" in html
    assert "https://cdnjs.cloudflare.com/ajax/libs/sigma.js/2.4.0/" in html
    assert (
        'window.__EMBEDDED_DATA__ = {"raw_nodes": [], "raw_edges": [], '
        '"mode": "raw-hypergraph"};'
    ) in html


def test_sigma_template_exposes_semantic_interaction_and_cdc_hooks() -> None:
    html = _read(_SIGMA_TEMPLATE)

    # Library and rendering contract. This is static semantic coverage, not visual QA.
    assert "graphology/0.25.4/graphology.umd.min.js" in html
    assert "sigma.js/2.4.0/sigma.min.js" in html
    assert "new graphology.MultiDirectedGraph" in html
    assert "new Sigma(" in html

    for test_id in (
        "mode-select",
        "sigma-stage",
        "search",
        "relation-filter",
        "run-filter",
        "cdc-toggle",
        "mode-notice",
        "inspector",
        "event-log",
        "zoom-in",
        "zoom-out",
        "camera-reset",
        "relayout",
    ):
        assert f'data-testid="{test_id}"' in html

    for semantic_hook in (
        'value="reify"',
        'value="compact"',
        'value="projection"',
        "function canonicalEdge(",
        'source_ids:uniq(',
        'target_ids:uniq(',
        'source_edge_ids:uniq(',
        'target_edge_ids:uniq(',
        "function applyEvent(",
        "function connectCdc(",
        "function placeIncremental(",
        "function directionRamp(",
        "c.createLinearGradient(sp.x,sp.y,x,y)",
        "gradient.addColorStop(0,ramp[0])",
        "gradient.addColorStop(.24,ramp[0])",
        "gradient.addColorStop(.62,ramp[1])",
        "gradient.addColorStop(1,ramp[2])",
        "state.renderer.setGraph(g)",
        'u.searchParams.set("stream",window.__BUNDLE_GRAPH_TYPE__)',
        'u.searchParams.set("since",String(raw.lastSeq||0))',
        "window.__KOGWISTAR_SIGMA__={raw,state,applyEvent,rebuild,selectNode}",
    ):
        assert semantic_hook in html


def test_forge_template_exposes_canonical_hyperedge_and_scenario_hooks() -> None:
    html = _read(_FORGE_TEMPLATE)

    for test_id in (
        "operation-switcher",
        "op-node-upsert",
        "op-node-remove",
        "op-edge-upsert",
        "op-edge-remove",
        "event-form",
        "source-node-ids",
        "target-node-ids",
        "source-edge-ids",
        "target-edge-ids",
        "json-preview",
        "send-event",
        "scenario-generic",
        "scenario-endpoint-first",
        "scenario-burst",
        "request-status",
        "activity-log",
    ):
        assert f'data-testid="{test_id}"' in html

    for semantic_hook in (
        'fetch("/ingest"',
        "source_ids: sourceIds",
        "target_ids: targetIds",
        "source_edge_ids: sourceEdgeIds",
        "target_edge_ids: targetEdgeIds",
        "function scenarioGeneric(",
        "function scenarioEndpointFirst(",
        "function scenarioBurst(",
        'data-scenario="endpoint-first"',
        'ordering_test: "edge-before-node"',
    ):
        assert semantic_hook in html


def test_debug_routes_serve_lab_sigma_and_forge(tmp_path: Path) -> None:
    with TestClient(create_app(oplog_file=tmp_path / "routes.jsonl")) as client:
        home = client.get("/debug")
        sigma = client.get("/debug/sigma?stream=workflow&mode=projection")
        fallback = client.get("/debug/sigma?stream=conversation&mode=unknown")
        forge = client.get("/debug/forge")
        graphology_asset = client.get(
            "/debug/assets/graphology-0.25.4.umd.min.js"
        )
        sigma_asset = client.get("/debug/assets/sigma-2.4.0.min.js")

    assert home.status_code == 200
    assert 'href="/debug/sigma"' in home.text
    assert 'href="/debug/forge"' in home.text

    assert sigma.status_code == 200
    assert "{{" not in sigma.text and "{%" not in sigma.text
    assert 'window.__BUNDLE_GRAPH_TYPE__ = "workflow";' in sigma.text
    assert 'window.__INITIAL_MODE__ = "projection";' in sigma.text
    assert "window.__CDC_ENABLED__ = true;" in sigma.text
    assert "window.__CDC_WS_URL__ = null;" in sigma.text
    assert 'src="/debug/assets/graphology-0.25.4.umd.min.js"' in sigma.text
    assert 'src="/debug/assets/sigma-2.4.0.min.js"' in sigma.text

    assert fallback.status_code == 200
    assert 'window.__BUNDLE_GRAPH_TYPE__ = "conversation";' in fallback.text
    assert 'window.__INITIAL_MODE__ = "reify";' in fallback.text

    assert forge.status_code == 200
    assert 'data-testid="event-form"' in forge.text
    assert 'fetch("/ingest"' in forge.text

    assert graphology_asset.status_code == 200
    assert "graphology" in graphology_asset.text.lower()
    assert sigma_asset.status_code == 200
    assert "sigma" in sigma_asset.text.lower()


def test_bridge_ingest_replay_preserves_canonical_multi_endpoint_hyperedge(
    tmp_path: Path,
) -> None:
    events = [
        _event(
            seq=index,
            op="node.upsert",
            kind="node",
            entity_id=node_id,
            payload={"id": node_id, "label": node_id.upper(), "type": "entity"},
        )
        for index, node_id in enumerate(("source-a", "source-b", "target-a", "target-b"), 1)
    ]
    hyperedge_payload = {
        "id": "edge-many-to-many",
        "label": "Canonical many-to-many",
        "type": "relationship",
        "relation": "corroborates",
        "source_ids": ["source-a", "source-b"],
        "target_ids": ["target-a", "target-b"],
        "source_edge_ids": [],
        "target_edge_ids": [],
        "props": {"confidence": 0.93},
    }
    events.append(
        _event(
            seq=5,
            op="edge.upsert",
            kind="edge",
            entity_id=hyperedge_payload["id"],
            payload=hyperedge_payload,
        )
    )

    with TestClient(create_app(oplog_file=tmp_path / "replay.jsonl")) as client:
        response = client.post("/ingest", json={"events": events})
        assert response.status_code == 200, response.text
        assert response.json() == {"ok": True, "accepted": 5, "last_seq": 5}

        # Connect after ingest: received messages therefore exercise durable replay,
        # rather than only live in-process broadcast.
        with client.websocket_connect("/changes/ws?since=4&stream=knowledge") as ws:
            replayed = json.loads(ws.receive_text())

    assert replayed["seq"] == 5
    assert replayed["op"] == "edge.upsert"
    assert replayed["entity"] == {
        "kind": "edge",
        "id": "edge-many-to-many",
        "kg_graph_type": "knowledge",
        "url": None,
    }
    assert replayed["payload"] == hyperedge_payload
    assert replayed["payload"]["source_ids"] == ["source-a", "source-b"]
    assert replayed["payload"]["target_ids"] == ["target-a", "target-b"]
