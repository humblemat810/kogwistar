from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.ci_full]

playwright = pytest.importorskip("playwright.sync_api")


def _browser_errors(page) -> list[str]:
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
    page.on(
        "console",
        lambda msg: errors.append(f"console: {msg.text}")
        if msg.type == "error"
        else None,
    )
    return errors


def test_sigma_forge_visual_interaction_acceptance() -> None:
    """Human-visible flow: forge real CDC events, then interact with Sigma canvas.

    Set ``KOGWISTAR_VISUAL_ARTIFACTS`` to retain screenshots outside pytest's
    temporary directory for manual visual review.
    """

    from fastapi.testclient import TestClient

    from kogwistar.cdc.change_bridge import create_app

    with TemporaryDirectory() as temp_dir:
        app = create_app(oplog_file=Path(temp_dir) / "cdc.jsonl")

        with TestClient(app) as probe:
            assert probe.get("/debug/sigma").status_code == 200

        # TestClient cannot expose its ASGI socket to Chromium. The subprocess
        # harness is intentionally kept in the test so this flow matches humans.
        import socket
        import subprocess
        import sys
        import time
        from urllib.request import urlopen

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        live_server_url = f"http://127.0.0.1:{port}"
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "kogwistar.cdc.change_bridge",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--oplog-file",
                str(Path(temp_dir) / "browser-cdc.jsonl"),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(100):
                try:
                    with urlopen(f"{live_server_url}/openapi.json", timeout=0.2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.05)
            else:
                raise AssertionError("CDC bridge did not start")

            _run_browser_acceptance(live_server_url)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def _run_browser_acceptance(live_server_url: str) -> None:
    artifacts = Path(os.environ.get("KOGWISTAR_VISUAL_ARTIFACTS", ".tmp_sigma_visuals"))
    artifacts.mkdir(parents=True, exist_ok=True)

    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1500, "height": 920})
        viewer = context.new_page()
        forge = context.new_page()
        errors = _browser_errors(viewer) + _browser_errors(forge)

        viewer.goto(f"{live_server_url}/debug/sigma?stream=knowledge")
        forge.goto(f"{live_server_url}/debug/forge")
        viewer.wait_for_function(
            "window.__KOGWISTAR_SIGMA__ && document.querySelector('#cdcState').textContent === 'live'"
        )

        forge.get_by_test_id("scenario-generic").click()
        forge.get_by_test_id("request-status").wait_for(state="visible")
        forge.wait_for_function(
            "document.querySelector('[data-testid=request-status]').textContent === 'accepted'"
        )
        forge.screenshot(path=artifacts / "00-event-forge.png", full_page=True)
        viewer.wait_for_function(
            "window.__KOGWISTAR_SIGMA__.raw.nodes.size === 6 && window.__KOGWISTAR_SIGMA__.raw.edges.size === 5"
        )
        viewer.wait_for_function("window.__KOGWISTAR_SIGMA__.state.changed.size === 0")
        viewer.screenshot(path=artifacts / "01-reify-live.png", full_page=True)

        # Direction remains readable when arrowheads are lost in a dense region:
        # the overlay follows the D3 contract of higher value at the source and
        # lower value at the target. Verify actual canvas pixels for both an
        # incoming/source incidence and an outgoing/target incidence.
        gradient_luminance = viewer.evaluate(
            """() => {
              const lab=window.__KOGWISTAR_SIGMA__,canvas=document.querySelector('#shapeOverlay'),ctx=canvas.getContext('2d'),dpr=window.devicePixelRatio||1;
              function luminanceAt(key,fraction){
                const g=lab.state.graph,s=g.source(key),t=g.target(key),sp=lab.state.renderer.graphToViewport(g.getNodeAttributes(s)),tp=lab.state.renderer.graphToViewport(g.getNodeAttributes(t)),angle=Math.atan2(tp.y-sp.y,tp.x-sp.x),radius=g.getNodeAttribute(t,'kind')==='edge'?14:10,end={x:tp.x-Math.cos(angle)*radius,y:tp.y-Math.sin(angle)*radius},x=Math.round((sp.x+(end.x-sp.x)*fraction)*dpr),y=Math.round((sp.y+(end.y-sp.y)*fraction)*dpr);
                let best={a:-1,l:0};
                for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){const p=ctx.getImageData(x+dx,y+dy,1,1).data,l=.2126*p[0]+.7152*p[1]+.0722*p[2];if(p[3]>best.a)best={a:p[3],l};}
                return best.l;
              }
              return {
                source:[luminanceAt('e-observed-by:sn:obs-radar',.12),luminanceAt('e-observed-by:sn:obs-radar',.82)],
                target:[luminanceAt('e-observed-by:tn:analyst-mira',.12),luminanceAt('e-observed-by:tn:analyst-mira',.82)]
              };
            }"""
        )
        assert gradient_luminance["source"][0] > gradient_luminance["source"][1] + 100
        assert gradient_luminance["target"][0] > gradient_luminance["target"][1] + 100

        # CDC updates must not re-run a global force layout or reset the camera.
        # Measure actual viewport coordinates, not only graph-model coordinates.
        stable_before = viewer.evaluate(
            """() => {
              const lab=window.__KOGWISTAR_SIGMA__, ids=['n:analyst-mira','n:obs-radar','e:e-corroborates'];
              return {camera:lab.state.renderer.getCamera().getState(),points:Object.fromEntries(ids.map(id=>{const a=lab.state.graph.getNodeAttributes(id),p=lab.state.renderer.graphToViewport(a);return [id,{x:a.x,y:a.y,vx:p.x,vy:p.y}]}))};
            }"""
        )
        forge.get_by_test_id("entity-id").fill("incremental-signal")
        forge.get_by_test_id("entity-label").fill("Incremental signal")
        forge.get_by_test_id("send-event").click()
        viewer.wait_for_function(
            "window.__KOGWISTAR_SIGMA__.raw.nodes.has('incremental-signal')"
        )
        forge.get_by_test_id("op-edge-upsert").click()
        forge.get_by_test_id("entity-id").fill("e-incremental-link")
        forge.get_by_test_id("entity-label").fill("Incremental link")
        forge.get_by_test_id("source-node-ids").fill("analyst-mira")
        forge.get_by_test_id("target-node-ids").fill("incremental-signal")
        forge.get_by_test_id("send-event").click()
        viewer.wait_for_function(
            "window.__KOGWISTAR_SIGMA__.raw.edges.has('e-incremental-link')"
        )
        stable_after = viewer.evaluate(
            """() => {
              const lab=window.__KOGWISTAR_SIGMA__, ids=['n:analyst-mira','n:obs-radar','e:e-corroborates'];
              return {camera:lab.state.renderer.getCamera().getState(),points:Object.fromEntries(ids.map(id=>{const a=lab.state.graph.getNodeAttributes(id),p=lab.state.renderer.graphToViewport(a);return [id,{x:a.x,y:a.y,vx:p.x,vy:p.y}]}))};
            }"""
        )
        assert stable_after["camera"] == stable_before["camera"]
        for node_id, before in stable_before["points"].items():
            after = stable_after["points"][node_id]
            assert abs(after["x"] - before["x"]) < 1e-9
            assert abs(after["y"] - before["y"]) < 1e-9
            assert abs(after["vx"] - before["vx"]) < 0.25
            assert abs(after["vy"] - before["vy"]) < 0.25
        viewer.screenshot(
            path=artifacts / "01b-incremental-stability.png", full_page=True
        )

        viewer.get_by_test_id("mode-select").select_option("compact")
        assert "mixed projection" in viewer.get_by_test_id("mode-notice").inner_text()
        assert viewer.evaluate("window.__KOGWISTAR_SIGMA__.state.graph.hasNode('e:e-observed-by')") is False
        viewer.screenshot(path=artifacts / "02-compact.png", full_page=True)

        viewer.get_by_test_id("mode-select").select_option("projection")
        assert "lossy" in viewer.get_by_test_id("mode-notice").inner_text()
        assert viewer.evaluate("window.__KOGWISTAR_SIGMA__.state.graph.hasEdge('e-observed-by:obs-radar->analyst-mira')") is True
        viewer.screenshot(path=artifacts / "03-projection.png", full_page=True)

        viewer.get_by_test_id("mode-select").select_option("reify")
        viewer.get_by_test_id("relation-filter").select_option("corroborates")
        assert viewer.evaluate("window.__KOGWISTAR_SIGMA__.state.graph.hasNode('e:e-recommends')") is False
        viewer.get_by_test_id("relation-filter").select_option("")

        drag_before = viewer.evaluate(
            """() => {
              const lab=window.__KOGWISTAR_SIGMA__, id='n:obs-radar';
              const p=lab.state.renderer.graphToViewport(lab.state.graph.getNodeAttributes(id));
              const box=document.querySelector('#sigma').getBoundingClientRect();
              return {screenX:box.left+p.x,screenY:box.top+p.y,x:lab.state.graph.getNodeAttribute(id,'x'),y:lab.state.graph.getNodeAttribute(id,'y')};
            }"""
        )
        viewer.mouse.move(drag_before["screenX"], drag_before["screenY"])
        viewer.mouse.down()
        viewer.mouse.move(drag_before["screenX"] + 70, drag_before["screenY"] - 35, steps=8)
        viewer.mouse.up()
        drag_after = viewer.evaluate(
            """() => { const g=window.__KOGWISTAR_SIGMA__.state.graph; return {x:g.getNodeAttribute('n:obs-radar','x'),y:g.getNodeAttribute('n:obs-radar','y')}; }"""
        )
        assert abs(drag_after["x"] - drag_before["x"]) > 0.01 or abs(drag_after["y"] - drag_before["y"]) > 0.01

        viewer.evaluate("window.__KOGWISTAR_SIGMA__.selectNode('e:e-corroborates')")
        viewer.wait_for_function(
            "document.querySelector('[data-testid=inspector]').textContent.includes('Signals corroborate incident')"
        )
        viewer.screenshot(path=artifacts / "04-selected-hyperedge.png", full_page=True)

        viewer.get_by_test_id("search").fill("probe")
        viewer.get_by_test_id("zoom-in").click()
        viewer.get_by_test_id("zoom-out").click()
        viewer.get_by_test_id("camera-reset").click()

        forge.get_by_test_id("scenario-endpoint-first").click()
        forge.wait_for_function(
            "document.querySelector('[data-testid=request-status]').textContent === 'accepted'"
        )
        viewer.wait_for_function(
            "window.__KOGWISTAR_SIGMA__.raw.edges.has('e-early-arrival') && !window.__KOGWISTAR_SIGMA__.raw.nodes.get('future-signal-a')._placeholder"
        )
        viewer.get_by_test_id("search").fill("")
        viewer.evaluate("window.__KOGWISTAR_SIGMA__.selectNode('e:e-early-arrival')")
        viewer.wait_for_function("window.__KOGWISTAR_SIGMA__.state.changed.size === 0")
        viewer.screenshot(path=artifacts / "05-endpoint-ordering.png", full_page=True)

        seq_before_pause = viewer.evaluate("window.__KOGWISTAR_SIGMA__.raw.lastSeq")
        viewer.get_by_test_id("cdc-toggle").click()
        viewer.wait_for_function("document.querySelector('#cdcState').textContent === 'paused'")
        forge.get_by_test_id("scenario-burst").click()
        forge.wait_for_function(
            "document.querySelector('[data-testid=request-status]').textContent === 'accepted'"
        )
        assert viewer.evaluate("window.__KOGWISTAR_SIGMA__.raw.lastSeq") == seq_before_pause
        viewer.get_by_test_id("cdc-toggle").click()
        viewer.wait_for_function(
            "window.__KOGWISTAR_SIGMA__.raw.edges.has('e-burst-final') && window.__KOGWISTAR_SIGMA__.raw.lastSeq > %d"
            % seq_before_pause
        )
        viewer.wait_for_function("window.__KOGWISTAR_SIGMA__.state.changed.size === 0")
        viewer.screenshot(path=artifacts / "06-burst-replay.png", full_page=True)

        assert viewer.locator("#sigma canvas").count() >= 3
        assert "n ·" in viewer.locator("#visibleCount").inner_text()
        assert not errors, "browser errors:\n" + "\n".join(errors)
        browser.close()
