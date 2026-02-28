
"""
Developer debug harness (full script): start CDC bridge, generate CDC-enabled D3 bundles
(using graph_knowledge_engine.utils.kge_debug_dump.dump_d3_bundle), capture WS stream,
run pytest, and validate captured events.

This version generates THREE pages (conversation/workflow/knowledge) with optional
bridge-side stream filtering via `?stream=...`.

LOCAL DEV ONLY. Do NOT place under tests/.

Example:
  python debug_cdc_answer_workflow_v2_v4.py --open-browser \
    --pytest "-k test_answer_workflow_v2_runs_end_to_end -s" \
    --bridge-app "change_bridge_v2:app" \
    --template "d3.html"
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from graph_knowledge_engine.engine import EngineType
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_http_up(url: str, timeout_s: float = 6.0) -> None:
    import urllib.request

    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=0.7) as resp:
                _ = resp.read(50)
            return
        except Exception as e:
            last = e
            time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for server: {url} (last={last})")


def _wait_tcp_open(host: str, port: int, timeout_s: float = 16.0) -> None:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=0.7):
                return
        except Exception as e:
            last = e
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for TCP {host}:{port} (last={last})")


# -------------------------
# CDC capture + validation
# -------------------------

@dataclass
class CdcValidationReport:
    count: int
    first_seq: Optional[int]
    last_seq: Optional[int]
    non_monotonic: int
    missing_fields: int
    bad_types: int


def _validate_event_shape(ev: Dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(ev, dict):
        return False, "event is not a dict"
    if "op" not in ev:
        return False, "missing op"
    if "seq" not in ev:
        return False, "missing seq"
    if not isinstance(ev.get("op"), str):
        return False, "op not str"
    if not isinstance(ev.get("seq"), int):
        try:
            int(ev.get("seq"))
        except Exception:
            return False, "seq not int-like"
    return True, "ok"


def validate_cdc_stream_jsonl(path: Path) -> CdcValidationReport:
    if not path.exists():
        raise RuntimeError(f"CDC stream file not found: {path}")
    count = 0
    first_seq = None
    last_seq = None
    non_mono = 0
    missing = 0
    bad_types = 0

    prev_seq = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                ev = json.loads(line)
            except Exception:
                missing += 1
                continue
            ok, _ = _validate_event_shape(ev if isinstance(ev, dict) else {})
            if not ok:
                missing += 1
                continue
            try:
                seq = int(ev["seq"])
            except Exception:
                bad_types += 1
                continue
            if first_seq is None:
                first_seq = seq
            last_seq = seq
            if prev_seq is not None and seq <= prev_seq:
                non_mono += 1
            prev_seq = seq

    return CdcValidationReport(
        count=count,
        first_seq=first_seq,
        last_seq=last_seq,
        non_monotonic=non_mono,
        missing_fields=missing,
        bad_types=bad_types,
    )


async def _ws_capture_task(ws_url: str, out_path: Path, stop_evt: threading.Event) -> None:
    try:
        import websockets  # type: ignore
    except Exception as e:
        raise RuntimeError("websockets is required for CDC capture. Install: pip install websockets") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as out:
        async with websockets.connect(ws_url) as ws:

            async def keepalive() -> None:
                while not stop_evt.is_set():
                    try:
                        await ws.send("ping")
                    except Exception:
                        return
                    await asyncio.sleep(1.0)

            ka = asyncio.create_task(keepalive())
            try:
                while not stop_evt.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if isinstance(msg, (bytes, bytearray)):
                        try:
                            msg = msg.decode("utf-8", errors="replace")
                        except Exception:
                            continue
                    out.write(str(msg).strip() + "\n")
                    out.flush()
            finally:
                ka.cancel()
                with contextlib.suppress(Exception):
                    await ka


def start_ws_capture_in_thread(ws_url: str, out_path: Path) -> tuple[threading.Thread, threading.Event]:
    stop_evt = threading.Event()

    def runner() -> None:
        asyncio.run(_ws_capture_task(ws_url, out_path, stop_evt))

    t = threading.Thread(target=runner, daemon=True, name = "async th _ws_capture_task")
    t.start()
    return t, stop_evt


# -------------------------
# Servers
# -------------------------


def start_uvicorn(app_import: str, host: str, port: int, app_dir: str, log_path: Path) -> subprocess.Popen:
    """Start uvicorn and persist combined stdout/stderr to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)

    cmd = [
        sys.executable,
        os.path.join(*app_import.rsplit(':',1)[0].split('.'))+".py", #app_import,
        "--app-dir",
        app_dir,
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "info",
        "--access-log",
        "--reset-oplog"
    ]
    p = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
    # Keep handle alive so the fd doesn't get closed early.
    setattr(p, "_log_fh", log_fh)
    setattr(p, "_log_path", str(log_path))
    return p

def stop_proc(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(timeout=3)
    except Exception:
        p.kill()


def start_static_server(
    root_dir: Path, host: str, port: int
) -> tuple[threading.Thread, http.server.ThreadingHTTPServer]:
    root = str(root_dir.resolve())
    
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
    httpd = http.server.ThreadingHTTPServer((host, port), handler)

    t = threading.Thread(
        target=httpd.serve_forever,
        daemon=True,
        name="static cdc viewer http server",
    )
    t.start()
    return t, httpd


# -------------------------
# Bundle rendering via kge_debug_dump
# -------------------------

def dump_bundle(*, template_path: Path, out_html: Path, engine_type: EngineType, ws_url: str) -> None:
    try:
        from graph_knowledge_engine.utils.kge_debug_dump import dump_d3_bundle  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import graph_knowledge_engine.utils.kge_debug_dump.dump_d3_bundle. "
            "Run this from repo root with your venv activated."
        ) from e
    
    
    template_html = template_path.read_text(encoding="utf-8")
    dump_d3_bundle(
        engine=None,
        engine_type=engine_type,
        template_html=template_html,
        out_html=out_html,
        doc_id=None,
        mode="reify",
        insertion_method=None,
        bundle_meta={"note": "CDC debug bundle", "engine_type": engine_type, "ws": ws_url},
        cdc_enabled=True,
        cdc_ws_url=ws_url,
        embed_empty=True,
    )


def main() -> int:
    default_app_path = f"{str((Path(__file__).absolute().parent.parent.stem))}.cdc.change_bridge:app"
    default_app_dir = str(Path(__file__).absolute().parent.parent.parent)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge-app", default=default_app_path, help="Uvicorn app import path")
    ap.add_argument("--bridge-dir", default=default_app_dir, help="Uvicorn app import app dir")
    ap.add_argument("--bridge-host", default="127.0.0.1")
    ap.add_argument("--bridge-port", type=int, default=8787, help="0 = choose free port")
    ap.add_argument("--pytest", default="-k test_answer_workflow_v2_runs_end_to_end -s", help="Pytest args string")
    ap.add_argument("--repo-root", default=".", help="Repo root (where pytest is run)")
    ap.add_argument("--template", default="graph_knowledge_engine/templates/d3.html", help="Path to templates/d3.html")
    ap.add_argument("--out-dir", default=".cdc_debug", help="Output directory for bundle + logs")
    ap.add_argument("--open-browser", action="store_true", help="Open the CDC page automatically")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bridge_port = args.bridge_port or _find_free_port()
    bridge_http = f"http://{args.bridge_host}:{bridge_port}"
    ws_base = f"ws://{args.bridge_host}:{bridge_port}/changes/ws"

    stream_urls = {
        "conversation": ws_base + "?since=0&stream=conversation",
        "workflow": ws_base + "?since=0&stream=workflow",
        "knowledge": ws_base + "?since=0&stream=knowledge",
    }

    template_path = (repo_root / args.template).resolve()
    if not template_path.exists():
        raise RuntimeError(f"d3.html template not found: {template_path}")

    bundle_paths = {
        "conversation": out_dir / "conversation.bundle.cdc.html",
        "workflow": out_dir / "workflow.bundle.cdc.html",
        "knowledge": out_dir / "kg.bundle.cdc.html",
    }
    dump_bundle(template_path=template_path, out_html=bundle_paths["conversation"], engine_type="conversation", ws_url=stream_urls["conversation"])
    dump_bundle(template_path=template_path, out_html=bundle_paths["workflow"], engine_type="workflow", ws_url=stream_urls["workflow"])
    dump_bundle(template_path=template_path, out_html=bundle_paths["knowledge"], engine_type="knowledge", ws_url=stream_urls["knowledge"])

    page_port = _find_free_port()
    _, httpd = start_static_server(out_dir, "127.0.0.1", page_port)
    page_urls = {k: f"http://127.0.0.1:{page_port}/{p.name}" for k, p in bundle_paths.items()}

    print(f"[CDC] Output dir: {out_dir}")
    print(f"[CDC] Bridge ingest: {bridge_http}/ingest")
    for k in ("conversation", "workflow", "knowledge"):
        print(f"[CDC] {k} WS: {stream_urls[k]}")
        print(f"[CDC] {k} page: {page_urls[k]}")
    bridge_log_path = out_dir / "cdc_bridge.log"
    bridge = start_uvicorn(args.bridge_app, args.bridge_host, bridge_port, args.bridge_dir, bridge_log_path)
    try:
        _wait_tcp_open(args.bridge_host, page_port, timeout_s=10.0)
        _wait_http_up(f"{bridge_http}/docs", timeout_s=10.0)
    except Exception:
        if bridge.stdout:
            print("---- bridge logs ----")
            try:
                for _ in range(200):
                    line = bridge.stdout.readline()
                    if not line:
                        break
                    print(line.rstrip())
            except Exception:
                pass
            print("---- end bridge logs ----")
        stop_proc(bridge)
        raise

    cdc_log_path = out_dir / "cdc_stream.jsonl"
    _, stop_evt = start_ws_capture_in_thread(ws_base, cdc_log_path)

    if args.open_browser:
        for k in ("conversation", "workflow", "knowledge"):
            webbrowser.open(page_urls[k])

    env = os.environ.copy()
    env.setdefault("CDC_BRIDGE_ENDPOINT", bridge_http)
    env.setdefault("CDC_WS_URL", stream_urls["conversation"])
    env.setdefault("CDC_OUT_DIR", str(out_dir))
    env.setdefault("CDC_PAGE_CONVERSATION", page_urls["conversation"])
    env.setdefault("CDC_PAGE_WORKFLOW", page_urls["workflow"])
    env.setdefault("CDC_PAGE_KNOWLEDGE", page_urls["knowledge"])

    print(f"[pytest] running: pytest {args.pytest}")
    # test_answer_workflow_v2_runs_end_to_end
    p = subprocess.run([sys.executable, "-m", "pytest", *args.pytest.split()], cwd=str(repo_root), env=env)
    rc = int(p.returncode)

    stop_evt.set()
    time.sleep(0.3)
    httpd.shutdown()
    stop_proc(bridge)

    try:
        rep = validate_cdc_stream_jsonl(cdc_log_path)
        print("[CDC] validation:", rep)
        if rep.count == 0:
            print("[CDC] WARNING: captured 0 events. Is engine wired to CDC_BRIDGE_ENDPOINT?")
        if rep.non_monotonic:
            print(f"[CDC] WARNING: non-monotonic seq count={rep.non_monotonic}. Engine may reset seq on restart.")
        if rep.missing_fields or rep.bad_types:
            print(f"[CDC] WARNING: malformed events missing={rep.missing_fields} bad_types={rep.bad_types}")
    except Exception as e:
        print("[CDC] validation failed:", e)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
