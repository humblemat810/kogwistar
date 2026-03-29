from __future__ import annotations

import asyncio
import importlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import threading
import uuid
import socket
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import requests

pytest_plugins = ["tests.core._async_chroma_real"]
pytest.importorskip("fastapi")
pytest.importorskip("fastmcp")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient

import kogwistar.server_mcp_with_admin as server
from kogwistar.conversation.models import ConversationNode
from kogwistar.engine_core.models import Grounding, Span
from kogwistar.runtime.models import RunSuccess
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.runtime.runtime import WorkflowRuntime
from kogwistar.server.auth_middleware import claims_ctx
from kogwistar.server.chat_service import AnswerRunRequest
from kogwistar.server.chat_service import RuntimeRunRequest

pytestmark = pytest.mark.ci_full


class FakeEmbeddingFunction:
    @staticmethod
    def name() -> str:
        return "default"

    def __init__(self, dim: int = 3):
        self._dim = dim
        self.is_legacy = False

    def __call__(self, input):
        return [[0.01] * self._dim for _ in input]


@pytest.fixture(params=["chroma", "pg"], ids=["chroma-async", "pg-async"])
def async_backend_kind(request) -> str:
    return str(request.param)


def _reload_real_server_app(
    *,
    monkeypatch: pytest.MonkeyPatch,
    backend_kind: str,
    tmp_path: Path,
    request: pytest.FixtureRequest,
):
    debug_dir = Path.cwd() / ".tmp_runtime_sse_debug"
    debug_log = debug_dir / f"{backend_kind}.jsonl"
    if debug_log.exists():
        debug_log.unlink()

    monkeypatch.setenv("GKE_PERSIST_DIRECTORY", str(tmp_path / "server-data"))
    monkeypatch.setenv("GKE_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("AUTH_MODE", "dev")
    monkeypatch.setenv("JWT_ALG", "HS256")
    monkeypatch.setenv("JWT_SECRET", "kogwistar-test-secret")
    monkeypatch.setenv("KOGWISTAR_RUNTIME_SSE_DEBUG_LOG", str(debug_log))
    bridge_endpoint = _start_cdc_bridge(
        monkeypatch=monkeypatch,
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        request=request,
    )
    monkeypatch.setenv("CDC_PUBLISH_ENDPOINT", bridge_endpoint)

    if backend_kind == "chroma":
        pytest.importorskip("chromadb")
        chroma_server = request.getfixturevalue("real_chroma_server")
        monkeypatch.setenv("GKE_BACKEND", "chroma")
        monkeypatch.setenv("GKE_CHROMA_ASYNC", "1")
        monkeypatch.setenv("GKE_CHROMA_HOST", str(chroma_server.host))
        monkeypatch.setenv("GKE_CHROMA_PORT", str(chroma_server.port))
    elif backend_kind == "pg":
        pg_dsn = request.getfixturevalue("pg_dsn")
        if not pg_dsn:
            pytest.skip("async pg fixtures are unavailable in this environment")
        monkeypatch.setenv("GKE_BACKEND", "pg")
        monkeypatch.setenv("GKE_PG_ASYNC", "1")
        monkeypatch.setenv("GKE_PG_URL", str(pg_dsn))
    else:
        raise ValueError(f"unknown backend_kind: {backend_kind!r}")

    import kogwistar.server.auth_middleware as auth_middleware_mod
    import kogwistar.server.resources as server_resources_mod
    import kogwistar.server_mcp_with_admin as server_mod

    auth_middleware_mod = importlib.reload(auth_middleware_mod)
    server_resources_mod = importlib.reload(server_resources_mod)
    server_mod = importlib.reload(server_mod)

    globals()["server"] = server_mod
    globals()["server_resources"] = server_resources_mod
    return server_mod


@contextmanager
def _claims(role: str, ns: str, sub: str | None = None):
    claims = {"role": role, "ns": ns}
    if sub is not None:
        claims["sub"] = sub
    token = claims_ctx.set(claims)
    try:
        yield
    finally:
        claims_ctx.reset(token)


def _token_header(
    client: TestClient, *, role: str, ns: str, username: str = "tester"
) -> dict[str, str]:
    resp = client.post(
        "/auth/dev-token",
        json={"username": username, "role": role, "ns": ns},
    )
    resp.raise_for_status()
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def _token_header_http(
    session, base_url: str, *, role: str, ns: str, username: str = "tester"
) -> dict[str, str]:
    resp = session.post(
        f"{base_url}/auth/dev-token",
        json={"username": username, "role": role, "ns": ns},
        timeout=10.0,
    )
    resp.raise_for_status()
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def _make_slow_answer_runner(
    *,
    ready: threading.Event,
    release: threading.Event,
):
    def _runner(req: AnswerRunRequest) -> dict[str, Any]:
        req.publish("run.stage", {"stage": "retrieve"})
        req.publish(
            "reasoning.summary",
            {"stage": "retrieve", "summary": "Waiting in the long-running pipeline."},
        )
        ready.set()
        if not release.wait(timeout=30.0):
            raise AssertionError("run release event was not signaled")

        assistant_text = f"Assistant reply: {req.user_text}"
        assistant_turn_node_id = f"assistant|{uuid.uuid4().hex}"
        embedding = req.conversation_engine.iterative_defensive_emb(assistant_text)
        req.conversation_engine.write.add_node(
            ConversationNode(
                id=assistant_turn_node_id,
                label="Assistant turn",
                type="entity",
                summary=assistant_text,
                conversation_id=req.conversation_id,
                role="assistant",  # type: ignore[arg-type]
                turn_index=req.prev_turn_meta_summary.tail_turn_index + 1,
                properties={"content": assistant_text, "entity_type": "assistant_turn"},
                mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
                metadata={
                    "level_from_root": 0,
                    "entity_type": "assistant_turn",
                    "in_conversation_chain": True,
                    "in_ui_chain": True,
                },
                domain_id=None,
                canonical_entity_id=None,
                embedding=embedding.tolist() if hasattr(embedding, "tolist") else embedding,
            )
        )
        req.prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(
            assistant_text
        )
        req.prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
        req.prev_turn_meta_summary.tail_turn_index += 1
        return {
            "assistant_turn_node_id": assistant_turn_node_id,
            "assistant_text": assistant_text,
            "workflow_status": "succeeded",
        }

    return _runner


def _predicate_always_false(_workflow_info, _state, _last_result) -> bool:
    return False


def _make_looping_sleep_runtime_runner():
    def _runner(req: RuntimeRunRequest) -> dict[str, Any]:
        resolver = MappingStepResolver()

        @resolver.register("start")
        def _start(ctx):
            with ctx.state_write as state:
                state["sleep_ticks"] = int(ctx.state_view.get("sleep_ticks") or 0)
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"sleep_ticks": int(ctx.state_view.get("sleep_ticks") or 0)})],
            )

        @resolver.register("sleep")
        def _sleep(ctx):
            deps = dict(ctx.state_view.get("_deps") or {})
            publish = deps.get("publish")
            tick = int(ctx.state_view.get("sleep_ticks") or 0) + 1
            if callable(publish):
                publish(
                    "sleep.tick",
                    {
                        "workflow_id": ctx.workflow_id,
                        "workflow_node_id": ctx.workflow_node_id,
                        "tick": tick,
                    },
                )
            time.sleep(1.0)
            with ctx.state_write as state:
                state["sleep_ticks"] = tick
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"sleep_ticks": tick})],
            )

        @resolver.register("end")
        def _end(ctx):
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    (
                        "u",
                        {
                            "done": True,
                            "sleep_ticks": int(ctx.state_view.get("sleep_ticks") or 0),
                        },
                    )
                ],
            )

        initial_state = dict(req.initial_state or {})
        deps = initial_state.get("_deps")
        if not isinstance(deps, dict):
            deps = {}
        deps.setdefault("conversation_engine", req.conversation_engine)
        deps.setdefault("knowledge_engine", req.knowledge_engine)
        deps.setdefault("workflow_engine", req.workflow_engine)
        deps.setdefault("publish", req.publish)
        initial_state["_deps"] = deps

        def _cancel_requested(run_id: str) -> bool:
            _ = run_id
            return req.is_cancel_requested()

        runtime = WorkflowRuntime(
            workflow_engine=req.workflow_engine,
            conversation_engine=req.conversation_engine,
            step_resolver=resolver,
            predicate_registry={"always_false": _predicate_always_false},
            checkpoint_every_n_steps=1,
            max_workers=1,
            cancel_requested=_cancel_requested,
        )
        run_result = runtime.run(
            workflow_id=req.workflow_id,
            conversation_id=req.conversation_id,
            turn_node_id=req.turn_node_id,
            initial_state=initial_state,
            run_id=req.run_id,
        )
        final_state = dict(getattr(run_result, "final_state", {}) or {})
        final_state.pop("_deps", None)
        return {
            "workflow_status": str(getattr(run_result, "status", "succeeded") or "succeeded"),
            "final_state": final_state,
        }

    return _runner


def _register_looping_sleep_workflow(
    client: TestClient,
    *,
    workflow_id: str,
    headers: dict[str, str],
    designer_id: str = "tester",
) -> None:
    start_id = f"wf|{workflow_id}|start"
    sleep_id = f"wf|{workflow_id}|sleep"
    end_id = f"wf|{workflow_id}|end"

    for payload in (
        {
            "designer_id": designer_id,
            "node_id": start_id,
            "label": "Start",
            "op": "start",
            "start": True,
        },
        {
            "designer_id": designer_id,
            "node_id": sleep_id,
            "label": "Sleep",
            "op": "sleep",
        },
        {
            "designer_id": designer_id,
            "node_id": end_id,
            "label": "End",
            "op": "end",
            "terminal": True,
        },
    ):
        resp = client.post(f"/api/workflow/design/{workflow_id}/nodes", json=payload, headers=headers)
        resp.raise_for_status()

    for payload in (
        {
            "designer_id": designer_id,
            "edge_id": f"wf|{workflow_id}|e|start_sleep",
            "src": start_id,
            "dst": sleep_id,
            "relation": "wf_next",
            "is_default": True,
        },
        {
            "designer_id": designer_id,
            "edge_id": f"wf|{workflow_id}|e|sleep_end",
            "src": sleep_id,
            "dst": end_id,
            "relation": "wf_next",
            "predicate": "always_false",
            "priority": 0,
        },
        {
            "designer_id": designer_id,
            "edge_id": f"wf|{workflow_id}|e|sleep_sleep",
            "src": sleep_id,
            "dst": sleep_id,
            "relation": "wf_next",
            "is_default": True,
            "priority": 1,
        },
    ):
        resp = client.post(f"/api/workflow/design/{workflow_id}/edges", json=payload, headers=headers)
        resp.raise_for_status()


def _register_looping_sleep_workflow_http(
    session,
    base_url: str,
    *,
    workflow_id: str,
    headers: dict[str, str],
    designer_id: str = "tester",
) -> None:
    start_id = f"wf|{workflow_id}|start"
    sleep_id = f"wf|{workflow_id}|sleep"
    end_id = f"wf|{workflow_id}|end"

    for payload in (
        {
            "designer_id": designer_id,
            "node_id": start_id,
            "label": "Start",
            "op": "start",
            "start": True,
        },
        {
            "designer_id": designer_id,
            "node_id": sleep_id,
            "label": "Sleep",
            "op": "sleep",
        },
        {
            "designer_id": designer_id,
            "node_id": end_id,
            "label": "End",
            "op": "end",
            "terminal": True,
        },
    ):
        resp = session.post(
            f"{base_url}/api/workflow/design/{workflow_id}/nodes",
            json=payload,
            headers=headers,
            timeout=20.0,
        )
        assert resp.ok, (
            f"workflow node upsert failed for workflow_id={workflow_id}: "
            f"status={resp.status_code} body={resp.text}"
        )

    for payload in (
        {
            "designer_id": designer_id,
            "edge_id": f"wf|{workflow_id}|e|start_sleep",
            "src": start_id,
            "dst": sleep_id,
            "relation": "wf_next",
            "is_default": True,
        },
        {
            "designer_id": designer_id,
            "edge_id": f"wf|{workflow_id}|e|sleep_end",
            "src": sleep_id,
            "dst": end_id,
            "relation": "wf_next",
            "predicate": "always_false",
            "priority": 0,
        },
        {
            "designer_id": designer_id,
            "edge_id": f"wf|{workflow_id}|e|sleep_sleep",
            "src": sleep_id,
            "dst": sleep_id,
            "relation": "wf_next",
            "is_default": True,
            "priority": 1,
        },
    ):
        resp = session.post(
            f"{base_url}/api/workflow/design/{workflow_id}/edges",
            json=payload,
            headers=headers,
            timeout=20.0,
        )
        assert resp.ok, (
            f"workflow edge upsert failed for workflow_id={workflow_id}: "
            f"status={resp.status_code} body={resp.text}"
        )


def _stream_sse_events(
    session,
    *,
    base_url: str,
    path: str,
    headers: dict[str, str],
    seen_five_sleep_events: threading.Event,
    collected: list[tuple[str, dict[str, Any]]],
    stream_errors: list[BaseException],
    debug_log_path: Path | None = None,
) -> None:
    _stream_sse_events_until_counts(
        session,
        base_url=base_url,
        path=path,
        headers=headers,
        target_counts={"sleep.tick": 5},
        target_event=seen_five_sleep_events,
        collected=collected,
        stream_errors=stream_errors,
        debug_log_path=debug_log_path,
    )


def _stream_sse_events_until_counts(
    session,
    *,
    base_url: str,
    path: str,
    headers: dict[str, str],
    target_counts: dict[str, int],
    target_event: threading.Event,
    collected: list[tuple[str, dict[str, Any]]],
    stream_errors: list[BaseException],
    debug_log_path: Path | None = None,
    stop_after_event_types: set[str] | None = None,
) -> None:
    current_event: str | None = None
    event_counts: dict[str, int] = defaultdict(int)
    try:
        _append_case_debug_log(
            debug_log_path,
            {
                "ts_ms": int(time.time() * 1000),
                "stage": "sse_open",
                "path": path,
                "target_counts": target_counts,
            },
        )
        with session.get(
            f"{base_url}{path}",
            headers=headers,
            stream=True,
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("event: "):
                    current_event = line.split(": ", 1)[1]
                    _append_case_debug_log(
                        debug_log_path,
                        {
                            "ts_ms": int(time.time() * 1000),
                            "stage": "sse_event",
                            "path": path,
                            "event_type": current_event,
                        },
                    )
                elif line.startswith("data: ") and current_event is not None:
                    payload = json.loads(line.split(": ", 1)[1])
                    collected.append((current_event, payload))
                    event_counts[current_event] += 1
                    _append_case_debug_log(
                        debug_log_path,
                        {
                            "ts_ms": int(time.time() * 1000),
                            "stage": "sse_data",
                            "path": path,
                            "event_type": current_event,
                            "event_count": event_counts[current_event],
                            "payload": payload,
                        },
                    )
                    if all(
                        event_counts[event_type] >= count
                        for event_type, count in target_counts.items()
                    ):
                        target_event.set()
                        _append_case_debug_log(
                            debug_log_path,
                            {
                                "ts_ms": int(time.time() * 1000),
                                "stage": "sse_target_reached",
                                "path": path,
                                "event_counts": dict(event_counts),
                            },
                        )
                    if (
                        stop_after_event_types is not None
                        and current_event in stop_after_event_types
                    ):
                        _append_case_debug_log(
                            debug_log_path,
                            {
                                "ts_ms": int(time.time() * 1000),
                                "stage": "sse_stop_after_event",
                                "path": path,
                                "event_type": current_event,
                            },
                        )
                        return
    except BaseException as exc:  # pragma: no cover - surfaced by the test
        stream_errors.append(exc)
        _append_case_debug_log(
            debug_log_path,
            {
                "ts_ms": int(time.time() * 1000),
                "stage": "sse_error",
                "path": path,
                "error": repr(exc),
            },
        )


def _wait_for_status_http(
    session,
    base_url: str,
    run_id: str,
    headers: dict[str, str],
    expected: set[str],
    *,
    path_template: str = "/api/workflow/runs/{run_id}",
) -> dict[str, Any]:
    deadline = time.time() + 30.0
    while time.time() < deadline:
        resp = session.get(
            f"{base_url}{path_template.format(run_id=run_id)}",
            headers=headers,
            timeout=10.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload["status"] in expected:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Run {run_id} did not reach one of {sorted(expected)}")


def _wait_for_conversation_tail_http(
    session,
    base_url: str,
    conversation_id: str,
    headers: dict[str, str],
    *,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            resp = session.get(
                f"{base_url}/api/conversations/{conversation_id}",
                headers=headers,
                timeout=10.0,
            )
            if resp.ok:
                data = resp.json()
                if str(data.get("tail_node_id") or ""):
                    return data
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.25)
    raise AssertionError(
        "conversation tail never became visible after create; "
        f"conversation_id={conversation_id!r} base_url={base_url!r} "
        f"last_error={last_error!r}"
    )


def _append_case_debug_log(log_path: Path | None, record: dict[str, Any]) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str))
        fh.write("\n")


def _wait_for_run_events_poll_http(
    session,
    base_url: str,
    run_id: str,
    headers: dict[str, str],
    expected_event_types: set[str],
    *,
    after_seq: int = 0,
    limit: int = 20,
    path_template: str = "/api/runs/{run_id}/events/poll",
    debug_log_path: Path | None = None,
) -> dict[str, Any]:
    deadline = time.time() + 30.0
    last: dict[str, Any] | None = None
    while time.time() < deadline:
        resp = session.get(
            f"{base_url}{path_template.format(run_id=run_id)}",
            params={"after_seq": int(after_seq), "limit": int(limit)},
            headers=headers,
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()
        events = data.get("events", [])
        event_names = [
            str(evt.get("event_type"))
            for evt in events
            if isinstance(evt, dict) and evt.get("event_type")
        ]
        _append_case_debug_log(
            debug_log_path,
            {
                "ts_ms": int(time.time() * 1000),
                "stage": "poll",
                "run_id": run_id,
                "after_seq": int(after_seq),
                "limit": int(limit),
                "event_names": event_names,
                "expected": sorted(expected_event_types),
            },
        )
        if expected_event_types.issubset(set(event_names)):
            return data
        last = data if isinstance(data, dict) else None
        time.sleep(0.25)
    raise AssertionError(
        f"run {run_id} did not expose expected events {sorted(expected_event_types)} "
        f"before timeout; last poll={last}"
    )


def _wait_for_workflow_trace_rows(trace_db: Path, *, timeout_s: float = 20.0) -> int:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        if trace_db.exists():
            try:
                with sqlite3.connect(trace_db) as conn:
                    row = conn.execute("SELECT COUNT(*) FROM wf_trace_events").fetchone()
                count = int(row[0] if row else 0)
                if count > 0:
                    return count
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        time.sleep(0.1)
    raise AssertionError(
        f"workflow trace DB did not record rows in time: {trace_db}\n"
        f"last_error={last_error!r}"
    )


def _manual_server_base_url() -> str:
    return os.getenv("KOGWISTAR_MANUAL_SERVER_BASE_URL", "http://127.0.0.1:28110")


def _manual_cdc_oplog() -> Path:
    return Path(
        os.getenv("KOGWISTAR_MANUAL_CDC_OPLOG", ".cdc_debug/data/cdc_oplog.jsonl")
    )


def _wait_for_manual_health(
    session: requests.Session, base_url: str, *, timeout_s: float = 5.0
) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            resp = session.get(f"{base_url}/health", timeout=1.0)
            if resp.ok:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.25)
    pytest.skip(
        f"manual server at {base_url} is not running. "
        f"Start the manual debug server or set KOGWISTAR_MANUAL_SERVER_BASE_URL. "
        f"Last error: {last_error!r}"
    )


def _post_json_with_case_log(
    session: requests.Session,
    url: str,
    *,
    json_body: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    case_log: Path,
    label: str,
):
    case_log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with case_log.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "label": f"{label}.start",
                    "url": url,
                    "timeout": timeout,
                    "body_keys": sorted(json_body.keys()),
                    "started_at": started,
                },
                sort_keys=True,
            )
            + "\n"
        )
    try:
        resp = session.post(url, json=json_body, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        with case_log.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "label": f"{label}.error",
                        "url": url,
                        "elapsed_s": round(time.time() - started, 3),
                        "error": repr(exc),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        raise
    with case_log.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "label": f"{label}.done",
                    "url": url,
                    "status_code": resp.status_code,
                    "elapsed_s": round(time.time() - started, 3),
                },
                sort_keys=True,
            )
            + "\n"
        )
    return resp


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_cdc_bridge(
    *,
    monkeypatch: pytest.MonkeyPatch | None,
    backend_kind: str,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> str:
    host = "127.0.0.1"
    port = _pick_free_port()
    oplog_file = tmp_path / f"{backend_kind}.cdc_oplog.jsonl"
    log_file = Path.cwd() / ".tmp_runtime_sse_debug" / f"{backend_kind}.cdc_bridge.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.unlink()

    cmd = [
        sys.executable,
        "-m",
        "kogwistar.cdc.change_bridge",
        "--host",
        host,
        "--port",
        str(port),
        "--oplog-file",
        str(oplog_file),
        "--reset-oplog",
        "--log-level",
        "warning",
    ]
    log_buf: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def _reader() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            stripped = line.rstrip()
            log_buf.append(stripped)
            try:
                with log_file.open("a", encoding="utf-8") as fh:
                    fh.write(stripped + "\n")
            except Exception:
                pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    def _cleanup() -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15.0)
        t.join(timeout=5.0)

    request.addfinalizer(_cleanup)

    endpoint = f"http://{host}:{port}"
    deadline = time.time() + 30.0
    last_err: Exception | None = None
    with requests.Session() as session:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    "CDC bridge exited before becoming healthy.\n"
                    f"exit={proc.returncode}\n"
                    + "\n".join(log_buf[-80:])
                )
            try:
                health = session.get(f"{endpoint}/openapi.json", timeout=1.5)
                if health.ok:
                    if monkeypatch is not None:
                        monkeypatch.setenv("CDC_PUBLISH_ENDPOINT", endpoint)
                    return endpoint
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            time.sleep(0.2)

    raise RuntimeError(
        f"CDC bridge at {endpoint} did not become healthy.\n"
        f"last_error={last_err!r}\n"
        f"log_tail=\n" + "\n".join(log_buf[-80:])
    )


def _wait_for_cdc_oplog_entries(oplog_file: Path, *, min_entries: int = 1, timeout_s: float = 20.0) -> int:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        if oplog_file.exists():
            try:
                with oplog_file.open("r", encoding="utf-8") as fh:
                    line_count = sum(1 for line in fh if line.strip())
                entry_count = max(0, line_count - 1)
                if entry_count >= min_entries:
                    return entry_count
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        time.sleep(0.1)
    raise AssertionError(
        f"CDC oplog did not record {min_entries} entries in time: {oplog_file}\n"
        f"last_error={last_error!r}"
    )


def _count_cdc_oplog_entries(oplog_file: Path) -> int:
    if not oplog_file.exists():
        return 0
    with oplog_file.open("r", encoding="utf-8") as fh:
        line_count = sum(1 for line in fh if line.strip())
    return max(0, line_count - 1)


@contextmanager
def _real_server_base_url(
    *,
    backend_kind: str,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    runtime_runner_import: str | None = None,
):
    import requests

    port = _pick_free_port()
    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"

    debug_dir = Path.cwd() / ".tmp_runtime_sse_debug"
    debug_log = debug_dir / f"{backend_kind}.jsonl"
    server_log = debug_dir / f"{backend_kind}-server.log"
    if debug_log.exists():
        debug_log.unlink()
    if server_log.exists():
        server_log.unlink()

    env = os.environ.copy()
    env["GKE_PERSIST_DIRECTORY"] = str(tmp_path / "server-data")
    env["GKE_INDEX_DIR"] = str(tmp_path / "index")
    env["AUTH_MODE"] = "dev"
    env["JWT_ALG"] = "HS256"
    env["JWT_SECRET"] = "kogwistar-test-secret"
    env["KOGWISTAR_RUNTIME_SSE_DEBUG_LOG"] = str(debug_log)
    if runtime_runner_import:
        env["KOGWISTAR_TEST_RUNTIME_RUNNER_IMPORT"] = str(runtime_runner_import)
    env["CDC_PUBLISH_ENDPOINT"] = _start_cdc_bridge(
        monkeypatch=None,
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        request=request,
    )
    if backend_kind == "chroma":
        pytest.importorskip("chromadb")
        chroma_server = request.getfixturevalue("real_chroma_server")
        env["GKE_BACKEND"] = "chroma"
        env["GKE_CHROMA_ASYNC"] = "1"
        env["GKE_CHROMA_HOST"] = str(chroma_server.host)
        env["GKE_CHROMA_PORT"] = str(chroma_server.port)
    elif backend_kind == "pg":
        pg_dsn = request.getfixturevalue("pg_dsn")
        if not pg_dsn:
            pytest.skip("async pg fixtures are unavailable in this environment")
        env["GKE_BACKEND"] = "pg"
        env["GKE_PG_ASYNC"] = "1"
        env["GKE_PG_URL"] = str(pg_dsn)
    else:
        raise ValueError(f"unknown backend_kind: {backend_kind!r}")

    port_file = tmp_path / f"{backend_kind}.port"
    env["PORT"] = str(port)
    env["HOST"] = host

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "kogwistar.server_mcp_with_admin:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    log_buf: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def _reader() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            stripped = line.rstrip()
            log_buf.append(stripped)
            try:
                with server_log.open("a", encoding="utf-8") as fh:
                    fh.write(stripped + "\n")
            except Exception:
                pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    try:
        deadline = time.time() + 90.0
        last_err: Exception | None = None
        ready = False
        with requests.Session() as session:
            while time.time() < deadline:
                if proc.poll() is not None:
                    joined = "\n".join(log_buf)
                    raise RuntimeError(
                        f"server exited before becoming healthy (exit={proc.returncode}).\n{joined}"
                    )
                try:
                    health = session.get(f"{base_url}/health", timeout=1.5)
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    time.sleep(0.25)
                    continue
                if health.ok:
                    ready = True
                    break
                time.sleep(0.25)
        if not ready:
            raise RuntimeError(
                f"server at {base_url} did not become healthy.\n"
                f"last_error={last_err!r}\n"
                f"log_tail=\n" + "\n".join(log_buf[-80:])
            )
        port_file.write_text(str(port), encoding="utf-8")
        yield base_url
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15.0)
        t.join(timeout=5.0)


def test_chat_rest_events_poll_sees_live_updates_for_async_backends(
    monkeypatch, async_backend_kind, tmp_path, request
):
    server_mod = _reload_real_server_app(
        monkeypatch=monkeypatch,
        backend_kind=async_backend_kind,
        tmp_path=tmp_path,
        request=request,
    )
    ready = threading.Event()
    release = threading.Event()
    server_mod.chat_service.get().answer_runner = _make_slow_answer_runner(
        ready=ready, release=release
    )

    with TestClient(server_mod.app) as client:
        rw_headers = _token_header(client, role="rw", ns="conversation")
        ro_headers = _token_header(client, role="ro", ns="conversation")

        created = client.post(
            "/api/conversations", json={"user_id": f"u-{async_backend_kind}"}, headers=rw_headers
        )
        assert created.status_code == 200, created.text
        conversation_id = str(created.json()["conversation_id"])

        submit = client.post(
            f"/api/conversations/{conversation_id}/turns:answer",
            json={"user_id": f"u-{async_backend_kind}", "text": "hello from the live poll test"},
            headers=rw_headers,
        )
        assert submit.status_code == 202
        run_id = str(submit.json()["run_id"])

        assert ready.wait(timeout=20.0), "runner did not reach its live-update checkpoint"

        start = time.monotonic()
        polled = client.get(
            f"/api/runs/{run_id}/events/poll?after_seq=0&limit=20",
            headers=ro_headers,
        )
        elapsed = time.monotonic() - start
        polled.raise_for_status()
        assert elapsed < 2.0

        payload = polled.json()
        event_names = [evt["event_type"] for evt in payload["events"]]
        assert payload["run_id"] == run_id
        assert "run.created" in event_names
        assert "run.started" in event_names
        assert "run.stage" in event_names
        assert "reasoning.summary" in event_names

        mid_run = client.get(f"/api/runs/{run_id}", headers=ro_headers)
        mid_run.raise_for_status()
        assert mid_run.json()["status"] == "running"

        release.set()
        deadline = time.time() + 20.0
        while time.time() < deadline:
            final_run = client.get(f"/api/runs/{run_id}", headers=ro_headers)
            final_run.raise_for_status()
            if final_run.json()["status"] == "succeeded":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("run did not reach succeeded status")


@pytest.mark.asyncio
async def test_mcp_run_events_sees_live_updates_for_async_backends(
    monkeypatch, async_backend_kind, tmp_path, request
):
    server_mod = _reload_real_server_app(
        monkeypatch=monkeypatch,
        backend_kind=async_backend_kind,
        tmp_path=tmp_path,
        request=request,
    )
    ready = threading.Event()
    release = threading.Event()
    server_mod.chat_service.get().answer_runner = _make_slow_answer_runner(
        ready=ready, release=release
    )

    with TestClient(server_mod.app) as client:
        rw_headers = _token_header(client, role="rw", ns="conversation")
        created = client.post(
            "/api/conversations", json={"user_id": f"u-{async_backend_kind}"}, headers=rw_headers
        )
        assert created.status_code == 200, created.text
        conversation_id = str(created.json()["conversation_id"])

        submit = client.post(
            f"/api/conversations/{conversation_id}/turns:answer",
            json={"user_id": f"u-{async_backend_kind}", "text": "hello from the mcp live test"},
            headers=rw_headers,
        )
        assert submit.status_code == 202
        run_id = str(submit.json()["run_id"])

        assert ready.wait(timeout=20.0), "runner did not reach its live-update checkpoint"

        with _claims("ro", "workflow"):
            result = await server_mod.mcp.call_tool(
                "workflow.run_events",
                {"run_id": run_id, "after_seq": 0, "limit": 20},
            )

        data = getattr(result, "structuredContent", None) or getattr(
            result, "structured_content", None
        )
        if data is None and hasattr(result, "model_dump"):
            data = result.model_dump()
        assert isinstance(data, dict)
        event_names = [evt["event_type"] for evt in data["events"]]
        assert data["run_id"] == run_id
        assert "run.created" in event_names
        assert "run.started" in event_names
        assert "run.stage" in event_names
        assert "reasoning.summary" in event_names

        release.set()
        deadline = time.time() + 20.0
        while time.time() < deadline:
            with _claims("ro", "conversation"):
                status = await server_mod.mcp.call_tool(
                    "conversation.run_status", {"run_id": run_id}
                )
            status_data = getattr(status, "structuredContent", None) or getattr(
                status, "structured_content", None
            )
            if status_data is None and hasattr(status, "model_dump"):
                status_data = status.model_dump()
            if isinstance(status_data, dict) and status_data.get("status") == "succeeded":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("run did not reach succeeded status")


@pytest.mark.asyncio
async def test_workflow_runtime_sse_cancel_after_sleep_ticks_for_async_backends(
    monkeypatch, async_backend_kind, tmp_path, request
):
    with _real_server_base_url(
        backend_kind=async_backend_kind,
        tmp_path=tmp_path,
        request=request,
        runtime_runner_import=(
            "tests.server._runtime_runner_overrides:make_looping_sleep_runtime_runner"
        ),
    ) as base_url, requests.Session() as api_session, requests.Session() as sse_session:
        debug_log_path = Path.cwd() / ".tmp_runtime_sse_debug" / f"{async_backend_kind}.jsonl"
        cdc_oplog = tmp_path / f"{async_backend_kind}.cdc_oplog.jsonl"
        conv_rw = _token_header_http(api_session, base_url, role="rw", ns="conversation")
        wf_rw = _token_header_http(api_session, base_url, role="rw", ns="workflow")
        wf_ro = _token_header_http(api_session, base_url, role="ro", ns="workflow")

        created = api_session.post(
            f"{base_url}/api/conversations",
            json={"user_id": f"u-wf-sse-{async_backend_kind}"},
            headers=conv_rw,
            timeout=20.0,
        )
        assert created.status_code == 200, created.text
        conversation_id = str(created.json()["conversation_id"])
        _wait_for_cdc_oplog_entries(cdc_oplog, min_entries=1, timeout_s=20.0)

        workflow_id = f"wf.runtime.sleep.loop.{async_backend_kind}.{uuid.uuid4().hex}"
        _register_looping_sleep_workflow_http(
            api_session, base_url, workflow_id=workflow_id, headers=wf_rw
        )

        submitted = api_session.post(
            f"{base_url}/api/workflow/runs",
            json={
                "workflow_id": workflow_id,
                "conversation_id": conversation_id,
                "initial_state": {"seed": async_backend_kind},
            },
            headers=wf_rw,
            timeout=20.0,
        )
        assert submitted.status_code == 202, submitted.text
        run_id = str(submitted.json()["run_id"])

        seen_five_sleep_events = threading.Event()
        collected: list[tuple[str, dict[str, Any]]] = []
        stream_errors: list[BaseException] = []
        stream_thread = threading.Thread(
            target=_stream_sse_events,
            kwargs={
                "session": sse_session,
                "base_url": base_url,
                "path": f"/api/workflow/runs/{run_id}/events",
                "headers": wf_ro,
                "seen_five_sleep_events": seen_five_sleep_events,
                "collected": collected,
                "stream_errors": stream_errors,
                "debug_log_path": debug_log_path,
            },
            daemon=True,
            name=f"sse-stream-{async_backend_kind}",
        )
        stream_thread.start()
        deadline = time.time() + 45.0
        while time.time() < deadline:
            if seen_five_sleep_events.is_set():
                break
            if stream_errors:
                raise AssertionError(f"SSE stream failed before threshold: {stream_errors!r}")
            await asyncio.sleep(0.1)
        else:
            raise AssertionError(
                "did not observe five sleep.tick SSE events in time; "
                f"stream_errors={stream_errors!r}; debug_log_path={debug_log_path}"
            )

        cancelling = await asyncio.to_thread(
            api_session.post,
            f"{base_url}/api/workflow/runs/{run_id}/cancel",
            headers=wf_rw,
            timeout=20.0,
        )
        assert cancelling.status_code == 202, cancelling.text
        assert cancelling.json()["status"] == "cancelling"

        stream_thread.join(timeout=20.0)
        assert not stream_thread.is_alive(), "SSE stream did not close after cancellation"
        assert not stream_errors, f"SSE stream failed: {stream_errors!r}"

        final_run = await asyncio.to_thread(
            _wait_for_status_http,
            api_session,
            base_url,
            run_id,
            wf_ro,
            {"cancelled"},
            path_template="/api/workflow/runs/{run_id}",
        )
        assert final_run["status"] == "cancelled"

        event_names = [event_type for event_type, _payload in collected]
        assert event_names.count("sleep.tick") >= 5
        assert "run.cancelling" in event_names
        assert "run.cancelled" in event_names

        trace_db = tmp_path / "server-data" / "workflow" / "wf_trace.sqlite"
        row_count = _wait_for_workflow_trace_rows(trace_db)
        assert row_count > 0, f"workflow trace DB is empty: {trace_db}"


@pytest.mark.asyncio
async def test_manual_async_server_streams_busy_answer_run_events():
    base_url = _manual_server_base_url()
    cdc_oplog = _manual_cdc_oplog()
    case_log = Path.cwd() / ".tmp_runtime_sse_debug" / "manual_answer_run.jsonl"
    if case_log.exists():
        case_log.unlink()
    if not cdc_oplog.exists():
        pytest.skip(f"manual CDC oplog not found: {cdc_oplog}")

    busy_prompt = (
        "Please answer with a detailed multi-paragraph response about why streaming "
        "server-sent events is useful for debugging, and keep the explanation long "
        "enough that the run remains active for a noticeable amount of time."
    )

    with requests.Session() as api_session, requests.Session() as sse_session:
        _wait_for_manual_health(api_session, base_url, timeout_s=5.0)
        conv_rw = _token_header_http(api_session, base_url, role="rw", ns="conversation")
        conv_ro = _token_header_http(api_session, base_url, role="ro", ns="conversation")

        cdc_before = _count_cdc_oplog_entries(cdc_oplog)

        created = _post_json_with_case_log(
            api_session,
            f"{base_url}/api/conversations",
            json_body={"user_id": "manual-sse-busy-run"},
            headers=conv_rw,
            timeout=60.0,
            case_log=case_log,
            label="manual_create_conversation",
        )
        assert created.status_code == 200, created.text
        conversation_id = str(created.json()["conversation_id"])
        _wait_for_conversation_tail_http(
            api_session,
            base_url,
            conversation_id,
            conv_ro,
            timeout_s=30.0,
        )

        cdc_after_create = _wait_for_cdc_oplog_entries(
            cdc_oplog, min_entries=cdc_before + 1, timeout_s=30.0
        )
        assert cdc_after_create >= cdc_before + 1

        submitted = _post_json_with_case_log(
            api_session,
            f"{base_url}/api/conversations/{conversation_id}/turns:answer",
            json_body={"user_id": "manual-sse-busy-run", "text": busy_prompt},
            headers=conv_rw,
            timeout=60.0,
            case_log=case_log,
            label="manual_submit_answer",
        )
        assert submitted.status_code == 202, submitted.text
        run_id = str(submitted.json()["run_id"])

        poll_data = _wait_for_run_events_poll_http(
            api_session,
            base_url,
            run_id,
            conv_ro,
            {"run.started"},
            after_seq=0,
            limit=20,
            path_template="/api/runs/{run_id}/events/poll",
            debug_log_path=case_log,
        )
        poll_event_names = [
            str(evt.get("event_type"))
            for evt in (poll_data.get("events") or [])
            if isinstance(evt, dict) and evt.get("event_type")
        ]
        assert "run.started" in poll_event_names, poll_event_names

        run_started = threading.Event()
        collected: list[tuple[str, dict[str, Any]]] = []
        stream_errors: list[BaseException] = []
        stream_thread = threading.Thread(
            target=_stream_sse_events_until_counts,
            kwargs={
                "session": sse_session,
                "base_url": base_url,
                "path": f"/api/runs/{run_id}/events",
                "headers": conv_ro,
                "target_counts": {"run.started": 1},
                "target_event": run_started,
                "collected": collected,
                "stream_errors": stream_errors,
                "debug_log_path": case_log,
                "stop_after_event_types": {
                    "run.completed",
                    "run.failed",
                    "run.cancelled",
                },
            },
            daemon=True,
            name="manual-sse-busy-run",
        )
        stream_thread.start()

        saw_started = await asyncio.to_thread(run_started.wait, 20.0)
        assert saw_started, (
            "manual SSE stream did not surface run.started even though the poll "
            f"endpoint did; poll_event_names={poll_event_names}; "
            f"case_log={case_log}"
        )

        running_state = await asyncio.to_thread(
            _wait_for_status_http,
            api_session,
            base_url,
            run_id,
            conv_ro,
            {"running", "succeeded"},
            path_template="/api/runs/{run_id}",
        )
        assert running_state["status"] in {"running", "succeeded"}

        final_state = await asyncio.to_thread(
            _wait_for_status_http,
            api_session,
            base_url,
            run_id,
            conv_ro,
            {"succeeded"},
            path_template="/api/runs/{run_id}",
        )
        assert final_state["status"] == "succeeded"

        stream_thread.join(timeout=20.0)
        assert not stream_thread.is_alive(), "manual SSE stream did not close"
        assert not stream_errors, f"manual SSE stream failed: {stream_errors!r}"

        event_names = [event_type for event_type, _payload in collected]
        assert "run.started" in event_names
        assert "run.completed" in event_names
        assert event_names.index("run.started") < event_names.index("run.completed")

        cdc_after_run = _wait_for_cdc_oplog_entries(
            cdc_oplog, min_entries=cdc_after_create + 1, timeout_s=60.0
        )
        assert cdc_after_run > cdc_after_create
        _append_case_debug_log(
            case_log,
            {
                "ts_ms": int(time.time() * 1000),
                "stage": "manual_done",
                "run_id": run_id,
                "event_names": event_names,
                "poll_event_names": poll_event_names,
            },
        )
