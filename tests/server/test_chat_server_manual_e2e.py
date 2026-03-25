from __future__ import annotations
""" test_chat_server_manual_e2e.py
"""
import json
import os
import time
from typing import Any

import pytest
import requests
from jose import jwt

from kogwistar.conversation.agentic_answering_design import (
    AGENTIC_ANSWERING_WORKFLOW_ID,
    DEBUG_RAG_WORKFLOW_ID,
    agentic_answering_expected_ops,
    build_agentic_answering_backend_payload,
    build_agentic_answering_frontend_payload,
)
from kogwistar.runtime.models import WorkflowDesignArtifact
from tests._helpers.span_consistent_seed import build_span_consistent_debug_rag_seed

pytest.importorskip("fastapi")
pytest.importorskip("fastmcp")
pytest.importorskip("sqlalchemy")

pytestmark = [pytest.mark.manual]


def _manual_base_url() -> str:
    return os.environ.get("KOGWISTAR_MANUAL_BASE_URL", "http://127.0.0.1:28110")


def _wait_for_health(
    session: requests.Session, base_url: str, timeout_s: float = 5.0
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
    raise RuntimeError(
        f"Manual server at {base_url} is not healthy. "
        f"Start `knowledge-mcp` first or set KOGWISTAR_MANUAL_BASE_URL. "
        f"Last error: {last_error}"
    )


def _dev_token(
    session: requests.Session, base_url: str, *, role: str, ns: str = "conversation"
) -> tuple[str, str]:
    username = "manual-e2e"
    resp = session.post(
        f"{base_url}/auth/dev-token",
        json={"username": username, "role": role, "ns": ns},
        timeout=10.0,
    )
    resp.raise_for_status()
    token = str(resp.json()["token"])
    subject = str(jwt.get_unverified_claims(token).get("sub") or username)
    return token, subject


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _wait_for_run_terminal(
    session: requests.Session,
    base_url: str,
    run_id: str,
    headers: dict[str, str],
    *,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = session.get(
                f"{base_url}/api/runs/{run_id}",
                headers=headers,
                timeout=(20.0, 10000.0),
            )
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("status") in {"succeeded", "failed", "cancelled"}:
                return payload

        except requests.RequestException:
            if time.time() >= deadline:
                raise
        time.sleep(0.5)
    raise AssertionError(f"Run {run_id} did not reach a terminal status within {timeout_s}s")


def _node_op(node: dict[str, Any]) -> str:
    metadata = node.get("metadata") or {}
    if isinstance(metadata, dict):
        op = metadata.get("wf_op") or metadata.get("op")
        if op:
            return str(op)
    props = node.get("properties") or {}
    if isinstance(props, dict):
        op = props.get("wf_op") or props.get("op")
        if op:
            return str(op)
    op = node.get("op") or node.get("wf_op")
    return str(op or "")


def _ensure_workflow_design_materialized(
    session: requests.Session,
    base_url: str,
    headers: dict[str, str],
    *,
    workflow_id: str,
    designer_id: str = "manual-e2e",
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    backend_payload = build_agentic_answering_backend_payload(workflow_id=workflow_id)
    frontend_payload = build_agentic_answering_frontend_payload(
        workflow_id=workflow_id
    )
    _payload_roundtrip = WorkflowDesignArtifact.model_validate(backend_payload)
    expected_ops = set(agentic_answering_expected_ops())

    frontend_nodes = frontend_payload.get("nodes") or []
    frontend_edges = frontend_payload.get("edges") or []

    for node in frontend_nodes:
        node_md = node.get("metadata") or {}
        resp = session.post(
            f"{base_url}/api/workflow/design/{workflow_id}/nodes",
            headers=headers,
            json={
                "designer_id": designer_id,
                "node_id": node.get("id"),
                "label": node.get("label"),
                "op": node_md.get("wf_op"),
                "start": bool(node_md.get("wf_start")),
                "terminal": bool(node_md.get("wf_terminal")),
                "fanout": bool(node_md.get("wf_fanout")),
                "metadata": node_md,
            },
            timeout=1000.0,
        )
        resp.raise_for_status()

    for edge in frontend_edges:
        edge_md = edge.get("metadata") or {}
        resp = session.post(
            f"{base_url}/api/workflow/design/{workflow_id}/edges",
            headers=headers,
            json={
                "designer_id": designer_id,
                "edge_id": edge.get("id"),
                "src": (edge.get("source_ids") or [None])[0],
                "dst": (edge.get("target_ids") or [None])[0],
                "relation": edge.get("relation"),
                "predicate": edge_md.get("wf_predicate"),
                "priority": int(edge_md.get("wf_priority", 100)),
                "is_default": bool(edge_md.get("wf_is_default")),
                "multiplicity": str(edge_md.get("wf_multiplicity") or "one"),
                "metadata": edge_md,
            },
            timeout=1000.0,
        )
        resp.raise_for_status()

    deadline = time.time() + timeout_s
    last_payload: dict[str, Any] | None = None
    need_refresh = True
    while time.time() < deadline:
        resp = session.get(
            f"{base_url}/api/workflow/design/{workflow_id}/graph",
            params={"refresh": "true" if need_refresh else "false"},
            headers=headers,
            timeout=1000.0,
        )
        need_refresh = False
        if resp.status_code in {404, 409}:
            time.sleep(0.5)
            continue
        resp.raise_for_status()
        payload = resp.json()
        last_payload = payload

        materialization_status = str(payload.get("materialization_status") or "")
        nodes = payload.get("nodes") or []
        ops = {_node_op(node) for node in nodes if isinstance(node, dict)}

        if materialization_status == "ready" and expected_ops.issubset(ops):
            end_count = sum(1 for node in nodes if _node_op(node) == "end")
            if end_count == 1:
                return payload

        time.sleep(0.5)

    raise AssertionError(
        "Workflow design did not become ready before the answer run. "
        f"workflow_id={workflow_id!r}, last_payload={last_payload!r}"
    )


def _collect_sse_events(
    session: requests.Session, base_url: str, run_id: str, headers: dict[str, str]
) -> list[tuple[str, dict[str, Any]]]:
    resp = session.get(
        f"{base_url}/api/runs/{run_id}/events",
        headers=headers,
        stream=True,
        timeout=None, #30.0,
    )
    resp.raise_for_status()

    events: list[tuple[str, dict[str, Any]]] = []
    current_event: str | None = None
    try:
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if raw_line.startswith("event: "):
                current_event = raw_line.split(": ", 1)[1]
                continue
            if raw_line.startswith("data: ") and current_event is not None:
                payload = json.loads(raw_line.split(": ", 1)[1])
                events.append((current_event, payload))
                current_event = None
    finally:
        resp.close()
    return events


def _seed_debug_rag_knowledge(
    session: requests.Session,
    base_url: str,
    *,
    headers: dict[str, str],
    doc_id: str = "manual-debug-rag-doc",
) -> None:
    seed = build_span_consistent_debug_rag_seed(doc_id=doc_id)
    doc_resp = session.post(
        f"{base_url}/api/document",
        json={
            "doc_id": seed.document.id,
            "doc_type": seed.document.type,
            "insertion_method": "manual_debug_rag_seed",
            "content": seed.document.content,
        },
        headers=headers,
        timeout=60.0,
    )
    doc_resp.raise_for_status()
    resp = session.post(
        f"{base_url}/api/document.upsert_tree",
        json=seed.as_document_upsert_payload(),
        headers=headers,
        timeout=60.0,
    )
    resp.raise_for_status()


def test_manual_insert_workflow_end_to_end() -> None:
    """Manual smoke test for a live `knowledge-mcp` server.

    Start `knowledge-mcp` first, then run this test against the live server.
    The flow is:
    1. Mint a dev token with `POST /auth/dev-token`
    2. Create a conversation with `POST /api/conversations`
    3. Submit a user message with `POST /api/conversations/{conversation_id}/turns:answer`
    4. Poll run status with `GET /api/runs/{run_id}` until terminal
    5. Stream end-to-end SSE from `GET /api/runs/{run_id}/events`

    Put the breakpoint in `POST /api/conversations/{conversation_id}/turns:answer`.
    If you want to watch the consumer side of the stream, also breakpoint
    `GET /api/runs/{run_id}/events`.
    
    workflow_id
        'agentic_answering.v2'
    resolved_designer_id
        'manual-e2e'
    """

    base_url = _manual_base_url()
    with requests.Session() as session:
        _wait_for_health(session, base_url)

        workflow_token, workflow_subject = _dev_token(
            session, base_url, role="rw", ns="workflow"
        )
        rw_token, _ = _dev_token(session, base_url, role="rw", ns="conversation")
        ro_token, _ = _dev_token(session, base_url, role="ro", ns="conversation")

        workflow_rw_headers = _auth_headers(workflow_token)
        

        _ensure_workflow_design_materialized(
            session,
            base_url,
            workflow_rw_headers,
            workflow_id=AGENTIC_ANSWERING_WORKFLOW_ID,
            designer_id=workflow_subject,
            timeout_s=30000.0,
        )

def test_manual_login_chat_and_sse_end_to_end() -> None:
    """Manual smoke test for a live `knowledge-mcp` server.

    Start `knowledge-mcp` first, then run this test against the live server.
    The flow is:
    1. Mint a dev token with `POST /auth/dev-token`
    2. Create a conversation with `POST /api/conversations`
    3. Submit a user message with `POST /api/conversations/{conversation_id}/turns:answer`
    4. Poll run status with `GET /api/runs/{run_id}` until terminal
    5. Stream end-to-end SSE from `GET /api/runs/{run_id}/events`

    Put the breakpoint in `POST /api/conversations/{conversation_id}/turns:answer`.
    If you want to watch the consumer side of the stream, also breakpoint
    `GET /api/runs/{run_id}/events`.
    """

    base_url = _manual_base_url()
    with requests.Session() as session:
        _wait_for_health(session, base_url)

        rw_token, _ = _dev_token(session, base_url, role="rw", ns="conversation")
        ro_token, _ = _dev_token(session, base_url, role="ro", ns="conversation")

        rw_headers = _auth_headers(rw_token)
        ro_headers = _auth_headers(ro_token)

        resp = session.get(
            f"{base_url}/api/workflow/design/{AGENTIC_ANSWERING_WORKFLOW_ID}/graph",
            params={"refresh": "false"},
            headers=ro_headers,
            timeout=10000.0,
        )
        assert resp.json()['materialization_status'] =='ready', "workflow prepopulation incomplete"
        resp.raise_for_status()
        created = session.post(
            f"{base_url}/api/conversations",
            json={"user_id": "manual-e2e-user"},
            headers=rw_headers,
            timeout=30.0,
        )
        created.raise_for_status()
        conversation_id = str(created.json()["conversation_id"])

        submit = session.post(
            f"{base_url}/api/conversations/{conversation_id}/turns:answer",
            json={"user_id": "manual-e2e-user", "text": "hello from manual e2e"},
            headers=rw_headers,
            timeout=None,
        )
        assert submit.status_code == 202
        run_id = str(submit.json()["run_id"])

        final_run = _wait_for_run_terminal(
            session, base_url, run_id, ro_headers, timeout_s=12000
        )
        assert final_run["status"] == "succeeded"
        assert "last_step_seq" not in final_run
        assert "step_count" not in final_run

        transcript = session.get(
            f"{base_url}/api/conversations/{conversation_id}/turns",
            headers=ro_headers,
            timeout=None,
        )
        transcript.raise_for_status()
        turns = transcript.json().get("turns", [])
        assert [turn.get("role") for turn in turns][-2:] == ["user", "assistant"]

        events = _collect_sse_events(session, base_url, run_id, ro_headers)
        names = [name for name, _payload in events]
        assert "run.created" in names
        assert "run.started" in names
        assert "output.completed" in names
        assert names[-1] == "run.completed"


def test_manual_debug_rag_chat_and_sse_end_to_end() -> None:
    """Manual smoke test for the server-known deterministic debug RAG workflow."""

    base_url = _manual_base_url()
    with requests.Session() as session:
        _wait_for_health(session, base_url)

        docs_token, _ = _dev_token(session, base_url, role="rw", ns="docs")
        rw_token, _ = _dev_token(session, base_url, role="rw", ns="conversation")
        ro_token, _ = _dev_token(session, base_url, role="ro", ns="conversation")

        docs_headers = _auth_headers(docs_token)
        rw_headers = _auth_headers(rw_token)
        ro_headers = _auth_headers(ro_token)

        _seed_debug_rag_knowledge(
            session,
            base_url,
            headers=docs_headers,
            doc_id="manual-debug-rag-doc",
        )

        created = session.post(
            f"{base_url}/api/conversations",
            json={"user_id": "manual-debug-rag-user"},
            headers=rw_headers,
            timeout=30.0,
        )
        created.raise_for_status()
        conversation_id = str(created.json()["conversation_id"])

        submit = session.post(
            f"{base_url}/api/conversations/{conversation_id}/turns:answer",
            json={
                "user_id": "manual-debug-rag-user",
                "text": "show me the deterministic debug rag answer",
                "workflow_id": DEBUG_RAG_WORKFLOW_ID,
            },
            headers=rw_headers,
            timeout=None,
        )
        submit.raise_for_status()
        run_id = str(submit.json()["run_id"])

        final_run = _wait_for_run_terminal(
            session, base_url, run_id, ro_headers, timeout_s=180.0
        )
        assert final_run["status"] == "succeeded"

        transcript = session.get(
            f"{base_url}/api/conversations/{conversation_id}/turns",
            headers=ro_headers,
            timeout=None,
        )
        transcript.raise_for_status()
        turns = transcript.json().get("turns", [])
        assistant_text = str(turns[-1].get("content") or "")
        assert "Debug RAG response" in assistant_text
        assert "manual-debug-rag-alpha" in assistant_text

        events = _collect_sse_events(session, base_url, run_id, ro_headers)
        names = [name for name, _payload in events]
        assert "run.created" in names
        assert "run.stage" in names
        assert "reasoning.summary" in names
        assert "output.completed" in names
        assert names[-1] == "run.completed"
