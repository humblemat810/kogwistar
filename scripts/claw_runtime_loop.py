from __future__ import annotations

"""Standalone tutorial: OpenClaw-style event runtime for this repo.

Highlights:
- Provenance-heavy graph writes via Grounding/Span (see `_make_grounding`).
- Workflow graph design in storage (`_wf_node`, `_wf_edge`).
- Strongly event-sourced input/output (SQLite inbox/outbox).
- Loop-budget TTL + `clock.tick` events to avoid infinite self-loops.
- Tool registry:
  - `add_knowledge` writes to KG with provenance.
  - `llm_route` asks an LLM (fallback if unavailable) whether to self-loop or emit output.
- CDC support:
  - Run bridge from this script (`run-cdc-bridge`).
  - Render CDC-enabled workflow/graph pages (`render-cdc-pages`).

Important model note:
- In this repo, `knowledge`, `conversation`, and `workflow` are all graphs.
- Hypergraph relations are supported: edges can point to edges (`source_edge_ids` / `target_edge_ids`).

TTL semantics in this file:
- `ttl` means "remaining loop budget" (how many internal self-emits are allowed).
- It is NOT wall-clock expiration.
- A time-based TTL can coexist (for example `expires_at_ms` in payload), while
  `ttl` still controls recursion depth.
"""

import argparse
import json
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import (
    Document,
    Edge,
    Grounding,
    MentionVerification,
    Node,
    Span,
)
from kogwistar.runtime.models import RunSuccess, WorkflowEdge, WorkflowNode
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.runtime.runtime import WorkflowRuntime
from kogwistar.utils.kge_debug_dump import dump_paired_bundles
from kogwistar.extraction import (
    find_all_exact,
    fuzzy_find_best_spans,
    pick_nearest,
    refresh_context,
)

# Default loop budget for internal self-loop emits. Times To Loop
DEFAULT_TTL = 4


def _now_ms() -> int:
    return int(time.time() * 1000)


## Start with some helpers

# This is a provenance heavy system, it tracks and record where the data come from as first class primitives.


def _make_grounding(doc_id: str, excerpt: str) -> Grounding:
    """Repo-core feature: every write carries Grounding + Span provenance."""
    return Grounding(
        spans=[
            Span(
                collection_page_url=f"document_collection/{doc_id}",
                document_page_url=f"document/{doc_id}",
                doc_id=doc_id,
                insertion_method="claw_runtime_loop",
                page_number=1,
                start_char=0,
                end_char=max(1, len(excerpt)),
                excerpt=excerpt[:512],
                context_before="",
                context_after="",
                chunk_id=None,
                source_cluster_id=None,
                verification=MentionVerification(
                    method="system", is_verified=True, score=1.0, notes="claw-runtime"
                ),
            )
        ]
    )


# Workflow design Node factory function


def _wf_node(
    *,
    workflow_id: str,
    node_id: str,
    op: str,
    start: bool = False,
    terminal: bool = False,
) -> WorkflowNode:
    """Workflow graph builder helper (nodes)."""
    return WorkflowNode(
        id=node_id,
        label=node_id,
        type="entity",
        doc_id=f"wf:{workflow_id}",
        summary=f"op={op}",
        mentions=[
            _make_grounding(f"wf:{workflow_id}", op)
        ],  # workflow design itself can be viewed as a document, imagine like a Canva
        properties={},
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": op,
            "wf_version": "v1",
            "wf_start": bool(start),
            "wf_terminal": bool(terminal),
            "wf_fanout": False,
        },
        domain_id=None,
        canonical_entity_id=None,
        level_from_root=None,
    )


# Workflow design Edge factory function


def _wf_edge(*, workflow_id: str, edge_id: str, src: str, dst: str) -> WorkflowEdge:
    """Workflow graph builder helper (edges)."""
    return WorkflowEdge(
        id=edge_id,
        source_ids=[src],
        target_ids=[dst],
        relation="wf_next",
        label="wf_next",
        type="relationship",
        summary=f"{src}->{dst}",
        doc_id=f"wf:{workflow_id}",
        mentions=[_make_grounding(f"wf:{workflow_id}", f"{src}->{dst}")],
        properties={},
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_predicate": None,
            "wf_priority": 100,
            "wf_is_default": True,
            "wf_multiplicity": "one",
        },
        source_edge_ids=[],
        target_edge_ids=[],
        domain_id=None,
        canonical_entity_id=None,
        level_from_root=None,
    )


# A simple class emulate major Claw features.
# Note: "clawbot" is just tutorial skin; the reusable pattern is event-sourced
# inbox/outbox with explicit status transitions and durable audit rows.


class ClawEventStore:
    """Durable event stream (`in` + `out`) with status transitions."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS claw_events(
                event_id TEXT PRIMARY KEY, direction TEXT NOT NULL, conversation_id TEXT NOT NULL,
                event_type TEXT NOT NULL, payload_json TEXT NOT NULL, status TEXT, source_event_id TEXT,
                run_id TEXT, error TEXT, created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL)"""
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_claw_events_dir_status_time ON claw_events(direction,status,created_at_ms)"
            )

    def enqueue_input(
        self, *, conversation_id: str, event_type: str, payload: Dict[str, Any]
    ) -> str:
        eid = f"in|{uuid.uuid4()}"
        now = _now_ms()
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "INSERT INTO claw_events VALUES(?, 'in', ?, ?, ?, 'pending', NULL, NULL, NULL, ?, ?)",
                (
                    eid,
                    conversation_id,
                    event_type,
                    json.dumps(payload, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return eid

    def claim_next_pending(self) -> Optional[Dict[str, Any]]:
        # "claim" is stateful: pending -> processing in one transaction.
        # This is different from read-only queue probes like has_pending_user_message.
        with sqlite3.connect(self.db_path) as c:
            c.row_factory = sqlite3.Row
            c.execute("BEGIN IMMEDIATE")
            r = c.execute(
                "SELECT event_id,conversation_id,event_type,payload_json FROM claw_events WHERE direction='in' AND status='pending' ORDER BY created_at_ms, rowid LIMIT 1"
            ).fetchone()
            if r is None:
                c.commit()
                return None
            now = _now_ms()
            c.execute(
                "UPDATE claw_events SET status='processing',updated_at_ms=? WHERE event_id=? AND status='pending'",
                (now, r["event_id"]),
            )
            c.commit()
            return {
                "event_id": r["event_id"],
                "conversation_id": r["conversation_id"],
                "event_type": r["event_type"],
                "payload": json.loads(r["payload_json"]),
            }

    def mark_done(self, event_id: str, run_id: str) -> None:
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "UPDATE claw_events SET status='done',run_id=?,updated_at_ms=? WHERE event_id=?",
                (run_id, _now_ms(), event_id),
            )

    def mark_failed(self, event_id: str, err: str) -> None:
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "UPDATE claw_events SET status='failed',error=?,updated_at_ms=? WHERE event_id=?",
                (err[:4000], _now_ms(), event_id),
            )

    def append_output(
        self,
        *,
        conversation_id: str,
        event_type: str,
        payload: Dict[str, Any],
        source_event_id: str,
        run_id: str,
    ) -> str:
        oid = f"out|{uuid.uuid4()}"
        now = _now_ms()
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "INSERT INTO claw_events VALUES(?, 'out', ?, ?, ?, 'done', ?, ?, NULL, ?, ?)",
                (
                    oid,
                    conversation_id,
                    event_type,
                    json.dumps(payload, ensure_ascii=False),
                    source_event_id,
                    run_id,
                    now,
                    now,
                ),
            )
        return oid

    def has_pending_user_message(
        self, *, conversation_id: str, exclude_event_id: str
    ) -> bool:
        # Read-only probe: does NOT claim or mutate any event.
        with sqlite3.connect(self.db_path) as c:
            r = c.execute(
                "SELECT 1 FROM claw_events WHERE direction='in' AND status='pending' AND conversation_id=? AND event_type='user.message' AND event_id!=? LIMIT 1",
                (conversation_id, exclude_event_id),
            ).fetchone()
        return r is not None

    def count_pending_events(self, *, conversation_id: str, event_type: str) -> int:
        with sqlite3.connect(self.db_path) as c:
            r = c.execute(
                "SELECT COUNT(1) FROM claw_events WHERE direction='in' AND status='pending' AND conversation_id=? AND event_type=?",
                (conversation_id, event_type),
            ).fetchone()
        return int((r[0] if r else 0) or 0)

    def list_events(self, *, direction: str, limit: int) -> list[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT * FROM claw_events WHERE direction=? ORDER BY created_at_ms DESC LIMIT ?",
                (direction, limit),
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    k: (json.loads(r[k]) if k == "payload_json" else r[k])
                    for k in r.keys()
                }
            )
        return out


def _llm_route(payload: Dict[str, Any], ttl: int) -> Dict[str, Any]:
    """Tool: LLM routing policy for self-loop vs output gate.

    Instruction includes self-emit semantics and TTL safety.
    Falls back to deterministic behavior if model deps/credentials are missing.

    This repo is design is vendor agnostic, so the example here is adaptor design
    to variosu LLM vendors,
    though we still need to choose an adaptor vendor.
    """
    system = (
        "You are routing an event-sourced agent. Output strict JSON with keys: "
        "route(self|output), reason, next_event_type, next_payload, output. "
        "Route controls enqueue behavior. If route=output and next_payload exists, it is deferred metadata only. "
        "If route=self and ttl>0, runtime may enqueue continuation; output can still be emitted in same decision. "
        "Treat ttl as loop-budget, not wall-clock lifetime."
    )
    try:
        # Default provider section: Azure OpenAI
        from langchain_openai import AzureChatOpenAI  # type: ignore

        m = AzureChatOpenAI(
            deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4_1"),
            model_name=os.getenv("OPENAI_MODEL_NAME_GPT4_1"),
            azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1"),
            openai_api_key=os.getenv("OPENAI_API_KEY_GPT4_1"),
            api_version="2024-08-01-preview",
            temperature=0.1,
            openai_api_type="azure",
        )

        # Alternative provider section: Gemini
        # Uncomment this block and comment out the Azure block above to use Gemini.
        #
        # from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        # m = ChatGoogleGenerativeAI(
        #     model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"),
        #     google_api_key=os.getenv("GOOGLE_API_KEY"),
        #     temperature=0.1,
        # )

        raw = m.invoke(
            [
                ("system", system),
                (
                    "human",
                    json.dumps({"payload": payload, "ttl": ttl}, ensure_ascii=False),
                ),
            ]
        )
        # note this is a tutorial level script to introduce the feature, this loop is not hardened, at least need a contract validator.
        x = json.loads(str(getattr(raw, "content", raw)))
        if str(x.get("route")) not in {"self", "output"}:
            x["route"] = "output"
        return x
    except Exception:
        unfinished = bool(payload.get("unfinished"))
        if unfinished and ttl > 0:
            return {
                "route": "self",
                "reason": "fallback-unfinished",
                "next_event_type": "agent.loop",
                "next_payload": {"text": payload.get("text", ""), "unfinished": False},
                "output": {"message": "internal continue"},
            }
        return {
            "route": "output",
            "reason": "fallback-output",
            "next_event_type": None,
            "next_payload": None,
            "output": {"message": str(payload.get("text") or "ack")},
        }


class ClawResolver(MappingStepResolver):
    """Resolver with output gate semantics.

    Output op chooses:
    - internal gate: enqueue next input event (self-loop)
    - external gate: append final outbox output

    `ttl` in state is a loop counter budget.
    """

    def __init__(self) -> None:
        super().__init__(
            handlers={
                "ingest_event": self.ingest_event,
                "decide_action": self.decide_action,
                "execute_action": self.execute_action,
                "persist_outbox": self.persist_outbox,
                "end": self.end,
            }
        )

    def ingest_event(self, ctx) -> RunSuccess:
        e = dict(ctx.state_view["event"])
        p = dict(e.get("payload") or {})
        with ctx.state_write as s:
            s["event"] = e
            s["payload"] = p
            s["ttl"] = int(p.get("ttl", DEFAULT_TTL))
        return RunSuccess(conversation_node_id=None, state_update=[])

    def decide_action(self, ctx) -> RunSuccess:
        e = dict(ctx.state_view["event"])
        p = dict(ctx.state_view["payload"])
        if e["event_type"] == "clock.tick":
            action = "poll_stream"
        elif str(p.get("tool") or "") == "add_knowledge":
            action = "tool:add_knowledge"
        else:
            action = "tool:llm_route"
        with ctx.state_write as s:
            s["action"] = action
        return RunSuccess(conversation_node_id=None, state_update=[])

    def execute_action(self, ctx) -> RunSuccess:
        deps = dict(ctx.state_view["_deps"])
        store: ClawEventStore = deps["event_store"]
        tools: Dict[str, Callable[..., Dict[str, Any]]] = deps["tool_registry"]
        e = dict(ctx.state_view["event"])
        p = dict(ctx.state_view["payload"])
        ttl = int(ctx.state_view["ttl"])
        if ctx.state_view["action"] == "poll_stream":
            if (
                store.has_pending_user_message(
                    conversation_id=e["conversation_id"], exclude_event_id=e["event_id"]
                )
                and ttl > 0
            ):
                d = {
                    "route": "self",
                    "reason": "clock-found-work",
                    "next_event_type": "agent.loop",
                    "next_payload": {"tool": "llm_route", "text": "tick follow-up"},
                    "output": {"message": "tick"},
                }
            else:
                d = {
                    "route": "output",
                    "reason": "clock-idle",
                    "next_event_type": None,
                    "next_payload": None,
                    "output": {"message": "idle"},
                }
        elif str(ctx.state_view["action"]).startswith("tool:"):
            t = str(ctx.state_view["action"]).split(":", 1)[1]
            d = tools[t](payload=p, ttl=ttl)
        else:
            d = {
                "route": "output",
                "reason": "ack",
                "next_event_type": None,
                "next_payload": None,
                "output": {"message": "ack"},
            }
        with ctx.state_write as s:
            s["decision"] = d
        return RunSuccess(conversation_node_id=None, state_update=[])

    def persist_outbox(self, ctx) -> RunSuccess:
        deps = dict(ctx.state_view["_deps"])
        store: ClawEventStore = deps["event_store"]
        e = dict(ctx.state_view["event"])
        d = dict(ctx.state_view["decision"])
        ttl = int(ctx.state_view["ttl"])
        if str(d.get("route")) == "self" and ttl > 0:
            nxt = dict(d.get("next_payload") or {})
            nxt["ttl"] = ttl - 1
            nxt["parent_event_id"] = e["event_id"]
            next_id = store.enqueue_input(
                conversation_id=e["conversation_id"],
                event_type=str(d.get("next_event_type") or "agent.loop"),
                payload=nxt,
            )
            otype = "claw.gate.internal"
            payload = {
                "route": "self",
                "next_event_id": next_id,
                "reason": d.get("reason"),
            }
        else:
            otype = "claw.gate.output"
            payload = {
                "route": "output",
                "reason": d.get("reason"),
                "result": d.get("output"),
            }
        out_id = store.append_output(
            conversation_id=e["conversation_id"],
            event_type=otype,
            payload=payload,
            source_event_id=e["event_id"],
            run_id=ctx.run_id,
        )
        with ctx.state_write as s:
            s["out_id"] = out_id
        return RunSuccess(conversation_node_id=None, state_update=[])

    def end(self, ctx) -> RunSuccess:
        return RunSuccess(conversation_node_id=None, state_update=[])


@dataclass
class ClawRuntimeApp:
    data_dir: Path
    workflow_id: str = "claw.loop.v1"
    cdc_publish_endpoint: Optional[str] = None

    def __post_init__(self) -> None:
        if self.cdc_publish_endpoint:
            os.environ["CDC_PUBLISH_ENDPOINT"] = self.cdc_publish_endpoint
        self.workflow_engine = GraphKnowledgeEngine(
            persist_directory=str(self.data_dir / "wf"), kg_graph_type="workflow"
        )
        self.conversation_engine = GraphKnowledgeEngine(
            persist_directory=str(self.data_dir / "conv"), kg_graph_type="conversation"
        )
        self.knowledge_engine = GraphKnowledgeEngine(
            persist_directory=str(self.data_dir / "kg"), kg_graph_type="knowledge"
        )
        self.event_store = ClawEventStore(self.data_dir / "claw_events.sqlite")
        self.resolver = ClawResolver()
        self.tool_registry = self._tool_registry()
        self.runtime = WorkflowRuntime(
            workflow_engine=self.workflow_engine,
            conversation_engine=self.conversation_engine,
            step_resolver=self.resolver.resolve,
            predicate_registry={},
            checkpoint_every_n_steps=1,
            max_workers=1,
        )
        self._ensure_workflow()

    def _tool_registry(self) -> Dict[str, Callable[..., Dict[str, Any]]]:
        """Register tools (KG writer + LLM router)."""

        def add_knowledge(*, payload: Dict[str, Any], ttl: int) -> Dict[str, Any]:
            doc_id = str(payload.get("doc_id") or f"doc:claw:{uuid.uuid4().hex[:8]}")
            subject = str(payload.get("subject") or "Subject")
            relation = str(payload.get("relation") or "related_to")
            obj = str(payload.get("object") or "Object")
            text = str(payload.get("text") or f"{subject} {relation} {obj}")
            self.knowledge_engine.write.add_document(
                Document(
                    id=doc_id,
                    content=text,
                    type="text",
                    metadata={"source": "tool_add_knowledge"},
                    embeddings=None,
                    source_map=None,
                    domain_id=None,
                    processed=False,
                )
            )
            n1 = Node(
                id=f"kg|n|{uuid.uuid4().hex[:10]}",
                label=subject,
                type="entity",
                doc_id=doc_id,
                summary=subject,
                mentions=[_make_grounding(doc_id, subject)],
                properties={},
                metadata={},
                domain_id=None,
                canonical_entity_id=None,
            )
            n2 = Node(
                id=f"kg|n|{uuid.uuid4().hex[:10]}",
                label=obj,
                type="entity",
                doc_id=doc_id,
                summary=obj,
                mentions=[_make_grounding(doc_id, obj)],
                properties={},
                metadata={},
                domain_id=None,
                canonical_entity_id=None,
            )
            e = Edge(
                id=f"kg|e|{uuid.uuid4().hex[:10]}",
                source_ids=[n1.id],
                target_ids=[n2.id],
                relation=relation,
                label=relation,
                type="relationship",
                summary=text,
                doc_id=doc_id,
                mentions=[_make_grounding(doc_id, text)],
                properties={},
                metadata={},
                source_edge_ids=[],
                target_edge_ids=[],
                domain_id=None,
                canonical_entity_id=None,
            )
            self.knowledge_engine.write.add_node(n1)
            self.knowledge_engine.write.add_node(n2)
            self.knowledge_engine.write.add_edge(e)
            return {
                "route": "output",
                "reason": "kg-write-done",
                "next_event_type": None,
                "next_payload": None,
                "output": {
                    "doc_id": doc_id,
                    "node_ids": [n1.id, n2.id],
                    "edge_id": e.id,
                    "ttl": ttl,
                },
            }

        return {"add_knowledge": add_knowledge, "llm_route": _llm_route}

    def _ensure_workflow(self) -> None:
        if self.workflow_engine.read.get_nodes(
            where={
                "$and": [
                    {"entity_type": "workflow_node"},
                    {"workflow_id": self.workflow_id},
                ]
            },
            limit=1,
        ):
            return
        ids = [
            f"wf|{self.workflow_id}|{n}"
            for n in ("ingest", "decide", "execute", "persist", "end")
        ]
        nodes = [
            _wf_node(
                workflow_id=self.workflow_id,
                node_id=ids[0],
                op="ingest_event",
                start=True,
            ),
            _wf_node(workflow_id=self.workflow_id, node_id=ids[1], op="decide_action"),
            _wf_node(workflow_id=self.workflow_id, node_id=ids[2], op="execute_action"),
            _wf_node(workflow_id=self.workflow_id, node_id=ids[3], op="persist_outbox"),
            _wf_node(
                workflow_id=self.workflow_id, node_id=ids[4], op="end", terminal=True
            ),
        ]
        edges = [
            _wf_edge(
                workflow_id=self.workflow_id,
                edge_id=f"{ids[0]}->1",
                src=ids[0],
                dst=ids[1],
            ),
            _wf_edge(
                workflow_id=self.workflow_id,
                edge_id=f"{ids[1]}->2",
                src=ids[1],
                dst=ids[2],
            ),
            _wf_edge(
                workflow_id=self.workflow_id,
                edge_id=f"{ids[2]}->3",
                src=ids[2],
                dst=ids[3],
            ),
            _wf_edge(
                workflow_id=self.workflow_id,
                edge_id=f"{ids[3]}->4",
                src=ids[3],
                dst=ids[4],
            ),
        ]
        for n in nodes:
            self.workflow_engine.write.add_node(n)
        for e in edges:
            self.workflow_engine.write.add_edge(e)

    def ensure_tutorial_workflow(
        self, workflow_id: str = "wf.tutorial.blocking.v2"
    ) -> str:
        """Workflow design for beginner tutorial (blocking get_input loop)."""
        if self.workflow_engine.read.get_nodes(
            where={
                "$and": [{"entity_type": "workflow_node"}, {"workflow_id": workflow_id}]
            },
            limit=1,
        ):
            return workflow_id
        # Use readable IDs so the CDC graph is self-explanatory for beginners.
        # In this repo, workflow topology itself is persisted as graph knowledge.
        n_get = f"wf|{workflow_id}|get_input"
        n_decide = f"wf|{workflow_id}|decide"
        n_execute = f"wf|{workflow_id}|execute"
        n_emit = f"wf|{workflow_id}|emit_output"
        n_end = f"wf|{workflow_id}|end"
        nodes = [
            _wf_node(
                workflow_id=workflow_id, node_id=n_get, op="get_input", start=True
            ),
            _wf_node(workflow_id=workflow_id, node_id=n_decide, op="decide"),
            _wf_node(workflow_id=workflow_id, node_id=n_execute, op="execute"),
            _wf_node(workflow_id=workflow_id, node_id=n_emit, op="emit_output"),
            _wf_node(workflow_id=workflow_id, node_id=n_end, op="end", terminal=True),
        ]
        for n in nodes:
            self.workflow_engine.write.add_node(n)
        edges = [
            _wf_edge(
                workflow_id=workflow_id,
                edge_id=f"{n_get}__to__{n_decide}",
                src=n_get,
                dst=n_decide,
            ),
            _wf_edge(
                workflow_id=workflow_id,
                edge_id=f"{n_decide}__to__{n_execute}",
                src=n_decide,
                dst=n_execute,
            ),
            _wf_edge(
                workflow_id=workflow_id,
                edge_id=f"{n_execute}__to__{n_emit}",
                src=n_execute,
                dst=n_emit,
            ),
            WorkflowEdge(
                id=f"{n_emit}__to__{n_get}__continue",
                source_ids=[n_emit],
                target_ids=[n_get],
                relation="wf_next",
                label="wf_next",
                type="relationship",
                summary="emit_output -> get_input (continue)",
                doc_id=f"wf:{workflow_id}",
                mentions=[_make_grounding(f"wf:{workflow_id}", "continue loop")],
                properties={},
                metadata={
                    "entity_type": "workflow_edge",
                    "workflow_id": workflow_id,
                    "wf_predicate": "should_continue",
                    "wf_priority": 0,
                    "wf_is_default": False,
                    "wf_multiplicity": "one",
                },
                source_edge_ids=[],
                target_edge_ids=[],
                domain_id=None,
                canonical_entity_id=None,
            ),
            _wf_edge(
                workflow_id=workflow_id,
                edge_id=f"{n_emit}__to__{n_end}__default",
                src=n_emit,
                dst=n_end,
            ),
        ]
        # last edge is default end path
        edges[-1].metadata["wf_is_default"] = True
        edges[-1].metadata["wf_priority"] = 100
        for e in edges:
            self.workflow_engine.write.add_edge(e)
        return workflow_id

    def purge_doc_graph(self, *, doc_id: str) -> Dict[str, int]:
        """CR-safe cleanup for one document's graph footprint before reseeding.
        In normal heavy provenance usage, delete is only an escape hatch or demo reset.

        This removes:
        - edges tied to doc_id
        - node documents/index rows tied to doc_id
        - nodes tied to doc_id
        - document row itself
        """
        node_ids = list(self.knowledge_engine.read.node_ids_by_doc(doc_id))
        edge_ids = list(self.knowledge_engine.read.edge_ids_by_doc(doc_id))

        if edge_ids:
            self.knowledge_engine.backend.edge_delete(ids=edge_ids)
            self.knowledge_engine.backend.edge_endpoints_delete(
                where={"edge_id": {"$in": edge_ids}}
            )

        # Best-effort cleanup for node_docs projection/index rows.
        try:
            self.knowledge_engine.backend.node_docs_delete(where={"doc_id": doc_id})
        except Exception:
            pass

        if node_ids:
            self.knowledge_engine.backend.node_delete(ids=node_ids)

        try:
            self.knowledge_engine.backend.document_delete(ids=[doc_id])
        except Exception:
            pass

        return {"deleted_nodes": len(node_ids), "deleted_edges": len(edge_ids)}

    def _repair_mentions_in_memory(
        self, *, content: str, items: list[Any]
    ) -> Dict[str, int]:
        """Repair mention spans in-memory for nodes/edges before first persist."""
        fixed_items = 0
        fixed_spans = 0
        failed_spans = 0
        for it in items:
            changed = False
            for g in it.mentions or []:
                repaired_spans = []
                for sp in g.spans:
                    new_sp, span_changed, mode = self._repair_one_span(
                        content=content, span=sp
                    )
                    repaired_spans.append(new_sp)
                    if span_changed:
                        fixed_spans += 1
                        changed = True
                    elif mode == "failed":
                        failed_spans += 1
                g.spans = repaired_spans
            if changed:
                fixed_items += 1
        return {
            "fixed_items": fixed_items,
            "fixed_spans": fixed_spans,
            "failed_spans": failed_spans,
        }

    def seed_background_hypergraph(self) -> Dict[str, Any]:
        """Notebook-style background knowledge seeding with hypergraph primitives.

        This intentionally uses explicit `# Step` comments so readers can follow
        it like a linear notebook, but still run it as normal Python.
        """
        # Step 1) CR-only friendly: allocate a new document id per run (no deletes).
        doc_id = f"doc:background:hypergraph:{_now_ms()}"
        sid = doc_id.rsplit(":", 1)[-1]

        # Step 2) Create a background document in the knowledge graph.
        # Keep seeded text aligned with tutorial excerpts so provenance repair can succeed.
        # Phrases below intentionally include:
        # "provides service to", "must pay", "USD 10,000", "payment clause",
        # "constrains", "reifies", and "owes amount".
        doc_text = (
            "Alice signs a consulting contract with Bob. "
            "Alice provides service to Bob, and Bob must pay Alice USD 10,000. "
            "The payment clause constrains obligations, reifies the payment edge, and owes amount terms are specified."
        )
        document = Document(
            id=doc_id,
            content=doc_text,
            type="text",
            metadata={"source": "seed_background_hypergraph"},
            embeddings=None,
            source_map=None,
            domain_id=None,
            processed=False,
        )

        # Step 3) Add primitive nodes.
        n_alice = Node(
            id=f"kg:bg:{sid}:alice",
            label="Alice",
            type="entity",
            doc_id=doc_id,
            summary="Contract party",
            mentions=[_make_grounding(doc_id, "Alice")],
            properties={},
            metadata={"kind": "person"},
            domain_id=None,
            canonical_entity_id=None,
        )
        n_bob = Node(
            id=f"kg:bg:{sid}:bob",
            label="Bob",
            type="entity",
            doc_id=doc_id,
            summary="Contract party",
            mentions=[_make_grounding(doc_id, "Bob")],
            properties={},
            metadata={"kind": "person"},
            domain_id=None,
            canonical_entity_id=None,
        )
        n_amount = Node(
            id=f"kg:bg:{sid}:usd_10000",
            label="USD 10,000",
            type="entity",
            doc_id=doc_id,
            summary="Contract payment amount",
            mentions=[_make_grounding(doc_id, "USD 10,000")],
            properties={"currency": "USD", "amount": 10000},
            metadata={"kind": "money"},
            domain_id=None,
            canonical_entity_id=None,
        )
        n_clause = Node(
            id=f"kg:bg:{sid}:clause_payment",
            label="Payment Clause",
            type="entity",
            doc_id=doc_id,
            summary="Clause that constrains obligations",
            mentions=[_make_grounding(doc_id, "payment clause")],
            properties={},
            metadata={"kind": "clause"},
            domain_id=None,
            canonical_entity_id=None,
        )
        nodes = [n_alice, n_bob, n_amount, n_clause]

        # Step 4) Add primitive node->node edges.
        e_service = Edge(
            id=f"kg:bg:{sid}:e_service",
            source_ids=[n_alice.id],
            target_ids=[n_bob.id],
            relation="provides_service_to",
            label="provides_service_to",
            type="relationship",
            summary="Alice provides consulting service to Bob",
            doc_id=doc_id,
            mentions=[_make_grounding(doc_id, "provides service to")],
            properties={},
            metadata={"kind": "obligation"},
            source_edge_ids=[],
            target_edge_ids=[],
            domain_id=None,
            canonical_entity_id=None,
        )
        e_payment = Edge(
            id=f"kg:bg:{sid}:e_payment",
            source_ids=[n_bob.id],
            target_ids=[n_alice.id],
            relation="must_pay",
            label="must_pay",
            type="relationship",
            summary="Bob must pay Alice",
            doc_id=doc_id,
            mentions=[_make_grounding(doc_id, "must pay")],
            properties={},
            metadata={"kind": "payment_obligation"},
            source_edge_ids=[],
            target_edge_ids=[],
            domain_id=None,
            canonical_entity_id=None,
        )
        edges = [e_service, e_payment]

        # Step 5) Add hypergraph edge: edge -> edge relation.
        # This represents "payment obligation constrains service obligation".
        e_constrains = Edge(
            id=f"kg:bg:{sid}:e_constrains",
            source_ids=[],
            target_ids=[],
            relation="constrains",
            label="constrains",
            type="relationship",
            summary="Payment obligation constrains service obligation",
            doc_id=doc_id,
            mentions=[_make_grounding(doc_id, "constrains")],
            properties={},
            metadata={"kind": "hyperedge"},
            source_edge_ids=[e_payment.id],
            target_edge_ids=[e_service.id],
            domain_id=None,
            canonical_entity_id=None,
        )
        edges.append(e_constrains)

        # Step 6) Show "edge is also a graph endpoint": node -> edge relation.
        e_clause_reifies = Edge(
            id=f"kg:bg:{sid}:e_clause_reifies",
            source_ids=[n_clause.id],
            target_ids=[],
            relation="reifies",
            label="reifies",
            type="relationship",
            summary="Payment clause reifies payment edge",
            doc_id=doc_id,
            mentions=[_make_grounding(doc_id, "reifies")],
            properties={},
            metadata={"kind": "node_to_edge"},
            source_edge_ids=[],
            target_edge_ids=[e_payment.id],
            domain_id=None,
            canonical_entity_id=None,
        )
        edges.append(e_clause_reifies)

        # Step 7) Optional extra link to amount primitive.
        e_amount = Edge(
            id=f"kg:bg:{sid}:e_amount",
            source_ids=[n_bob.id],
            target_ids=[n_amount.id],
            relation="owes_amount",
            label="owes_amount",
            type="relationship",
            summary="Bob owes USD 10,000",
            doc_id=doc_id,
            mentions=[_make_grounding(doc_id, "owes amount")],
            properties={},
            metadata={"kind": "amount_binding"},
            source_edge_ids=[],
            target_edge_ids=[],
            domain_id=None,
            canonical_entity_id=None,
        )
        edges.append(e_amount)

        # Step 8) Add one intentionally noisy span to demonstrate repair utilities.
        noisy_span = Span(
            collection_page_url=f"document_collection/{doc_id}",
            document_page_url=f"document/{doc_id}",
            doc_id=doc_id,
            insertion_method="seed_background_hypergraph",
            page_number=1,
            start_char=3,  # intentionally wrong offset for excerpt "Alice"
            end_char=8,
            excerpt="Alice",
            context_before="",
            context_after="",
            chunk_id=None,
            source_cluster_id=None,
            verification=MentionVerification(
                method="system",
                is_verified=False,
                score=0.0,
                notes="intentional noisy span",
            ),
        )
        n_noisy = Node(
            id=f"kg:bg:{sid}:noisy_alice",
            label="Noisy Alice Mention",
            type="entity",
            doc_id=doc_id,
            summary="A node with intentionally wrong span offsets for repair demo",
            mentions=[Grounding(spans=[noisy_span])],
            properties={},
            metadata={"kind": "noisy_demo"},
            domain_id=None,
            canonical_entity_id=None,
        )
        nodes.append(n_noisy)

        # Step 8) Repair provenance in-memory before first persist (CR semantics).
        node_repair = self._repair_mentions_in_memory(content=doc_text, items=nodes)
        edge_repair = self._repair_mentions_in_memory(content=doc_text, items=edges)

        # Step 9) Persist repaired document/nodes/edges.
        self.knowledge_engine.write.add_document(document)
        for n in nodes:
            self.knowledge_engine.write.add_node(n)
        for e in edges:
            self.knowledge_engine.write.add_edge(e)

        return {
            "doc_id": doc_id,
            "node_ids": [n_alice.id, n_bob.id, n_amount.id, n_clause.id, n_noisy.id],
            "edge_ids": [
                e_service.id,
                e_payment.id,
                e_constrains.id,
                e_clause_reifies.id,
                e_amount.id,
            ],
            "provenance_repair": {
                "mode": "pre_persist_in_memory",
                "fixed_nodes": node_repair["fixed_items"],
                "fixed_edges": edge_repair["fixed_items"],
                "fixed_spans": int(node_repair["fixed_spans"])
                + int(edge_repair["fixed_spans"]),
                "failed_spans": int(node_repair["failed_spans"])
                + int(edge_repair["failed_spans"]),
            },
        }

    def _repair_one_span(self, *, content: str, span: Span) -> tuple[Span, bool, str]:
        """Repair one span using extraction matching helpers (exact first, then fuzzy)."""
        excerpt = span.excerpt or ""
        if not excerpt:
            return span, False, "empty_excerpt"

        start = int(span.start_char or 0)
        end = int(span.end_char or start)
        if 0 <= start < end <= len(content) and content[start:end] == excerpt:
            return span, False, "already_correct"

        exact_hits = find_all_exact(content, excerpt)
        nearest = pick_nearest(exact_hits, start)
        if nearest is not None:
            fixed_start = int(nearest)
            fixed_end = fixed_start + len(excerpt)
            before, after = refresh_context(content, fixed_start, fixed_end)
            fixed = span.model_copy(
                update={
                    "start_char": fixed_start,
                    "end_char": fixed_end,
                    "context_before": before,
                    "context_after": after,
                    "verification": MentionVerification(
                        method="regex",
                        is_verified=True,
                        score=1.0,
                        notes="exact match repair",
                    ),
                }
            )
            return fixed, True, "exact"

        hits = fuzzy_find_best_spans(content, excerpt, start, max_hits=1)
        if hits:
            h = hits[0]
            fixed_excerpt = content[h.start : h.end]
            before, after = refresh_context(content, h.start, h.end)
            fixed = span.model_copy(
                update={
                    "start_char": int(h.start),
                    "end_char": int(h.end),
                    "excerpt": fixed_excerpt,
                    "context_before": before,
                    "context_after": after,
                    "verification": MentionVerification(
                        method="levenshtein",
                        is_verified=True,
                        score=float(h.score) / 100.0,
                        notes="fuzzy match repair",
                    ),
                }
            )
            return fixed, True, "fuzzy"

        failed = span.model_copy(
            update={
                "verification": MentionVerification(
                    method="heuristic",
                    is_verified=False,
                    score=0.0,
                    notes="unable to repair span",
                )
            }
        )
        return failed, False, "failed"

    def repair_provenance_for_doc(self, doc_id: str) -> Dict[str, Any]:
        """Refine span correctness for all mentions in nodes/edges of a document."""
        doc = self.knowledge_engine.get_document(doc_id)
        content = doc.content or ""
        fixed_nodes = 0
        fixed_edges = 0
        fixed_spans = 0
        failed_spans = 0

        nodes = self.knowledge_engine.read.get_nodes(
            where={"doc_id": doc_id}, limit=2000
        )
        for n in nodes:
            node_changed = False
            for g in n.mentions or []:
                repaired_spans = []
                for sp in g.spans:
                    new_sp, changed, mode = self._repair_one_span(
                        content=content, span=sp
                    )
                    repaired_spans.append(new_sp)
                    if changed:
                        fixed_spans += 1
                        node_changed = True
                    elif mode == "failed":
                        failed_spans += 1
                g.spans = repaired_spans
            if node_changed:
                self.knowledge_engine.write.add_node(n)
                fixed_nodes += 1

        edges = self.knowledge_engine.read.get_edges(
            where={"doc_id": doc_id}, limit=3000
        )
        for e in edges:
            edge_changed = False
            for g in e.mentions or []:
                repaired_spans = []
                for sp in g.spans:
                    new_sp, changed, mode = self._repair_one_span(
                        content=content, span=sp
                    )
                    repaired_spans.append(new_sp)
                    if changed:
                        fixed_spans += 1
                        edge_changed = True
                    elif mode == "failed":
                        failed_spans += 1
                g.spans = repaired_spans
            if edge_changed:
                self.knowledge_engine.write.add_edge(e)
                fixed_edges += 1

        return {
            "doc_id": doc_id,
            "fixed_nodes": fixed_nodes,
            "fixed_edges": fixed_edges,
            "fixed_spans": fixed_spans,
            "failed_spans": failed_spans,
        }

    def enqueue(
        self, *, conversation_id: str, event_type: str, payload: Dict[str, Any]
    ) -> str:
        # `ttl` here is loop budget; callers may also include `expires_at_ms` separately.
        if "ttl" not in payload:
            payload["ttl"] = DEFAULT_TTL
        return self.event_store.enqueue_input(
            conversation_id=conversation_id, event_type=event_type, payload=payload
        )

    def run_once(self) -> bool:
        ev = self.event_store.claim_next_pending()
        if ev is None:
            return False
        rid = f"claw|{ev['event_id']}"
        try:
            self.runtime.run(
                workflow_id=self.workflow_id,
                conversation_id=str(ev["conversation_id"]),
                turn_node_id=f"turn|{ev['event_id']}",
                initial_state={
                    "event": ev,
                    "_deps": {
                        "event_store": self.event_store,
                        "tool_registry": self.tool_registry,
                    },
                },
                run_id=rid,
            )
            self.event_store.mark_done(event_id=str(ev["event_id"]), run_id=rid)
            return True
        except Exception:
            self.event_store.mark_failed(
                event_id=str(ev["event_id"]), err=traceback.format_exc()
            )
            return True

    def run_loop(
        self,
        *,
        sleep_ms: int,
        max_iterations: int,
        clock_interval_ms: int,
        clock_conversation_id: str,
        max_pending_ticks: int = 1,
    ) -> None:
        # to rewrite this part into workflow design graph, put the init count states in initial_state
        # of the run time, add another node before end, update loop state, and add predicate to either
        # route back to start or go to end.
        
        i = 0  
        last_clock = 0
        while True:
            if clock_interval_ms > 0 and _now_ms() - last_clock >= clock_interval_ms:
                pending_ticks = self.event_store.count_pending_events(
                    conversation_id=clock_conversation_id,
                    event_type="clock.tick",
                )
                if pending_ticks < max(0, int(max_pending_ticks)):
                    self.enqueue(
                        conversation_id=clock_conversation_id,
                        event_type="clock.tick",
                        payload={"source": "timer", "ttl": 1},
                    )
                last_clock = _now_ms()
            worked = self.run_once()
            if not worked:
                time.sleep(max(0, sleep_ms) / 1000.0)
            i += 1
            if max_iterations > 0 and i >= max_iterations:
                break

    def render_cdc_pages(
        self, *, out_dir: Path, cdc_ws_url: str, embed_empty: bool
    ) -> Dict[str, Any]:
        template_html = (
            ROOT / "kogwistar" / "templates" / "d3.html"
        ).read_text(encoding="utf-8")
        meta = dump_paired_bundles(
            kg_engine=None if embed_empty else self.knowledge_engine,
            conversation_engine=None if embed_empty else self.conversation_engine,
            workflow_engine=None if embed_empty else self.workflow_engine,
            template_html=template_html,
            out_dir=out_dir,
            cdc_ws_url=cdc_ws_url,
            embed_empty=embed_empty,
        )
        return {
            "ok": True,
            "out_dir": str(out_dir.resolve()),
            "workflow_bundle": str((out_dir / "workflow.bundle.html").resolve()),
            "meta": meta,
        }

    def check_ollama(self) -> Dict[str, Any]:
        """Step -1: verify Ollama is installed/reachable for local LLM workflows."""
        try:
            p = subprocess.run(
                ["ollama", "--version"], capture_output=True, text=True, timeout=5
            )
            return {"ok": p.returncode == 0, "output": (p.stdout or p.stderr).strip()}
        except Exception as e:
            return {"ok": False, "output": str(e)}

    def get_hypergraph_snapshot(self) -> Dict[str, Any]:
        """Return compact hypergraph snapshot for tutorial display."""
        nodes = self.knowledge_engine.read.get_nodes(limit=200)
        edges = self.knowledge_engine.read.get_edges(limit=300)
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "hyper_edges": [
                {
                    "id": e.id,
                    "relation": e.relation,
                    "source_ids": list(e.source_ids or []),
                    "target_ids": list(e.target_ids or []),
                    "source_edge_ids": list(e.source_edge_ids or []),
                    "target_edge_ids": list(e.target_edge_ids or []),
                }
                for e in edges
                if (e.source_edge_ids or e.target_edge_ids)
            ],
        }

    def get_coerced_relationship_view(self) -> list[str]:
        """Flatten edges for beginner display while keeping hypergraph storage intact."""
        out: list[str] = []
        for e in self.knowledge_engine.read.get_edges(limit=300):
            src = list(e.source_ids or []) + [
                f"[edge:{x}]" for x in (e.source_edge_ids or [])
            ]
            dst = list(e.target_ids or []) + [
                f"[edge:{x}]" for x in (e.target_edge_ids or [])
            ]
            out.append(
                f"{','.join(src) or '(none)'} -[{e.relation}]-> {','.join(dst) or '(none)'}"
            )
        return out


def _parse_payload(s: str) -> Dict[str, Any]:
    x = json.loads(s)
    if not isinstance(x, dict):
        raise ValueError("--payload must be JSON object")
    return x


def _is_tcp_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


@dataclass
class ManagedCdcBridge:
    """Background CDC bridge lifecycle manager.

    Behavior:
    - If bridge port is already open, reuse external bridge (do not own lifecycle).
    - Otherwise spawn bridge subprocess and own shutdown on exit.
    """

    host: str = "127.0.0.1"
    port: int = 8787
    oplog_file: str = ".cdc_debug/data/cdc_oplog.jsonl"
    reset_oplog: bool = False
    proc: subprocess.Popen | None = None
    owned: bool = False

    def start(self, timeout_s: float = 8.0) -> dict[str, Any]:
        if _is_tcp_open(self.host, self.port):
            self.owned = False
            return {
                "started": True,
                "owned": False,
                "message": "Reusing already-running CDC bridge.",
            }

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run-cdc-bridge",
            "--host",
            str(self.host),
            "--port",
            str(self.port),
            "--oplog-file",
            str(self.oplog_file),
        ]
        if self.reset_oplog:
            cmd.append("--reset-oplog")

        self.proc = subprocess.Popen(cmd)
        self.owned = True

        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            if _is_tcp_open(self.host, self.port):
                return {
                    "started": True,
                    "owned": True,
                    "message": "CDC bridge started in background.",
                }
            if self.proc.poll() is not None:
                return {
                    "started": False,
                    "owned": True,
                    "message": f"CDC bridge exited early with code={self.proc.returncode}.",
                }
            time.sleep(0.15)
        return {
            "started": False,
            "owned": True,
            "message": "Timed out waiting for CDC bridge to become ready.",
        }

    def stop(self, timeout_s: float = 5.0) -> None:
        if not self.owned:
            return
        if self.proc is None:
            return
        if self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=2.0)


class TutorialResolver(MappingStepResolver):
    """Tutorial runtime resolver:
    - `get_input` blocks until an input event is available.
    - operations update state dict.
    - output stage emits output and may enqueue continuation.
    - `route` is authoritative for enqueue behavior:
      route=self => may enqueue continuation; route=output => never auto-enqueue.
    - `next_payload` under route=output is persisted as deferred metadata only.
    """

    def __init__(self, app: ClawRuntimeApp, stop_event: threading.Event) -> None:
        self.app = app
        self.stop_event = stop_event
        super().__init__(
            handlers={
                "get_input": self.get_input,
                "decide": self.decide,
                "execute": self.execute,
                "emit_output": self.emit_output,
                "end": self.end,
            }
        )

    def get_input(self, ctx) -> RunSuccess:
        # Blocking input op: waits for queued event (or stop signal).
        # No event means no work: this demonstrates event-sourced "idle by default".
        while not self.stop_event.is_set():
            ev = self.app.event_store.claim_next_pending()
            if ev is not None:
                payload = dict(ev.get("payload") or {})
                with ctx.state_write as s:
                    s["current_event"] = ev
                    s["current_payload"] = payload
                    s["current_ttl"] = int(payload.get("ttl", DEFAULT_TTL))
                return RunSuccess(conversation_node_id=None, state_update=[])
            time.sleep(0.25)
        with ctx.state_write as s:
            s["current_event"] = {
                "event_id": "stop",
                "conversation_id": "tutorial",
                "event_type": "system.stop",
                "payload": {"ttl": 0},
            }
            s["current_payload"] = {"ttl": 0}
            s["current_ttl"] = 0
        return RunSuccess(conversation_node_id=None, state_update=[])

    def decide(self, ctx) -> RunSuccess:
        e = dict(ctx.state_view.get("current_event") or {})
        p = dict(ctx.state_view.get("current_payload") or {})
        if e.get("event_type") == "clock.tick":
            op = "tick"
        elif str(p.get("tool") or "") == "add_knowledge":
            op = "add_knowledge"
        elif e.get("event_type") == "system.stop":
            op = "stop"
        else:
            op = "llm_route"
        with ctx.state_write as s:
            s["op"] = op
        return RunSuccess(conversation_node_id=None, state_update=[])

    def execute(self, ctx) -> RunSuccess:
        e = dict(ctx.state_view.get("current_event") or {})
        p = dict(ctx.state_view.get("current_payload") or {})
        ttl = int(ctx.state_view.get("current_ttl") or 0)
        op = str(ctx.state_view.get("op") or "llm_route")

        if op == "add_knowledge":
            decision = self.app.tool_registry["add_knowledge"](payload=p, ttl=ttl)
        elif op == "tick":
            if self.app.event_store.has_pending_user_message(
                conversation_id=str(e.get("conversation_id") or "tutorial"),
                exclude_event_id=str(e.get("event_id") or ""),
            ):
                decision = {
                    "route": "self",
                    "reason": "tick-found-work",
                    "next_event_type": "agent.loop",
                    "next_payload": {"tool": "llm_route", "text": "tick-followup"},
                    "output": {"message": "tick found work"},
                }
            else:
                decision = {
                    "route": "output",
                    "reason": "tick-idle",
                    "next_event_type": None,
                    "next_payload": None,
                    "output": {"message": "tick idle"},
                }
        elif op == "stop":
            decision = {
                "route": "output",
                "reason": "stop",
                "next_event_type": None,
                "next_payload": None,
                "output": {"message": "stopped"},
            }
        else:
            decision = self.app.tool_registry["llm_route"](payload=p, ttl=ttl)

        with ctx.state_write as s:
            s["decision"] = decision
        return RunSuccess(conversation_node_id=None, state_update=[])

    def emit_output(self, ctx) -> RunSuccess:
        e = dict(ctx.state_view.get("current_event") or {})
        d = dict(ctx.state_view.get("decision") or {})
        ttl = int(ctx.state_view.get("current_ttl") or 0)
        loops_done = int(ctx.state_view.get("demo_self_requeues_done") or 0)
        max_loops = int(ctx.state_view.get("max_demo_loops") or 2)
        route = str(d.get("route") or "output")
        next_payload_raw = d.get("next_payload")
        has_next_payload = isinstance(next_payload_raw, dict)
        should_attempt_enqueue = route == "self"
        budget_ok = (
            loops_done < max_loops
            and ttl > 0
            and str(e.get("event_type")) != "system.stop"
        )
        did_enqueue = False
        next_id: Optional[str] = None

        # route is authoritative:
        # - route=self  -> may enqueue continuation
        # - route=output -> never auto-enqueue (next_payload is deferred metadata)
        if should_attempt_enqueue and budget_ok:
            # If LLM chose continue but omitted payload, synthesize minimal continuation input.
            nxt = dict(
                next_payload_raw
                or {
                    "tool": "llm_route",
                    "text": "",
                    "source": "auto-continue-empty-next-payload",
                }
            )
            nxt["ttl"] = ttl - 1
            nxt["parent_event_id"] = e.get("event_id")
            next_id = self.app.event_store.enqueue_input(
                conversation_id=str(e.get("conversation_id") or "tutorial"),
                event_type=str(d.get("next_event_type") or "agent.loop"),
                payload=nxt,
            )
            did_enqueue = True

        out_payload: Dict[str, Any] = {"route": route, "reason": d.get("reason")}
        if d.get("output") is not None:
            out_payload["result"] = d.get("output")
        if next_id is not None:
            out_payload["next_event_id"] = next_id
            out_payload["next_event_type"] = str(
                d.get("next_event_type") or "agent.loop"
            )
        if route == "output" and has_next_payload:
            # Persist deferred payload for audit/replay, but do not auto-consume it.
            out_payload["deferred_next_payload"] = dict(next_payload_raw or {})
            out_payload["deferred_note"] = (
                "deferred only; requires future external event to be used"
            )

        self.app.event_store.append_output(
            conversation_id=str(e.get("conversation_id") or "tutorial"),
            event_type="claw.gate.output",
            payload=out_payload,
            source_event_id=str(e.get("event_id") or "unknown"),
            run_id=ctx.run_id,
        )

        # Demo guardrail counts only internal self-requeues, not all processed events.
        if did_enqueue:
            loops_done += 1
        allow_continue = (
            did_enqueue
            and loops_done < max_loops
            and str(e.get("event_type")) != "system.stop"
        )
        with (
            ctx.state_write as s
        ):  # this example use state_write lock to update state in op level, you can also submit up updates in RunSuccess
            s["demo_self_requeues_done"] = loops_done
            s["continue_loop"] = allow_continue
        return RunSuccess(conversation_node_id=None, state_update=[])

    def end(self, ctx) -> RunSuccess:
        # with ctx.state_write as s:
        #     s["continue_loop"] = False
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", "continue_loop")]
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenClaw-style runtime tutorial")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default=".gke-data/claw-loop")
    common.add_argument(
        "--cdc-publish-endpoint", default=os.getenv("CDC_PUBLISH_ENDPOINT", "")
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", parents=[common])
    pe = sub.add_parser("enqueue", parents=[common])
    pe.add_argument("--conversation-id", required=True)
    pe.add_argument("--event-type", default="user.message")
    pe.add_argument("--payload", required=True)
    pc = sub.add_parser("enqueue-clock", parents=[common])
    pc.add_argument("--conversation-id", default="__clock__")
    pc.add_argument(
        "--ttl",
        type=int,
        default=1,
        help="Loop budget for this event, not wall-clock TTL.",
    )
    sub.add_parser("run-once", parents=[common])
    pl = sub.add_parser("run-loop", parents=[common])
    pl.add_argument("--sleep-ms", type=int, default=500)
    pl.add_argument("--max-iterations", type=int, default=0)
    pl.add_argument("--clock-interval-ms", type=int, default=0)
    pl.add_argument("--clock-conversation-id", default="__clock__")
    pl.add_argument(
        "--max-pending-ticks",
        type=int,
        default=1,
        help="Cap queued pending clock.tick events for auto clock producer.",
    )
    ps = sub.add_parser("list-events", parents=[common])
    ps.add_argument("--direction", choices=["in", "out"], default="in")
    ps.add_argument("--limit", type=int, default=20)
    pr = sub.add_parser("render-cdc-pages", parents=[common])
    pr.add_argument("--out-dir", default=".cdc_debug/pages")
    pr.add_argument("--cdc-ws-url", default="ws://127.0.0.1:8787/changes/ws")
    pr.add_argument("--empty", action="store_true")
    pb = sub.add_parser("run-cdc-bridge", parents=[common])
    pb.add_argument("--host", default="127.0.0.1")
    pb.add_argument("--port", type=int, default=8787)
    pb.add_argument("--oplog-file", default=".cdc_debug/data/cdc_oplog.jsonl")
    pb.add_argument("--reset-oplog", action="store_true")
    sub.add_parser("seed-background", parents=[common])
    pp = sub.add_parser("repair-provenance", parents=[common])
    pp.add_argument("--doc-id", default="doc:background:hypergraph:001")
    pt = sub.add_parser("tutorial", parents=[common])
    pt.add_argument("--open-browser", action="store_true")
    pt.add_argument(
        "--max-demo-loops",
        type=int,
        default=2,
        help="Guardrail: max internal self-requeues before tutorial worker exits.",
    )
    pt.add_argument("--cdc-host", default="127.0.0.1")
    pt.add_argument("--cdc-port", type=int, default=8787)
    pt.add_argument("--cdc-oplog-file", default=".cdc_debug/data/cdc_oplog.jsonl")
    pt.add_argument("--reset-cdc-oplog", action="store_true")
    pt.add_argument(
        "--auto-cdc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-start/stop CDC bridge during tutorial.",
    )
    return p


def main() -> None:
    a = build_parser().parse_args()
    if a.cmd == "tutorial" and not a.cdc_publish_endpoint:
        a.cdc_publish_endpoint = f"http://{a.cdc_host}:{int(a.cdc_port)}/ingest"
    if a.cmd == "run-cdc-bridge":
        from kogwistar.cdc.change_bridge import main as bridge_main

        argv = [
            "--host",
            str(a.host),
            "--port",
            str(a.port),
            "--oplog-file",
            str(a.oplog_file),
        ]
        if a.reset_oplog:
            argv.append("--reset-oplog")
        raise SystemExit(bridge_main(argv))
    app = ClawRuntimeApp(
        data_dir=Path(a.data_dir), cdc_publish_endpoint=(a.cdc_publish_endpoint or None)
    )
    if a.cmd == "init":
        print(
            json.dumps(
                {
                    "ok": True,
                    "data_dir": str(Path(a.data_dir).resolve()),
                    "workflow_id": app.workflow_id,
                    "cdc_publish_endpoint": app.cdc_publish_endpoint,
                },
                indent=2,
            )
        )
        return
    if a.cmd == "enqueue":
        print(
            json.dumps(
                {
                    "ok": True,
                    "event_id": app.enqueue(
                        conversation_id=a.conversation_id,
                        event_type=a.event_type,
                        payload=_parse_payload(a.payload),
                    ),
                },
                indent=2,
            )
        )
        return
    if a.cmd == "enqueue-clock":
        print(
            json.dumps(
                {
                    "ok": True,
                    "event_id": app.enqueue(
                        conversation_id=a.conversation_id,
                        event_type="clock.tick",
                        payload={"source": "manual", "ttl": int(a.ttl)},
                    ),
                },
                indent=2,
            )
        )
        return
    if a.cmd == "run-once":
        print(
            json.dumps({"ok": True, "processed_event": bool(app.run_once())}, indent=2)
        )
        return
    if a.cmd == "seed-background":
        seeded = app.seed_background_hypergraph()
        print(json.dumps({"ok": True, "seeded": seeded}, indent=2))
        return
    if a.cmd == "repair-provenance":
        report = app.repair_provenance_for_doc(a.doc_id)
        print(json.dumps({"ok": True, "report": report}, indent=2))
        return
    if a.cmd == "tutorial":
        cdc_mgr: ManagedCdcBridge | None = None
        ws_url = f"ws://{a.cdc_host}:{int(a.cdc_port)}/changes/ws"

        # -1) Make sure Ollama.
        ollama = app.check_ollama()
        print(f"[Step -1] Ollama check: ok={ollama['ok']} output={ollama['output']}")

        # 0) Start CDC bridge (or reuse existing one) before first write.
        if a.auto_cdc:
            cdc_mgr = ManagedCdcBridge(
                host=str(a.cdc_host),
                port=int(a.cdc_port),
                oplog_file=str(a.cdc_oplog_file),
                reset_oplog=bool(a.reset_cdc_oplog),
            )
            status = cdc_mgr.start()
            print(f"[Step 0] {status['message']}")
            if not status["started"]:
                print(
                    "[Step 0] CDC bridge did not start. You can run with --no-auto-cdc and start it manually."
                )
        else:
            print(
                "[Step 0] Auto CDC disabled. Ensure bridge is running manually before first write."
            )

        # 1) Render CDC HTML and optionally open browser.
        pages = app.render_cdc_pages(
            out_dir=Path(".cdc_debug/pages"), cdc_ws_url=ws_url, embed_empty=True
        )
        print(f"[Step 1] Rendered CDC pages under {pages['out_dir']}")
        if a.open_browser:
            webbrowser.open(str((Path(pages["out_dir"]) / "kg.bundle.html").resolve()))
            webbrowser.open(
                str((Path(pages["out_dir"]) / "workflow.bundle.html").resolve())
            )

        # 2) Seed hypergraph data + retrieval.
        seeded = app.seed_background_hypergraph()
        print(
            f"[Step 2] Seeded hypergraph nodes/edges: {json.dumps(seeded, ensure_ascii=False)}"
        )
        print(
            f"[Step 2] Provenance span repair report (pre-persist): {json.dumps(seeded.get('provenance_repair', {}), ensure_ascii=False)}"
        )
        snap = app.get_hypergraph_snapshot()
        print(f"[Step 2] Hypergraph snapshot: {json.dumps(snap, ensure_ascii=False)}")
        print("[Step 2] Open KG CDC viewer: .cdc_debug/pages/kg.bundle.html")

        # 3) Explain event-sourcing.
        print(
            "[Step 3] All writes/events are event-sourced and CDC-streamed to viewer."
        )

        # 4) Coerced relationship view while storage remains hypergraph.
        print(
            "[Step 4] Coerced relationship view (edge endpoints can be node or edge):"
        )
        for line in app.get_coerced_relationship_view()[:12]:
            print(f"  - {line}")

        # 5) Design workflow with TTL guard + exit queue semantics.
        wfid = app.ensure_tutorial_workflow()
        print(f"[Step 5] Tutorial workflow ready: {wfid}")
        print(
            "[Step 5] Predicate abstraction: if ttl<=0, no self-requeue; output to exit queue."
        )

        # 6) Workflow CDC view.
        print(
            "[Step 6] Open workflow CDC viewer: .cdc_debug/pages/workflow.bundle.html"
        )

        # 7) Explain ops/state/resolver + vendor-agnostic adapters.
        print(
            "[Step 7] Ops=get_input/decide/execute/emit_output, state=dict, resolver=TutorialResolver."
        )
        print(
            "[Step 7] LLM adapter example uses Azure by default; Gemini section is commented in _llm_route."
        )
        print(
            "[Step 7] get_input blocks when queue is empty; worker runs in separate thread."
        )
        print(
            "[Step 7] Policy: route=self may enqueue continuation; route=output stores next_payload as deferred metadata only."
        )

        # 8) Run worker thread; user can enqueue questions/ticks.
        stop_evt = threading.Event()
        pred = {
            "should_continue": lambda _wf, st, _r: bool(st.get("continue_loop", False))
        }
        rt = WorkflowRuntime(
            workflow_engine=app.workflow_engine,
            conversation_engine=app.conversation_engine,
            step_resolver=TutorialResolver(app=app, stop_event=stop_evt).resolve,
            predicate_registry=pred,
            checkpoint_every_n_steps=1,
            max_workers=1,
        )

        def _worker() -> None:
            rt.run(
                workflow_id=wfid,
                conversation_id="tutorial-conversation",
                turn_node_id="tutorial-turn-0",
                initial_state={
                    "max_demo_loops": int(a.max_demo_loops),
                    "demo_self_requeues_done": 0,
                    "continue_loop": True,
                },
                run_id=f"tutorial|{uuid.uuid4()}",
            )

        t = threading.Thread(
            target=_worker,
            daemon=True,
            name="tutorial-conversation-worker|TutorialResolver",
        )
        t.start()
        print(
            "[Step 8] Worker started. Type a knowledge question; /tick to enqueue clock.tick; /quit to stop."
        )

        # 9) Conversation graph viewer guidance.
        print(
            "[Step 9] Open conversation CDC viewer: .cdc_debug/pages/conversation.bundle.html"
        )
        print(
            "[Step 9] Graph may look messy; backbone is expected. Build custom viewer filters as needed."
        )

        # 10) Keep loop running until quit or guardrail exit.
        # Important tutorial behavior:
        # this input loop is bound to worker liveness, so it is NOT infinite by default.
        # If no self-requeue path is taken, worker may finish and CLI loop exits.
        try:
            while t.is_alive():
                user_text = input("tutorial> ").strip()
                if not user_text:
                    continue
                if user_text.lower() == "/quit":
                    stop_evt.set()
                    app.enqueue(
                        conversation_id="tutorial-conversation",
                        event_type="system.stop",
                        payload={"ttl": 0},
                    )
                    break
                if user_text.lower() == "/tick":
                    app.enqueue(
                        conversation_id="tutorial-conversation",
                        event_type="clock.tick",
                        payload={"ttl": 1, "source": "manual"},
                    )
                    continue
                app.enqueue(
                    conversation_id="tutorial-conversation",
                    event_type="user.message",
                    payload={"text": user_text, "tool": "llm_route", "ttl": 2},
                )
            t.join(timeout=3)
        except KeyboardInterrupt:
            stop_evt.set()
            app.enqueue(
                conversation_id="tutorial-conversation",
                event_type="system.stop",
                payload={"ttl": 0},
            )
        finally:
            if cdc_mgr is not None:
                cdc_mgr.stop()
        print("[Step 10] Tutorial finished. Loop can be cancelled anytime.")
        return
    if a.cmd == "run-loop":
        app.run_loop(
            sleep_ms=a.sleep_ms,
            max_iterations=a.max_iterations,
            clock_interval_ms=a.clock_interval_ms,
            clock_conversation_id=a.clock_conversation_id,
            max_pending_ticks=a.max_pending_ticks,
        )
        print(json.dumps({"ok": True}, indent=2))
        return
    if a.cmd == "list-events":
        print(
            json.dumps(
                {
                    "ok": True,
                    "events": app.event_store.list_events(
                        direction=a.direction, limit=a.limit
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if a.cmd == "render-cdc-pages":
        print(
            json.dumps(
                app.render_cdc_pages(
                    out_dir=Path(a.out_dir),
                    cdc_ws_url=a.cdc_ws_url,
                    embed_empty=bool(a.empty),
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    raise RuntimeError(f"Unknown command: {a.cmd}")


# Cheatsheet (copy-paste)
# python scripts/claw_runtime_loop.py run-cdc-bridge --host 127.0.0.1 --port 8787 --reset-oplog
# python scripts/claw_runtime_loop.py init --data-dir .gke-data/claw-loop --cdc-publish-endpoint http://127.0.0.1:8787/ingest
# python scripts/claw_runtime_loop.py render-cdc-pages --data-dir .gke-data/claw-loop --out-dir .cdc_debug/pages --cdc-ws-url ws://127.0.0.1:8787/changes/ws --empty
# python scripts/claw_runtime_loop.py seed-background --data-dir .gke-data/claw-loop
# python scripts/claw_runtime_loop.py repair-provenance --data-dir .gke-data/claw-loop --doc-id doc:background:hypergraph:001
# python scripts/claw_runtime_loop.py enqueue --data-dir .gke-data/claw-loop --conversation-id conv-demo --event-type user.message --payload "{\"text\":\"draft\",\"unfinished\":true,\"tool\":\"llm_route\",\"ttl\":3}"
# python scripts/claw_runtime_loop.py enqueue --data-dir .gke-data/claw-loop --conversation-id conv-demo --event-type user.message --payload "{\"tool\":\"add_knowledge\",\"doc_id\":\"doc:demo\",\"subject\":\"OpenClaw\",\"relation\":\"teaches\",\"object\":\"event sourcing\",\"text\":\"fact\",\"ttl\":2}"
# python scripts/claw_runtime_loop.py run-loop --data-dir .gke-data/claw-loop --sleep-ms 300 --clock-interval-ms 3000
# python scripts/claw_runtime_loop.py list-events --data-dir .gke-data/claw-loop --direction in --limit 30
# python scripts/claw_runtime_loop.py list-events --data-dir .gke-data/claw-loop --direction out --limit 30
# python scripts/claw_runtime_loop.py tutorial --data-dir .gke-data/claw-loop --open-browser
# python scripts/claw_runtime_loop.py tutorial --data-dir .gke-data/claw-loop --open-browser --no-auto-cdc


if __name__ == "__main__":
    main()
