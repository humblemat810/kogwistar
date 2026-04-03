from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

from kogwistar.engine_core.scoped_seq import (
    ScopedSeqHookConfig,
    install_scoped_seq_hooks,
    maybe_assign_node_scoped_seq,
)
from kogwistar.graph_kinds import KIND_CHAT

from .models import (
    ConversationEdge,
    ConversationNode,
    CONVERSATION_EDGE_CAUSAL_TYPE_BY_RELATION,
)


def infer_conversation_edge_causal_type(relation: str) -> str:
    return CONVERSATION_EDGE_CAUSAL_TYPE_BY_RELATION.get(relation, "reference")


if TYPE_CHECKING:
    from kogwistar.engine_core.engine import GraphKnowledgeEngine


def where_and(*clauses: dict) -> dict:
    flat = [c for c in clauses if c]
    if not flat:
        return {}
    if len(flat) == 1:
        return flat[0]
    return {"$and": flat}


def edge_endpoints_exists(engine: "GraphKnowledgeEngine", *, where: dict) -> bool:
    res = engine.backend.edge_endpoints_get(where=where, include=["metadatas"], limit=1)
    mds = res.get("metadatas") or []
    return bool(mds and mds[0])


def edge_endpoints_first_edge_id(
    engine: "GraphKnowledgeEngine", *, where: dict
) -> str | None:
    res = engine.backend.edge_endpoints_get(where=where, include=["metadatas"], limit=1)
    mds = res.get("metadatas") or []
    if not mds or not mds[0]:
        return None
    md = mds[0]
    eid = md.get("edge_id")
    if isinstance(eid, str) and eid:
        return eid
    rid = md.get("id")
    if isinstance(rid, str) and "::" in rid:
        return rid.split("::", 1)[0]
    return None


def doc_id_for_edge(edge: ConversationEdge) -> str | None:
    doc_id = getattr(edge, "doc_id", None)
    if isinstance(doc_id, str) and doc_id:
        return doc_id
    md = edge.metadata or {}
    conv_id = (
        md.get("conversation_id")
        or md.get("conv_id")
        or getattr(edge, "conversation_id", None)
    )
    if isinstance(conv_id, str) and conv_id:
        return f"conv:{conv_id}"
    return None


def is_duplicate_next_turn_noop(
    engine: "GraphKnowledgeEngine", edge: ConversationEdge
) -> bool:
    if engine.kg_graph_type != KIND_CHAT:
        return False
    if edge.relation != "next_turn":
        return False

    src = (edge.source_ids or [None])[0]
    tgt = (edge.target_ids or [None])[0]
    if not src or not tgt:
        return False

    doc_id = doc_id_for_edge(edge)
    w_out = where_and(
        {"relation": "next_turn"},
        {"role": "src"},
        {"endpoint_type": "node"},
        {"endpoint_id": src},
        ({"doc_id": doc_id} if doc_id else {}),
    )
    existing_eid = edge_endpoints_first_edge_id(engine, where=w_out)
    if not existing_eid:
        return False

    got = engine.backend.edge_get(ids=[existing_eid], include=["documents"])
    docs = got.get("documents") or []
    if not docs:
        return False
    try:
        obj = json.loads(docs[0])
    except Exception:
        return False

    if obj.get("relation") != "next_turn":
        return False
    if obj.get("doc_id") != doc_id and doc_id is not None:
        return False

    same_src = obj.get("source_ids") == [src]
    same_tgt = obj.get("target_ids") == [tgt]
    same_src_edges = (obj.get("source_edge_ids") or []) == (
        getattr(edge, "source_edge_ids", []) or []
    )
    same_tgt_edges = (obj.get("target_edge_ids") or []) == (
        getattr(edge, "target_edge_ids", []) or []
    )
    return bool(same_src and same_tgt and same_src_edges and same_tgt_edges)


def normalize_edge_metadata(edge: ConversationEdge) -> None:
    md = dict(edge.metadata or {})
    if md.get("causal_type") is None:
        md["causal_type"] = infer_conversation_edge_causal_type(edge.relation)
    edge.metadata = md


def validate_edge_add(engine: "GraphKnowledgeEngine", edge: ConversationEdge) -> None:
    if engine.kg_graph_type != KIND_CHAT:
        return

    normalize_edge_metadata(edge)
    md = edge.metadata or {}
    causal_type = md.get("causal_type") or infer_conversation_edge_causal_type(
        edge.relation
    )
    doc_id = doc_id_for_edge(edge)

    src = (edge.source_ids or [None])[0]
    tgt = (edge.target_ids or [None])[0]

    if edge.relation == "next_turn" or causal_type == "chain":
        if src is None or tgt is None:
            raise ValueError("next_turn requires single source_id and single target_id")
        if (getattr(edge, "source_edge_ids", []) or []) or (
            getattr(edge, "target_edge_ids", []) or []
        ):
            raise ValueError("next_turn must be node-to-node only (no edge endpoints)")

        w_out = where_and(
            {"relation": "next_turn"},
            {"role": "src"},
            {"endpoint_type": "node"},
            {"endpoint_id": src},
            ({"doc_id": doc_id} if doc_id else {}),
        )
        if edge_endpoints_exists(engine, where=w_out):
            raise ValueError(f"next_turn outgoing already exists for source_id={src}")

        w_in = where_and(
            {"relation": "next_turn"},
            {"role": "tgt"},
            {"endpoint_type": "node"},
            {"endpoint_id": tgt},
            ({"doc_id": doc_id} if doc_id else {}),
        )
        if edge_endpoints_exists(engine, where=w_in):
            raise ValueError(f"next_turn incoming already exists for target_id={tgt}")

    if causal_type == "dependency":
        if tgt is None:
            raise ValueError("dependency edge requires single target_id")

        w_used_chain = where_and(
            {"role": "src"},
            {"endpoint_type": "node"},
            {"endpoint_id": tgt},
            {"causal_type": "chain"},
            ({"doc_id": doc_id} if doc_id else {}),
        )
        w_used_dep = where_and(
            {"role": "src"},
            {"endpoint_type": "node"},
            {"endpoint_id": tgt},
            {"causal_type": "dependency"},
            ({"doc_id": doc_id} if doc_id else {}),
        )
        if edge_endpoints_exists(engine, where=w_used_chain) or edge_endpoints_exists(
            engine, where=w_used_dep
        ):
            raise ValueError(
                f"Cannot add dependency incoming edge into already-used node {tgt}"
            )


def _conversation_scope_id(
    engine: "GraphKnowledgeEngine", subject: Any
) -> str | None:
    _ = engine
    conv_id = getattr(subject, "conversation_id", None)
    if conv_id is None:
        conv_id = (getattr(subject, "metadata", {}) or {}).get("conversation_id")
    if conv_id is None:
        return None
    return str(conv_id)


def _conversation_should_stamp(
    engine: "GraphKnowledgeEngine", subject: Any
) -> bool:
    _ = subject
    return bool(getattr(engine, "kg_graph_type", None) == KIND_CHAT)


_CONVERSATION_SEQ_HOOK_CONFIG = ScopedSeqHookConfig(
    metadata_field="seq",
    should_stamp_node=_conversation_should_stamp,
    scope_id_for_node=_conversation_scope_id,
)


def maybe_assign_seq(engine: "GraphKnowledgeEngine", node: Any) -> None:
    maybe_assign_node_scoped_seq(engine, node, config=_CONVERSATION_SEQ_HOOK_CONFIG)


def get_last_seq_node(engine: "GraphKnowledgeEngine", conversation_id, min_seq=None):
    if min_seq is None:
        min_seq = engine.meta_sqlite.next_user_seq(conversation_id)
    if engine.kg_graph_type != KIND_CHAT:
        raise RuntimeError("chat-only call")
    got = engine.backend.node_get(
        where={
            "$and": [{"conversation_id": conversation_id}]
            + [{"seq": {"$gte": min_seq or 0}}],
        },
        include=["documents", "metadatas", "embeddings"],
    )
    if not got["ids"]:
        return None
    nodes: list[ConversationNode] = engine.read.nodes_from_single_or_id_query_result(
        got, node_type=ConversationNode
    )
    nodes.sort(key=lambda n: n.metadata.get("seq") or -1)
    return nodes[-1]


def get_chat_tail(
    engine: "GraphKnowledgeEngine",
    conversation_id: str,
    min_turn_index: int | None = None,
    tail_search_includes: list[str] = [
        "conversation_start",
        "conversation_turn",
        "conversation_summary",
        "assistant_turn",
    ],
) -> Optional[ConversationNode]:
    if engine.kg_graph_type != KIND_CHAT:
        raise RuntimeError("chat-only call")
    got = engine.backend.node_get(
        where={
            "$and": [
                {"conversation_id": conversation_id},
                {"in_conversation_chain": True},
            ]
            + (
                [{"turn_index": {"$gte": min_turn_index}}]
                if min_turn_index is not None
                else []
            )
        },
        include=["documents", "metadatas", "embeddings"],
    )
    if not got["ids"]:
        return None
    nodes: list[ConversationNode] = engine.read.nodes_from_single_or_id_query_result(
        got, node_type=ConversationNode
    )
    nodes2 = [x for x in nodes if x.metadata.get("entity_type") in tail_search_includes]
    if not nodes2:
        return None
    nodes2.sort(key=lambda n: n.turn_index or -1)
    return nodes2[-1]


def last_summary_of_node(engine: "GraphKnowledgeEngine", node: ConversationNode):
    summaries = engine.read.get_nodes(
        where=where_and(
            {"conversation_id": node.conversation_id},
            {"entity_type": "conversation_summary"},
        ),
        node_type=ConversationNode,
        limit=20000,
    )
    best = None
    best_idx = -1
    for s in summaries or []:
        ti = getattr(s, "turn_index", None)
        if ti is None:
            continue
        if node.turn_index is not None and ti <= node.turn_index and ti > best_idx:
            best = s
            best_idx = ti
    return [best] if best is not None else []


def install_engine_hooks(engine: "GraphKnowledgeEngine") -> None:
    if getattr(engine, "_chat_policy_hooks_ready", False):
        return
    install_scoped_seq_hooks(
        engine,
        _CONVERSATION_SEQ_HOOK_CONFIG,
        ready_attr="_chat_policy_scoped_seq_hooks_ready",
    )

    def _edge_hook(edge: Any) -> bool:
        if not isinstance(edge, ConversationEdge):
            return False
        if is_duplicate_next_turn_noop(engine, edge):
            return True
        validate_edge_add(engine, edge)
        return False

    def _edge_hook_pure(edge: Any) -> bool:
        if not isinstance(edge, ConversationEdge):
            return False
        validate_edge_add(engine, edge)
        return False

    def _allow_missing_doc(edge: Any) -> bool:
        return isinstance(edge, ConversationEdge)

    edge_hooks = getattr(engine, "pre_add_edge_hooks", None)
    if isinstance(edge_hooks, list):
        edge_hooks.append(_edge_hook)
    pure_edge_hooks = getattr(engine, "pre_add_pure_edge_hooks", None)
    if isinstance(pure_edge_hooks, list):
        pure_edge_hooks.append(_edge_hook_pure)
    missing_doc_hooks = getattr(
        engine, "allow_missing_doc_id_on_endpoint_rows_hooks", None
    )
    if isinstance(missing_doc_hooks, list):
        missing_doc_hooks.append(_allow_missing_doc)

    setattr(engine, "_chat_policy_hooks_ready", True)
