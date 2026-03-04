from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Self, Sequence, TypeAlias 
from .typing_interfaces import EngineLike
from typing import Iterable

from .models import ConversationEdge
from .models import ConversationNode
import json

Role: TypeAlias =  Literal["system", "user", "assistant", "tool"]
# system include kg graph, internal summary, filtering thinking, reasoning from llm call
# user include user input

Source = Literal[
    "live_turn",
    "history_turn",
    "conversation_summary",
    "memory_summary",
    "memory_pinned",
    "kg_ref",
    "kg_ref_pinned",
    "system",
    "summary",
]

ItemKind = Literal[
    "system_prompt",
    "head_summary",
    "tail_turn",
    "memory_context",
    "pinned_memory",
    "kg_ref",
    "pinned_kg_ref",
    "tool_state",
]

DropReason = Literal[
    "over_budget",
    "superseded",
    "duplicate",
    "too_old",
    "low_priority",
    "compressed",
]

@dataclass(frozen=True)
class ContextMessage:
    role: Role
    content: str
    # provenance
    node_id: str | None = None
    source: Source = "history_turn"


@dataclass(frozen=True)
class ContextItem:
    """
    A single packable unit (turn / summary / memory / KG ref).
    This is the unit of budgeting + traceability.
    subclass in use: memory context with memctx_ids
    """
    kind: ItemKind
    text: str
    # Conversation
    role : Role
    
    extra: dict|None = None
    
    # provenance / tracing
    node_id: str | None = None
    edge_ids: tuple[str, ...] = ()
    pointer_ids: tuple[str, ...] = ()     # e.g. span ids, doc pointers
    cluster_ids: tuple[str, ...] = ()     # your pX_cY clusters if relevant

    # packing controls
    priority: int = 100                  # lower = more important
    pinned: bool = False                 # pinned => harder to drop
    max_tokens: int | None = None        # optional per-item cap (for compression)
    source: Source = "history_turn"

    # accounting (filled by builder)
    token_cost: int = 0


@dataclass(frozen=True)
class DroppedItem:
    kind: ItemKind
    node_id: str | None
    reason: DropReason
    token_cost: int


from typing import Protocol, Callable, Mapping

# -------------------------
# Phase 2B: Pluggable ordering strategies (snapshot-friendly)
# -------------------------

class ContextOrderingStrategy(Protocol):
    """Ordering policy for context items.

    IMPORTANT: This influences only *new* context construction.
    Replay must use the persisted snapshot order (ordinals / used_node_ids).
    """
    name: str

    def pre_pack(self, items: list[ContextItem]) -> list[ContextItem]:
        """Return the iteration order used for packing/budgeting."""

    def post_pack(self, kept: list[ContextItem]) -> list[ContextItem]:
        """Return final order fed into the renderer."""

class OrderingRegistry:
    def __init__(self) -> None:
        self._m: dict[str, ContextOrderingStrategy] = {}

    def register(self, strat: ContextOrderingStrategy) -> None:
        self._m[strat.name] = strat

    def get(self, name: str | None) -> ContextOrderingStrategy:
        if not name:
            name = "default"
        if name not in self._m:
            raise KeyError(f"Unknown ordering strategy: {name!r}. Known: {sorted(self._m)}")
        return self._m[name]

ORDERING_REGISTRY = OrderingRegistry()

class _DefaultOrdering:
    """Matches the current repository behavior (do not change defaults)."""
    name = "default"
    def pre_pack(self, items: list[ContextItem]) -> list[ContextItem]:
        out = list(items)
        # Pinned first, then priority (lower=more important)
        out.sort(key=lambda x: (not x.pinned, x.priority, x.kind, (x.node_id or "")))
        return out
    def post_pack(self, kept: list[ContextItem]) -> list[ContextItem]:
        # Keep as chosen by packing (stable)
        return list(kept)

class _GraphDerivedOrdering:
    """Preserve gather order as much as possible (stable, minimal policy)."""
    name = "graph_derived"
    def pre_pack(self, items: list[ContextItem]) -> list[ContextItem]:
        return list(items)
    def post_pack(self, kept: list[ContextItem]) -> list[ContextItem]:
        return list(kept)

class _GroupedPolicyOrdering:
    """Group by kind then stable within group (human-friendly prompts)."""
    name = "grouped_policy"
    _rank: dict[str, int] = {
        "system_prompt": 0,
        "head_summary": 1,
        "memory_context": 2,
        "pinned_memory": 2,
        "pinned_kg_ref": 3,
        "kg_ref": 3,
        "tool_state": 4,
        "tail_turn": 5,
    }
    def pre_pack(self, items: list[ContextItem]) -> list[ContextItem]:
        out = list(items)
        out.sort(key=lambda x: (not x.pinned, x.priority, self._rank.get(x.kind, 99), (x.node_id or "")))
        return out
    def post_pack(self, kept: list[ContextItem]) -> list[ContextItem]:
        out = list(kept)
        # turns in chronological order if provided
        def turn_ix(it: ContextItem) -> int:
            if it.kind != "tail_turn":
                return 10**9
            extra = it.extra or {}
            return int(extra.get("turn_index", 10**9))
        out.sort(key=lambda x: (self._rank.get(x.kind, 99), turn_ix(x), (x.node_id or "")))
        return out

ORDERING_REGISTRY.register(_DefaultOrdering())
ORDERING_REGISTRY.register(_GraphDerivedOrdering())
ORDERING_REGISTRY.register(_GroupedPolicyOrdering())

def apply_ordering(*, items: list[ContextItem], ordering: str | None, phase: Literal["pre_pack","post_pack"]) -> list[ContextItem]:
    strat = ORDERING_REGISTRY.get(ordering)
    if phase == "pre_pack":
        return strat.pre_pack(items)
    return strat.post_pack(items)
from pydantic import BaseModel, model_validator


class PromptContext(BaseModel):
    """A rendered prompt context for an LLM call.

    Mental model:
    - This object is the *LLM-facing* view: an ordered list of `ContextItem`s
      plus the final `ContextMessage`s that will be sent to the model.
    - It is primarily a **debug/telemetry artifact** produced at runtime.
      It is safe to persist as a *snapshot* when you need replayability.
    - It is NOT meant to represent the entire conversation graph.

    Naming:
    - This used to be called `ConversationContextView`.
      We keep a backwards-compatible alias below.
    """
    conversation_id: str
    purpose: str

    token_budget: int
    tokens_used: int

    # the *plan*
    items: Sequence[ContextItem]

    # final LLM-ready messages
    messages: Sequence[ContextMessage]

    # trace
    included_node_ids: tuple[str, ...] = ()
    included_edge_ids: tuple[str, ...] = ()
    included_pointer_ids: tuple[str, ...] = ()
    dropped: Sequence[DroppedItem] = field(default_factory=tuple)

    # helpful partitions for debugging/telemetry
    head_summary_ids: tuple[str, ...] = ()
    tail_turn_ids: tuple[str, ...] = ()
    active_memory_context_ids: tuple[str, ...] = ()
    pinned_memory_ids: tuple[str, ...] = ()
    pinned_kg_ref_ids: tuple[str, ...] = ()
    @model_validator(mode='after')
    def assert_valid(self) -> Self:
        # Basic budget invariant
        if self.token_budget < 0:
            raise ValueError(f"token_budget must be >= 0, got {self.token_budget}")
        if self.tokens_used < 0:
            raise ValueError(f"tokens_used must be >= 0, got {self.tokens_used}")
        if self.tokens_used > self.token_budget:
            raise ValueError(f"Context overflow: {self.tokens_used} > {self.token_budget}")

        # Must have at least one message (system prelude or turns)
        if not self.messages or len(self.messages) == 0:
            raise ValueError("Empty context view: messages is empty")

        # Items/messages should not be empty at the same time (plan must exist)
        if getattr(self, "items", None) is not None and len(self.items) == 0:
            raise ValueError("Empty context view: items is empty")

        # If you have items with token_cost populated, ensure accounting is sane
        # (Don’t force equality because renderer may add small overhead; allow a small slack.)
        if getattr(self, "items", None):
            item_sum = 0
            missing_cost = False
            for it in self.items:
                tc = getattr(it, "token_cost", None)
                if tc is None:
                    missing_cost = True
                    break
                item_sum += int(tc)

            if not missing_cost:
                # Allow small slack for role/format overhead or tokenizer mismatch
                slack = 64
                if abs(item_sum - int(self.tokens_used)) > slack:
                    raise ValueError(
                        f"Token accounting mismatch: sum(items.token_cost)={item_sum} "
                        f"but tokens_used={self.tokens_used} (slack={slack})"
                    )

        # Trace sanity: included_node_ids should be subset of item node_ids
        if getattr(self, "included_node_ids", None) and getattr(self, "items", None):
            item_nodes = {it.node_id for it in self.items if getattr(it, "node_id", None)}
            bad = [nid for nid in self.included_node_ids if nid not in item_nodes]
            if bad:
                raise ValueError(f"included_node_ids contains IDs not present in items: {bad[:10]}")

        # Dropped trace sanity (optional)
        if getattr(self, "dropped", None):
            for d in self.dropped:
                if getattr(d, "reason", None) not in {
                    "over_budget", "superseded", "duplicate", "too_old",
                    "low_priority", "compressed"
                }:
                    raise ValueError(f"Unknown drop reason: {getattr(d, 'reason', None)}")
        return self


# Backwards-compatibility: older code may still import ConversationContextView.
ConversationContextView = PromptContext

class ContextRenderer:
    def render(self, items, *, purpose: str):

        system_prompt_parts: list[str] = []
        head_summaries: list[str] = []
        mem_contexts: list[str] = []
        pinned_refs: list[str] = []
        turn_items = []

        for it in items:
            if it.kind == "system_prompt":
                system_prompt_parts.append(it.text.strip())
            elif it.kind == "head_summary" and it.text.strip():
                head_summaries.append(it.text.strip())
            elif it.kind in ("memory_context", "pinned_memory") and it.text.strip():
                mem_contexts.append(it.text.strip())
            elif it.kind in ("pinned_kg_ref", "kg_ref") and it.text.strip():
                pinned_refs.append(it.text.strip())
            elif it.kind == "tail_turn":
                turn_items.append(it)

        prelude_chunks: list[str] = []
        if system_prompt_parts:
            prelude_chunks.append("\n\n".join(system_prompt_parts).strip())

        if head_summaries:
            prelude_chunks.append("Conversation summary:\n" + "\n\n".join(head_summaries))

        if mem_contexts:
            prelude_chunks.append("Memory context:\n" + "\n\n".join(mem_contexts))

        if pinned_refs:
            bullets = "\n".join([f"- {r.replace(chr(10), ' | ')}" for r in pinned_refs])
            prelude_chunks.append("Pinned knowledge references:\n" + bullets)

        messages: list[ContextMessage] = []
        if prelude_chunks:
            messages.append(
                ContextMessage(role="system", content="\n\n".join(prelude_chunks).strip(), source="system")
            )

        for it in turn_items:
            role = it.role or "user"
            messages.append(
                ContextMessage(role=role, content=it.text, node_id=it.node_id, source=it.source)
            )

        if not messages:
            messages.append(ContextMessage(role="system", content="", source="system"))

        return messages
class ConversationContextBuilder:
    def __init__(self, *, sources, tokenizer, renderer):
        """
        sources: gathers candidate ContextItems (turns/summaries/memory/kg refs)
        tokenizer: count_tokens()
        renderer: converts items -> final LLM messages
        """
        self.sources = sources
        self.tokenizer = tokenizer
        self.renderer = renderer

    def build(self, *, conversation_id: str, purpose: str, budget_tokens: int, ordering_strategy: str | None = None) -> PromptContext:
        # 1) gather candidates
        candidates: list[ContextItem] = self.sources.gather(
            conversation_id=conversation_id,
            purpose=purpose,
        )

        # 2) price them
        priced: list[ContextItem] = []
        for it in candidates:
            cost = self.tokenizer.count_tokens(it.text)
            priced.append(ContextItem(**{**it.__dict__, "token_cost": cost}))

        # 3) deterministic ordering: pinned first, then priority, then recency if needed
        # (sources should encode recency into priority or provide stable tie-break keys)
        priced = apply_ordering(items=priced, ordering=ordering_strategy, phase="pre_pack")

        # 4) pack
        kept: list[ContextItem] = []
        dropped: list[DroppedItem] = []
        used = 0

        for it in priced:
            if used + it.token_cost <= budget_tokens:
                kept.append(it)
                used += it.token_cost
            else:
                # attempt compression if allowed
                if it.max_tokens is not None and it.max_tokens < it.token_cost:
                    # compress by truncation or summarizer hook (your choice)
                    # Here: cheap truncation as placeholder
                    compressed_text = self._truncate_to_tokens(it.text, it.max_tokens)
                    compressed_cost = self.tokenizer.count_tokens(compressed_text)
                    if used + compressed_cost <= budget_tokens:
                        kept.append(ContextItem(**{**it.__dict__, "text": compressed_text, "token_cost": compressed_cost}))
                        used += compressed_cost
                        dropped.append(DroppedItem(kind=it.kind, node_id=it.node_id, reason="compressed", token_cost=it.token_cost))
                        continue

                # drop (but do not drop pinned system prompt)
                if it.kind == "system_prompt":
                    # last resort: hard fail; system prompt must fit
                    raise ValueError("System prompt alone exceeds budget")
                dropped.append(DroppedItem(kind=it.kind, node_id=it.node_id, reason="over_budget", token_cost=it.token_cost))

        # 5) final ordering for rendering
        kept = apply_ordering(items=list(kept), ordering=ordering_strategy, phase="post_pack")

        # 6) render to LLM messages
        messages = self.renderer.render(kept, purpose=purpose)

        # 6) build trace fields
        included_node_ids = tuple(sorted({i.node_id for i in kept if i.node_id}))
        included_edge_ids = tuple(sorted({e for i in kept for e in i.edge_ids}))
        included_pointer_ids = tuple(sorted({p for i in kept for p in i.pointer_ids}))

        head_summary_ids = tuple(i.node_id for i in kept if i.kind == "head_summary" and i.node_id)
        tail_turn_ids = tuple(i.node_id for i in kept if i.kind == "tail_turn" and i.node_id)
        active_memory_context_ids = tuple(i.node_id for i in kept if i.kind == "memory_context" and i.node_id)
        pinned_memory_ids = tuple(i.node_id for i in kept if i.kind == "pinned_memory" and i.node_id)
        pinned_kg_ref_ids = tuple(i.node_id for i in kept if i.kind == "pinned_kg_ref" and i.node_id)

        return PromptContext(
            conversation_id=conversation_id,
            purpose=purpose,
            token_budget=budget_tokens,
            tokens_used=used,
            items=tuple(kept),
            messages=tuple(messages),
            included_node_ids=included_node_ids,
            included_edge_ids=included_edge_ids,
            included_pointer_ids=included_pointer_ids,
            dropped=tuple(dropped),
            head_summary_ids=head_summary_ids,
            tail_turn_ids=tail_turn_ids,
            active_memory_context_ids=active_memory_context_ids,
            pinned_memory_ids=pinned_memory_ids,
            pinned_kg_ref_ids=pinned_kg_ref_ids,
        )

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        # placeholder: you’ll replace with a proper token-aware truncation or LLM compression
        return text[: max(1, max_tokens * 4)]
    


@dataclass(frozen=True)
class _EdgeSelection:
    memctx_ids: set[str]
    ptr_ids: set[str]
    edge_ids_for_memctx: set[str]
    edge_ids_for_ptr: set[str]
class ContextSources:
    """Gather *candidate* context items from the conversation graph.

    This is a convenience/debug layer.

    It does **not** mean "persist the entire conversation graph".
    The returned items are merely inputs to prompt construction.

    If you need replayable prompt inputs, persist a **ContextSnapshot** that
    captures the final PromptContext (messages/items) and any evidence-pack
    digests used for answering.
    """
    def __init__(
        self,
        *,
        conversation_engine: "EngineLike",
        tail_turns: int = 8,
        include_summaries: bool = True,
        include_memory_context: bool = True,
        include_pinned_kg_refs: bool = True,
    ) -> None:
        self.engine = conversation_engine
        self.tail_turns = tail_turns
        self.include_summaries = include_summaries
        self.include_memory_context = include_memory_context
        self.include_pinned_kg_refs = include_pinned_kg_refs

    def gather(self, *, conversation_id: str, purpose: str):
        # Phase 1: load nodes
        ids, docs, metas = self._load_node_rows(conversation_id)

        # Phase 2: decode nodes
        by_id, meta_by_id = self._decode_nodes(ids, docs, metas)

        # Phase 3: compute turn ids & summary id
        tail_turn_ids = self._select_tail_turn_ids(by_id, meta_by_id)
        head_summary_id = self._select_head_summary_id(by_id) if self.include_summaries else None

        # Phase 4: edge-based expansions
        edge_sel = self._select_edges_for_tail_turns(conversation_id, tail_turn_ids)

        # Phase 5: build ContextItems
        items: list[ContextItem] = []
        self._append_head_summary(items, by_id, head_summary_id)
        self._append_memory_context(items, by_id, edge_sel)
        self._append_pinned_refs(items, by_id, edge_sel)
        self._append_tail_turns(items, by_id, tail_turn_ids)

        return items

    # -------------------------
    # Phase 1
    # -------------------------
    def _load_node_rows(self, conversation_id: str) -> tuple[list[Any], list[Any], list[Any]]:
        got = self.engine.backend.node_get(
            where={"conversation_id": conversation_id},
            include=["documents", "metadatas"],
        )
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        return ids, docs, metas

    # -------------------------
    # Phase 2
    # -------------------------
    def _decode_nodes(
        self,
        ids: Iterable[Any],
        docs: Iterable[Any],
        metas: Iterable[Any],
    ) -> tuple[dict[str, ConversationNode], dict[str, dict]]:
        by_id: dict[str, ConversationNode] = {}
        meta_by_id: dict[str, dict] = {}

        for nid, doc, meta in zip(ids, docs, metas):
            base = self._safe_json_dict(doc)
            if isinstance(meta, dict):
                base["metadata"] = {**(base.get("metadata") or {}), **meta}

            n = self._safe_validate_conversation_node(base)
            if n is None:
                continue

            sid = str(nid)
            by_id[sid] = n
            meta_by_id[sid] = (base.get("metadata") or {})
        return by_id, meta_by_id

    def _safe_json_dict(self, doc: Any) -> dict:
        if not isinstance(doc, str):
            return {}
        try:
            out = json.loads(doc)
            return out if isinstance(out, dict) else {}
        except Exception:
            return {}

    def _safe_validate_conversation_node(self, payload: dict) -> ConversationNode | None:
        try:
            return ConversationNode.model_validate(payload)
        except Exception:
            return None

    # -------------------------
    # Phase 3
    # -------------------------
    def _entity_type(self, nid: str, meta_by_id: dict[str, dict]) -> str:
        m = meta_by_id.get(nid) or {}
        return str(m.get("entity_type") or m.get("type") or "")

    def _turn_index(self, n: ConversationNode) -> int:
        return int(getattr(n, "turn_index", -1) or -1)

    def _select_tail_turn_ids(self, by_id: dict[str, ConversationNode], meta_by_id: dict[str, dict]) -> list[str]:
        turn_ids = [nid for nid in by_id.keys() if self._entity_type(nid, meta_by_id) == "conversation_turn"]
        turn_ids.sort(key=lambda nid: self._turn_index(by_id[nid]))
        return turn_ids[-self.tail_turns :] if self.tail_turns > 0 else []

    def _select_head_summary_id(self, by_id: dict[str, ConversationNode]) -> str | None:
        best_ix = -10**9
        head_summary_id: str | None = None
        for nid, n in by_id.items():
            if str(getattr(n, "type", "")) == "memory_summary":
                ix = self._turn_index(n)
                if ix >= best_ix:
                    best_ix = ix
                    head_summary_id = nid
        return head_summary_id

    # -------------------------
    # Phase 4
    # -------------------------
    def _select_edges_for_tail_turns(self, conversation_id: str, tail_turn_ids: list[str]) -> _EdgeSelection:
        memctx_ids: set[str] = set()
        ptr_ids: set[str] = set()
        edge_ids_for_memctx: set[str] = set()
        edge_ids_for_ptr: set[str] = set()

        if not tail_turn_ids:
            return _EdgeSelection(memctx_ids, ptr_ids, edge_ids_for_memctx, edge_ids_for_ptr)
        if not (self.include_memory_context or self.include_pinned_kg_refs):
            return _EdgeSelection(memctx_ids, ptr_ids, edge_ids_for_memctx, edge_ids_for_ptr)

        egot = self.engine.backend.edge_get(
            where={"doc_id": f"conv:{conversation_id}"},
            include=["metadatas"],
        )
        eids = egot.get("ids") or []
        emetas = egot.get("metadatas") or []

        tail_turn_set = set(tail_turn_ids)

        for eid, em in zip(eids, emetas):
            if not isinstance(em, dict):
                continue

            rel = str(em.get("relation") or "")
            src = self._first_id_from_json_list(em.get("source_ids"))
            if src is None or src not in tail_turn_set:
                continue

            tids = self._ids_from_json_list(em.get("target_ids"))
            if not tids:
                continue

            if rel == "has_memory_context" and self.include_memory_context:
                memctx_ids.update(tids)
                edge_ids_for_memctx.add(str(eid))

            elif rel == "references" and self.include_pinned_kg_refs:
                ptr_ids.update(tids)
                edge_ids_for_ptr.add(str(eid))

        return _EdgeSelection(memctx_ids, ptr_ids, edge_ids_for_memctx, edge_ids_for_ptr)

    def _ids_from_json_list(self, raw: Any) -> list[str]:
        try:
            xs = json.loads(raw or "[]")
        except Exception:
            xs = []
        if not isinstance(xs, list):
            return []
        return [str(x) for x in xs if x is not None]

    def _first_id_from_json_list(self, raw: Any) -> str | None:
        ids = self._ids_from_json_list(raw)
        return ids[0] if ids else None

    # -------------------------
    # Phase 5 (builders)
    # -------------------------
    def _append_head_summary(self, items: list[ContextItem], by_id: dict[str, ConversationNode], head_summary_id: str | None) -> None:
        if not head_summary_id:
            return
        n = by_id.get(head_summary_id)
        if n is None:
            return
        items.append(
            ContextItem(
                kind="head_summary",
                role="system",
                text=str(getattr(n, "summary", "") or ""),
                node_id=head_summary_id,
                priority=20,
                pinned=True,
                max_tokens=800,
                source="summary",
            )
        )

    def _append_memory_context(self, items: list[ContextItem], by_id: dict[str, ConversationNode], edge_sel: _EdgeSelection) -> None:
        for mid in sorted(edge_sel.memctx_ids):
            n = by_id.get(mid)
            if n is None:
                continue
            items.append(
                ContextItem(
                    kind="memory_context",
                    role="system",
                    text=str(getattr(n, "summary", "") or ""),
                    node_id=mid,
                    edge_ids=tuple(sorted(edge_sel.edge_ids_for_memctx)),
                    priority=30,
                    pinned=True,
                    max_tokens=600,
                    source="memory_pinned",
                    extra={"source_node_ids": (getattr(n, "properties", {}) or {}).get("source_node_ids")},
                )
            )

    def _append_pinned_refs(self, items: list[ContextItem], by_id: dict[str, ConversationNode], edge_sel: _EdgeSelection) -> None:
        for pid in sorted(edge_sel.ptr_ids):
            n = by_id.get(pid)
            if n is None:
                continue
            props = getattr(n, "properties", {}) or {}
            refers_to = props.get("refers_to_id")
            label = getattr(n, "label", "") or ""
            summary = getattr(n, "summary", "") or ""
            txt = f"{label}\n{summary}".strip()
            items.append(
                ContextItem(
                    role="system",
                    kind="pinned_kg_ref",
                    text=txt,
                    node_id=pid,
                    edge_ids=tuple(sorted(edge_sel.edge_ids_for_ptr)),
                    pointer_ids=(str(refers_to),) if isinstance(refers_to, str) and refers_to else (),
                    priority=40,
                    pinned=True,
                    max_tokens=500,
                    source="kg_ref",
                    extra={"refers_to_id": refers_to},
                )
            )

    def _append_tail_turns(self, items: list[ContextItem], by_id: dict[str, ConversationNode], tail_turn_ids: list[str]) -> None:
        # chronological order; newest gets lowest priority value
        for idx, tid in enumerate(tail_turn_ids):
            n = by_id.get(tid)
            if n is None:
                continue
            role: Role = getattr(n, "role", None) or "user"
            text = str(getattr(n, "summary", "") or "")
            age_from_newest = max(0, (len(tail_turn_ids) - 1 - idx))
            items.append(
                ContextItem(
                    kind="tail_turn",
                    text=text,
                    node_id=tid,
                    priority=50 + age_from_newest,
                    pinned=False,
                    role=role,  # type: ignore
                    source="history_turn",
                    extra={"turn_index": int(getattr(n, "turn_index", -1) or -1)},
                )
            )



@dataclass
class EngineConversationStore:
    engine: "EngineLike"

    def get_turns(self, conversation_id: str) -> list[ContextMessage]:
        # 1) fetch all conversation nodes for this conversation_id
        # NOTE: adapt include/where to your Chroma wrapper API
        res = self.engine.backend.node_get(
            where={"conversation_id": conversation_id},
            include=["metadatas", "documents", "ids"],  # or whatever your wrapper uses
        )

        # 2) materialize + sort
        turns: list[ConversationNode] = []
        for nid, meta, doc in zip(res["ids"], res["metadatas"], res.get("documents", [None]*len(res["ids"]))):
            # You might store content in doc, or in summary/properties.
            # If your storage puts text in `documents`, use doc.
            # Otherwise, parse from meta/properties as you do elsewhere.
            node = ConversationNode(
                id=nid,
                metadata=meta,
                # other required fields for Node/GraphEntityRefBase if needed…
                # If ConversationNode requires label/type/summary/mentions, you may need to pull those too.
            )
            turns.append(node)

        turns.sort(key=lambda n: (n.turn_index or 0))

        # 3) convert to ContextMessage
        out: list[ContextMessage] = []
        for n in turns:
            # Choose ONE canonical “text for LLM” location:
            # - if you stored it in summary: use n.summary
            # - if you stored it in metadata/properties/documents: use that
            text = getattr(n, "summary", None) or ""
            out.append(
                ContextMessage(
                    role=(n.role or "user"),     # role is on node via ConversationRoleMixin
                    content=text,
                    node_id=n.id,
                    source="history_turn",
                )
            )
        return out