from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Self, Sequence

Role = Literal["system", "user", "assistant", "tool"]
Source = Literal[
    "live_turn",
    "history_turn",
    "conversation_summary",
    "memory_summary",
    "memory_pinned",
    "kg_ref",
    "kg_ref_pinned",
    "system",
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
    """
    kind: ItemKind
    text: str
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

from pydantic import BaseModel, model_validator
# @dataclass(frozen=True)
class ConversationContextView(BaseModel):
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

    def build(self, *, conversation_id: str, purpose: str, budget_tokens: int) -> ConversationContextView:
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
        priced.sort(key=lambda x: (not x.pinned, x.priority))

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

        # 5) render to LLM messages
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

        return ConversationContextView(
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