from __future__ import annotations

from typing import Final

# Canonical graph-kind tokens.
KIND_KG: Final[str] = "knowledge"
KIND_CHAT: Final[str] = "conversation"
KIND_FLOW: Final[str] = "workflow"


def normalize_graph_kind(kind: str | None) -> str:
    raw = str(kind or KIND_KG).strip().lower()
    if raw in {KIND_KG, "kg"}:
        return KIND_KG
    if raw in {KIND_FLOW, "wf"}:
        return KIND_FLOW
    if raw in {KIND_CHAT, "chat", "conv", "dialog"}:
        return KIND_CHAT
    return raw
