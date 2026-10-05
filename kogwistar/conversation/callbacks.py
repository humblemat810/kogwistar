"""Protocols shared by conversation retrieval components."""

from __future__ import annotations

from typing import Protocol

from kogwistar.llm_tasks import LLMTaskSet

from .models import FilteringResult, RetrievalResult


class RetrievalFilteringCallback(Protocol):
    """Select relevant candidate IDs after retrieval context is prepared."""

    def __call__(
        self,
        llm_tasks: LLMTaskSet,
        user_text: str,
        candidate_nodes: str,
        candidate_edges: str,
        node_ids: list[str],
        edge_ids: list[str],
        context_text: str,
    ) -> tuple[FilteringResult | RetrievalResult, str]: ...


class MemorySummarizeCallback(Protocol):
    """Summarize selected memory using the configured LLM task set."""

    def __call__(
        self,
        llm_tasks: LLMTaskSet,
        user_text: str,
        selected: RetrievalResult,
    ) -> str: ...


__all__ = ["MemorySummarizeCallback", "RetrievalFilteringCallback"]
