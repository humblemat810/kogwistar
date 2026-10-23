"""Bounded advisory context assembled before an agent model call."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, model_validator

from kogwistar.json_types import JsonValue

from .read_tools import AgentReadTools, ReadScope


class WhisperConfig(BaseModel):
    """Opt-in retrieval limits for the model-call Whisper context hook."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    knowledge_depth: Literal[0, 1] = 1
    tool_limit: int = 3
    knowledge_limit: int = 10
    seed_limit: int = 3
    summary_max_chars: int = 320
    payload_max_chars: int = 6000

    @model_validator(mode="after")
    def _validate_limits(self) -> WhisperConfig:
        if not 0 <= self.tool_limit <= 3:
            raise ValueError("tool_limit must be between 0 and 3")
        if not 0 <= self.knowledge_limit <= 10:
            raise ValueError("knowledge_limit must be between 0 and 10")
        if not 0 <= self.seed_limit <= 3:
            raise ValueError("seed_limit must be between 0 and 3")
        if not 1 <= self.summary_max_chars <= 320:
            raise ValueError("summary_max_chars must be between 1 and 320")
        if not 128 <= self.payload_max_chars <= 6000:
            raise ValueError("payload_max_chars must be between 128 and 6000")
        return self


class WhisperToolHint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    kind: str
    summary: str


class WhisperKnowledgeHint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str
    summary: str
    source_refs: tuple[str, ...] = ()


class WhisperResult(BaseModel):
    """Prompt-safe, inspection-friendly result; never an authorization grant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ok", "empty", "degraded"]
    tool_hints: tuple[WhisperToolHint, ...] = ()
    knowledge: tuple[WhisperKnowledgeHint, ...] = ()
    truncated: bool = False
    degraded_reason: str | None = None

    def as_json(self) -> JsonValue:
        return cast(JsonValue, self.model_dump(mode="json"))


def _text(item: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _source_refs(item: Mapping[str, object]) -> tuple[str, ...]:
    refs: list[str] = []
    raw_refs = item.get("source_refs")
    if isinstance(raw_refs, (list, tuple)):
        refs.extend(str(value) for value in raw_refs if isinstance(value, str) and value)
    for key in ("source_ref", "source_id", "source_revision_id", "locator"):
        value = item.get(key)
        if isinstance(value, str) and value:
            refs.append(value)
    return tuple(dict.fromkeys(refs))


def _project_knowledge(
    item: Mapping[str, object], *, summary_max_chars: int
) -> WhisperKnowledgeHint | None:
    entity_id = _text(item, "entity_id", "node_id", "id", "memory_id")
    if not entity_id:
        return None
    title = _text(item, "title", "name", "term") or entity_id
    summary = _text(item, "summary", "description", "definition", "content")
    return WhisperKnowledgeHint(
        id=entity_id,
        title=title[:summary_max_chars],
        summary=summary[:summary_max_chars],
        source_refs=_source_refs(item),
    )


def _project_tool(item: Mapping[str, object], *, summary_max_chars: int) -> WhisperToolHint | None:
    entry_id = _text(item, "logical_id")
    kind = _text(item, "kind").lower()
    if not entry_id or kind not in {"tool", "mcp"}:
        return None
    return WhisperToolHint(
        id=entry_id,
        name=_text(item, "name") or entry_id,
        kind=kind,
        summary=_text(item, "summary")[:summary_max_chars],
    )


def _within_payload_budget(
    result: WhisperResult, max_chars: int
) -> WhisperResult:
    """Trim whole records and then summaries while retaining valid JSON."""

    tools = list(result.tool_hints)
    knowledge = list(result.knowledge)
    truncated = result.truncated
    while True:
        candidate = result.model_copy(
            update={"tool_hints": tuple(tools), "knowledge": tuple(knowledge), "truncated": truncated}
        )
        encoded = json.dumps(
            candidate.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(encoded) <= max_chars:
            return candidate
        if knowledge:
            knowledge.pop()
            truncated = True
            continue
        if tools:
            tools.pop()
            truncated = True
            continue
        if result.degraded_reason is not None:
            return WhisperResult(status="degraded", degraded_reason="payload_limit")
        return WhisperResult(status="empty", truncated=True)


def build_whisper_context(
    *,
    query: str,
    config: WhisperConfig,
    read_tools: AgentReadTools,
    scope: ReadScope,
    effective_capabilities: tuple[str, ...] = (),
) -> WhisperResult:
    """Retrieve compact tool and knowledge hints through scoped read adapters."""

    if not config.enabled:
        return WhisperResult(status="empty")
    try:
        tool_hints: list[WhisperToolHint] = []
        retrieval_truncated = False
        if config.tool_limit:
            tool_page = read_tools.catalog_search(
                query, scope=scope, mode="bm25", limit=read_tools.max_limit
            )
            retrieval_truncated = tool_page.truncated
            for item in tool_page.items:
                kind = _text(item, "kind").lower()
                if kind not in {"tool", "mcp"}:
                    continue
                required = item.get("required_capabilities", [])
                required_caps = (
                    {str(value) for value in required}
                    if isinstance(required, (list, tuple, set))
                    else set()
                )
                if not required_caps or not required_caps.issubset(effective_capabilities):
                    continue
                projected = _project_tool(item, summary_max_chars=config.summary_max_chars)
                if projected is not None:
                    tool_hints.append(projected)
            if len(tool_hints) > config.tool_limit:
                retrieval_truncated = True
                tool_hints = tool_hints[: config.tool_limit]

        knowledge_hints: list[WhisperKnowledgeHint] = []
        if config.knowledge_limit:
            seed_limit = (
                config.knowledge_limit
                if config.knowledge_depth == 0
                else config.seed_limit
            )
            seed_page = (
                read_tools.knowledge_search(query, scope=scope, limit=seed_limit)
                if seed_limit
                else None
            )
            seeds = seed_page.items if seed_page is not None else ()
            retrieval_truncated = retrieval_truncated or bool(
                seed_page is not None and seed_page.truncated
            )
            ordered_items: list[Mapping[str, object]] = list(seeds)
            if config.knowledge_depth == 1:
                for seed in seeds[: config.seed_limit]:
                    seed_id = _text(seed, "entity_id", "node_id", "id", "memory_id")
                    if not seed_id:
                        continue
                    remaining = config.knowledge_limit - len(ordered_items)
                    if remaining <= 0:
                        break
                    neighbor_page = read_tools.knowledge_expand(
                        seed_id, scope=scope, depth=1, limit=remaining
                    )
                    retrieval_truncated = retrieval_truncated or neighbor_page.truncated
                    ordered_items.extend(neighbor_page.items)

            seen: set[str] = set()
            for item in ordered_items:
                hint = _project_knowledge(item, summary_max_chars=config.summary_max_chars)
                if hint is None or hint.id in seen:
                    continue
                seen.add(hint.id)
                knowledge_hints.append(hint)
                if len(knowledge_hints) > config.knowledge_limit:
                    retrieval_truncated = True
                    knowledge_hints = knowledge_hints[: config.knowledge_limit]
                    break

        result = WhisperResult(
            status="ok" if tool_hints or knowledge_hints else "empty",
            tool_hints=tuple(tool_hints),
            knowledge=tuple(knowledge_hints),
            truncated=retrieval_truncated,
        )
        return _within_payload_budget(result, config.payload_max_chars)
    except Exception:
        # Provider/backend details can contain sensitive IDs and diagnostics.
        # They stay in host logs, never in model-visible context.
        return WhisperResult(status="degraded", degraded_reason="retrieval_unavailable")


__all__ = [
    "WhisperConfig",
    "WhisperKnowledgeHint",
    "WhisperResult",
    "WhisperToolHint",
    "build_whisper_context",
]
