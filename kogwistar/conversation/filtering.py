from __future__ import annotations

import re

from kogwistar.engine_core.utils import AliasBook
from kogwistar.llm_tasks import FilterCandidatesTaskRequest, LLMTaskSet

from .models import FilteringResult


def _alias_candidate_text(text: object, aliases: dict[str, str]) -> str:
    """Replace whole candidate-ID tokens without touching ordinary prose."""
    rendered = str(text or "")
    for canonical, alias in sorted(aliases.items(), key=lambda item: -len(item[0])):
        rendered = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(canonical)}(?![A-Za-z0-9_])",
            alias,
            rendered,
        )
    return rendered


def candiate_filtering_callback(
    llm_tasks: LLMTaskSet,
    conversation_content,
    cand_node_list_str,
    cand_edge_list_str,
    candidate_node_ids: list[str],
    candidate_edge_ids: list[str],
    context_text,
):
    book = AliasBook.deterministic(candidate_node_ids, candidate_edge_ids)
    node_aliases = {item: book.alias_for_node(item) for item in candidate_node_ids}
    edge_aliases = {item: book.alias_for_edge(item) for item in candidate_edge_ids}
    node_alias_ids = tuple(node_aliases.values())
    edge_alias_ids = tuple(edge_aliases.values())
    candidate_nodes_text = _alias_candidate_text(cand_node_list_str, node_aliases)
    candidate_edges_text = _alias_candidate_text(cand_edge_list_str, edge_aliases)
    max_retry = 3
    err_messages: list[str] = []
    for _retry in range(max_retry):
        resp = llm_tasks.filter_candidates(
            FilterCandidatesTaskRequest(
                conversation_content=str(conversation_content or ""),
                context_text=str(context_text or ""),
                candidate_nodes_text=candidate_nodes_text,
                candidate_edges_text=candidate_edges_text,
                candidate_node_ids=node_alias_ids,
                candidate_edge_ids=edge_alias_ids,
                retry_error_messages=tuple(err_messages),
            )
        )
        if resp.parsing_error:
            err_messages.append(str(resp.parsing_error))
            continue

        not_node_candidate = set(resp.node_ids).difference(set(node_alias_ids))
        not_edge_candidate = set(resp.edge_ids).difference(set(edge_alias_ids))
        if not_node_candidate or not_edge_candidate:
            if not_node_candidate:
                err_messages.append(
                    f"Non candidates node ids returned: {sorted(not_node_candidate)}"
                )
            if not_edge_candidate:
                err_messages.append(
                    f"Non candidates edge ids returned: {sorted(not_edge_candidate)}"
                )
            continue
        return (
            FilteringResult(
                node_ids=[book.resolve_node(item) for item in resp.node_ids],
                edge_ids=[book.resolve_edge(item) for item in resp.edge_ids],
            ),
            str(resp.reasoning or ""),
        )
    raise RuntimeError("Exhausted all models")
