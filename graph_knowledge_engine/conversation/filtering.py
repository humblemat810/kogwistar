from __future__ import annotations

from graph_knowledge_engine.llm_tasks import FilterCandidatesTaskRequest, LLMTaskSet

from .models import FilteringResult


def candiate_filtering_callback(
    llm_tasks: LLMTaskSet,
    conversation_content,
    cand_node_list_str,
    cand_edge_list_str,
    candidate_node_ids: list[str],
    candidate_edge_ids: list[str],
    context_text,
):
    max_retry = 3
    err_messages: list[str] = []
    for _retry in range(max_retry):
        resp = llm_tasks.filter_candidates(
            FilterCandidatesTaskRequest(
                conversation_content=str(conversation_content or ""),
                context_text=str(context_text or ""),
                candidate_nodes_text=str(cand_node_list_str or ""),
                candidate_edges_text=str(cand_edge_list_str or ""),
                candidate_node_ids=tuple(candidate_node_ids),
                candidate_edge_ids=tuple(candidate_edge_ids),
                retry_error_messages=tuple(err_messages),
            )
        )
        if resp.parsing_error:
            err_messages.append(str(resp.parsing_error))
            continue

        not_node_candidate = set(resp.node_ids).difference(set(candidate_node_ids))
        not_edge_candidate = set(resp.edge_ids).difference(set(candidate_edge_ids))
        if not_node_candidate or not_edge_candidate:
            if not_node_candidate:
                err_messages.append(f"Non candidates node ids returned: {sorted(not_node_candidate)}")
            if not_edge_candidate:
                err_messages.append(f"Non candidates edge ids returned: {sorted(not_edge_candidate)}")
            continue
        return (
            FilteringResult(node_ids=list(resp.node_ids), edge_ids=list(resp.edge_ids)),
            str(resp.reasoning or ""),
        )
    raise RuntimeError("Exhausted all models")
