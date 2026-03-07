from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from .models import FilteringResponse, FilteringResult


def candiate_filtering_callback(
    llm: BaseChatModel,
    conversation_content,
    cand_node_list_str,
    cand_edge_list_str,
    candidate_node_ids: list[str],
    candidate_edge_ids: list[str],
    context_text,
):
    max_retry = 3
    err_messages: list[tuple[str, str]] = []
    for _retry in range(max_retry):
        filter_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant filtering knowledge graph nodes."),
                (
                    "human",
                    f"User Input: {conversation_content}\n\n"
                    + (f"Context: {context_text}\n\n" if context_text else "")
                    + f"Candidate Nodes:\n{cand_node_list_str}\n\n"
                    + f"Candidate Edges:\n{cand_edge_list_str}\n\n"
                    + "Return a JSON list of IDs for nodes and edges that are RELEVANT to the user input. ",
                ),
            ]
            + err_messages
        )
        chain = filter_prompt | llm.with_structured_output(FilteringResponse, include_raw=True)
        resp: dict[str, Any] | BaseModel = chain.invoke({})
        if isinstance(resp, BaseModel):
            raise RuntimeError("unreachable response shape")

        if err := resp.get("parsing_error"):
            err_messages.append(("system", f"error: {str(err)}"))
            continue

        parsed: BaseModel | None = resp.get("parsed")
        if parsed is None:
            raise RuntimeError("unreachable response shape")
        parsed2 = FilteringResponse.model_validate(parsed)
        not_node_candidate = set(parsed2.relevant_ids.node_ids).difference(set(candidate_node_ids))
        not_edge_candidate = set(parsed2.relevant_ids.edge_ids).difference(set(candidate_edge_ids))
        if not_node_candidate or not_edge_candidate:
            if not_node_candidate:
                err_messages.append(("system", str(Exception(f"Non candidates ids returned {not_node_candidate}"))))
            if not_edge_candidate:
                err_messages.append(("system", str(Exception(f"Non candidates ids returned {not_edge_candidate}"))))
            continue
        return (
            FilteringResult(node_ids=parsed2.relevant_ids.node_ids, edge_ids=parsed2.relevant_ids.edge_ids),
            parsed2.reasoning,
        )
    raise RuntimeError("Exhausted all models")
