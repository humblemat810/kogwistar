from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel

from .contracts import (
    AdjudicateBatchTaskRequest,
    AdjudicateBatchTaskResult,
    AdjudicatePairTaskRequest,
    AdjudicatePairTaskResult,
    AnswerWithCitationsTaskRequest,
    AnswerWithCitationsTaskResult,
    ExtractGraphTaskRequest,
    ExtractGraphTaskResult,
    FilterCandidatesTaskRequest,
    FilterCandidatesTaskResult,
    LLMTaskProviderHints,
    LLMTaskSet,
    RepairCitationsTaskRequest,
    RepairCitationsTaskResult,
    SummarizeContextTaskRequest,
    SummarizeContextTaskResult,
)
from .errors import ProviderDependencyError

ProviderName = Literal["gemini", "openai", "ollama"]


@dataclass(frozen=True)
class DefaultTaskProviderConfig:
    extract_graph_provider: ProviderName = "gemini"
    adjudicate_pair_provider: ProviderName = "gemini"
    adjudicate_batch_provider: ProviderName = "gemini"
    filter_candidates_provider: ProviderName = "gemini"
    summarize_context_provider: ProviderName = "gemini"
    answer_with_citations_provider: ProviderName = "gemini"
    repair_citations_provider: ProviderName = "gemini"
    gemini_model_name: str = "gemini-2.5-pro"
    ollama_model_name: str = "qwen3:4b"  # Default for ollama


class _Runner:
    # the abstracted shape for llm provider, override for multimodel or custom behaviour
    def invoke_structured(
        self,
        *,
        messages: Sequence[tuple[str, str]],
        schema: type[BaseModel],
        variables: Mapping[str, object],
        prefer_json_schema: bool = False,
    ) -> tuple[object | None, object | None, object | None]:
        raise NotImplementedError

    def invoke_text(
        self,
        *,
        messages: Sequence[tuple[str, str]],
        variables: Mapping[str, object],
    ) -> str:
        raise NotImplementedError


class _MissingRunner(_Runner):
    def __init__(self, message: str) -> None:
        self._message = message

    def invoke_structured(
        self,
        *,
        messages: Sequence[tuple[str, str]],
        schema: type[BaseModel],
        variables: Mapping[str, object],
        prefer_json_schema: bool = False,
    ) -> tuple[object | None, object | None, object | None]:
        _ = (messages, schema, variables, prefer_json_schema)
        raise ProviderDependencyError(self._message)

    def invoke_text(
        self,
        *,
        messages: Sequence[tuple[str, str]],
        variables: Mapping[str, object],
    ) -> str:
        _ = (messages, variables)
        raise ProviderDependencyError(self._message)


class _LangChainRunner(_Runner):
    def __init__(self, model: object) -> None:
        self._model = model

    def invoke_structured(
        self,
        *,
        messages: Sequence[tuple[str, str]],
        schema: type[BaseModel],
        variables: Mapping[str, object],
        prefer_json_schema: bool = False,
    ) -> tuple[object | None, object | None, object | None]:
        try:
            from langchain_core.prompts import ChatPromptTemplate
        except Exception as e:
            raise ProviderDependencyError(
                "Default task provider needs 'langchain-core'. Install with: pip install 'kogwistar[full]'"
            ) from e

        prompt = ChatPromptTemplate.from_messages(list(messages))
        if prefer_json_schema:
            try:
                structured = self._model.with_structured_output(
                    schema, method="json_schema", include_raw=True
                )
            except TypeError:
                structured = self._model.with_structured_output(
                    schema, include_raw=True
                )
        else:
            structured = self._model.with_structured_output(schema, include_raw=True)
        result = (prompt | structured).invoke(dict(variables))

        if isinstance(result, dict):
            return result.get("raw"), result.get("parsed"), result.get("parsing_error")
        return None, result, None

    def invoke_text(
        self,
        *,
        messages: Sequence[tuple[str, str]],
        variables: Mapping[str, object],
    ) -> str:
        try:
            from langchain_core.prompts import ChatPromptTemplate
        except Exception as e:
            raise ProviderDependencyError(
                "Default task provider needs 'langchain-core'. Install with: pip install 'kogwistar[full]'"
            ) from e
        prompt = ChatPromptTemplate.from_messages(list(messages))
        result = (prompt | self._model).invoke(dict(variables))
        if isinstance(result, str):
            return result
        content = getattr(result, "content", None)
        return str(content if content is not None else result)


def _provider_extra_name(provider: ProviderName) -> str:
    if provider == "ollama":
        return "ollama"
    return "gemini" if provider == "gemini" else "openai"


def _missing_provider_message(provider: ProviderName) -> str:
    extra = _provider_extra_name(provider)
    return (
        f"LLM provider '{provider}' is not available. "
        f"Install with: pip install 'kogwistar[{extra}]'"
    )


def _build_runner(provider: ProviderName, config: DefaultTaskProviderConfig) -> _Runner:
    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception:
            return _MissingRunner(_missing_provider_message(provider))
        return _LangChainRunner(
            ChatGoogleGenerativeAI(
                model=config.gemini_model_name,
                temperature=0.1,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except Exception:
            return _MissingRunner(_missing_provider_message(provider))
        return _LangChainRunner(
            ChatOllama(
                model=config.ollama_model_name,
                temperature=0.1,
            )
        )

    try:
        from langchain_openai import AzureChatOpenAI
    except Exception:
        return _MissingRunner(_missing_provider_message(provider))
    return _LangChainRunner(
        AzureChatOpenAI(
            deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4_1"),
            model_name=os.getenv("OPENAI_MODEL_NAME_GPT4_1"),
            azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1"),
            cache=None,
            openai_api_key=os.getenv("OPENAI_API_KEY_GPT4_1"),
            api_version="2024-08-01-preview",
            model_version=os.getenv("OPENAI_DEPLOYMENT_VERSION_GPT4_1"),
            temperature=0.1,
            max_tokens=30000,
            openai_api_type="azure",
        )
    )


def _payload(parsed: object | None) -> Mapping[str, object] | None:
    if parsed is None:
        return None
    if isinstance(parsed, BaseModel):
        dumped = parsed.model_dump(mode="python")
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if isinstance(parsed, Mapping):
        return dict(parsed)
    return None


def _err_text(parsing_error: object | None) -> str | None:
    if parsing_error is None:
        return None
    return str(parsing_error)


def _extract_schema_for_mode(schema_mode: str) -> tuple[type[BaseModel], bool]:
    from graph_knowledge_engine.engine_core.models import (
        AssocFlattenedLLMGraphExtraction,
        LLMGraphExtraction,
    )

    if schema_mode == "full":
        return LLMGraphExtraction["llm"], False
    if schema_mode == "lean":
        return LLMGraphExtraction["llm_in"], False
    if schema_mode == "flattened_lean":
        return AssocFlattenedLLMGraphExtraction["llm_in"], True
    if schema_mode == "flattened_full":
        return AssocFlattenedLLMGraphExtraction["llm"], True
    raise ValueError(f"Unsupported extract_graph schema mode: {schema_mode!r}")


def _build_task_set_from_runner_getter(
    *, get_runner_for_task, provider_hints: LLMTaskProviderHints
) -> LLMTaskSet:
    def _extract_graph(request: ExtractGraphTaskRequest) -> ExtractGraphTaskResult:
        schema, prefer_json_schema = _extract_schema_for_mode(request.schema_mode)
        messages: list[tuple[str, str]] = [
            (
                "system",
                "You are an expert knowledge graph extractor. "
                "You are an information extraction system that converts legal contracts into a knowledge graph. "
                "Your task: extract ALL entities (nodes) and relationships (edges) from the text. "
                f"{request.instruction}"
                "Allow multiple edge between the same nodes. Allow hypergraph. Allow edge pointing to other edge. "
                "Allow same label but different content. "
                "Breakdown A is obligated to do work for B as A -> B : relation = do work for 100 dollar. "
                "Build relationship triplets of SVO. "
                "Also pay attention to monetary terms, numbers. "
                "For any signatories with blank to sign, or signed. They are equally important to note. "
                "Extract all nodes and edges from the following contract section. "
                "Be exhaustive and granular. "
                "Important: Span.excerpt must match zero-indexed start/end slice exactly. "
                f"{request.prompt_rules}",
            ),
            (
                "human",
                "Aliases (delta for this turn):\n{alias_nodes}\n\n{alias_edges}\n\n"
                "Document:\n```{document}```\n\n"
                "Return only the structured JSON for the schema.",
            ),
        ]
        if request.last_error:
            last_parsed = request.last_parsed or {}
            messages.append(
                (
                    "system",
                    f"last answer has error\n\nLast attempt: ```{last_parsed}```\n\n"
                    f"error from last attempt: ```{request.last_error}```",
                )
            )

        raw, parsed, parsing_error = get_runner_for_task(
            "extract_graph"
        ).invoke_structured(
            messages=messages,
            schema=schema,
            variables={
                "alias_nodes": request.alias_nodes,
                "alias_edges": request.alias_edges,
                "document": request.content,
                "_DOC_ALIAS": request.doc_alias,
            },
            prefer_json_schema=prefer_json_schema,
        )
        return ExtractGraphTaskResult(
            raw=raw,
            parsed_payload=_payload(parsed),
            parsing_error=_err_text(parsing_error),
        )

    def _adjudicate_pair(
        request: AdjudicatePairTaskRequest,
    ) -> AdjudicatePairTaskResult:
        from graph_knowledge_engine.engine_core.models import LLMMergeAdjudication

        if request.question == "node_edge_equivalence":
            guidance = (
                "Interpret 'node_edge_equivalence' as: determine whether the NODE names, reifies, or denotes "
                "the specific EDGE relation instance. A positive verdict is allowed even though one side is a "
                "node and the other side is an edge, but only when the meaning and relation instance align."
            )
        elif request.question == "same_relation":
            guidance = (
                "Interpret 'same_relation' as: determine whether two EDGEs represent the same logical relation "
                "instance, including compatible endpoints and direction."
            )
        else:
            guidance = "Interpret 'same_entity' as: determine whether two NODEs refer to the same real-world entity."

        raw, parsed, parsing_error = get_runner_for_task(
            "adjudicate_pair"
        ).invoke_structured(
            messages=[
                (
                    "system",
                    "You adjudicate whether two objects satisfy the requested relationship. "
                    f"{guidance} Be conservative and return a structured JSON verdict only.",
                ),
                (
                    "human",
                    "Question: {question}\n\n"
                    "Left:\n{left}\n\nRight:\n{right}\n\n"
                    "Return only the structured JSON for the schema.",
                ),
            ],
            schema=LLMMergeAdjudication,
            variables={
                "question": request.question,
                "left": dict(request.left),
                "right": dict(request.right),
            },
        )
        return AdjudicatePairTaskResult(
            verdict_payload=_payload(parsed),
            raw=raw,
            parsing_error=_err_text(parsing_error),
        )

    def _adjudicate_batch(
        request: AdjudicateBatchTaskRequest,
    ) -> AdjudicateBatchTaskResult:
        from graph_knowledge_engine.engine_core.models import BatchAdjudications

        raw, parsed, parsing_error = get_runner_for_task(
            "adjudicate_batch"
        ).invoke_structured(
            messages=[
                (
                    "system",
                    "You adjudicate candidate pairs (entity-relation cross-type allowed). "
                    "Use the mapping table to interpret question_code. "
                    "Return only structured JSON and keep short ids exactly.",
                ),
                ("human", "Mapping table:\n{mapping}\n\nPairs:\n{pairs}"),
            ],
            schema=BatchAdjudications,
            variables={
                "mapping": list(request.mapping),
                "pairs": list(request.pairs),
            },
        )

        payload = _payload(parsed) or {}
        verdict_payloads: list[Mapping[str, object]] = []
        items = payload.get("items")
        if isinstance(items, list):
            verdict_payloads = [dict(x) for x in items if isinstance(x, Mapping)]
        else:
            merge_items = payload.get("merge_adjudications")
            if isinstance(merge_items, list):
                verdict_payloads = [
                    dict(x) for x in merge_items if isinstance(x, Mapping)
                ]

        return AdjudicateBatchTaskResult(
            verdict_payloads=verdict_payloads,
            raw=raw,
            parsing_error=_err_text(parsing_error),
        )

    def _filter_candidates(
        request: FilterCandidatesTaskRequest,
    ) -> FilterCandidatesTaskResult:
        from graph_knowledge_engine.conversation.models import FilteringResponse

        err_messages = [("system", f"error: {m}") for m in request.retry_error_messages]
        raw, parsed, parsing_error = get_runner_for_task(
            "filter_candidates"
        ).invoke_structured(
            messages=[
                (
                    "system",
                    "You are a helpful assistant filtering knowledge graph nodes.",
                ),
                (
                    "human",
                    f"User Input: {request.conversation_content}\n\n"
                    + (
                        f"Context: {request.context_text}\n\n"
                        if request.context_text
                        else ""
                    )
                    + f"Candidate Nodes:\n{request.candidate_nodes_text}\n\n"
                    + f"Candidate Edges:\n{request.candidate_edges_text}\n\n"
                    + "Return a JSON list of IDs for nodes and edges that are RELEVANT to the user input. ",
                ),
                *err_messages,
            ],
            schema=FilteringResponse,
            variables={},
        )

        if parsing_error:
            return FilterCandidatesTaskResult(
                node_ids=(),
                edge_ids=(),
                reasoning="",
                raw=raw,
                parsing_error=_err_text(parsing_error),
            )

        payload = _payload(parsed) or {}
        validated = FilteringResponse.model_validate(payload)
        return FilterCandidatesTaskResult(
            node_ids=tuple(validated.relevant_ids.node_ids),
            edge_ids=tuple(validated.relevant_ids.edge_ids),
            reasoning=str(validated.reasoning or ""),
            raw=raw,
            parsing_error=None,
        )

    def _summarize_context(
        request: SummarizeContextTaskRequest,
    ) -> SummarizeContextTaskResult:
        text = get_runner_for_task("summarize_context").invoke_text(
            messages=[
                (
                    "system",
                    "Summarize this conversation segment into a concise memory.",
                ),
                ("human", "{full_text}"),
            ],
            variables={"full_text": request.full_text},
        )
        return SummarizeContextTaskResult(text=text)

    def _answer_with_citations(
        request: AnswerWithCitationsTaskRequest,
    ) -> AnswerWithCitationsTaskResult:
        raw, parsed, parsing_error = get_runner_for_task(
            "answer_with_citations"
        ).invoke_structured(
            messages=[
                ("system", request.system_prompt or "You are a helpful assistant."),
                (
                    "human",
                    "Answer the user using ONLY the provided evidence when making factual claims.\n\n"
                    "User question:\n{question}\n\n"
                    "Evidence pack (cite using SpanRef indices):\n{evidence}\n\n"
                    "Requirements:\n"
                    "- Provide a helpful answer in `text`.\n"
                    "- Provide `claims` as key factual claims.\n"
                    "- For each claim, include citations to exact evidence spans.\n"
                    "Return JSON that matches the provided schema.\n",
                ),
            ],
            schema=request.response_model,
            variables={"question": request.question, "evidence": request.evidence},
        )
        return AnswerWithCitationsTaskResult(
            answer_payload=_payload(parsed),
            raw=raw,
            parsing_error=_err_text(parsing_error),
        )

    def _repair_citations(
        request: RepairCitationsTaskRequest,
    ) -> RepairCitationsTaskResult:
        raw, parsed, parsing_error = get_runner_for_task(
            "repair_citations"
        ).invoke_structured(
            messages=[
                ("system", request.system_prompt or "You are a helpful assistant."),
                (
                    "human",
                    "Your previous citations contained invalid references.\n\n"
                    "User question:\n{question}\n\n"
                    "Evidence pack:\n{evidence}\n\n"
                    "Previous answer text:\n{answer_text}\n\n"
                    "Please return corrected JSON that matches the provided schema.\n",
                ),
            ],
            schema=request.response_model,
            variables={
                "question": request.question,
                "evidence": request.evidence,
                "answer_text": request.answer_text,
            },
        )
        return RepairCitationsTaskResult(
            answer_payload=_payload(parsed),
            raw=raw,
            parsing_error=_err_text(parsing_error),
        )

    return LLMTaskSet(
        extract_graph=_extract_graph,
        adjudicate_pair=_adjudicate_pair,
        adjudicate_batch=_adjudicate_batch,
        filter_candidates=_filter_candidates,
        summarize_context=_summarize_context,
        answer_with_citations=_answer_with_citations,
        repair_citations=_repair_citations,
        provider_hints=provider_hints,
    )


def build_default_llm_tasks(
    config: DefaultTaskProviderConfig | None = None,
) -> LLMTaskSet:
    cfg = config or DefaultTaskProviderConfig()
    runner_cache: dict[str, _Runner] = {}
    provider_by_task: dict[str, ProviderName] = {
        "extract_graph": cfg.extract_graph_provider,
        "adjudicate_pair": cfg.adjudicate_pair_provider,
        "adjudicate_batch": cfg.adjudicate_batch_provider,
        "filter_candidates": cfg.filter_candidates_provider,
        "summarize_context": cfg.summarize_context_provider,
        "answer_with_citations": cfg.answer_with_citations_provider,
        "repair_citations": cfg.repair_citations_provider,
    }

    def _runner_for_provider(provider: ProviderName) -> _Runner:
        # get llm configured
        if provider not in runner_cache:
            runner_cache[provider] = _build_runner(provider, cfg)
        return runner_cache[provider]

    def _get_runner_for_task(task_name: str) -> _Runner:
        return _runner_for_provider(provider_by_task[task_name])

    hints = LLMTaskProviderHints(
        extract_graph_provider=cfg.extract_graph_provider,
        adjudicate_pair_provider=cfg.adjudicate_pair_provider,
        adjudicate_batch_provider=cfg.adjudicate_batch_provider,
        filter_candidates_provider=cfg.filter_candidates_provider,
        summarize_context_provider=cfg.summarize_context_provider,
        answer_with_citations_provider=cfg.answer_with_citations_provider,
        repair_citations_provider=cfg.repair_citations_provider,
    )
    return _build_task_set_from_runner_getter(
        get_runner_for_task=_get_runner_for_task,
        provider_hints=hints,
    )
