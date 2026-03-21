from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

pytestmark = pytest.mark.ci

from graph_knowledge_engine.llm_tasks import (
    AdjudicateBatchTaskResult,
    AdjudicatePairTaskResult,
    AnswerWithCitationsTaskResult,
    ExtractGraphTaskResult,
    FilterCandidatesTaskResult,
    LLMTaskSet,
    MissingTaskError,
    RepairCitationsTaskResult,
    SummarizeContextTaskResult,
    validate_llm_task_set,
)


def _valid_task_set() -> LLMTaskSet:
    return LLMTaskSet(
        extract_graph=lambda _req: ExtractGraphTaskResult(
            raw=None, parsed_payload=None, parsing_error=None
        ),
        adjudicate_pair=lambda _req: AdjudicatePairTaskResult(
            verdict_payload=None, raw=None, parsing_error=None
        ),
        adjudicate_batch=lambda _req: AdjudicateBatchTaskResult(
            verdict_payloads=(), raw=None, parsing_error=None
        ),
        filter_candidates=lambda _req: FilterCandidatesTaskResult(
            node_ids=(), edge_ids=(), reasoning="", raw=None, parsing_error=None
        ),
        summarize_context=lambda req: SummarizeContextTaskResult(text=req.full_text),
        answer_with_citations=lambda _req: AnswerWithCitationsTaskResult(
            answer_payload=None, raw=None, parsing_error=None
        ),
        repair_citations=lambda _req: RepairCitationsTaskResult(
            answer_payload=None, raw=None, parsing_error=None
        ),
    )


def test_validate_llm_task_set_accepts_complete_set() -> None:
    task_set = _valid_task_set()
    assert validate_llm_task_set(task_set) is task_set


def test_validate_llm_task_set_rejects_missing_task_callable() -> None:
    task_set = replace(_valid_task_set(), filter_candidates=cast(object, None))
    with pytest.raises(MissingTaskError):
        validate_llm_task_set(task_set)
