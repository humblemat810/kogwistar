"""Versioned wire contract for document-scoped LLM graph extraction.

Entity payloads stay JSON objects because the legacy Python rehydrator accepts
normal and flattened reference layouts. This validates the transport envelope;
the selected write owner must still rehydrate entities and validate evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from kogwistar.json_types import JsonObject

GRAPH_EXTRACTION_WIRE_VERSION = 1


class GraphExtractionContractError(ValueError):
    """Raised when an extraction request cannot cross the wire seam."""


class GraphExtractionRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal[1]
    doc_id: StrictStr
    content: StrictStr | None = None
    doc_type: Literal["text", "ocr"] = "text"
    insertion_method: StrictStr = "api_upsert"
    nodes: list[JsonObject] = Field(default_factory=list)
    edges: list[JsonObject] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_header(self) -> GraphExtractionRequestV1:
        if not self.doc_id.strip():
            raise GraphExtractionContractError(
                "graph extraction document id must not be empty"
            )
        if not self.insertion_method.strip():
            raise GraphExtractionContractError(
                "graph extraction insertion method must not be empty"
            )
        return self

    def validate_source_spans(self, source_texts: Mapping[str, str]) -> None:
        """Verify text evidence against host-supplied authoritative documents.

        Offsets use Python character indexing. Callers must supply source text
        for every cited document; model-provided excerpts are never authority.
        """

        for entity in (*self.nodes, *self.edges):
            spans = _entity_text_spans(entity)
            for span in spans:
                _validate_text_span(span, self.doc_id, source_texts)


def _entity_text_spans(entity: JsonObject) -> list[JsonObject]:
    found: list[JsonObject] = []
    references = entity.get("references")
    if isinstance(references, list):
        found.extend(item for item in references if isinstance(item, dict))
        if any(not isinstance(item, dict) for item in references):
            raise GraphExtractionContractError("graph extraction reference must be an object")

    mentions = entity.get("mentions", entity.get("groundings"))
    if isinstance(mentions, list):
        for mention in mentions:
            if not isinstance(mention, dict):
                raise GraphExtractionContractError("graph extraction grounding must be an object")
            spans = mention.get("spans")
            if isinstance(spans, list):
                if any(not isinstance(span, dict) for span in spans):
                    raise GraphExtractionContractError("graph extraction span must be an object")
                found.extend(spans)
            elif isinstance(mention.get("span_ids"), list):
                raise GraphExtractionContractError(
                    "flattened span IDs require root span records not present in this request contract"
                )
            elif "start_char" in mention:
                found.append(mention)
    return found


def _validate_text_span(
    span: JsonObject,
    default_doc_id: str,
    source_texts: Mapping[str, str],
) -> None:
    raw_doc_id = span.get("doc_id", default_doc_id)
    if not isinstance(raw_doc_id, str) or not raw_doc_id:
        raise GraphExtractionContractError("graph extraction span doc_id must be a non-empty string")
    source = source_texts.get(raw_doc_id)
    if source is None:
        raise GraphExtractionContractError(
            f"authoritative source text is unavailable for span document {raw_doc_id!r}"
        )
    start = span.get("start_char")
    end = span.get("end_char")
    excerpt = span.get("excerpt", "")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise GraphExtractionContractError("graph extraction span start_char must be non-negative")
    if isinstance(end, bool) or not isinstance(end, int):
        raise GraphExtractionContractError("graph extraction span end_char must be an integer")
    if not isinstance(excerpt, str):
        raise GraphExtractionContractError("graph extraction span excerpt must be a string")
    resolved_end = start + len(excerpt) if end == -1 else end
    if resolved_end <= start or resolved_end > len(source) or source[start:resolved_end] != excerpt:
        raise GraphExtractionContractError(
            f"graph extraction span bounds or excerpt do not match source document {raw_doc_id!r}"
        )


def validate_graph_extraction_request(payload: object) -> GraphExtractionRequestV1:
    """Validate and return the versioned extraction wire payload."""

    try:
        return GraphExtractionRequestV1.model_validate(payload)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, GraphExtractionContractError):
            raise
        raise GraphExtractionContractError(str(exc)) from exc
