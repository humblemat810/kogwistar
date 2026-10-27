"""Versioned wire contract for document-scoped LLM graph extraction.

Entity payloads stay JSON objects because the legacy Python rehydrator accepts
normal and flattened reference layouts. This validates the transport envelope;
the selected write owner must still rehydrate entities and validate evidence.
"""

from __future__ import annotations

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


def validate_graph_extraction_request(payload: object) -> GraphExtractionRequestV1:
    """Validate and return the versioned extraction wire payload."""

    try:
        return GraphExtractionRequestV1.model_validate(payload)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, GraphExtractionContractError):
            raise
        raise GraphExtractionContractError(str(exc)) from exc
