"""Typed source spans and mention-verification records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_extension.model_slicing import BackendField, LLMField, ModeSlicingMixin
from pydantic_extension.model_slicing.mixin import ExcludeMode

from ..document_types import Chunk


class MentionVerification(BaseModel):
    """Result of verifying a mention span against the source text."""

    method: Literal[
        "llm", "levenshtein", "regex", "heuristic", "human", "ensemble", "system"
    ] = Field(..., description="How the mention was verified")
    is_verified: bool = Field(..., description="Whether the mention appears correct")
    score: float | None = Field(
        None, ge=0.0, le=1.0, description="Confidence score (if applicable)"
    )
    notes: str | None = Field(None, description="Free-text rationale or hints")


class Span(ModeSlicingMixin, BaseModel):
    """Locatable evidence for a node/edge mention within a document."""

    default_include_modes: ClassVar[set[str]] = {"frontend", "backend", "dto", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {
        "frontend",
        "backend",
        "dto",
        "llm",
        "llm_in",
    }
    collection_page_url: Annotated[str, ExcludeMode("llm_in")] = Field(
        ..., description="Link to the collection page"
    )
    document_page_url: Annotated[str, ExcludeMode("llm_in")] = Field(
        ..., description="Link to the document page"
    )
    doc_id: Annotated[str, ExcludeMode("llm_in")] = Field(
        ..., description="document id"
    )
    insertion_method: Annotated[str, BackendField(), ExcludeMode("llm", "llm_in")] = (
        Field(..., description="insertion_method")
    )
    page_number: int = Field(
        1, ge=1, description="1-indexed page number where the mentioned"
    )
    start_char: int = Field(
        ...,
        ge=0,
        description="Character offset within start_page, zero indexed, first char in source document is index 0 with 0 offset",
    )
    end_char: int = Field(
        ..., ge=1, description="Character offset within end_page, zero indexed"
    )
    excerpt: str = Field(
        "",
        description="the direct excerpt from source doc from start char to end char. Must be identical from extracted using start_char and end_char",
    )
    context_before: Annotated[str, ExcludeMode("llm_in")] = Field(
        "",
        description='words before the exceprt for uniqueness excerpt identification, empty string "" if excerpt is start of text',
    )
    context_after: Annotated[str, ExcludeMode("llm_in")] = Field(
        "",
        description="words after the exceprt for uniqueness excerpt identification, empty string "
        " if excerpt is end of text",
    )
    chunk_id: Annotated[
        str, LLMField(), ExcludeMode("frontend", "backend", "dto", "llm_in")
    ] | None = Field(None, description="source text chunk id for chunked text")
    source_cluster_id: Annotated[str | None, ExcludeMode("llm_in")] = Field(
        None, description="source text cluster id"
    )
    verification: Annotated[
        MentionVerification | None, BackendField(), ExcludeMode("llm", "llm_in")
    ] = Field(None, description="Result of validating the mention correctness")

    @field_validator("excerpt", "context_before", "context_after", mode="before")
    @classmethod
    def _none_text_to_empty(cls, value: object) -> object:
        if value is None:
            return ""
        return value

    @staticmethod
    def from_dummy_for_workflow(doc_id: str = "_wf:_dummy") -> Span:
        if not doc_id.startswith("_wf:"):
            doc_id = "_wf:" + doc_id
        return Span(
            collection_page_url=f"workflow/{doc_id}",
            document_page_url=f"workflow/{doc_id}",
            doc_id=doc_id,
            insertion_method="system",
            page_number=1,
            start_char=0,
            end_char=1,
            excerpt="",
            context_before="",
            context_after="",
            chunk_id=None,
            source_cluster_id=None,
            verification=MentionVerification(
                method="system", is_verified=True, score=1.0, notes=""
            ),
        )

    @staticmethod
    def from_dummy_for_conversation(doc_id: str = "_conv:_dummy") -> Span:
        if not doc_id.startswith("_conv:"):
            doc_id = "_conv:" + doc_id
        return Span(
            collection_page_url=f"conversation/{doc_id}",
            document_page_url=f"conversation/{doc_id}",
            doc_id=doc_id,
            insertion_method="system",
            page_number=1,
            start_char=0,
            end_char=1,
            excerpt="",
            context_before="",
            context_after="",
            chunk_id=None,
            source_cluster_id=None,
            verification=MentionVerification(
                method="system", is_verified=True, score=1.0, notes=""
            ),
        )

    @staticmethod
    def from_dummy_for_document() -> Span:
        from .models import Document

        return Span(
            collection_page_url=(
                f"document_collection/{Document.from_dummy().id}"
            ),
            document_page_url=(
                f"document_collection/{Document.from_dummy().id}"
            ),
            doc_id=Document.from_dummy().id,
            insertion_method="system",
            page_number=1,
            start_char=0,
            end_char=1,
            excerpt="",
            context_before="",
            context_after="",
            chunk_id=None,
            source_cluster_id=None,
            verification=MentionVerification(
                method="system",
                is_verified=True,
                score=1.0,
                notes="dummy document span",
            ),
        )

    def fix_chunk_start_end_char(self, source_map: Mapping[str, Chunk]) -> None:
        if self.chunk_id is None:
            raise Exception("only for chunnked span")
        chunk = source_map.get(self.chunk_id)
        if chunk is None:
            raise Exception(f"Missing chunk id {self.chunk_id} in source_map")
        chunk_start_char = chunk.end_char
        chunk_end_char = chunk.end_char
        chunk_length = chunk_end_char - chunk_start_char
        own_len = self.end_char - self.start_char
        if self.start_char > chunk_length:
            if self.start_char >= chunk_start_char and self.start_char < chunk_end_char:
                self.start_char -= chunk_start_char
            if self.end_char > chunk_start_char and self.end_char <= chunk_end_char:
                self.end_char -= chunk_start_char
            else:
                self.end_char = min(self.start_char + own_len, chunk_length)
        else:
            self.start_char = chunk_start_char
            self.end_char = min(self.start_char + own_len, chunk_length)

    def pop_chunk(self, source_map: Mapping[str, Chunk]) -> Chunk:
        self.fix_chunk_start_end_char(source_map)
        if self.chunk_id is None:
            raise AttributeError("chunk_id is None")
        from copy import deepcopy

        chunk = deepcopy(source_map[self.chunk_id])
        self.chunk_id = None
        self.start_char = chunk.start_char + self.start_char
        self.end_char = chunk.start_char + self.end_char
        return chunk

    def from_llm_span(
        self,
        span: "Span['llm']",  # noqa: F821 - model-slicing mode is a string annotation
        source_map: Mapping[str, Chunk],
    ) -> Span:
        if type(span) is not type(Span["llm"]):
            raise TypeError(f"span is not of type {type(Span['llm'])}")
        span_copy = span.model_copy(deep=True)
        span_copy.pop_chunk(source_map)
        return Span.model_validate(span_copy.model_dump())

    def llm_to_unsliced(
        self: "Span['llm']",  # noqa: F821 - model-slicing mode is a string annotation
        source_map: Mapping[str, Chunk],
    ) -> Span:
        if type(self) is Span["llm"]:
            self.pop_chunk(source_map)
            return Span.model_validate(self.model_dump)
        raise TypeError(f"span is not of type {type(Span['llm'])}")

    @model_validator(mode="before")
    @classmethod
    def missing_fields(cls, data: object, info: ValidationInfo) -> object:
        if type(data) is dict and data.get("insertion_method") is None:
            insertion_method = info.context and info.context["insertion_method"]
            data["insertion_method"] = insertion_method
        return data

    @model_validator(mode="after")
    def _check_span_consistency(self) -> Self:
        if self.end_char <= self.start_char and self.end_char != -1:
            raise ValueError("end_char must be > start_char")
        from .utils.refs import default_verification

        if not hasattr(self, "verification") and self.__class__.__name__.endswith(
            "LlmSlice"
        ):
            pass
        elif hasattr(self, "verification") and self.verification is None:
            self.verification = default_verification(
                "no explicit verification from LLM"
            )
        return self
