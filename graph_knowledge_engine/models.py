# ✅ Models for Chroma + LLM with optional embeddings and adjudication

from typing import List, Literal, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator, field_validator
import uuid
from enum import IntEnum

JsonPrimitive = Union[str, int, float, bool, None]


class AdjudicationQuestionCode(IntEnum):
    SAME_ENTITY = 1
    SAME_EVENT = 2
    CONTRADICTION = 3
    HIERARCHY = 4
    CAUSAL = 5
    ATTR_EQ = 6

QUESTION_KEY = {
    AdjudicationQuestionCode.SAME_ENTITY: "same_entity",
    AdjudicationQuestionCode.SAME_EVENT: "same_event",
    AdjudicationQuestionCode.CONTRADICTION: "contradiction",
    AdjudicationQuestionCode.HIERARCHY: "hierarchical_relation",
    AdjudicationQuestionCode.CAUSAL: "causal_relation",
    AdjudicationQuestionCode.ATTR_EQ: "attribute_equivalence",
}

QUESTION_DESC = {
    AdjudicationQuestionCode.SAME_ENTITY: "Do the two nodes refer to the same real-world entity (aliases, abbreviations, synonyms allowed)?",
    AdjudicationQuestionCode.SAME_EVENT: "Are these mentions the same event in time/space, possibly phrased differently?",
    AdjudicationQuestionCode.CONTRADICTION: "Do the statements contradict each other?",
    AdjudicationQuestionCode.HIERARCHY: "Is one a subtype/subclass/member of the other?",
    AdjudicationQuestionCode.CAUSAL: "Does one cause or lead to the other?",
    AdjudicationQuestionCode.ATTR_EQ: "Are these property values equivalent (unit/format differences allowed)?",
}




# -------------------------
# Utilities
# -------------------------

def generate_id() -> str:
    return str(uuid.uuid1())

# -------------------------
# Provenance / reference (with optional spans)
# -------------------------
class MentionVerification(BaseModel):
    """Result of verifying a mention span against the source text."""
    method: Literal["llm", "levenshtein", "regex", "heuristic", "human"] = Field(
        ..., description="How the mention was verified"
    )
    is_verified: bool = Field(..., description="Whether the mention appears correct")
    score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score (if applicable)"
    )
    notes: Optional[str] = Field(None, description="Free-text rationale or hints")

class ReferenceSession(BaseModel):
    """Locatable evidence for a node/edge mention within a specific document."""
    collection_page_url: str = Field(..., description="Link to the collection page")
    document_page_url: str = Field(..., description="Link to the document page")
    # Required locators (may span pages)
    start_page: int = Field(..., ge=1, description="1-based page index where the mention starts")
    end_page: int = Field(..., ge=1, description="1-based page index where the mention ends (>= start_page)")
    start_char: int = Field(..., ge=0, description="Character offset within start_page")
    end_char: int = Field(..., ge=0, description="Character offset within end_page")
    # Optional extras
    snippet: Optional[str] = Field(None, description="Short text snippet for quick preview")
    verification: Optional[MentionVerification] = Field(
        None, description="Result of validating the mention correctness"
    )

    @model_validator(mode="after")
    def _check_span_consistency(self):
        if self.end_page < self.start_page:
            raise ValueError("end_page must be >= start_page")
        if self.start_page == self.end_page and self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char when start_page == end_page")
        return self

# -------------------------
# Domain
# -------------------------
class Domain(BaseModel):
    id: str = Field(default_factory=generate_id, description="Unique identifier for the domain")
    name: str = Field(..., description="Name of the domain")
    description: Optional[str] = Field(None, description="Optional description of the domain")

# -------------------------
# Core graph entities
# -------------------------
class GraphEntityBase(BaseModel):
    label: str = Field(..., description="Human-readable label for the node or edge")
    type: Literal['entity', 'relationship'] = Field(..., description="Type of entity")
    summary: str = Field(..., description="Summary of the node/relationship")
    domain_id: Optional[str] = Field(None, description="Domain ID this entity belongs to")
    canonical_entity_id: Optional[str] = Field(
        None, description="Canonical ID to link equivalents (e.g., Wikidata QID or internal UUID)"
    )
    properties: Optional[Dict[str, JsonPrimitive]] = Field(
        None, description="Optional flat properties (JSON primitives only)"
    )
    # 🔴 NOW REQUIRED: LLM must always provide locatable evidence
    references: List[ReferenceSession] = Field(
        ..., min_items=1, description="One or more locatable mentions supporting this entity"
    )

    @field_validator("references")
    @classmethod
    def _require_non_empty_refs(cls, refs: List[ReferenceSession]):
        if not refs:
            raise ValueError("At least one ReferenceSession is required")
        return refs


class EdgeMixin(BaseModel):
    source_ids: List[str] = Field(..., description="List of source node IDs")
    target_ids: List[str] = Field(..., description="List of target node IDs")
    relation: str = Field(..., description="Type of relationship between source and target nodes")

# -------------------------
# Storage-facing mixin (embedding OPTIONAL)
# -------------------------
class ChromaMixin(BaseModel):
    id: str = Field(default_factory=generate_id, description="Unique identifier")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding for the entity")
    # Optional but handy to keep JSON and Chroma metadata aligned
    doc_id: Optional[str] = Field(None, description="Document ID from which this entity was extracted")


# -------------------------
# LLM-facing mixin (NO embedding field to keep schema tight)
# -------------------------
class LLMMixin(BaseModel):
    id: Optional[str] = Field(None, description="None for new object; use existing IDs to upsert")
    # No embedding in LLM schema to avoid bloating the output


# -------------------------
# Final models
# -------------------------
class Node(ChromaMixin, GraphEntityBase):
    pass

class Edge(ChromaMixin, EdgeMixin, GraphEntityBase):
    pass

class LLMNode(LLMMixin, GraphEntityBase):
    """
    Represents a node extracted by an LLM from a document.
    Contains label, type, summary, optional domain, and properties.
    ID is optional and will be added post-processing.
    Node extracted by the LLM. Must include at least one ReferenceSession with precise span.
    """
    pass

class LLMEdge(LLMMixin, EdgeMixin, GraphEntityBase):
    """
    Represents an edge extracted by an LLM from a document.
    Inherits node fields and adds source/target relationships and relation type.
    ID is optional and will be added post-processing.
    Edge extracted by the LLM. Must include at least one ReferenceSession with precise span.
    """
    pass

class LLMGraphExtraction(BaseModel):
    """
    Top-level structured output from LLM for knowledge graph extraction.
    Contains lists of nodes and edges.
    """
    nodes: List[LLMNode] = Field(..., description="List of extracted nodes")
    edges: List[LLMEdge] = Field(..., description="List of extracted edges")

# -------------------------
# Adjudication structures
# -------------------------
class AdjudicationCandidate(BaseModel):
    left: Node = Field(..., description="First node candidate")
    right: Node = Field(..., description="Second node candidate")
    question_code: int = Field(
        AdjudicationQuestionCode.SAME_ENTITY,
        description="Integer question code from the mapping table"
    )

class AdjudicationVerdict(BaseModel):
    """Structured decision returned by an adjudicator (LLM+rules)."""
    same_entity: bool = Field(..., description="True if the two candidates refer to the same real-world entity")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the adjudication")
    reason: str = Field(..., description="Natural language rationale for the decision")
    canonical_entity_id: Optional[str] = Field(
        None,
        description="If applicable, the canonical ID both should map to (new or existing)",
    )

class LLMMergeAdjudication(BaseModel):
    """LLM-structured output wrapper for a single merge adjudication."""
    verdict: AdjudicationVerdict = Field(..., description="Final adjudication verdict")

# -------------------------
# Document
# -------------------------
class Document(BaseModel):
    id: str = Field(default_factory=generate_id, description="Unique document identifier")
    content: str = Field(..., description="Text content of the document")
    type: str = Field(..., description="Type of document, e.g., 'ocr', 'pdf'")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the document")
    domain_id: Optional[str] = Field(None, description="Optional domain this document belongs to")
    processed: bool = Field(False, description="Whether the document has been processed")

# LLM-facing candidate (int code, not Literal)
