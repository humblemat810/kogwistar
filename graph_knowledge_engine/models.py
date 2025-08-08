# ✅ Models for Chroma + LLM with optional embeddings and adjudication

from typing import List, Literal, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
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
class ReferenceSession(BaseModel):
    collection_page_url: str = Field(..., description="Link to the collection page")
    document_page_url: str = Field(..., description="Link to the document page")
    page: Optional[int] = Field(None, description="Page number in the source document, if applicable")
    start_char: Optional[int] = Field(None, description="Start character offset of the evidence span")
    end_char: Optional[int] = Field(None, description="End character offset of the evidence span")
    snippet: Optional[str] = Field(None, description="Short text snippet of the evidence")

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
    type: Literal['entity', 'relationship'] = Field(..., description="Type of entity.")
    summary: str = Field(..., description="Summary of the node/relationship")
    domain_id: Optional[str] = Field(None, description="Optional domain ID this entity belongs to")
    canonical_entity_id: Optional[str] = Field(
        None,
        description="External or internal canonical ID to link equivalent entities across documents (e.g., Wikidata QID or internal UUID)",
    )
    properties: Optional[Dict[str, JsonPrimitive]] = Field(
        None, description="Optional properties of the entity (flat primitives only)"
    )
    references: Optional[List[ReferenceSession]] = Field(None, description="References to information sources")

class EdgeMixin(BaseModel):
    source_ids: List[str] = Field(..., description="List of source node IDs")
    target_ids: List[str] = Field(..., description="List of target node IDs")
    relation: str = Field(..., description="Type of relationship between source and target nodes")

# -------------------------
# Storage-facing mixin (embedding OPTIONAL)
# -------------------------
class ChromaMixin(BaseModel):
    id: str = Field(default_factory=generate_id, description="Unique identifier")
    # Embedding is OPTIONAL. Chroma can compute via its default embedding function or we can omit at retrieval time.
    embedding: Optional[List[float]] = Field(
        None, description="Vector embedding for the entity (optional; may be omitted)"
    )
    doc_id: Optional[str] = Field(None, description="Document ID from which this entity was extracted")

# -------------------------
# LLM-facing mixin (NO embedding field to keep schema tight)
# -------------------------
class LLMMixin(BaseModel):
    id: Optional[str] = Field(
        None,
        description="None if referring to a new object; use existing IDs for existing objects",
    )

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
    """
    pass

class LLMEdge(LLMMixin, EdgeMixin, GraphEntityBase):
    """
    Represents an edge extracted by an LLM from a document.
    Inherits node fields and adds source/target relationships and relation type.
    ID is optional and will be added post-processing.
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
