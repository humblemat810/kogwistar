# ✅ Refactored model with EdgeMixin, adjudication types, and clean inheritance

from typing import List, Literal, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import uuid

JsonPrimitive = Union[str, int, float, bool, None]

# ✅ Shared utility for ID generation

def generate_id() -> str:
    return str(uuid.uuid1())

# ✅ Reference for provenance (now includes optional span info for adjudication)
class ReferenceSession(BaseModel):
    collection_page_url: str = Field(..., description="Link to the collection page")
    document_page_url: str = Field(..., description="Link to the document page")
    page: Optional[int] = Field(None, description="Page number in the source document, if applicable")
    start_char: Optional[int] = Field(None, description="Start character offset of the evidence span")
    end_char: Optional[int] = Field(None, description="End character offset of the evidence span")
    snippet: Optional[str] = Field(None, description="Short text snippet of the evidence")

# ✅ Domain concept
class Domain(BaseModel):
    id: str = Field(default_factory=generate_id, description="Unique identifier for the domain")
    name: str = Field(..., description="Name of the domain")
    description: Optional[str] = Field(None, description="Optional description of the domain")

# ✅ Base for all graph entities (nodes and edges)
class GraphEntityBase(BaseModel):
    label: str = Field(..., description="Human-readable label for the node or edge")
    type: Literal['entity', 'relationship'] = Field(..., description="Type of entity.")
    summary: str = Field(..., description="Summary of the node/relationship")
    domain_id: Optional[str] = Field(None, description="Optional domain ID this entity belongs to")
    canonical_entity_id: Optional[str] = Field(
        None,
        description="External or internal canonical ID to link equivalent entities across documents (e.g., Wikidata QID or internal UUID)",
    )
    properties: Optional[Dict[str, JsonPrimitive]] = Field(None, description="Optional properties of the entity (flat primitives only)")
    references: Optional[List[ReferenceSession]] = Field(None, description="References to information sources")

# ✅ Edge-specific fields extracted into a reusable mixin
class EdgeMixin(BaseModel):
    source_ids: List[str] = Field(..., description="List of source node IDs")
    target_ids: List[str] = Field(..., description="List of target node IDs")
    relation: str = Field(..., description="Type of relationship between source and target nodes")

# ✅ ChromaDB-bound mixin: adds id and embedding
class ChromaMixin(BaseModel):
    id: str = Field(default_factory=generate_id, description="Unique identifier")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding for the entity")

# ✅ LLM output mixin: makes id optional
class LLMMixin(BaseModel):
    id: Optional[str] = Field(None, description="None if referring to a new object; use existing IDs for existing objects")
    embedding: Optional[List[float]] = Field(None, description="Optional embedding for similarity search (usually omitted in LLM output)")

# ✅ Node and Edge models for ChromaDB storage
class Node(ChromaMixin, GraphEntityBase):
    pass

class Edge(ChromaMixin, EdgeMixin, GraphEntityBase):
    pass

# ✅ Node and Edge models for LLM output
class LLMNode(LLMMixin, GraphEntityBase):
    """
    Represents a node extracted by an LLM from a document.
    Contains label, type, optional domain, and properties.
    ID and embedding are optional and will be added post-processing.
    """
    pass

class LLMEdge(LLMMixin, EdgeMixin, GraphEntityBase):
    """
    Represents an edge extracted by an LLM from a document.
    Inherits all node fields and adds source/target relationships and relation type.
    ID and embedding are optional and will be added post-processing.
    """
    pass

class LLMGraphExtraction(BaseModel):
    """
    Top-level structured output from LLM for knowledge graph extraction.
    Contains lists of nodes and edges.
    """
    nodes: List[LLMNode] = Field(..., description="List of extracted nodes")
    edges: List[LLMEdge] = Field(..., description="List of extracted edges")

# ✅ Adjudication models (for LLM or rule-based merge decisions)
class AdjudicationCandidate(BaseModel):
    """A pair of candidate entities/mentions to evaluate for equivalence or linkage."""
    left: Node = Field(..., description="First node candidate")
    right: Node = Field(..., description="Second node candidate")
    question: str = Field(
        "same_entity",
        description="Adjudication question type, e.g., 'same_entity', 'same_event', 'contradiction', etc.",
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

# ✅ Document structure for tracking sources
class Document(BaseModel):
    id: str = Field(default_factory=generate_id, description="Unique document identifier")
    content: str = Field(..., description="Text content of the document")
    type: str = Field(..., description="Type of document, e.g., 'ocr', 'pdf'")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the document")
    domain_id: Optional[str] = Field(None, description="Optional domain this document belongs to")
    processed: bool = Field(False, description="Whether the document has been processed")
