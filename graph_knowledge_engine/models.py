# ✅ Refactored model with EdgeMixin and clean inheritance

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
from typing import Union

JsonPrimitive = Union[str, int, float, bool, None]
# ✅ Shared utility for ID generation
def generate_id():
    return str(uuid.uuid1())

# ✅ Reference for provenance
class ReferenceSession(BaseModel):
    collection_page_url: str = Field(..., description="Link to the collection page")
    document_page_url: str = Field(..., description="Link to the document page")

# ✅ Domain concept
class Domain(BaseModel):
    id: str = Field(default_factory=generate_id, description="Unique identifier for the domain")
    name: str = Field(..., description="Name of the domain")
    description: Optional[str] = Field(None, description="Optional description of the domain")

# ✅ Base for all graph entities (nodes and edges)
class GraphEntityBase(BaseModel):
    label: str = Field(..., description="Human-readable label for the node or edge")
    type: Literal['entity', 'relationship'] = Field(..., description="Type of entity.")
    summary: str = Field(..., description = "summary of the node/ relationship")
    domain_id: Optional[str] = Field(None, description="Optional domain ID this entity belongs to")
    properties: Optional[Dict[str, JsonPrimitive]] = Field(None, description="Optional properties of the entity")
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
    id: Optional[str] = Field(None, description="None if refering to new object, use existing IDs for exising object. ")
    embedding: Optional[List[float]] = Field(None, description="Optional embedding for similarity search")

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

# ✅ Document structure for tracking sources
class Document(BaseModel):
    id: str = Field(default_factory=generate_id, description="Unique document identifier")
    content: str = Field(..., description="Text content of the document")
    type: str = Field(..., description="Type of document, e.g., 'ocr', 'pdf'")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the document")
    domain_id: Optional[str] = Field(None, description="Optional domain this document belongs to")
    processed: bool = Field(False, description="Whether the document has been processed")
