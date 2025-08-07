from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import uuid

class ReferenceSession(BaseModel):
    collection_page_url: str  # Link to the collection page (external/static)
    document_page_url: str    # Link to the document page

class Domain(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid1()))
    name: str
    description: Optional[str] = None

class Node(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid1()))
    label: str
    type: str  # e.g., 'entity', 'idea', 'relationship', etc.
    domain_id: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    references: Optional[List[ReferenceSession]] = None  # List of references to information sources

class Edge(Node):
    # In a hypergraph, edges are also nodes
    source_ids: List[str]
    target_ids: List[str]
    relation: str  # e.g., 'related_to', 'part_of', etc.
    # references inherited from Node

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid1()))
    content: str
    type: str  # e.g., 'ocr', 'pdf', 'image', etc.
    metadata: Optional[Dict[str, Any]] = None
    domain_id: Optional[str] = None
    processed: bool = False

# LLM output models (no id fields)
class LLMNode(BaseModel):
    label: str
    type: str
    domain_id: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    references: Optional[List[ReferenceSession]] = None

class LLMEdge(BaseModel):
    label: str
    type: str
    domain_id: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    references: Optional[List[ReferenceSession]] = None
    source_ids: List[str]
    target_ids: List[str]
    relation: str

class LLMGraphExtraction(BaseModel):
    nodes: List[LLMNode]
    edges: List[LLMEdge]
