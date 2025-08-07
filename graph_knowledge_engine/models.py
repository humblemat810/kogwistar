from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import uuid

class Domain(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None

class Node(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    type: str  # e.g., 'entity', 'idea', 'relationship', etc.
    domain_id: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None

class Edge(Node):
    # In a hypergraph, edges are also nodes
    source_ids: List[str]
    target_ids: List[str]
    relation: str  # e.g., 'related_to', 'part_of', etc.


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    type: str  # e.g., 'ocr', 'pdf', 'image', etc.
    metadata: Optional[Dict[str, Any]] = None
    domain_id: Optional[str] = None
    processed: bool = False
