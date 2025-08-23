# ✅ Models for Chroma + LLM with optional embeddings and adjudication

from typing import List, Literal, Optional, Dict, Any, Union, Annotated, ClassVar
from pydantic import BaseModel, Field, model_validator, field_validator, ValidationInfo
import uuid
from enum import IntEnum
from pydantic_extension.model_slicing import (ModeSlicingMixin, NotMode, FrontendField, BackendField, LLMField,
                DtoType,
                BackendType,
                FrontendType,
                LLMType,
                use_mode)
from pydantic_extension.model_slicing.mixin import ExcludeMode, DtoField
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
    method: Literal["llm", "levenshtein", "regex", "heuristic", "human", "ensemble"] = Field(
        ..., description="How the mention was verified"
    )
    is_verified: bool = Field(..., description="Whether the mention appears correct")
    score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score (if applicable)"
    )
    notes: Optional[str] = Field(None, description="Free-text rationale or hints")

class ReferenceSession(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"frontend", "backend", "dto", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"frontend", "backend", "dto", "llm"}
    """Locatable evidence for a node/edge mention within a specific document."""
    collection_page_url: str = Field(..., description="Link to the collection page")
    document_page_url: str = Field(..., description="Link to the document page")
    doc_id : str  = Field(..., description="document id")
    insertion_method : Annotated[str, BackendField(), ExcludeMode("llm")]  = Field(..., description="insertion_method")
    # Required locators (may span pages)
    start_page: int = Field(..., ge=1, description="1-based page index where the mention starts")
    end_page: int = Field(..., ge=1, description="1-based page index where the mention ends (>= start_page)")
    start_char: int = Field(..., ge=0, description="Character offset within start_page")
    end_char: int = Field(..., ge=0, description="Character offset within end_page")
    # Optional extras
    snippet: Optional[str] = Field(None, description="Short text snippet for quick preview")
    verification: Annotated[Optional[MentionVerification], BackendField(), ExcludeMode("llm")] = Field(
                                        None, description="Result of validating the mention correctness"
                                    )
    @model_validator(mode="before")
    @classmethod
    def missing_fields(cls, data, info: ValidationInfo):
        
        # data_dump = data.model_dump()
        if data.get('insertion_method') is None:
            insertion_method = info.context["insertion_method"]
            data['insertion_method'] = insertion_method
        return data
    @model_validator(mode="after")
    def _check_span_consistency(self):
        if self.end_page < self.start_page:
            raise ValueError("end_page must be >= start_page")
        if self.start_page == self.end_page and self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char when start_page == end_page")
        from .engine import _default_verification
        if (not hasattr(self, 'verification') and self.__class__.__name__.endswith("LlmSlice")):
            pass # ok LLM ok to have no such field
        elif hasattr(self, 'verification') and self.verification is None: # ok
            self.verification = _default_verification("no explicit verification from LLM")
        else:
            pass # already has verification filled in
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
class GraphEntityBase(ModeSlicingMixin, BaseModel):
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
    
    references: Annotated[List[ReferenceSession], FrontendField(),BackendField(),DtoField(),LLMField()] = Field(
        ..., min_items=1, description="One or more locatable mentions supporting this entity"
    )

    @field_validator("references")
    @classmethod
    def _require_non_empty_refs(cls, refs: List[ReferenceSession],info: ValidationInfo):
        for r in refs:
            pass
        if not refs:
            raise ValueError("At least one ReferenceSession is required")
        return refs


class EdgeMixin(ModeSlicingMixin, BaseModel):
    source_ids: List[str] = Field(..., description="List of source node IDs")
    target_ids: List[str] = Field(..., description="List of target node IDs")
    relation: str = Field(..., description="Type of relationship between source and target nodes")
    source_edge_ids: Optional[List[str]] = Field(..., description="List of source edge IDs")
    target_edge_ids: Optional[List[str]] = Field(..., description="List of target edge IDs")
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
class LLMMixin(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"llm"}
    id: Optional[str] = Field(None, description="None for new object; use existing IDs to upsert")
    # No embedding in LLM schema to avoid bloating the output
    local_id: Optional[str] = Field(
        None,
        description=("Optional within-output temp id for new edge, e.g., 'ne:moon', use `nn:` for new nodes (nn stand for new node, ne stand for new edge. ). "
                     "Set it when this edge is referred by other edges. ")
    )
    

# -------------------------
# Final models
# -------------------------


class DocNodeMixin():
    type: Literal['page', 'chunk', "summary"] = Field(..., description = "type of data")
    doc_id: str = Field(..., description = "document ID")
    pass
class DocNode(DocNodeMixin, GraphEntityBase):
    pass
class Node(ChromaMixin, GraphEntityBase):
    pass

class Edge(ChromaMixin, EdgeMixin, GraphEntityBase):
    # @model_validator(mode="before")
    # @classmethod
    # def inject_context_on_children_before(cls, data: dict, info: ValidationInfo):
    #     # You can inspect context and even transform incoming data
    #     # (not required—context is already auto-propagated)
    #     _ = info.context or {}
        
    #     return data
    # @model_validator(mode="after")
    # def inject_context_on_children_after(self, info: ValidationInfo):
    #     # You can inspect context and even transform incoming data
    #     # (not required—context is already auto-propagated)
    #     _ = info.context or {}
    #     edge = self
    #     if not (
    #         (edge.source_ids or edge.source_edge_ids)
    #         and (edge.target_ids or edge.target_edge_ids)
    #     ):
    #         raise ValueError(
    #             f"Edge {edge.relation} ({edge.label}) must have at least one source and one target"
    #         )
    #     return self
    pass
class LLMNode( LLMMixin, GraphEntityBase):
    """
    Represents a node extracted by an LLM from a document.
    Contains label, type, summary, optional domain, and properties.
    ID is optional and will be added post-processing.
    Node extracted by the LLM. Must include at least one ReferenceSession with precise span.
    If this is a new node, set either:
      - id = "nn:<slug>"  (preferred), or
      - leave id empty and set local_id = "nn:<slug>"
    If this references an existing node, set id to the provided alias (e.g., N3, N~abc, or UUID).
    """
    
    pass

class LLMEdge( LLMMixin, EdgeMixin, GraphEntityBase):
    """
    Represents an edge extracted by an LLM from a document.
    Inherits node fields and adds source/target relationships and relation type.
    ID is optional and will be added post-processing.
    Edge extracted by the LLM. Must include at least one ReferenceSession with precise span.
    For a new edge, set id='ne:<slug>' or leave empty.
    For endpoints, use existing node aliases (N#/UUID) or temp ids 'nn:<slug>'.
    If this is a new edge, set either:
      - id = "ne:<slug>"  (preferred), or
      - leave id empty and set local_id = "ne:<slug>"
    If this references an existing edge, set id to the provided alias (e.g., N3, N~abc, or UUID).
    """
    
    pass

class LLMGraphExtraction(ModeSlicingMixin, BaseModel):
    """
    Top-level structured output from LLM for knowledge graph extraction.
    Contains lists of nodes and edges.
    """
    include_unmarked_for_modes: ClassVar[set[str]] = {"llm"}
    nodes: List[LLMNode] = Field(..., description="List of extracted nodes")
    edges: List[LLMEdge] = Field(..., description="List of extracted edges")
    @model_validator(mode="before")
    @classmethod
    def inject_context_on_children_before(cls, data: dict, info: ValidationInfo):
        # You can inspect context and even transform incoming data
        # (not required—context is already auto-propagated)
        _ = info.context or {}
        return data
    @model_validator(mode="after")
    def inject_context_on_children_after(self, info: ValidationInfo):
        # You can inspect context and even transform incoming data
        # (not required—context is already auto-propagated)
        _ = info.context or {}
        return self
    @classmethod
    def FromLLMSlice(cls, sliced, insertion_method):
        sliced: LLMGraphExtraction['llm']
        if isinstance(sliced, BaseModel):
            dumped = sliced.model_dump()
        else:
            raise(ValueError("Unsupported type for 'sliced'"))
        for ne in dumped['nodes'] + dumped['edges']:
            for r in ne['references']:
                r['insertion_method'] = insertion_method
        return cls.model_validate(dumped, context = {})
                
# -------------------------
# Adjudication structures
# -------------------------
# class AdjudicationCandidate(BaseModel):
#     left: Node = Field(..., description="First node candidate")
#     right: Node = Field(..., description="Second node candidate")
#     question_code: int = Field(
#         AdjudicationQuestionCode.SAME_ENTITY,
#         description="Integer question code from the mapping table"
#     )

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
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_extension.model_slicing import ModeSlicingMixin, DtoType, BackendType
class AdjudicationTarget(BaseModel):
    """A reference to either a node or an edge to be adjudicated."""
    kind: Literal["node", "edge"] = Field(..., description="What is being adjudicated")
    id: str = Field(..., description="UUID of the node or edge")
    # Optional snapshot fields help the LLM (and offline rules) decide without re-fetching.
    label: Optional[str] = None
    type: Optional[str] = None
    summary: Optional[str] = None
    relation: Optional[str] = None                 # for edges
    source_ids: Optional[List[str]] = None         # for edges
    target_ids: Optional[List[str]] = None         # for edges
    source_edge_ids: Optional[List[str]] = None    # for meta-edges
    target_edge_ids: Optional[List[str]] = None    # for meta-edges
    domain_id: Optional[str] = None
    canonical_entity_id: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None

class BatchAdjudications(BaseModel):
    merge_adjudications: List[LLMMergeAdjudication]
class AdjudicationCandidate(BaseModel):
    """A pair of adjudication targets (same kind: node↔node or edge↔edge)."""
    left: AdjudicationTarget
    right: AdjudicationTarget
    question: str = Field(
        "same_entity",
        description="Question key, e.g., 'same_entity' for nodes or 'same_relation' for edges"
    )
# -------------------------
# Document
# -------------------------
class Document(ModeSlicingMixin, BaseModel):
    id: BackendType[str] = Field(default_factory=generate_id, description="Unique document identifier")
    content: BackendType[DtoType[str]] = Field(..., description="Text content of the document")
    type: BackendType[DtoType[str]] = Field(..., description="Type of document, e.g., 'ocr', 'pdf'")
    metadata: BackendType[DtoType[Optional[Dict[str, Any]]]] = Field(None, description="Additional metadata for the document")
    domain_id: BackendType[Optional[str]] = Field(None, description="Optional domain this document belongs to")
    processed: BackendType[bool] = Field(False, description="Whether the document has been processed")

# LLM-facing candidate (int code, not Literal)
