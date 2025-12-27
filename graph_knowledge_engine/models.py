# ✅ Models for Chroma + LLM with optional embeddings and adjudication
import logging
import os
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.debug("loading models")
from typing import List, Literal, Optional, Dict, Any, Type, TypeAlias, Union, Annotated, ClassVar
from pydantic import BaseModel, Field, model_validator, field_validator, ValidationInfo
import uuid
from enum import IntEnum
import os
import json
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

class Span(ModeSlicingMixin, BaseModel):
    """A single span"""
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
    end_char: int = Field(..., ge=-1, description="Character offset within end_page")
    # Optional extras
    source_cluster_id: Optional[str] = Field(None, description = 'source text cluster id')
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
        if self.start_page == self.end_page and (self.end_char < self.start_char) and not self.end_char == -1:
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



class Grounding(ModeSlicingMixin, BaseModel):
    spans: Annotated[List[Span], FrontendField(),BackendField(),DtoField(),LLMField()] = Field(
        ..., min_items=1, description="One or more locatable mentions supporting this entity"
    )
    def validate_span(self, span:Span):
        
        pass
    def __init__(self, span : Span | list[Span]):
        if type(span) is Span:
            self.spans = [span]
        elif type(span) is list:
            self.spans = span
        pass
    def validate_from_source(self):
        for sp in self.spans:
            self.validate_span(sp)
        pass

class GraphEntityExtractionBase(GraphEntityBase):
    # the mentions warps Grounding
    
    groundings: Annotated[Grounding, FrontendField(),BackendField(),DtoField(),LLMField()] = Field(
        ..., min_items=1, description="Mentioning of the idea across possibly multiple paragraphs"
    )
    #NEED-FIX
    @field_validator("groundings")
    @classmethod
    def _require_non_empty_groundings(cls, mentions: List[Grounding], info: ValidationInfo):
        
        if not mentions:
            raise ValueError("At least one grounding is required")
        for g in mentions:
            g.validate_from_source()
        return mentions
    def to_type(self, type: Type):
        if type is GraphEntityRefBase:
            return self.coerce_to_db()
        else:
            raise(ValueError("unrecognised type"))
    def coerce_to_db(self):
        # works for single extraction, shield the inner db representation
        # convert the groundings -> mentions List[Grounding]
        temp = self.model_dump()
        temp['mentions'] = [temp.pop('groundings')]
        return GraphEntityRefBase.model_validate(temp)

class GraphEntityRefBase(GraphEntityBase):
    # the mentions warps Grounding
    
    mentions: Annotated[List[Grounding], FrontendField(),BackendField(),DtoField(),LLMField()] = Field(
        ..., min_items=1, description="Mentioning of the idea across possibly multiple paragraphs/ data sources"
    )
    #NEED-FIX
    @field_validator("mentions")
    @classmethod
    def _require_non_empty_refs(cls, mentions: List[Grounding], info: ValidationInfo):
        
        if not mentions:
            raise ValueError("At least one mentions is required")
        for g in mentions:
            g.validate_from_source()
        return mentions


class EdgeMixin(ModeSlicingMixin, BaseModel):
    source_ids: List[str] = Field(..., description="List of source node IDs")
    target_ids: List[str] = Field(..., description="List of target node IDs")
    relation: str = Field(..., description="Type of relationship between source and target nodes")
    source_edge_ids: Optional[List[str]] = Field(..., description="List of source edge IDs")
    target_edge_ids: Optional[List[str]] = Field(..., description="List of target edge IDs")
# -------------------------
# Storage-facing mixin (embedding OPTIONAL)
# -------------------------

from typing import Sequence
class ChromaMixin(BaseModel):
    id: str = Field(default_factory=generate_id, description="Unique identifier")
    embedding: Optional[Sequence[float]] = Field(None, description="Vector embedding for the entity")
    # Optional but handy to keep JSON and Chroma metadata aligned
    doc_id: Optional[str] = Field(None, description="Document ID from which this entity was extracted")
    metadata: dict = Field(
        {}, description="metadata"
    )

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
class DocNode(DocNodeMixin, GraphEntityRefBase): # type: ignore
    pass

class PureChromaNode(ChromaMixin,GraphEntityBase):
    # base node without reference enforced
    pass
class PureChromaEdge(ChromaMixin,EdgeMixin,GraphEntityBase):
    # base edge without reference enforced
    pass
class PureGraph(ModeSlicingMixin, BaseModel):
    nodes: List[PureChromaNode] = Field(..., description="List of refless nodes")
    edges: List[PureChromaEdge] = Field(..., description="List of refless edges")

class ChromaValidateSourceMixin(BaseModel):
    # provide methods to validate model reference/ source from chromadb
    def validate_from_source(self, source):
        pass

class Node(ChromaValidateSourceMixin, ChromaMixin, GraphEntityRefBase):
    # Node with ref session enforced
    pass

class Edge(ChromaValidateSourceMixin, ChromaMixin, EdgeMixin, GraphEntityRefBase):
    # Edge with ref session enforced
    
    pass
class LLMNode( LLMMixin, GraphEntityRefBase):
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

class LLMEdge( LLMMixin, EdgeMixin, GraphEntityRefBase):
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

class GroundginMandatorySnippet(Span):
    snippet: str = Field(..., description="Short text snippet for quick preview")

    pass
class LLMNodeExtraction(LLMNode):
    "extracted node information"
    
    groundings: Annotated[List[GroundginMandatorySnippet], FrontendField(),BackendField(),DtoField(),LLMField()] = Field(
        ..., min_items=1, description="One or more locatable mentions supporting this entity"
    )

class LLMEdgeExtraction(LLMEdge):
    "extracted edge information"
    
    groundings: Annotated[List[GroundginMandatorySnippet], FrontendField(),BackendField(),DtoField(),LLMField()] = Field(
        ..., min_items=1, description="One or more locatable mentions supporting this entity"
    )

class GraphExtractionWithIDs(ModeSlicingMixin, BaseModel):
    """represent a graph extracted by external tool and all ids are imported

    Args:
        ModeSlicingMixin (_type_): _description_
        BaseModel (_type_): _description_
    """
    nodes: List[Node] = Field(..., description="List of nodes")
    edges: List[Edge] = Field(..., description="List of edges")
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
        sliced: LLMGraphExtraction['llm'] | dict
        if isinstance(sliced, BaseModel):
            dumped = sliced.model_dump()
        elif type(sliced) is dict:
            dumped = sliced
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
    @staticmethod
    def from_text(text: str, **kwarg):
        return Document(content = text, type = "text", metadata = {}, **kwarg)
        
    def get_content_by_span(self, span: Span)-> str:
        if self.type == 'ocr':
            return self.get_content_as_ocr_doc_by_span(span)
        elif self.type == 'text':
            return self.get_content_as_text_doc_by_span(span)
        else:
            raise Exception ("Unknown document type")

    def get_content_as_text_doc_by_span(self, span: Span)-> str:
        if self.type != 'text':
            raise Exception("only can call method when doc is plain text type")
        return self.content[span.start_char:span.end_char]
        return ""
    
    def get_content_as_ocr_doc_by_span(self, span: Span)-> str:
        if self.type != 'ocr':
            raise Exception("only can call method when doc is ocr type")
        return ""
    
    
#========================= OCR DOC

class box_2d(BaseModel):
    box_2d: list[int] = Field(description = 'box y min, x min, y max and x max')
    label : str = Field(description = 'text in the box')
    id: int  = Field(description = 'id of the text box in the page, autoincrement from 0')    
class TextCluster(ModeSlicingMixin, BaseModel):
    """a text cluster along with spatial information"""
    text: DtoType[str] = Field(description='the text content of the text cluster')
    bb_x_min: DtoType[float]  = Field(description='the bounding box x min in pixel coordinate of the text_cluster. ')
    bb_x_max: DtoType[float]  = Field(description='the bounding box x max in pixel coordinate of the text_cluster. ')
    bb_y_min: DtoType[float]  = Field(description='the bounding box y min in pixel coordinate of the text_cluster. ')
    bb_y_max: DtoType[float]  = Field(description='the bounding box y max in pixel coordinate of the text_cluster. ')
    cluster_number: int = Field(description="per page unique number of the cluster, starting from 0")
class NonTextCluster(ModeSlicingMixin, BaseModel):
    """Recognised meaningful objects other than OCR characters, include image, figures. """
    description: DtoType[str] = Field(description='the description or summary of the non-OCR object')
    bb_x_min: DtoType[float]  = Field(description='the bounding box x min in pixel coordinate of the non-OCR object. ')
    bb_x_max: DtoType[float]  = Field(description='the bounding box x max in pixel coordinate of the non-OCR object. ')
    bb_y_min: DtoType[float]  = Field(description='the bounding box y min in pixel coordinate of the non-OCR object. ')
    bb_y_max: DtoType[float]  = Field(description='the bounding box y max in pixel coordinate of the non-OCR object. ')
    cluster_number: int = Field(description="per page unique number of the cluster, starting from 0")

class OCRClusterResponse(ModeSlicingMixin, BaseModel):
    """cluster and signatore share the same unique set of cluster number, i.e. if cluster 1 is signature, no other text cluster can take 1"""
    OCR_text_clusters: DtoType[list[TextCluster]] = Field(description="the OCR text results. Share cluster number uniqueness with non-OCR objects. ")
    non_text_objects:  DtoType[list[NonTextCluster]] = Field(description="the non-OCR object results. Share cluster number uniqueness with OCR texts. ")
    is_empty_page: DtoType[Optional[bool]] = Field(default = False, description="true if the whole page is empty without recognisable text.")
    printed_page_number: DtoType[Optional[str]] = Field(description='the page number identified from OCR texts, can be in form of roman numerals such as "i", "ii", "iii", "iv"...; ' 
                                       'Arabic numeral such as 1, 2, 3... or letter such as "a", "b", "c"...\n'
                                       'Sometimes the are surrounded by symbols such as "- 1 -", "- 2 -"'
                                       r"Can be null/none if there is no page order assigned and printed and found in the scanned texts. Do not assign page number. Only use page number found.")
    meaningful_ordering : DtoType[list[int]] = Field(description="The correct meaningful ordering of the identified text clusters. Must cover all OCR_text_clusters once and only once. ")
    
    # block_signed: DtoType[list[bool]]= Field(default = [], description="Same array length as signature_blocks, corresponding to signed or not")
    page_x_min : DtoType[float]=Field(description='the page x min in pixel coordinate. ')
    page_x_max : DtoType[float]=Field(description='the page x max in pixel coordinate. ')
    page_y_min : DtoType[float]=Field(description='the page y min in pixel coordinate. ')
    page_y_max : DtoType[float]=Field(description='the page y max in pixel coordinate. ')
    estimated_rotation_degrees : DtoType[float]=Field(description='the page estimated rotation degree using right hand rule. ')
    incomplete_words_on_edge: DtoType[bool] = Field(description='If there is any text being incomplete due to the scan does not scan the edges properly. ')
    incomplete_text: DtoType[bool]  = Field(description='Any incomplete text')
    data_loss_likelihood: DtoType[float] = Field(description='The likelihood (range from 0.0 to 1.0 inclusive) that the page has lost information by missing the scan data on the edges of the page.' )
    scan_quality: DtoType[Literal['low', 'medium', 'high']] = Field(description='The image quality of the scan. All qualities exclude signatures. '
                                                                                      '"low", "medium" or "high". '
                                'low: text barely legible. medium: Legible with non smooth due to pixelation. high: texts are easily and highly identifiable. ' )
    contains_table: DtoType[bool] = Field(description='Whether this page contains table. ')

    @model_validator(mode='after')
    def check_cluster_meaningful_ordering_agreement(self):
        assert bool(self.is_empty_page) ^ (len(self.OCR_text_clusters) > 0), f"is_empty_page value {self.is_empty_page} disagree with OCR_text_clusters len={len(self.OCR_text_clusters)}"
        if not len([i.cluster_number for i in (self.non_text_objects + self.OCR_text_clusters)]) == len(set(i.cluster_number for i in self.non_text_objects + self.OCR_text_clusters)):
            raise ValueError("cluster number from non_text_objects block and ocr text blocks must be ALL distinct. ")
        try:
            if not (len(self.meaningful_ordering) == len(set(self.meaningful_ordering))): # <= len(self.OCR_text_clusters)):
                raise ValueError("meaningful_order must cover each text cluster at most once")
        except Exception as e:
            raise e
        return self

    

class OCRClusterResponseBc(OCRClusterResponse):
    # backward compability layer, can use by overriding fields with union of previous versions
    
    pass
    # OCR_text_clusters: list[TextCluster | TextCluster_yolo_bb] = Field(description="the OCR text results, " # type: ignore
    #                                                                    "prefer min max bounding box to centre width height style")
    
class SplitPageMeta(BaseModel):
    ocr_model_name: str = Field(description="model that perform OCR")
    ocr_datetime: float = Field(description="unix timestamp when ocr is performed")
    ocr_json_version: str = Field(description = "the model does the OCR") 
    @field_validator('ocr_json_version', mode = "before")
    def version_to_str(cls, v):
        return str(v)
class SplitPage(OCRClusterResponseBc):
    # model not for LLM response
    pdf_page_num: int
    metadata: SplitPageMeta
    refined_version: Optional[OCRClusterResponse[DtoField]] = Field(default = None, description = "refined processed/ grouped/ merged version of ocr text clusters. ")
    def model_dump(self, *arg, **kwarg):
        return self.to_doc()
    def dump_raw(self):
        return super(SplitPage, self).model_dump(exclude = ["refined_version"])
    def dump_supercede_parse(self):
        return super(SplitPage, self).model_dump(exclude = ["refined_version", "metadata"])
    @model_validator(mode="after")
    def roundtrip_invariant(self, info: ValidationInfo) -> "SplitPage":
        # Context may be None if caller didn't pass it
        ctx = info.context or {}
        # Re-entrancy guard
        if ctx.get("_roundtrip_active", False):
            return self
        # First time from here
        self.to_doc()
        # Mark active for the nested validation
        nested_ctx = dict(ctx)
        nested_ctx["_roundtrip_active"] = True

        dumped = self.model_dump(mode="json")

        # IMPORTANT: pass context so nested validation sees the flag
        again = self.__class__.model_validate(dumped, context=nested_ctx)
        
        
        # Optional: assert equivalence (pick your definition)
        if again != self:
            raise ValueError("Roundtrip invariant failed: dump->validate changed the model")

        return self
    def to_doc(self):
        """Model to llm one-way serializer with manual slicing logic, can refactor using sliced view
        with some token saving logic. 
        """
        target = self.refined_version or self
        ocr_cluster = target.OCR_text_clusters
        non_ocr_cluster = target.non_text_objects
        if target.contains_table:
            id_sorted_text_cluster = []
            cluster_numbers = (i.cluster_number for i in ocr_cluster)
            assert len(set(cluster_numbers)) == len(ocr_cluster)
            cluster_lookup_by_number = {i.cluster_number : i for i in (ocr_cluster + non_ocr_cluster#+ target.signature_blocks
                                                                       )}
            for i in target.meaningful_ordering:
                c_p = cluster_lookup_by_number.get(i)
                if c_p is None:
                    raise KeyError(f"{i} does not exist")
                cluster_dump: dict = c_p.model_dump()
                cluster_dump.pop("cluster_number")
                id_sorted_text_cluster.append(cluster_dump)
            others = (set(cluster_numbers) - set(target.meaningful_ordering))
            for i in others:
                cluster_dump: dict
                cluster_dump = cluster_lookup_by_number[i].model_dump()
                cluster_dump.pop("cluster_number")
                id_sorted_text_cluster.append(cluster_dump)
            c_return = {}
            c_return['pdf_page_num'] = self.pdf_page_num
            c_return['printed_page_number'] = self.printed_page_number
            c_return['OCR_text_clusters'] = id_sorted_text_cluster
            c_return['contains_table'] = self.contains_table
            return c_return
        else:
            # isSorted = True
            expected_next = 0
            i_ocr = 0
            # i_sig = 0
            ocr_clus_nums = sorted([i.cluster_number for i in ocr_cluster])
            non_clus_nums = sorted([i.cluster_number for i in non_ocr_cluster])
            # sig_clus_nums = sorted([i.cluster_number for i in target.signature_blocks])
            # expected_next = min(ocr_clus_nums + sig_clus_nums)
            if ocr_clus_nums:
                start_num = min(ocr_clus_nums)
                assert start_num in [0, 1], "only allow 0-indexed based or 1-indexed based cluster numbers"
                expected_next = start_num
                while expected_next < start_num + (len(ocr_clus_nums) + len(non_clus_nums)):
                    if (i_ocr <len(ocr_cluster)) and expected_next == ocr_cluster[i_ocr].cluster_number:
                        i_ocr += 1
                    # elif (i_sig < len(target.signature_blocks)) and expected_next == target.signature_blocks[i_sig].cluster_number:
                    #     i_sig += 1
                    else:
                        raise Exception('expected index {expected_next} not found in both ocr_cluster nor signature_blocks')
                    expected_next += 1

            
            id_sorted_text_cluster = sorted(ocr_cluster, key=lambda x : x.cluster_number )
            assert len(id_sorted_text_cluster) == len(set(i.cluster_number for i in id_sorted_text_cluster))
            is_normal = True
            shift = 0 # try zero indexing sanity
            c : TextCluster #| TextCluster_yolo_bb
            for i, c in enumerate(id_sorted_text_cluster):
                
                if c.cluster_number != i:
                    is_normal = False
                    break
            c2: TextCluster #| TextCluster_yolo_bb
            if not is_normal:
                is_normal = True
                shift = -1 # try one-indexing sanity
                for i, c2 in enumerate(id_sorted_text_cluster):
                    if c2.cluster_number != i+1:
                        is_normal = False
                        break
            # sig_num = set([i.cluster_number for i in self.signature_blocks])
            assert set(self.meaningful_ordering) <= set(i.cluster_number for i in (self.OCR_text_clusters))
            if is_normal:
                # is sorted list where index is order + shift
                texts = '\n'.join(id_sorted_text_cluster[i+shift].text for i in self.meaningful_ordering)
            else:
                # general case
                tcd = {x.cluster_number:x   for x in id_sorted_text_cluster}
                texts = '\n'.join(tcd[i].text for i in self.meaningful_ordering)
            c_return = {}
            c_return['pdf_page_num'] = self.pdf_page_num
            c_return['printed_page_number'] = self.printed_page_number
            c_return['text'] = texts
            return c_return

    
PastCompatibleSplitPage: TypeAlias = SplitPage
def get_page_json(folder_path, page_num):
    with open(os.path.join(folder_path, 'page_'+str(page_num)+'.json'), 'r') as f:
        file_json_raw = json.load(f)
    return file_json_raw
def regen_page(file_json_raw, use_raw):
        # add compatible to union if want to compatible with past models
    """regen from json returned by SplitPage.to_doc(), can be view as SplitPage.FromJson(filepath)"""
    p = PastCompatibleSplitPage(**file_json_raw)
    if use_raw:
        return p.dump_supercede_parse()
    try:
        res = p.to_doc()
    except:
        raise
    return res
def regen_doc(folder_path, use_raw = False):
    pages_nums = sorted((int(i.rsplit(".json",1)[0].split("page_",1)[1]) for i in os.listdir(folder_path) if i.endswith('.json') and i.startswith("page_")))
    pages = []
    split_pages = []
    for pn in pages_nums:
        try:
            pages.append(get_page_json(folder_path, pn))
            split_pages.append(regen_page(pages[-1], use_raw = use_raw))
        except Exception as e:
            folder_path,pn
            print(f'error at page {pn}')
            print(f'in file {folder_path}')
            logger.error(f'error at page {pn}')
            logger.error(f'in file {folder_path}')
            raise
    
    # pages = map( partial(get_page_json, folder_path= folder_path), pages_nums)
    # split_pages = map(regen_page, pages)
    full_doc = list(split_pages)
    return full_doc
# class Doc(BaseModel):
#     file_full_path: str = Field(description = "file full path")
#     pages: list[dict[str, Any]]  = Field(description = "pages")
    
# def index_doc_group(doc_group_dumped):
#     """convert at a look up of tuple[filename, pagenum] -> doc content"""
#     doc_group_indexed = {(f,i['pdf_page_num']): i for f in doc_group_dumped['documents'] for i in  doc_group_dumped['documents'][f]}
#     return doc_group_indexed
# class DocumentGroup(BaseModel):
#     documents : dict[str, list|Doc] = Field(description = 'list of documents')
#     def to_doc_group_indexed(self):
#         doc_group = self.model_dump()
#         doc_group_indexed = index_doc_group(doc_group)
#         return doc_group_indexed
#     @staticmethod
#     def from_doc_folder(folder_path):
#         """ Assume the dir contains a list of folder with each folder with the filename
#         each subfolder contains a list of pages

#         Args:
#             folder_path (_type_): _description_
#         """
#         dirs = os.listdir(folder_path)
#         doc_group = {}
#         for d in dirs:
#             doc = regen_doc(os.path.join(folder_path, d)) 
#             doc_group[d] = doc
#         return DocumentGroup(documents= doc_group)