
import logging
import os
from dataclasses import dataclass

if True:
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    logger.debug("loading models")
from ..id_provider import new_id_str, new_event_id, stable_id
from typing import List, Literal, Optional, Dict, Any, Type, TypeAlias, TypedDict, Union, Annotated, ClassVar, Tuple, Self, cast
from pydantic import BaseModel, Field, model_validator, field_validator, ValidationInfo, ConfigDict
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
from ..id_provider import new_event_id, stable_id
from pydantic import BaseModel, Field
from typing import Optional, ClassVar, Literal
Role :TypeAlias = Literal["user", "assistant", "system", "tool"]
class IdPolicyMixin(BaseModel):
    id: Optional[str] = Field(default=None)
    id_policy: ClassVar[Literal["event","canonical"]] = "event"
    id_kind: ClassVar[str] = "model"

    def identity_key(self) -> Tuple[str, ...]:
        """
        Subclasses with id_policy="canonical" MUST override this.
        Should return stable, minimal identity parts.
        """
        return self.__class__.__name__,
    @model_validator(mode="after")
    def _ensure_id(self) -> Self:
        if self.id_policy == "canonical" and self.identity_key.__func__ is IdPolicyMixin.identity_key:
            raise TypeError("Canonical models must override identity_key()")
        if self.id is not None:
            return self        

        if self.id_policy == "event":
            self.id = str(new_id_str())
            return self

        # canonical
        key = self.identity_key()  # must be stable & non-empty
        self.id = str(stable_id(self.id_kind, *key))
        return self
@dataclass(frozen=True)
class ContextCost:
    """Canonical cost abstraction for prompt/context accounting."""
    char_count: int = 0
    token_count: int | None = None

    def __add__(self, other: "ContextCost") -> "ContextCost":
        tc = None
        if self.token_count is not None and other.token_count is not None:
            tc = self.token_count + other.token_count
        return ContextCost(char_count=self.char_count + other.char_count, token_count=tc)

    def to_flat_metadata(self, *, prefix: str = "cost") -> Dict[str, Any]:
        # You requested dot keys.
        return {
            f"{prefix}.char_count": int(self.char_count),
            f"{prefix}.token_count": (None if self.token_count is None else int(self.token_count)),
        }

    @staticmethod
    def from_flat_metadata(meta: Dict[str, Any], *, prefix: str = "cost") -> "ContextCost":
        cc_key = f"{prefix}.char_count"
        tc_key = f"{prefix}.token_count"

        char_count = int(meta.get(cc_key, 0) or 0)
        token_raw = meta.get(tc_key, None)
        token_count: Optional[int]
        if token_raw is None or token_raw == "":
            token_count = None
        else:
            token_count = int(token_raw)

        return ContextCost(char_count=char_count, token_count=token_count)

        # {
        #     "user_turn_node_id": turn_node_id,
        #     "response_turn_node_id": response_turn_node_id,
        #     "turn_index": new_index,
        #     "relevant_kg_node_ids": [i for i in (kg.selected.node_ids if kg.selected else [])],
        #     "relevant_kg_edge_ids": [i for i in (kg.selected.edge_ids if kg.selected else [])],
        #     "pinned_kg_pointer_node_ids": pinned_ptrs,
        #     "pinned_kg_edge_ids": pinned_edges,
        #     "memory_context_node_id": memory_pin.memory_context_node if memory_pin else None,
        #     "memory_context_edge_ids": memory_pin.pinned_edges if memory_pin else [],
        # }



# -------------------------
# Provenance / reference (with optional spans)
# -------------------------
class MentionVerification(BaseModel):
    """Result of verifying a mention span against the source text."""
    method: Literal["llm", "levenshtein", "regex", "heuristic", "human", "ensemble", "system"] = Field(
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
    # Required locators 
    page_number: int = Field(..., ge=1, description="1-indexed page number where the mentioned")
    # end_page: int = Field(..., ge=1, description="1-based page index where the mention ends (>= start_page)")
    start_char: int = Field(..., ge=0, description="Character offset within start_page, zero indexed, first char in source document is index 0 with 0 offset")
    end_char: int = Field(..., ge=1, description="Character offset within end_page, zero indexed")
    excerpt: str = Field(..., description="the direct excerpt from source doc from start char to end char. Must be identical from extracted using start_char and end_char")
    context_before: str = Field(..., description='words before the exceprt for uniqueness excerpt identification, empty string "" if excerpt is start of text')
    context_after: str  = Field(..., description="words after the exceprt for uniqueness excerpt identification, empty string "" if excerpt is end of text")
    # Optional extras
    # only for chunked text document
    chunk_id: Optional[Annotated[str, LLMField(), ExcludeMode("frontend", "backend", "dto")] ] = Field(None, description = 'source text chunk id for chunked text')
    source_cluster_id: Optional[str] = Field(None, description = 'source text cluster id')
    
    verification: Annotated[Optional[MentionVerification], BackendField(), ExcludeMode("llm")] = Field(
                                        None, description="Result of validating the mention correctness"
                                    )
    
    @staticmethod
    def from_dummy_for_workflow(doc_id = "_wf:_dummy"):
        if doc_id.startswith("_wf:"):
            pass
        else:
            doc_id = "_wf:"+doc_id
        dummy_span = Span(
            collection_page_url=f"workflow/{doc_id}",
            document_page_url=f"workflow/{doc_id}",
            doc_id=f"{doc_id}",
            insertion_method="system",
            page_number=1, start_char=0, end_char=1,
            excerpt="", context_before="", context_after="",
            chunk_id = None,
            source_cluster_id = None,
            verification=MentionVerification(method="system", is_verified=True, score=1.0, notes="")
        )
        return dummy_span
    @staticmethod
    def from_dummy_for_conversation(doc_id = "_conv:_dummy"):
        if doc_id.startswith("_conv:"):
            pass
        else:
            doc_id = "_conv:"+doc_id
        dummy_span = Span(
            collection_page_url=f"conversation/{doc_id}",
            document_page_url=f"conversation/{doc_id}",
            doc_id=f"{doc_id}",
            insertion_method="system",
            page_number=1, start_char=0, end_char=1,
            excerpt="", context_before="", context_after="",
            chunk_id = None,
            source_cluster_id = None,
            verification=MentionVerification(method="system", is_verified=True, score=1.0, notes="")
        )
        return dummy_span
    @staticmethod
    def from_dummy_for_document():
        dummy_span = Span(
            collection_page_url=f"document_collection/{Document.from_dummy().id}",
            document_page_url=f"document_collection/{Document.from_dummy().id}",
            doc_id=Document.from_dummy().id,
            insertion_method="system",
            page_number=1, start_char=0, end_char=1,
            excerpt="", context_before="", context_after="",
            chunk_id = None,
            source_cluster_id = None,
            verification=MentionVerification(method="system", is_verified=True, score=1.0, notes="dummy document span")
        )
        return dummy_span    
    def fix_chunk_start_end_char(self, source_map: dict[str, "Chunk"]):
        if self.chunk_id is None:
            raise Exception("only for chunnked span")
        chunk = source_map.get(self.chunk_id)
        if chunk is None:
            raise Exception(f"Missing chunk id {self.chunk_id} in source_map")
        chunk_start_char = chunk.end_char
        chunk_end_char = chunk.end_char
        chunk_length =  chunk_end_char-chunk_start_char
        own_len = self.end_char - self.start_char
        if self.start_char > chunk_length :
            # assume it used global id accidentally instead of local id
            if self.start_char >= chunk_start_char and self.start_char < chunk_end_char:
                self.start_char -= chunk_start_char
            if self.end_char > chunk_start_char and self.end_char <= chunk_end_char:
                self.end_char -= chunk_start_char
            else:
                self.end_char = min(self.start_char + own_len, chunk_length)
        else:
            self.start_char = chunk_start_char
            self.end_char = min(self.start_char + own_len, chunk_length)

    def pop_chunk(self, source_map):
        # pop chunk information and switch to global indexing
        self.fix_chunk_start_end_char(source_map)
        if self.chunk_id is not None:
            from copy import deepcopy
            chunk: Chunk = deepcopy(source_map[self.chunk_id])
            self.chunk_id = None
            self.start_char = chunk.start_char + self.start_char
            self.end_char = chunk.start_char + self.end_char
            return chunk
        else:
            raise AttributeError("chunk_id is None")
        
    
    def from_llm_span(self, span : "Span['llm']", source_map) -> "Span":
        if type(span) is not type(Span["llm"]):
            raise TypeError(f'span is not of type {type(Span["llm"])}')
        sp2 = span.model_copy(deep=True)
        sp2.pop_chunk(source_map)
        return Span.model_validate(sp2.model_dump())
        
    
    def llm_to_unsliced(self: "Span['llm']", source_map):
        # makesure chunk id is converted to global id for text case, need source map to resolve
        if type(self) is Span['llm']:
            _ = self.pop_chunk(source_map)
            return Span.model_validate(self.model_dump)
        else:
            raise TypeError(f'span is not of type {type(Span["llm"])}')
        
    
    @model_validator(mode="before")
    @classmethod
    def missing_fields(cls, data, info: ValidationInfo):
        
        # data_dump = data.model_dump()
        if type(data) is dict:
            if data.get('insertion_method') is None:
                insertion_method = info.context and info.context["insertion_method"]
                data['insertion_method'] = insertion_method
        return data
    @model_validator(mode="after")
    def _check_span_consistency(self):
        # if self.end_page < self.start_page:
        #     raise ValueError("end_page must be >= start_page")
        if (self.end_char <= self.start_char) and not self.end_char == -1:
            raise ValueError("end_char must be > start_char")
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
class Domain(IdPolicyMixin, BaseModel):
    id: str = Field(..., description="Unique identifier for the domain")
    name: str = Field(..., description="Name of the domain")
    description: Optional[str] = Field(None, description="Optional description of the domain")
    # id_policy: ClassVar[Literal["event", "canonical"]] = "canonical"
    # id_kind: ClassVar[str] = "model"  # override per subclass if you want stable separation

    def identity_key(self) -> Tuple[str, ...]:
        """
        Subclasses with id_policy="canonical" MUST override this.
        Should return stable, minimal identity parts.
        """
        return self.__class__.__name__, str(self.id_policy)

    
# -------------------------
# Core graph entities
# -------------------------
from typing import Mapping
class GraphEntityBase(ModeSlicingMixin, BaseModel):
    label: str = Field(..., description="Human-readable label for the node or edge")
    type: Literal['entity', 'relationship', 'reference_pointer'] = Field(..., description="Type of entity")
    summary: str = Field(..., description="Summary of the node/relationship")
    domain_id: Optional[str] = Field(None, description="Domain ID this entity belongs to")
    canonical_entity_id: Optional[str] = Field(
        None, description="Canonical ID to link equivalents (e.g., Wikidata QID or internal UUID)"
    )
    properties: Optional[Mapping[str, JsonPrimitive| list[JsonPrimitive] | Mapping[str, JsonPrimitive] ]] = Field(
        None, description="Optional flat properties (JSON primitives only)"
    )



class Grounding(ModeSlicingMixin, BaseModel):
    spans: Annotated[List[Span], FrontendField(),BackendField(),DtoField(),LLMField()] = Field(
        ..., min_items=1, description="One or more locatable span across chunks/text-clusters supporting this entity"
    )
    def validate_span(self, span:Span):
        
        pass
    @field_validator("spans")
    def spans_validate(cls, spans : Span | list[Span]):
        if type(spans) is Span:
            spans = [spans]
        elif type(spans) is list:
            spans = spans
        return spans
    def validate_from_source(self):
        for sp in self.spans:
            self.validate_span(sp)
        pass

class GraphEntityExtractionBase(GraphEntityBase):
    # the groundings wrap grouped supporting spans
    
    mentions: Annotated[List[Grounding], FrontendField(),BackendField(),DtoField(),LLMField()] = Field(
        ..., min_items=1, description="Mentioning of the idea across possibly multiple paragraphs"
    )
    #NEED-FIX
    @field_validator("mentions")
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
    def iter_span(self):
        for g in self.mentions:
            for sp in g.spans:
                yield sp
    #NEED-FIX
    @field_validator("mentions")
    @classmethod
    def _require_non_empty_refs(cls, mentions: List[Grounding], info: ValidationInfo):
        try:
            if not mentions:
                raise ValueError("At least one mentions is required")
            for g in mentions:
                g.validate_from_source()
        except Exception as _e:
            raise
        return mentions
    @model_validator(mode="before")
    @classmethod
    def end_char_minus_1_to_ending_index(cls, data):
        def check_and_update_span_inplace(span):
                        if type(span) is Span:
                            if span.end_char == -1:
                                span.end_char += len(span.excerpt)
                        else:
                            if type(span) is not dict:
                                raise ValueError("span has wrong data type of ")
                            if span.get("end_char") == -1:
                                span["end_char"] += len(span["excerpt"])
        try:
            mentions:list[Grounding]
            if type(data['mentions']) is str:
                mentions = [Grounding.model_validate(i) for i in json.loads(data['mentions'])]
            else:
                mentions = data['mentions']
            if type(mentions) is not list:
                raise TypeError("mentions should be a list of groundings")
            for mention in mentions:
                if type(mention) is Grounding:
                    for span in mention.spans:
                        check_and_update_span_inplace(span)
                else:
                    
                        
                    try:
                        mention_dict: dict = cast(dict, mention)
                        for span in (mention_dict.get('spans') or []):
                            check_and_update_span_inplace(span)
                    except Exception as _e:
                        raise
        except Exception as _e:
            raise
        return data

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
from pydantic import ConfigDict
class BaseNodeMetadata(BaseModel):
    """
    Excample fields
    # level_from_root: int = Field(..., ge=0)
    # entity_type: str
    # char_distance_from_last_summary: int
    # turn_distance_from_last_summary: int
    """
    model_config = ConfigDict(extra="allow")

# --- Phase 1: Conversation edge intent classification (causality) ---

CONVERSATION_EDGE_CAUSAL_TYPE_BY_RELATION: dict[str, str] = {
    # Canonical chain
    "next_turn": "chain",

    # Tool / retrieval wiring (non-causal references)
    "tool_call_entry_point": "reference",
    "run_result": "reference",
    "has_memory_context": "reference",
    "has_knowledge_context": "reference",

    # Summaries describe past, but do not causally "create" the past
    "summarizes": "summary",

    # Default catch-all for conversation edges
}

class ChromaMixin(BaseModel):
    id: Optional[str] = Field(default = None, description="Unique identifier")
    embedding: Optional[Sequence[float]] = Field(None, description="Vector embedding for the entity")
    # Optional but handy to keep JSON and Chroma metadata aligned
    doc_id: Optional[str] = Field(None, description="Document ID from which this entity was extracted")
    metadata: dict = Field(
        {}, description="metadata"
    )
    @field_validator('metadata')
    def check_metadata(cls, v):
        BaseNodeMetadata.model_validate(v)
        return v
    
# -------------------------
# LLM-facing mixin (NO embedding field to keep schema tight)
# -------------------------
class LLMMixin(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"llm"}
    id: Optional[str] = Field(default = None, description="None for new object; use existing IDs to upsert")
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
    def get_extra_update(self):
        return {}
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

class LevelAwareMixin(BaseModel):
    """Mixin to handle level_from_root synchronization with metadata"""
    level_from_root: Optional[int] = Field(None, description="Hierarchy level from root, metadata store final authoritative")

    @model_validator(mode="before")
    @classmethod
    def sync_level_from_root(cls, data: Any, info: ValidationInfo) -> Any:
        if isinstance(data, dict):
            metadata = data.get("metadata", {}) or {}
            # Pull from metadata if not explicitly set in data
            if "level_from_root" not in data and "level_from_root" in metadata:
                data["level_from_root"] = metadata["level_from_root"]
        return data

    @model_validator(mode="after")
    def push_level_to_metadata(self) -> "LevelAwareMixin":
        if self.level_from_root is not None:
            if not hasattr(self, "metadata"):
                # Should be mixed in with ChromaMixin or similar
                return self
            if self.metadata is None:
                self.metadata = {}
            self.metadata["level_from_root"] = self.level_from_root
        return self

class TombstoneMixin(BaseModel):
    
    @field_validator('metadata', check_fields=False)
    def check_tombstone_fields(cls, v):
        if "lifecycle_status" not in v:
            v["lifecycle_status"] = "active"
        else:
            assert v["lifecycle_status"] in ['active', 'deleted']
        if "redirect_to_id" not in v:
            v["redirect_to_id"] = None        
        return v

class Node(IdPolicyMixin, TombstoneMixin, LevelAwareMixin, ChromaValidateSourceMixin, ChromaMixin, GraphEntityRefBase):
    """
    Base **provenance-heavy** knowledge graph node.
    
    This is the fundamental unit of the knowledge graph. Unlike typical graph nodes,
    it enforces strict provenance tracking via `ReferenceSession` (spans, verification).
    It supports lifecycle management (tombstoning), hierarchy (level awareness),
    and vector embedding.
    """
    # Node with ref session enforced and level awareness
    id_kind: ClassVar[str] = "kg.node"
    def safe_get_id(self):
        return cast(str, self.id)
    def get_extra_update(self):
        return {}
    # id_policy: ClassVar[Literal["event", "canonical"]] = "canonical"
    # id_kind: ClassVar[str] = "model"  # override per subclass if you want stable separation

    def identity_key(self) -> Tuple[str, ...]:
        """
        Subclasses with id_policy="canonical" MUST override this.
        Should return stable, minimal identity parts.
        """
        return self.summary, str(self.doc_id)

    # @model_validator(mode="after")
    # def _ensure_id(self) -> Self:
    #     if self.node_id is not None:
    #         return self        

    #     if self.id_policy == "event":
    #         self.id = str(new_id_str())
    #         return self

    #     # canonical
    #     key = self.identity_key()  # must be stable & non-empty
    #     self.node_id = stable_id(self.id_kind, *key)
    #     return self

class Edge(IdPolicyMixin, ChromaValidateSourceMixin, ChromaMixin, EdgeMixin, GraphEntityRefBase):
    """
    Base **provenance-heavy** knowledge graph edge.

    Represents a relationship between nodes. Like `Node`, it carries full provenance
    metadata (where this relationship was asserted), supporting "hypergraph" features
    via edge-to-edge connections and rich metadata.
    """
    # Edge with ref session enforced
    id_kind: ClassVar[str] = "kg.edge"
    def safe_get_id(self):
        return cast(str, self.id)
    def get_extra_update(self):
        return {}
    # id_policy: ClassVar[Literal["event", "canonical"]] = "canonical"
    # id_kind: ClassVar[str] = "model"  # override per subclass if you want stable separation

    def identity_key(self) -> Tuple[str, ...]:
        """
        Subclasses with id_policy="canonical" MUST override this.
        Should return stable, minimal identity parts.
        """
        return self.summary, self.relation, str(self.source_ids), str(self.target_ids), str(self.source_edge_ids), str(self.target_edge_ids)

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

    def to_flattened(self, *, span_to_id: dict[tuple, str]) -> "FlattenedLLMNode":
        return FlattenedLLMNode.from_canonical(self, span_to_id=span_to_id)

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

    def to_flattened(self, *, span_to_id: dict[tuple, str]) -> "FlattenedLLMEdge":
        return FlattenedLLMEdge.from_canonical(self, span_to_id=span_to_id)

class GroundginMandatoryExcerpt(Span):
    excerpt: str = Field(..., description="the direct excerpt from source doc from start char to end char. "
                                            "Must be identical from extracted using start_char and end_char")  # type: ignore

class LLMNodeExtraction(LLMNode):
    "extracted node information"
    
    mentions: Annotated[List[GroundginMandatoryExcerpt], FrontendField(),BackendField(),DtoField(),LLMField()] = Field(
        min_items=1, description="One or more locatable mentions supporting this entity"
    )# type: ignore

class LLMEdgeExtraction(LLMEdge):
    "extracted edge information"
    
    mentions: Annotated[List[GroundginMandatoryExcerpt], FrontendField(),BackendField(),DtoField(),LLMField()] = Field(
        min_items=1, description="One or more locatable mentions supporting this entity"
    )# type: ignore



def _span_identity_key(span: Span) -> tuple:
    """Stable key for deduplicating canonical spans inside one flattened payload."""
    return (
        span.collection_page_url,
        span.document_page_url,
        span.doc_id,
        span.page_number,
        span.start_char,
        span.end_char,
        span.excerpt,
        span.context_before,
        span.context_after,
        span.chunk_id,
        span.source_cluster_id,
    )


class FlattenedSpan(Span):
    """LLM-facing flattened span with required temporary id."""
    include_unmarked_for_modes: ClassVar[set[str]] = {"llm"}

    id: str = Field(..., description="Required temporary span id within this extraction payload, e.g. 'sp:1'")

    @classmethod
    def from_canonical(
        cls,
        span: Span,
        *,
        span_id: str,
        insertion_method: Optional[str] = None,
    ) -> "FlattenedSpan":
        # Respect llm slice semantics: keep llm fields (e.g. chunk_id), drop backend-only fields.
        payload = span.model_dump(field_mode="llm")
        payload["id"] = span_id
        resolved_insertion_method = insertion_method if insertion_method is not None else span.insertion_method
        return cls.model_validate(payload, context={"insertion_method": resolved_insertion_method})

    def to_canonical(self, *, insertion_method: Optional[str] = None) -> Span:
        # Use llm mode so chunk_id survives and insertion_method remains an explicit reconstruction concern.
        payload = self.model_dump(field_mode="llm", exclude={"id"})
        resolved_insertion_method = insertion_method if insertion_method is not None else self.insertion_method
        payload["insertion_method"] = resolved_insertion_method
        return Span.model_validate(payload, context={"insertion_method": resolved_insertion_method})


class FlattenedGrounding(ModeSlicingMixin, BaseModel):
    """Grouped support that references one or more root-level spans by id."""
    include_unmarked_for_modes: ClassVar[set[str]] = {"llm"}

    span_ids: Annotated[List[str], FrontendField(),BackendField(),DtoField(),LLMField()] = Field(
        ..., min_items=1, description="One or more root-level span ids supporting this grounding"
    )

    @field_validator("span_ids")
    @classmethod
    def _require_non_empty_span_ids(cls, span_ids: List[str]):
        if not span_ids:
            raise ValueError("At least one span_id is required")
        return span_ids

    @classmethod
    def from_canonical(cls, grounding: Grounding, *, span_to_id: dict[tuple, str]) -> "FlattenedGrounding":
        return cls(span_ids=[span_to_id[_span_identity_key(span)] for span in grounding.spans])

    def to_canonical(self, *, span_by_id: dict[str, Span]) -> Grounding:
        spans: list[Span] = []
        for span_id in self.span_ids:
            try:
                spans.append(span_by_id[span_id])
            except KeyError as e:
                raise KeyError(f"Unknown span id referenced by grounding: {span_id}") from e
        return Grounding(spans=spans)


class FlattenedLLMNode(LLMMixin, GraphEntityBase):
    include_unmarked_for_modes: ClassVar[set[str]] = {"llm"}

    mentions: Annotated[List[FlattenedGrounding], FrontendField(),BackendField(),DtoField(),LLMField()] = Field(
        ..., min_items=1, description="One or more grouped evidence supports for this node"
    )

    @field_validator("mentions")
    @classmethod
    def _require_non_empty_groundings(cls, mentions: List[FlattenedGrounding]):
        if not mentions:
            raise ValueError("At least one grounding is required")
        return mentions

    @classmethod
    def from_canonical(cls, node: "LLMNode", *, span_to_id: dict[tuple, str]) -> "FlattenedLLMNode":
        return cls(
            id=node.id,
            local_id=node.local_id,
            label=node.label,
            type=node.type,
            summary=node.summary,
            domain_id=node.domain_id,
            canonical_entity_id=node.canonical_entity_id,
            properties=node.properties,
            mentions=[FlattenedGrounding.from_canonical(g, span_to_id=span_to_id) for g in node.mentions],
        )

    def to_canonical(self, *, span_by_id: dict[str, Span]) -> "LLMNode":
        return LLMNode(
            id=self.id,
            local_id=self.local_id,
            label=self.label,
            type=self.type,
            summary=self.summary,
            domain_id=self.domain_id,
            canonical_entity_id=self.canonical_entity_id,
            properties=self.properties,
            mentions=[g.to_canonical(span_by_id=span_by_id) for g in self.mentions],
        )


class FlattenedLLMEdge(LLMMixin, EdgeMixin, GraphEntityBase):
    include_unmarked_for_modes: ClassVar[set[str]] = {"llm"}

    mentions: Annotated[List[FlattenedGrounding], FrontendField(),BackendField(),DtoField(),LLMField()] = Field(
        ..., min_items=1, description="One or more grouped evidence supports for this edge"
    )

    @field_validator("mentions")
    @classmethod
    def _require_non_empty_groundings(cls, mentions: List[FlattenedGrounding]):
        if not mentions:
            raise ValueError("At least one grounding is required")
        return mentions

    @classmethod
    def from_canonical(cls, edge: "LLMEdge", *, span_to_id: dict[tuple, str]) -> "FlattenedLLMEdge":
        return cls(
            id=edge.id,
            local_id=edge.local_id,
            label=edge.label,
            type=edge.type,
            summary=edge.summary,
            domain_id=edge.domain_id,
            canonical_entity_id=edge.canonical_entity_id,
            properties=edge.properties,
            source_ids=edge.source_ids,
            target_ids=edge.target_ids,
            relation=edge.relation,
            source_edge_ids=edge.source_edge_ids,
            target_edge_ids=edge.target_edge_ids,
            mentions=[FlattenedGrounding.from_canonical(g, span_to_id=span_to_id) for g in edge.mentions],
        )

    def to_canonical(self, *, span_by_id: dict[str, Span]) -> "LLMEdge":
        return LLMEdge(
            id=self.id,
            local_id=self.local_id,
            label=self.label,
            type=self.type,
            summary=self.summary,
            domain_id=self.domain_id,
            canonical_entity_id=self.canonical_entity_id,
            properties=self.properties,
            source_ids=self.source_ids,
            target_ids=self.target_ids,
            relation=self.relation,
            source_edge_ids=self.source_edge_ids,
            target_edge_ids=self.target_edge_ids,
            mentions=[g.to_canonical(span_by_id=span_by_id) for g in self.mentions],
        )


class FlattenedLLMGraphExtraction(ModeSlicingMixin, BaseModel):
    include_unmarked_for_modes: ClassVar[set[str]] = {"llm"}

    spans: List[FlattenedSpan] = Field(..., description="Root-level spans referenced by node/edge groundings")
    nodes: List[FlattenedLLMNode] = Field(..., description="List of extracted flattened nodes")
    edges: List[FlattenedLLMEdge] = Field(..., description="List of extracted flattened edges")

    @field_validator("spans")
    @classmethod
    def _validate_unique_span_ids(cls, spans: List[FlattenedSpan]):
        seen: set[str] = set()
        for sp in spans:
            if sp.id in seen:
                raise ValueError(f"Duplicate span id found: {sp.id}")
            seen.add(sp.id)
        return spans

    @model_validator(mode="after")
    def _validate_references(self):
        span_ids = {sp.id for sp in self.spans}
        referenced: set[str] = set()
        for node in self.nodes:
            for grounding in node.mentions:
                for span_id in grounding.span_ids:
                    if span_id not in span_ids:
                        raise ValueError(f"Node references unknown span id: {span_id}")
                    referenced.add(span_id)
        for edge in self.edges:
            for grounding in edge.mentions:
                for span_id in grounding.span_ids:
                    if span_id not in span_ids:
                        raise ValueError(f"Edge references unknown span id: {span_id}")
                    referenced.add(span_id)
        orphaned = span_ids - referenced
        if orphaned:
            raise ValueError(f"Unreferenced spans are not allowed: {sorted(orphaned)}")
        return self

    @classmethod
    def from_canonical(
        cls,
        graph: "LLMGraphExtraction",
        *,
        insertion_method: Optional[str] = None,
    ) -> "FlattenedLLMGraphExtraction":
        span_to_id: dict[tuple, str] = {}
        flat_spans: list[FlattenedSpan] = []
        seq = 0

        def intern_span(span: Span) -> str:
            nonlocal seq
            key = _span_identity_key(span)
            existing = span_to_id.get(key)
            if existing is not None:
                return existing
            seq += 1
            span_id = f"sp:{seq}"
            span_to_id[key] = span_id
            resolved_insertion_method = insertion_method if insertion_method is not None else span.insertion_method
            flat_spans.append(
                FlattenedSpan.from_canonical(
                    span,
                    span_id=span_id,
                    insertion_method=resolved_insertion_method,
                )
            )
            return span_id

        for node in graph.nodes:
            for grounding in node.mentions:
                for span in grounding.spans:
                    intern_span(span)
        for edge in graph.edges:
            for grounding in edge.mentions:
                for span in grounding.spans:
                    intern_span(span)

        return cls(
            spans=flat_spans,
            nodes=[node.to_flattened(span_to_id=span_to_id) for node in graph.nodes],
            edges=[edge.to_flattened(span_to_id=span_to_id) for edge in graph.edges],
        )

    def to_canonical(self, *, insertion_method: Optional[str] = None) -> "LLMGraphExtraction":
        span_by_id: dict[str, Span] = {sp.id: sp.to_canonical(insertion_method=insertion_method) for sp in self.spans}
        return LLMGraphExtraction(
            nodes=[node.to_canonical(span_by_id=span_by_id) for node in self.nodes],
            edges=[edge.to_canonical(span_by_id=span_by_id) for edge in self.edges],
        )


class AssocFlattenedGroundingRow(ModeSlicingMixin, BaseModel):
    include_unmarked_for_modes: ClassVar[set[str]] = {"llm"}

    id: str = Field(..., description="Required temporary grounding id within this extraction payload, e.g. 'gr:1'")


class AssocNodeGroundingLink(ModeSlicingMixin, BaseModel):
    include_unmarked_for_modes: ClassVar[set[str]] = {"llm"}

    node_index: int = Field(..., ge=0, description="0-based node index into root-level `nodes` table")
    grounding_id: str = Field(..., description="Grounding id linked to this node")


class AssocEdgeGroundingLink(ModeSlicingMixin, BaseModel):
    include_unmarked_for_modes: ClassVar[set[str]] = {"llm"}

    edge_index: int = Field(..., ge=0, description="0-based edge index into root-level `edges` table")
    grounding_id: str = Field(..., description="Grounding id linked to this edge")


class AssocGroundingSpanLink(ModeSlicingMixin, BaseModel):
    include_unmarked_for_modes: ClassVar[set[str]] = {"llm"}

    grounding_id: str = Field(..., description="Grounding id linked to a span")
    span_id: str = Field(..., description="Span id linked to the grounding")


class AssocFlattenedLLMNode(LLMMixin, GraphEntityBase):
    include_unmarked_for_modes: ClassVar[set[str]] = {"llm"}

    @classmethod
    def from_canonical(cls, node: "LLMNode") -> "AssocFlattenedLLMNode":
        return cls(
            id=node.id,
            local_id=node.local_id,
            label=node.label,
            type=node.type,
            summary=node.summary,
            domain_id=node.domain_id,
            canonical_entity_id=node.canonical_entity_id,
            properties=node.properties,
        )

    def to_canonical(self, *, mentions: List[Grounding]) -> "LLMNode":
        return LLMNode(
            id=self.id,
            local_id=self.local_id,
            label=self.label,
            type=self.type,
            summary=self.summary,
            domain_id=self.domain_id,
            canonical_entity_id=self.canonical_entity_id,
            properties=self.properties,
            mentions=mentions,
        )


class AssocFlattenedLLMEdge(LLMMixin, EdgeMixin, GraphEntityBase):
    include_unmarked_for_modes: ClassVar[set[str]] = {"llm"}

    @classmethod
    def from_canonical(cls, edge: "LLMEdge") -> "AssocFlattenedLLMEdge":
        return cls(
            id=edge.id,
            local_id=edge.local_id,
            label=edge.label,
            type=edge.type,
            summary=edge.summary,
            domain_id=edge.domain_id,
            canonical_entity_id=edge.canonical_entity_id,
            properties=edge.properties,
            source_ids=edge.source_ids,
            target_ids=edge.target_ids,
            relation=edge.relation,
            source_edge_ids=edge.source_edge_ids,
            target_edge_ids=edge.target_edge_ids,
        )

    def to_canonical(self, *, mentions: List[Grounding]) -> "LLMEdge":
        return LLMEdge(
            id=self.id,
            local_id=self.local_id,
            label=self.label,
            type=self.type,
            summary=self.summary,
            domain_id=self.domain_id,
            canonical_entity_id=self.canonical_entity_id,
            properties=self.properties,
            source_ids=self.source_ids,
            target_ids=self.target_ids,
            relation=self.relation,
            source_edge_ids=self.source_edge_ids,
            target_edge_ids=self.target_edge_ids,
            mentions=mentions,
        )


class AssocFlattenedLLMGraphExtraction(ModeSlicingMixin, BaseModel):
    include_unmarked_for_modes: ClassVar[set[str]] = {"llm"}

    spans: List[FlattenedSpan] = Field(..., description="Root-level spans referenced by grounding-span link table")
    nodes: List[AssocFlattenedLLMNode] = Field(..., description="Flattened nodes without nested mentions")
    edges: List[AssocFlattenedLLMEdge] = Field(..., description="Flattened edges without nested mentions")
    groundings: List[AssocFlattenedGroundingRow] = Field(..., description="Root-level grounding rows")
    node_groundings: List[AssocNodeGroundingLink] = Field(
        ..., description="Link table: node index -> grounding id (ordered mention list for each node)"
    )
    edge_groundings: List[AssocEdgeGroundingLink] = Field(
        ..., description="Link table: edge index -> grounding id (ordered mention list for each edge)"
    )
    grounding_spans: List[AssocGroundingSpanLink] = Field(
        ..., description="Link table: grounding id -> span id (ordered span list for each grounding)"
    )

    @field_validator("spans")
    @classmethod
    def _validate_unique_span_ids(cls, spans: List[FlattenedSpan]):
        seen: set[str] = set()
        for sp in spans:
            if sp.id in seen:
                raise ValueError(f"Duplicate span id found: {sp.id}")
            seen.add(sp.id)
        return spans

    @field_validator("groundings")
    @classmethod
    def _validate_unique_grounding_ids(cls, groundings: List[AssocFlattenedGroundingRow]):
        seen: set[str] = set()
        for g in groundings:
            if g.id in seen:
                raise ValueError(f"Duplicate grounding id found: {g.id}")
            seen.add(g.id)
        return groundings

    @model_validator(mode="after")
    def _validate_references(self):
        span_ids = {sp.id for sp in self.spans}
        grounding_ids = {g.id for g in self.groundings}

        node_ref_counts = {idx: 0 for idx in range(len(self.nodes))}
        edge_ref_counts = {idx: 0 for idx in range(len(self.edges))}
        grounding_entity_ref_counts = {gid: 0 for gid in grounding_ids}
        grounding_span_counts = {gid: 0 for gid in grounding_ids}
        referenced_spans: set[str] = set()

        for link in self.node_groundings:
            if link.node_index not in node_ref_counts:
                raise ValueError(f"Node grounding link has invalid node_index: {link.node_index}")
            if link.grounding_id not in grounding_ids:
                raise ValueError(f"Node grounding link references unknown grounding id: {link.grounding_id}")
            node_ref_counts[link.node_index] += 1
            grounding_entity_ref_counts[link.grounding_id] += 1

        for link in self.edge_groundings:
            if link.edge_index not in edge_ref_counts:
                raise ValueError(f"Edge grounding link has invalid edge_index: {link.edge_index}")
            if link.grounding_id not in grounding_ids:
                raise ValueError(f"Edge grounding link references unknown grounding id: {link.grounding_id}")
            edge_ref_counts[link.edge_index] += 1
            grounding_entity_ref_counts[link.grounding_id] += 1

        for idx, count in node_ref_counts.items():
            if count == 0:
                raise ValueError(f"Node at index {idx} has no grounding links")
        for idx, count in edge_ref_counts.items():
            if count == 0 and len(self.edges) > 0:
                raise ValueError(f"Edge at index {idx} has no grounding links")

        for link in self.grounding_spans:
            if link.grounding_id not in grounding_ids:
                raise ValueError(f"Grounding-span link references unknown grounding id: {link.grounding_id}")
            if link.span_id not in span_ids:
                raise ValueError(f"Grounding-span link references unknown span id: {link.span_id}")
            grounding_span_counts[link.grounding_id] += 1
            referenced_spans.add(link.span_id)

        unreferenced_groundings = [gid for gid, count in grounding_entity_ref_counts.items() if count == 0]
        if unreferenced_groundings:
            raise ValueError(f"Unreferenced groundings are not allowed: {sorted(unreferenced_groundings)}")

        groundings_without_spans = [gid for gid, count in grounding_span_counts.items() if count == 0]
        if groundings_without_spans:
            raise ValueError(f"Groundings without span links are not allowed: {sorted(groundings_without_spans)}")

        orphaned_spans = span_ids - referenced_spans
        if orphaned_spans:
            raise ValueError(f"Unreferenced spans are not allowed: {sorted(orphaned_spans)}")

        return self

    @classmethod
    def from_canonical(
        cls,
        graph: "LLMGraphExtraction",
        *,
        insertion_method: Optional[str] = None,
    ) -> "AssocFlattenedLLMGraphExtraction":
        span_to_id: dict[tuple, str] = {}
        flat_spans: list[FlattenedSpan] = []
        flat_groundings: list[AssocFlattenedGroundingRow] = []
        node_grounding_links: list[AssocNodeGroundingLink] = []
        edge_grounding_links: list[AssocEdgeGroundingLink] = []
        grounding_span_links: list[AssocGroundingSpanLink] = []

        span_seq = 0
        grounding_seq = 0

        def intern_span(span: Span) -> str:
            nonlocal span_seq
            key = _span_identity_key(span)
            existing = span_to_id.get(key)
            if existing is not None:
                return existing
            span_seq += 1
            span_id = f"sp:{span_seq}"
            span_to_id[key] = span_id
            resolved_insertion_method = insertion_method if insertion_method is not None else span.insertion_method
            flat_spans.append(
                FlattenedSpan.from_canonical(
                    span,
                    span_id=span_id,
                    insertion_method=resolved_insertion_method,
                )
            )
            return span_id

        for node_idx, node in enumerate(graph.nodes):
            for grounding in node.mentions:
                grounding_seq += 1
                grounding_id = f"gr:{grounding_seq}"
                flat_groundings.append(AssocFlattenedGroundingRow(id=grounding_id))
                node_grounding_links.append(AssocNodeGroundingLink(node_index=node_idx, grounding_id=grounding_id))
                for span in grounding.spans:
                    grounding_span_links.append(
                        AssocGroundingSpanLink(grounding_id=grounding_id, span_id=intern_span(span))
                    )

        for edge_idx, edge in enumerate(graph.edges):
            for grounding in edge.mentions:
                grounding_seq += 1
                grounding_id = f"gr:{grounding_seq}"
                flat_groundings.append(AssocFlattenedGroundingRow(id=grounding_id))
                edge_grounding_links.append(AssocEdgeGroundingLink(edge_index=edge_idx, grounding_id=grounding_id))
                for span in grounding.spans:
                    grounding_span_links.append(
                        AssocGroundingSpanLink(grounding_id=grounding_id, span_id=intern_span(span))
                    )

        return cls(
            spans=flat_spans,
            nodes=[AssocFlattenedLLMNode.from_canonical(node) for node in graph.nodes],
            edges=[AssocFlattenedLLMEdge.from_canonical(edge) for edge in graph.edges],
            groundings=flat_groundings,
            node_groundings=node_grounding_links,
            edge_groundings=edge_grounding_links,
            grounding_spans=grounding_span_links,
        )

    def to_canonical(self, *, insertion_method: Optional[str] = None) -> "LLMGraphExtraction":
        span_by_id: dict[str, Span] = {sp.id: sp.to_canonical(insertion_method=insertion_method) for sp in self.spans}

        grounding_spans_by_id: dict[str, list[Span]] = {g.id: [] for g in self.groundings}
        for link in self.grounding_spans:
            grounding_spans_by_id[link.grounding_id].append(span_by_id[link.span_id])

        grounding_by_id: dict[str, Grounding] = {
            grounding_id: Grounding(spans=spans)
            for grounding_id, spans in grounding_spans_by_id.items()
        }

        node_mentions_by_index: dict[int, list[Grounding]] = {idx: [] for idx in range(len(self.nodes))}
        edge_mentions_by_index: dict[int, list[Grounding]] = {idx: [] for idx in range(len(self.edges))}

        for link in self.node_groundings:
            node_mentions_by_index[link.node_index].append(grounding_by_id[link.grounding_id])
        for link in self.edge_groundings:
            edge_mentions_by_index[link.edge_index].append(grounding_by_id[link.grounding_id])

        return LLMGraphExtraction(
            nodes=[
                node.to_canonical(mentions=node_mentions_by_index[idx])
                for idx, node in enumerate(self.nodes)
            ],
            edges=[
                edge.to_canonical(mentions=edge_mentions_by_index[idx])
                for idx, edge in enumerate(self.edges)
            ],
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
        _ = info.context or {}
        return data

    @model_validator(mode="after")
    def inject_context_on_children_after(self, info: ValidationInfo):
        _ = info.context or {}
        return self

    def to_flattened(self, *, insertion_method: Optional[str] = None) -> "FlattenedLLMGraphExtraction":
        return FlattenedLLMGraphExtraction.from_canonical(self, insertion_method=insertion_method)

    def to_assoc_flattened(self, *, insertion_method: Optional[str] = None) -> "AssocFlattenedLLMGraphExtraction":
        return AssocFlattenedLLMGraphExtraction.from_canonical(self, insertion_method=insertion_method)

    @classmethod
    def from_normal_llm(
        cls,
        sliced: "Union[LLMGraphExtraction['llm'], dict, BaseModel]",
        insertion_method,
    ) -> "LLMGraphExtraction":
        if isinstance(sliced, BaseModel):
            dumped = sliced.model_dump()
        elif type(sliced) is dict:
            dumped = sliced
        else:
            raise ValueError("Unsupported type for 'sliced'")

        for ne in (dumped.get('nodes') or []) + (dumped.get('edges') or []):
            if 'mentions' not in ne and 'references' in ne and ne.get('references') is not None:
                refs = ne.get('references') or []
                ne['mentions'] = [{"spans": refs}]

            if 'mentions' not in ne and 'groundings' in ne and ne.get('groundings') is not None:
                ne['mentions'] = ne.pop('groundings')

            for g in (ne.get('mentions') or []):
                spans = (g or {}).get('spans') or []
                for s in spans:
                    if isinstance(s, dict):
                        s['insertion_method'] = insertion_method

            if 'references' in ne:
                ne.pop('references', None)

        return cls.model_validate(dumped, context={"insertion_method": insertion_method})

    @classmethod
    def from_flattened_llm(
        cls,
        sliced: "Union[FlattenedLLMGraphExtraction, AssocFlattenedLLMGraphExtraction, dict, BaseModel]",
        insertion_method,
    ) -> "LLMGraphExtraction":
        if isinstance(sliced, AssocFlattenedLLMGraphExtraction):
            return sliced.to_canonical(insertion_method=insertion_method)
        if isinstance(sliced, FlattenedLLMGraphExtraction):
            return sliced.to_canonical(insertion_method=insertion_method)

        if isinstance(sliced, BaseModel):
            dumped = sliced.model_dump()
        elif type(sliced) is dict:
            dumped = sliced
        else:
            raise ValueError("Unsupported type for 'sliced'")

        if cls._looks_like_assoc_flattened_payload(dumped):
            assoc_dumped = cls._normalize_assoc_flattened_payload_dict(dumped)
            assoc = AssocFlattenedLLMGraphExtraction.model_validate(
                assoc_dumped,
                context={"insertion_method": insertion_method},
            )
            return assoc.to_canonical(insertion_method=insertion_method)

        flat_dumped = cls._normalize_flattened_payload_dict(dumped)
        flat = FlattenedLLMGraphExtraction.model_validate(
            flat_dumped,
            context={"insertion_method": insertion_method},
        )
        return flat.to_canonical(insertion_method=insertion_method)

    @staticmethod
    def _normalize_flattened_payload_dict(payload: dict) -> dict:
        normalized = dict(payload)
        for section in ("nodes", "edges"):
            entries = normalized.get(section) or []
            normalized_entries = []
            for entry in entries:
                if isinstance(entry, dict):
                    entry_dict = dict(entry)
                    if "mentions" not in entry_dict and entry_dict.get("groundings") is not None:
                        entry_dict["mentions"] = entry_dict["groundings"]
                    entry_dict.pop("groundings", None)
                    normalized_entries.append(entry_dict)
                else:
                    normalized_entries.append(entry)
            normalized[section] = normalized_entries
        return normalized

    @staticmethod
    def _normalize_assoc_flattened_payload_dict(payload: dict) -> dict:
        normalized = dict(payload)
        alias_map = {
            "grounding_rows": "groundings",
            "node_grounding_links": "node_groundings",
            "edge_grounding_links": "edge_groundings",
            "grounding_span_links": "grounding_spans",
        }
        for alias_key, canonical_key in alias_map.items():
            if canonical_key not in normalized and alias_key in normalized:
                normalized[canonical_key] = normalized[alias_key]
        return normalized

    @staticmethod
    def _looks_like_assoc_flattened_payload(payload: dict) -> bool:
        base_keys = {"spans", "nodes", "edges", "groundings", "grounding_spans"}
        if not base_keys.issubset(payload.keys()):
            return False

        has_node_link_table = ("node_groundings" in payload) or ("node_grounding_links" in payload)
        has_edge_link_table = ("edge_groundings" in payload) or ("edge_grounding_links" in payload)
        return has_node_link_table or has_edge_link_table

    @staticmethod
    def _looks_like_flattened_payload(payload: dict) -> bool:
        if LLMGraphExtraction._looks_like_assoc_flattened_payload(payload):
            return True

        if "spans" not in payload:
            return False

        for entry in (payload.get("nodes") or []) + (payload.get("edges") or []):
            if not isinstance(entry, dict):
                continue
            mentions = entry.get("mentions")
            if mentions is None:
                mentions = entry.get("groundings")
            if not isinstance(mentions, list):
                continue
            for grounding in mentions:
                if isinstance(grounding, dict) and "span_ids" in grounding:
                    return True
        return False

    @classmethod
    def FromLLMSlice(cls, sliced: "Union[LLMGraphExtraction['llm'], FlattenedLLMGraphExtraction, dict, BaseModel]", insertion_method) -> "LLMGraphExtraction":
        if isinstance(sliced, FlattenedLLMGraphExtraction):
            return cls.from_flattened_llm(sliced, insertion_method)

        if type(sliced) is dict:
            if cls._looks_like_flattened_payload(sliced):
                return cls.from_flattened_llm(sliced, insertion_method)

        return cls.from_normal_llm(sliced, insertion_method)


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

@dataclass(frozen=True)
class Chunk:
    doc_id: str
    start_char: int
    end_char: int  # exclusive
    text: str    
# -------------------------
# Document
# -------------------------
from typing import Tuple
class Document(ModeSlicingMixin, BaseModel):
    id: BackendType[str] = Field(..., description="Unique document identifier")
    content: BackendType[DtoType[str | dict]] = Field(..., description="Text content or OCR dict content of the document")
    # "text_chunked" is temp type for view as chunks and validate chunked results
    type: BackendType[DtoType[Literal["text", "ocr_document", "text_chunked"] | str]] = Field(..., description="Type of document, e.g., 'ocr', 'pdf'")
    metadata: BackendType[DtoType[Optional[Dict[str, Any]]]] = Field(None, description="Additional metadata for the document")
    domain_id: BackendType[Optional[str]] = Field(None, description="Optional domain this document belongs to")
    processed: BackendType[bool] = Field(False, description="Whether the document has been processed, source map is produced, set true only you are migrating from server or is directly from trusted ingestor")
    embeddings: BackendType[Optional[Any]] = Field(..., description="embedding for collection")
    source_map: BackendType[Optional[Dict[str, Any]]] = Field(..., description="source_map for chunk boundary, not for LLM ingestion, chunk id to character index range")
    @staticmethod
    def from_text(text: str, **kwarg):
        return Document(content = text, type = "text", metadata = {}, **kwarg)
    @staticmethod
    def from_dummy(text: str="", **kwarg):
        return Document(id="_dummy", content=text, type="text", metadata={}, embeddings=None, source_map=None, **kwarg)
    def validate_text_span(self, span: Span):
        if span.chunk_id is not None:
            raise Exception(f"text span should not have chunk ID, found chunk_id={span.chunk_id}")
        pass
    def validate_text_chunked_span(self, span: Span):
        if self.source_map is None:
            raise Exception("Source map should be available when span is checked")
        span.fix_chunk_start_end_char(self.source_map) # check if chunk id really in span
        pass
    def validate_span(self, span: Span):
        if self.type == "text":
            self.validate_text_span(span)
        elif self.type == "text_chunked":
            self.validate_text_chunked_span(span)
        
        pass
    @classmethod
    def from_ocr(cls, id: str, ocr_content: dict, type:str):
        def prepare_document_for_llm(doc_dict: Dict) -> Tuple[Dict, Dict[str, Dict]]:
            # Simple restructure of input format
            filename = list(doc_dict.keys())[0]
            pages_data = doc_dict[filename]
            source_cluster_map = {}
            for page in pages_data:
                if 'pdf_page_num' not in page or 'OCR_text_clusters' not in page:
                    continue
                page_num = page['pdf_page_num']
                for i, cluster in enumerate(page['OCR_text_clusters']):
                    cluster_id = f"p{page_num}_c{i}"
                    cluster['id'] = cluster_id
                    source_cluster_map[cluster_id] = cluster
                for i, cluster in enumerate(page['non_text_objects']):
                    cluster_id = f"p{page_num}_c{i}"
                    cluster['id'] = cluster_id
                    source_cluster_map[cluster_id] = cluster
            return {"document_filename": filename, "pages": pages_data}, source_cluster_map
        _, source_map = prepare_document_for_llm(ocr_content)
        return Document(id = id, content = ocr_content, type = "ocr_document", metadata = {},
                        embeddings = None, domain_id = None, processed = False, source_map = source_map)
    # helper for workflow  long doc -> chunked doc -> llm -> result span unchunked (chunk doc need not but can persist)
    def to_text_chunked(self):
        # convert simple text doc with long text content to text_chunked
        self.type = "text_chunked"
        self.update_source_map()
    

    def update_source_map(self):
        source_map: Dict[str, Any] = {}
        from splitter import split_doc_deterministic
        chunks = split_doc_deterministic(content = str(self.content), doc_id = self.id)
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            source_map[chunk_id] = chunk
        self.source_map = source_map
    @property
    def chunked_text(self):
        if self.source_map is None:
            self.update_source_map()
        if self.source_map is None:
            raise Exception("source map update failed")
        items = list(self.source_map.items())
        sorted_items = sorted(items, key=lambda x: x[0])
        sorted_items_str = [{"chunk_id": f"chunk_{i}", "chunk": str(self.get_chunk(chunk))} for i, chunk in sorted_items]
        return str(sorted_items_str)
    def get_chunk(self, chunk: Chunk )-> str:        
        return self.content[chunk.start_char: chunk.end_char]
    def __str__(self):
        if self.type == "text":
            return self.content
        elif self.type == "text_chunked":
            return self.chunked_text
        else:
            raise Exception("unsupported type")
    def get_content_by_span(self, span: Span)-> str:
        if self.type == 'ocr':
            return self.get_content_as_ocr_doc_by_span(span)
        elif self.type in ['text',  "text_chunked"]:
            return self.get_content_as_text_doc_by_span(span)
        else:
            raise Exception ("Unknown document type")

    def get_content_as_text_doc_by_span(self, span: Span)-> str:
        if self.type == 'text':
            return self.content[span.start_char:span.end_char]
        elif self.type == 'text_chunked':
            if span.chunk_id is None:
                raise ValueError("span must have chunk id")
            if self.source_map is None:
                raise Exception("source map must not be none when span already obtained")
            source_map = self.source_map
            return self.get_chunk(source_map[span.chunk_id])[span.start_char:span.end_char]
        else:
            raise Exception("only can call method when doc is plain text type")
        
    
    def get_content_as_ocr_doc_by_span(self, span: Span)-> str:
        if self.type != 'ocr':
            raise Exception("only can call method when doc is ocr type")
        return ""
    
    
#========================= OCR DOC

# pre-validation model
class box_2d(BaseModel):
    box_2d: list[int] = Field(description = 'box y min, x min, y max and x max')
    label : str = Field(description = 'text in the box')
    id: int  = Field(description = 'id of the text box in the page, autoincrement from 0')
class NonText_box_2d(ModeSlicingMixin, BaseModel):
    """Recognised meaningful objects other than OCR characters, include image, figures. """
    description: DtoType[str] = Field(description='the description or summary of the non-OCR object')
    box_2d: DtoType[list[int]] = Field(description = 'box y min, x min, y max and x max')
    id: int = Field(description="per page unique number of the cluster, starting from 0")

# post validation model
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
    """id/ cluster number must be all unique, for example, one of the ocr boxes_2d used id='1', the first image box id (cluster numebr) will be '2', the next signature will be '3' """
    OCR_text_clusters: DtoType[list[TextCluster]] = Field(description="the OCR text results. Share cluster number uniqueness with non-OCR objects. ")
    non_text_objects:  DtoType[list[NonTextCluster]] = Field(description="the non-OCR object results. Share cluster number uniqueness with OCR texts. ")
    is_empty_page: DtoType[Optional[bool]] = Field(default = False, description="true if the whole page is empty without recognisable text.")
    printed_page_number: DtoType[Optional[str]] = Field(description='the page number identified from OCR texts, can be in form of roman numerals such as "i", "ii", "iii", "iv"...; ' 
                    'Arabic numeral such as 1, 2, 3... or letter such as "a", "b", "c"...\n'
                    'Sometimes the are surrounded by symbols such as "- 1 -", "- 2 -"'
                    r"Can be null/none if there is no page order assigned and printed and found in the scanned texts. Do not assign page number. Only use page number found.")
    meaningful_ordering : DtoType[list[int]] = Field(description="The correct meaningful ordering of the identified text clusters. Must cover all OCR_text_clusters once and only once. ")
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

# -------------------------
# Workflow traces: run/step/checkpoint (persist in conversation_engine)
# -------------------------

# deserialize form db
# EntitiyTypeToNodeTypeMapping = {
#     "workflow_checkpoint" : WorkflowCheckpointNode,
#     "workflow_step_exec": WorkflowStepExecNode,
#     "workflow_run" : WorkflowRunNode,
#     "workflow_node": WorkflowNode,
#     "node" : Node,
#     "conversation_node": ConversationNode,
#     "conversation_start": ConversationNode,
#     "conversation_turn": ConversationNode,
#     "conversation_summary": ConversationNode,
#     "tool_result" : ConversationNode,
#     "agent_run"   : ConversationNode,
#     "knowledge_reference": ConversationNode,
# }

# EntitiyTypeToEdgeTypeMapping = {
    
#     "workflow_edge" : WorkflowEdge,
#     "edge" : Edge,
#     "conversation_edge": ConversationEdge,
# }
