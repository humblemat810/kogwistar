from dataclasses import asdict, dataclass, field

from kogwistar.engine_core.models import (
    BaseNodeMetadata,
    ContextCost,
    Edge,
    Node,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing import Any, ClassVar, Dict, List, Literal, Self, Tuple

# --- Phase 1: chat-edge intent classification (causality) ---

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
    # Default catch-all for chat edges
}


@dataclass
class MetaFromLastSummary:
    prev_node_char_distance_from_last_summary: int
    prev_node_distance_from_last_summary: int
    # distances are rough estimate, also serve as a stub for injection if accounting
    # calculations are done in the process and the two distances indicates how to inject
    # Do not rely on the two to make strict decisions

    tail_turn_index: int = 0  # works more like a node seq number

    # Back-compat shim: parts of the workflow/test stack treat this like a Pydantic model.
    def model_dump(self, *_args, **_kwargs) -> dict[str, int]:
        return asdict(self)


class ContextSnapshotMetadata(BaseModel):
    entity_type: Literal["context_snapshot"] = "context_snapshot"
    level_from_root: int = Field(0, ge=0)

    # Execution identity
    run_id: str
    run_step_seq: int = Field(..., ge=0)
    attempt_seq: int = Field(0, ge=0)

    # Snapshot content
    stage: str
    model_name: str = ""
    budget_tokens: int = Field(0, ge=0)
    tail_turn_index: int = Field(0, ge=0)

    # Determinism + audit
    used_node_ids: List[str] = Field(default_factory=list)
    rendered_context_hash: str
    cost: ContextCost = Field(default_factory=ContextCost)

    model_config = ConfigDict(extra="allow")

    # ---- Storage helpers for Chroma / flat metadata ----

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """
        Flatten to Chroma-friendly metadata (primitives only).
        Keeps all extra fields too, but flattens `cost`.
        """
        d = self.model_dump(mode="python", exclude={"cost"})
        if not d.get("used_node_ids"):
            d.pop("used_node_ids", None)
        d.update(self.cost.to_flat_metadata(prefix="cost"))
        return d

    @classmethod
    def from_chroma_metadata(cls, meta: Dict[str, Any]) -> "ContextSnapshotMetadata":
        """
        Reconstruct from flat Chroma metadata. Accepts either:
        - flattened cost keys (cost.char_count / cost.token_count), or
        - legacy nested `cost` dict (if any exists)
        """
        data = dict(meta)

        # If `cost` was stored nested for some reason, tolerate it.
        cost_val = data.pop("cost", None)
        if isinstance(cost_val, dict):
            cost = ContextCost(
                char_count=int(cost_val.get("char_count", 0) or 0),
                token_count=(
                    None
                    if cost_val.get("token_count", None) is None
                    else int(cost_val["token_count"])
                ),
            )
        else:
            cost = ContextCost.from_flat_metadata(data, prefix="cost")

        # Strip flattened fields so they don't end up as "extra"
        data.pop("cost.char_count", None)
        data.pop("cost.token_count", None)

        obj = cls(**data, cost=cost)
        return obj

    @model_validator(mode="before")
    @classmethod
    def _accept_flat_cost_on_direct_init(cls, values: Any) -> Any:
        """
        Optional: allows ContextSnapshotMetadata(**meta_from_chroma) directly,
        without calling from_chroma_metadata().
        """
        if not isinstance(values, dict):
            return values
        if "cost" in values:
            return values

        # If flattened cost present, assemble `cost`
        if "cost.char_count" in values or "cost.token_count" in values:
            cost = ContextCost.from_flat_metadata(values, prefix="cost")
            values = dict(values)
            values["cost"] = cost
            # Do NOT remove the keys here; let extra=allow keep them if desired.
            # If you prefer strictness, you can pop them here.
        return values


class EvidencePackDigest(BaseModel):
    """A compact, rehydratable description of an evidence pack.

    Mental model:
    - The *evidence pack* is a concrete JSON payload materialized from KG nodes
      (and their neighborhood) for citation picking.
    - This digest stores the parameters needed to rebuild that pack later.

    Rehydration is best-effort:
    - If the underlying KG changes, re-materialization may differ.
    - When `evidence_pack_hash` is present, callers can detect drift.
    """

    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    depth: str = Field(
        "shallow", description="Materialization depth hint (e.g. shallow/deep)"
    )
    max_chars_per_item: int = Field(0, ge=0)
    max_total_chars: int = Field(0, ge=0)
    evidence_pack_hash: str | None = None

    model_config = ConfigDict(extra="allow")


@dataclass(frozen=True)
class AddTurnResult:
    user_turn_node_id: str
    response_turn_node_id: (
        str | None
    )  # if add system node with this method, no response required but optional
    turn_index: int
    relevant_kg_node_ids: list[str] = field(default_factory=list)
    relevant_kg_edge_ids: list[str] = field(default_factory=list)
    pinned_kg_pointer_node_ids: list[str] = field(default_factory=list)
    pinned_kg_edge_ids: list[str] = field(default_factory=list)
    memory_context_node_id: str | None = None
    memory_context_edge_ids: list[str] = field(default_factory=list)
    prev_turn_meta_summary: MetaFromLastSummary = field(
        default_factory=MetaFromLastSummary
    )


class ConversationNodeMetadata(BaseNodeMetadata):
    """Metadata for conversation nodes.

    **Invariant (Phase 1):** summary-distance accounting is stored on edges (e.g. `next_turn`)
    rather than nodes. Nodes MUST NOT carry:
    - `char_distance_from_last_summary`
    - `turn_distance_from_last_summary`

    This keeps node identity stable and avoids accidental "rewriting the past" via node updates.
    """

    level_from_root: int = Field(..., ge=0)
    entity_type: str = Field("conversation_node")

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _forbid_summary_distance_fields(self) -> Self:
        # These must never appear on nodes. Accounting belongs to edges.
        # extra = self.model_fields_set or {}
        forbidden = {
            "char_distance_from_last_summary",
            "turn_distance_from_last_summary",
            "prev_node_char_distance_from_last_summary",
            "prev_node_distance_from_last_summary",
            "prev_turn_meta_summary",
        }
        present = forbidden.intersection(self.model_fields_set)
        if present:
            raise ValueError(
                f"ConversationNodeMetadata must not contain summary-distance fields: {sorted(present)}"
            )
        return self


class ConversationEdgeMetadata(BaseNodeMetadata):
    """Metadata for conversation edges.

    Phase 1 introduces `causal_type` to make edge intent explicit and enforceable.
    When absent, it can be inferred from `relation` via a mapping.
    """

    causal_type: (
        None
        | Literal[
            "chain",
            "dependency",
            "reuse",
            "annotation",
            "summary",
            "reference",
            "internal",
        ]
    ) = None

    # Accounting currently lives on edges (canonical counters will be normalized in Phase 2).
    char_distance_from_last_summary: None | int = None
    turn_distance_from_last_summary: None | int = None

    model_config = ConfigDict(extra="allow")

ConversationRole = Literal["user", "assistant", "system", "tool"]
class ConversationRoleMixin(BaseModel):
    """Mixin to handle conversation roles and context"""

    role: None | ConversationRole = Field(
        None, description="Role in conversation"
    )
    turn_index: None | int = Field(None, description="Sequential turn index")
    conversation_id: None | str = Field(None, description="Conversation thread ID")
    user_id: None | str = Field(
        None, description="User ID (cross-conversation memory scope)"
    )

    @model_validator(mode="before")
    @classmethod
    def sync_conversation_metadata(cls, data: Any, info: ValidationInfo) -> Any:
        if isinstance(data, dict):
            metadata = data.get("metadata", {}) or {}
            for field in ["role", "turn_index", "conversation_id", "user_id"]:
                if field not in data and field in metadata:
                    data[field] = metadata[field]
        return data

    @model_validator(mode="after")
    def push_conversation_metadata(self) -> "ConversationRoleMixin":
        if not hasattr(self, "metadata"):
            return self

        if self.metadata is None:
            self.metadata = {}

        for field in ["role", "turn_index", "conversation_id", "user_id"]:
            val = getattr(self, field, None)
            if val is not None:
                self.metadata[field] = val
        return self


class ConversationAIResponse(BaseModel):
    """Standard response model for AI conversation responses."""

    text: str = Field(default="", description="Assistant text response to the user.")
    llm_decision_need_summary: bool = Field(
        default=False, description="If True, request summarization this turn."
    )

    used_kg_node_ids: List[str] = Field(default_factory=list)
    used_memory_node_ids: List[str] = Field(default_factory=list)
    projected_conversation_node_ids: List[str] = Field(default_factory=list)
    projected_conversation_edge_ids: List[str] = Field(default_factory=list)
    run_trace_node_id: None | str = None
    response_node_id: str | None = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class ConversationEdge(Edge):
    """
    Specialized edge for conversation links, extending the base **knowledge graph**.

    Used for `next_turn` flow, `references` to knowledge, and `summarizes` relationships.
    Inherits provenance features from `Edge`.
    """

    metadata: dict  # ConversationNodeMetadata
    id_kind: ClassVar[str] = "conversation.edge"
    # id_policy: ClassVar[Literal["event", "canonical"]] = "canonical"
    # id_kind: ClassVar[str] = "model"  # override per subclass if you want stable separation

    def identity_key(self) -> Tuple[str, ...]:
        """
        Subclasses with id_policy="canonical" MUST override this.
        Should return stable, minimal identity parts.
        """
        return (
            self.summary,
            str(self.doc_id),
            str(self.source_ids),
            str(self.target_ids),
            str(self.source_edge_ids),
            str(self.target_edge_ids),
            str(self.metadata.get("entity_type")),
        )

    @field_validator("metadata")
    def check_fields(cls, v):
        # convertible to ConversationNodeMetadata but never materialize the conversion. just a checker model
        try:
            ConversationEdgeMetadata.model_validate(v)
        except Exception as _e:
            raise
        return v

    def get_extra_update(self) -> dict:
        try:
            updates = {
                "in_conversation_chain": self.metadata.get("in_conversation_chain")
            }
        except Exception as _e:
            raise
        return updates


class ConversationNode(ConversationRoleMixin, Node):
    """
    Specialized node for conversation elements, extending the base **knowledge graph**.

    This subclass represents conversation turns, summaries, and system messages
    as first-class citizens in the graph. It inherits the provenance capabilities of `Node`
    but adds conversation-specific metadata (role, turn_index, etc.).
    """

    # id_policy: ClassVar[Literal["event", "canonical"]] = "canonical"
    # id_kind: ClassVar[str] = "model"  # override per subclass if you want stable separation
    id_kind: ClassVar[str] = "conversation.node"

    def identity_key(self) -> Tuple[str, ...]:
        # from conversation_orchestrator import get_id_for_conversation_turn
        """
        Subclasses with id_policy="canonical" MUST override this.
        Should return stable, minimal identity parts.
        """
        return (
            self.id_kind,
            json.dumps(self.user_id),
            json.dumps(self.conversation_id),
            self.summary,
            json.dumps(self.turn_index),
            json.dumps(self.role),
            json.dumps(self.metadata.get("entity_type")),
            json.dumps(self.metadata.get("in_conversation_chain")),
        )
        # return self.summary, str(self.doc_id), str(self.user_id), str(self.conversation_id), str(self.metadata.get("entity_type"))

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

    metadata: dict  # ConversationNodeMetadata

    @field_validator("metadata")
    def check_fields(cls, v):
        # convertible to ConversationNodeMetadata but never materialize the conversion. just a checker model
        try:
            ConversationNodeMetadata.model_validate(v)
        except Exception as _e:
            raise
        return v

    def get_incoming_turn_edge(self, engine) -> "ConversationEdge | None":
        from kogwistar.engine_core.engine import GraphKnowledgeEngine

        engine2: GraphKnowledgeEngine = engine
        edges = engine2.query_edges(
            where={"relation": "next_turn", "target_id": self.id}
        )
        assert len(edges) <= 1
        return edges[0] if edges else None

    def get_extra_update(self) -> dict:
        try:
            updates = {
                "in_conversation_chain": self.metadata.get("in_conversation_chain")
            }
        except Exception as _e:
            raise
        return updates


@dataclass
class RetrievalResult:
    nodes: List[Node]
    edges: List[Edge]


@dataclass
class BaseToolResult:
    node_id_entry: (
        str | None
    )  # if a tool has created 1 or network of connected node, an entry point of reference


@dataclass(kw_only=True)
class MemoryRetrievalResult(BaseToolResult):
    # Cross-conversation memory candidates (by user_id)
    candidate: RetrievalResult
    selected: None | RetrievalResult
    reasoning: str

    # Derived artifacts
    memory_context_text: None | str
    seed_kg_node_ids: List[str]


@dataclass(kw_only=True)
class MemoryPinResult(BaseToolResult):
    memory_context_node: ConversationNode
    pinned_edges: List[ConversationEdge]


class FilteringResult(BaseModel):
    node_ids: list[str] = Field(description="list of relevant node ids")
    edge_ids: list[str] = Field(description="list of relevant edge ids")


class FilteringResponse(BaseModel):
    reasoning: str = Field(description="workspace for reasoning relevance filtering")
    relevant_ids: FilteringResult = Field(
        ..., description="a list of relevant node and edge ids"
    )


@dataclass(kw_only=True)
class KnowledgeRetrievalResult(BaseToolResult):
    candidate: RetrievalResult
    selected: FilteringResult | None
    reasoning: str

    def get_filtered_candidate(self):
        if self.selected:
            set_node_ids = set(self.selected.node_ids)
            set_edge_ids = set(self.selected.node_ids)
            return RetrievalResult(
                nodes=[i for i in self.candidate.nodes if i in set_node_ids],
                edges=[i for i in self.candidate.edges if i in set_edge_ids],
            )

        else:
            raise Exception(
                "selected field cannot be None when calling get_filtered_candidate"
            )
