from typing import Any, ClassVar, Literal, Optional, TypeAlias, TypedDict, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..engine_core.models import Edge, Node, Span


class WorkflowNodeMetadata(BaseModel):
    """
    Workflow *design* node metadata (stored in workflow_engine).

    Required keys:
      - entity_type="workflow_node"
      - workflow_id
      - wf_op (unless wf_terminal=True)

    Optional:
      - wf_version: used for replay compatibility + future cache invalidation
      - wf_fanout: allow multiple outgoing edges to fire in parallel
      - wf_start: marks the start node in this workflow_id
      - wf_terminal: marks terminal node
    """
    entity_type: str = Field("workflow_node", description='Must be "workflow_node"')
    workflow_id: str
    wf_op: str = "noop"
    wf_version: str = "v1"
    wf_start: bool = False
    wf_terminal: bool = False
    wf_fanout: bool = False
    wf_join: bool = False  # barrier/join node

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate(self) -> "WorkflowNodeMetadata":
        if self.entity_type != "workflow_node":
            raise ValueError("WorkflowNodeMetadata.entity_type must be 'workflow_node'")
        if not self.wf_terminal and (self.wf_op is None or str(self.wf_op).strip() == ""):
            raise ValueError("wf_op is required unless wf_terminal=True")
        return self


class WorkflowEdgeMetadata(BaseModel):
    """
    Workflow *design* edge metadata (stored in workflow_engine).

    Keys:
      - entity_type="workflow_edge"
      - workflow_id
      - wf_predicate: symbolic predicate name (None means unconditional)
      - wf_priority: lower first
      - wf_is_default: used if no predicate matches
      - wf_multiplicity: "one"|"many"  (allows fanout)
    """
    entity_type: str = Field("workflow_edge", description='Must be "workflow_edge"')
    workflow_id: str
    wf_predicate: None | str = None  # llm explanation: wf_predicate: “This path is meant for a specific condition.”
    wf_priority: int = 100
    wf_is_default: bool = False         # llm explanation: wf_is_default = what to do if the decision returns nothing
    wf_multiplicity: Literal["one", "many"] = "one"

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate(self) -> "WorkflowEdgeMetadata":
        if self.entity_type != "workflow_edge":
            raise ValueError("WorkflowEdgeMetadata.entity_type must be 'workflow_edge'")
        if self.wf_priority < 0:
            raise ValueError("wf_priority must be >= 0")
        return self


class WorkflowNode(Node):
    """
    Typed wrapper for **workflow design** nodes, extending the base **knowledge graph**.

    Workflow steps are stored as nodes in the graph. This allows the workflow definition
    itself to be managed, queried, and verified using the same engine as the knowledge data.
    """
    metadata: dict
    id_kind: ClassVar[str] = "workflow.node"
    @field_validator("metadata")
    def check_metadataWFN(cls, v):
        v2= WorkflowNodeMetadata.model_validate(v).model_dump()
        v.update(v2)
        return v

    @property
    def op(self):
        return self.metadata.get('wf_op') or "noop"
    @property
    def terminal(self):
        return self.metadata.get('wf_terminal') or False
    @property
    def start(self):
        return self.metadata.get('wf_start') or False
    @property
    def fanout(self):
        return self.metadata.get('wf_fanout') or False


class WorkflowEdge(Edge):
    """
    Typed wrapper for **workflow transitions**, extending the base **knowledge graph**.

    Represents transitions between workflow steps (nodes). Includes metadata for
    predicates, priority, and branching logic.
    """
    metadata: dict
    id_kind: ClassVar[str] = "workflow.edge"
    @field_validator("metadata")
    def check_workflow_edge_metadata(cls, v):
        v = WorkflowEdgeMetadata.model_validate(v).model_dump()
        return v
    @property
    def predicate(self):
        return self.metadata.get('wf_predicate')
    @property
    def multiplicity(self):
        return self.metadata.get('wf_multiplicity')
    @property
    def is_default(self):
        return self.metadata.get('wf_is_default')
    @property
    def priority(self):
        return int(self.metadata.get('wf_priority'))


# class PrevTurnMetaSummaryDict(TypedDict):
    # prev_node_char_distance_from_last_summary: int
    # prev_node_distance_from_last_summary: int
    # tail_turn_index: int


# class SummaryStateDict(TypedDict):
#     should_summarize: bool
#     did_summarize: bool
#     summary_node_id: Optional[str]


class WorkflowState(TypedDict):
    conversation_id: str
    user_id: str
    turn_node_id: str
    turn_index: int
    role: str
    user_text: str
    mem_id: str
    self_span: Span
    embedding: Any
    memory: Optional[Any]
    memory_raw: Optional[Any]
    kg: Optional[Any]
    memory_pin: Optional[Any]
    kg_pin: Optional[Any]
    answer: Optional[Any]
    # summary: SummaryStateDict
    # prev_turn_meta_summary: PrevTurnMetaSummaryDict
    _deps: dict[str, Any]
    _rt_join: dict[str, Any]

StateAppendUpdate = tuple[Literal['u'], Any]
StateOverwriteUpdate = tuple[Literal['a'], Any]

StateUpdate = Union[StateAppendUpdate , StateOverwriteUpdate]


class RunFailure(BaseModel):
    conversation_node_id: Optional[str]
    state_update: list[StateUpdate] # can still update, append an error message
    update: dict[str, Any] | None = None
    errors: list[str]
    next_step_names: list[str] = []  # empty will by default fan out all
    status: Literal["failure"] = "failure"


class RunSuspended(BaseModel):
    conversation_node_id: Optional[str] = None
    state_update: list[StateUpdate] = Field(default_factory=list)
    update: dict[str, Any] | None = None
    next_step_names: list[str] = Field(default_factory=list)
    status: Literal["suspended"] = "suspended"
    resume_payload: dict[str, Any] = Field(default_factory=dict)


class RunSuccess(BaseModel):
    conversation_node_id: str|None  # node id of the 'entry point' of the node cluster created in a resolver step, 
    #step can create multiple node edges but at least should expose a node to connect to the main node net
    state_update: list[StateUpdate]
    # Optional native update dict (schema-driven). This does NOT replace state_update.
    # When present, WorkflowRuntime.run() applies it using state_schema and then
    # falls back unknown keys into DSL ('u') overwrite semantics.
    update: dict[str, Any] | None = None
    next_step_names: list[str] = []  # empty will by default fan out all
    status: Literal["success"] = "success"

StepRunResult: TypeAlias = RunSuccess | RunFailure | RunSuspended


class WorkflowRunMetadata(BaseModel):
    entity_type: str = Field("workflow_run", description='Must be "workflow_run"')
    workflow_id: str
    workflow_version: str = "v1"
    run_id: str
    conversation_id: str
    turn_node_id: str
    status: str = "running"

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate(self) -> "WorkflowRunMetadata":
        if self.entity_type != "workflow_run":
            raise ValueError("WorkflowRunMetadata.entity_type must be 'workflow_run'")
        return self


class WorkflowStepExecMetadata(BaseModel):
    entity_type: str = Field("workflow_step_exec", description='Must be "workflow_step_exec"')
    run_id: str
    workflow_id: str
    workflow_node_id: str
    step_seq: int
    op: str
    status: str = "ok"
    duration_ms: int = 0
    result_json: str

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate(self) -> "WorkflowStepExecMetadata":
        if self.entity_type != "workflow_step_exec":
            raise ValueError("WorkflowStepExecMetadata.entity_type must be 'workflow_step_exec'")
        if self.step_seq < 0:
            raise ValueError("step_seq must be >= 0")
        return self


class WorkflowCheckpointMetadata(BaseModel):
    entity_type: str = Field("workflow_checkpoint", description='Must be "workflow_checkpoint"')
    run_id: str
    workflow_id: str
    step_seq: int
    state_json: str

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate(self) -> "WorkflowCheckpointMetadata":
        if self.entity_type != "workflow_checkpoint":
            raise ValueError("WorkflowCheckpointMetadata.entity_type must be 'workflow_checkpoint'")
        if self.step_seq < 0:
            raise ValueError("step_seq must be >= 0")
        return self


class WorkflowCompletedMetadata(BaseModel):
    entity_type: str = Field("workflow_completed", description='Must be "workflow_completed"')
    workflow_id: str
    run_id: str
    conversation_id: str
    accepted_step_seq: int

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate(self) -> "WorkflowCompletedMetadata":
        if self.entity_type != "workflow_completed":
            raise ValueError("WorkflowCompletedMetadata.entity_type must be 'workflow_completed'")
        return self


class WorkflowCancelledMetadata(BaseModel):
    entity_type: str = Field("workflow_cancelled", description='Must be "workflow_cancelled"')
    workflow_id: str
    run_id: str
    conversation_id: str
    accepted_step_seq: int

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate(self) -> "WorkflowCancelledMetadata":
        if self.entity_type != "workflow_cancelled":
            raise ValueError("WorkflowCancelledMetadata.entity_type must be 'workflow_cancelled'")
        return self


class WorkflowRuntimeEdgeMetadata(BaseModel):
    relation: str
    conversation_id: str | None = None
    run_id: str | None = None
    entity_type: str = "conversation_edge"

    model_config = ConfigDict(extra="allow")


class WorkflowRunNode(Node):
    metadata: dict

    @field_validator("metadata")
    def check_metadata(cls, v):
        WorkflowRunMetadata.model_validate(v)
        return v


class WorkflowStepExecNode(Node):
    metadata: dict

    @field_validator("metadata")
    def check_metadata(cls, v):
        WorkflowStepExecMetadata.model_validate(v)
        return v


class WorkflowCheckpointNode(Node):
    metadata: dict

    @field_validator("metadata")
    def check_metadata(cls, v):
        WorkflowCheckpointMetadata.model_validate(v)
        return v


class WorkflowCompletedNode(Node):
    metadata: dict

    @field_validator("metadata")
    def check_metadata(cls, v):
        WorkflowCompletedMetadata.model_validate(v)
        return v


class WorkflowCancelledNode(Node):
    metadata: dict

    @field_validator("metadata")
    def check_metadata(cls, v):
        WorkflowCancelledMetadata.model_validate(v)
        return v


class WorkflowRuntimeEdge(Edge):
    metadata: dict
    id_kind: ClassVar[str] = "workflow.runtime.edge"

    @field_validator("metadata")
    def check_metadata(cls, v):
        WorkflowRuntimeEdgeMetadata.model_validate(v)
        return v
