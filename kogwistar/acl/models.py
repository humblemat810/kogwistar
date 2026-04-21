from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..engine_core.models import Edge, Node


ACLGrainLiteral = Literal["document", "grounding", "span", "node", "edge", "artifact"]
ACLEdgeTypeLiteral = Literal["acl_supersedes", "acl_targets_truth", "acl_covers_usage"]


class ACLNodeMetadata(BaseModel):
    entity_type: Literal["acl_record"] = "acl_record"
    acl_truth_graph: str
    acl_target_entity_id: str
    acl_target_grain: ACLGrainLiteral
    acl_target_item_id: str = ""
    acl_version: int = Field(..., ge=1)
    acl_mode: Literal["private", "shared", "scope", "group", "public"]
    created_by: str | None = None
    owner_id: str | None = None
    security_scope: str | None = None
    tombstoned: bool = False
    supersedes_version: int | None = None
    level_from_root: int = 0

    model_config = ConfigDict(extra="allow")


class ACLEdgeMetadata(BaseModel):
    entity_type: ACLEdgeTypeLiteral
    acl_truth_graph: str
    acl_target_entity_id: str
    acl_target_grain: ACLGrainLiteral
    acl_target_item_id: str = ""
    level_from_root: int = 0

    model_config = ConfigDict(extra="allow")


class ACLNode(Node):
    metadata: dict
    id_kind: ClassVar[str] = "acl.node"

    @field_validator("metadata")
    @classmethod
    def check_acl_node_metadata(cls, value: dict):
        return ACLNodeMetadata.model_validate(value).model_dump()


class ACLEdge(Edge):
    metadata: dict
    id_kind: ClassVar[str] = "acl.edge"

    @field_validator("metadata")
    @classmethod
    def check_acl_edge_metadata(cls, value: dict):
        return ACLEdgeMetadata.model_validate(value).model_dump()
