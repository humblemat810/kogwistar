from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from kogwistar.engine_core.engine import GraphKnowledgeEngine


HISTORY_NAMESPACE = "bridge_governance_history"
PROJECTION_NAMESPACE = "bridge_governance"
PROJECTION_SCHEMA_VERSION = 1

PREDEFINED_GOVERNANCE_EVENTS: list[dict[str, Any]] = [
    {
        "interaction_id": "interaction-alpha",
        "event_type": "interaction_opened",
        "agent": "router",
        "status": "opened",
        "active_agents": ["router", "policy"],
        "policy_version": "v1",
        "decision": "collect_context",
    },
    {
        "interaction_id": "interaction-alpha",
        "event_type": "policy_reviewed",
        "agent": "policy",
        "status": "under_review",
        "active_agents": ["router", "policy"],
        "policy_version": "v2",
        "decision": "tighten_boundary",
    },
    {
        "interaction_id": "interaction-alpha",
        "event_type": "interaction_closed",
        "agent": "governor",
        "status": "approved",
        "active_agents": ["router", "policy", "governor"],
        "policy_version": "v2",
        "decision": "approved",
    },
    {
        "interaction_id": "interaction-beta",
        "event_type": "interaction_opened",
        "agent": "router",
        "status": "opened",
        "active_agents": ["router", "auditor"],
        "policy_version": "v1",
        "decision": "collect_context",
    },
    {
        "interaction_id": "interaction-beta",
        "event_type": "interaction_closed",
        "agent": "auditor",
        "status": "rejected",
        "active_agents": ["router", "auditor"],
        "policy_version": "v1",
        "decision": "rejected",
    },
]


class _DemoEmbeddingFunction:
    _name = "named-projection-governance-demo-embedding-v1"

    def name(self):
        return self._name

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in input:
            value = str(text or "")
            vectors.append(
                [
                    float((len(value) % 17) + 1),
                    float((sum(ord(ch) for ch in value) % 29) + 1),
                ]
            )
        return vectors


def _build_engine(
    *,
    data_dir: Path,
    backend_factory: Any | None = None,
) -> GraphKnowledgeEngine:
    kwargs: dict[str, Any] = {"embedding_function": _DemoEmbeddingFunction()}
    if backend_factory is not None:
        kwargs["backend_factory"] = backend_factory
    return GraphKnowledgeEngine(
        persist_directory=str(data_dir / "projection_demo"),
        kg_graph_type="conversation",
        **kwargs,
    )


def _sorted_unique(values: list[str]) -> list[str]:
    return sorted({str(value) for value in values if value})


def _fold_bridge_governance_events(
    interaction_id: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "interaction_id": str(interaction_id),
        "event_count": len(events),
        "event_types": [str(item["event_type"]) for item in events],
        "participants": _sorted_unique([str(item.get("agent") or "") for item in events]),
        "active_agents": _sorted_unique(
            [
                str(agent)
                for item in events
                for agent in list(item.get("active_agents") or [])
            ]
        ),
        "policy_versions": _sorted_unique(
            [str(item.get("policy_version") or "") for item in events]
        ),
        "latest_status": str(events[-1].get("status") or "unknown") if events else "unknown",
        "latest_decision": str(events[-1].get("decision") or "unknown") if events else "unknown",
        "last_event_seq": int(events[-1].get("seq") or 0) if events else 0,
    }
    return payload


class BridgeGovernanceProjectionService:
    def __init__(self, engine: GraphKnowledgeEngine) -> None:
        self._engine = engine
        self._meta = engine.meta_sqlite

    def record_predefined_history(self) -> list[dict[str, Any]]:
        recorded: list[dict[str, Any]] = []
        for index, item in enumerate(PREDEFINED_GOVERNANCE_EVENTS, start=1):
            payload = {
                "event_type": str(item["event_type"]),
                "agent": str(item["agent"]),
                "status": str(item["status"]),
                "active_agents": list(item["active_agents"]),
                "policy_version": str(item["policy_version"]),
                "decision": str(item["decision"]),
            }
            seq = self._meta.append_entity_event(
                namespace=HISTORY_NAMESPACE,
                event_id=f"{item['interaction_id']}::{index}",
                entity_kind="bridge_interaction",
                entity_id=str(item["interaction_id"]),
                op="UPSERT",
                payload_json=json.dumps(payload, sort_keys=True, separators=(",", ":")),
            )
            recorded.append(
                {
                    "interaction_id": str(item["interaction_id"]),
                    "seq": int(seq),
                    "payload": payload,
                }
            )
        return recorded

    def get_bridge_governance_projection(
        self, interaction_id: str
    ) -> dict[str, Any] | None:
        return self._meta.get_named_projection(PROJECTION_NAMESPACE, str(interaction_id))

    def list_bridge_governance_projections(self) -> list[dict[str, Any]]:
        return self._meta.list_named_projections(PROJECTION_NAMESPACE)

    def clear_bridge_governance_projection(self, interaction_id: str) -> None:
        self._meta.clear_named_projection(PROJECTION_NAMESPACE, str(interaction_id))

    def clear_bridge_governance_namespace(self) -> None:
        self._meta.clear_projection_namespace(PROJECTION_NAMESPACE)

    def refresh_bridge_governance_projection(
        self, interaction_id: str
    ) -> tuple[dict[str, Any], list[str]]:
        interaction_key = str(interaction_id)
        latest_authoritative_seq = self._meta.get_latest_entity_event_seq(
            namespace=HISTORY_NAMESPACE
        )
        existing = self.get_bridge_governance_projection(interaction_key)
        rebuilding_payload = (
            dict(existing.get("payload") or {})
            if isinstance(existing, dict)
            else {"interaction_id": interaction_key}
        )
        self._meta.replace_named_projection(
            PROJECTION_NAMESPACE,
            interaction_key,
            rebuilding_payload,
            last_authoritative_seq=latest_authoritative_seq,
            last_materialized_seq=int(
                (existing or {}).get("last_materialized_seq") or 0
            ),
            projection_schema_version=PROJECTION_SCHEMA_VERSION,
            materialization_status="rebuilding",
        )
        statuses = ["rebuilding"]

        relevant_events: list[dict[str, Any]] = []
        for seq, _entity_kind, entity_id, _op, payload_json in self._meta.iter_entity_events(
            namespace=HISTORY_NAMESPACE,
            from_seq=1,
        ):
            if str(entity_id) != interaction_key:
                continue
            payload = json.loads(str(payload_json))
            relevant_events.append(
                {
                    "seq": int(seq),
                    "event_type": str(payload.get("event_type") or "unknown"),
                    "agent": str(payload.get("agent") or ""),
                    "status": str(payload.get("status") or "unknown"),
                    "active_agents": list(payload.get("active_agents") or []),
                    "policy_version": str(payload.get("policy_version") or ""),
                    "decision": str(payload.get("decision") or "unknown"),
                }
            )

        materialized_seq = (
            int(relevant_events[-1]["seq"]) if relevant_events else 0
        )
        payload = _fold_bridge_governance_events(interaction_key, relevant_events)
        self._meta.replace_named_projection(
            PROJECTION_NAMESPACE,
            interaction_key,
            payload,
            last_authoritative_seq=latest_authoritative_seq,
            last_materialized_seq=materialized_seq,
            projection_schema_version=PROJECTION_SCHEMA_VERSION,
            materialization_status="ready",
        )
        statuses.append("ready")
        projection = self.get_bridge_governance_projection(interaction_key)
        if projection is None:
            raise RuntimeError("expected projection to exist after refresh")
        return projection, statuses


def run_named_projection_governance_demo(
    *,
    data_dir: str | Path,
    backend_factory: Any | None = None,
    reset_data: bool = True,
) -> dict[str, Any]:
    root = Path(data_dir)
    if reset_data and root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    engine = _build_engine(data_dir=root, backend_factory=backend_factory)
    service = BridgeGovernanceProjectionService(engine)
    recorded_history = service.record_predefined_history()

    alpha_projection, alpha_statuses = service.refresh_bridge_governance_projection(
        "interaction-alpha"
    )
    beta_projection, beta_statuses = service.refresh_bridge_governance_projection(
        "interaction-beta"
    )
    projections_before_clear = service.list_bridge_governance_projections()

    service.clear_bridge_governance_projection("interaction-alpha")
    assert service.get_bridge_governance_projection("interaction-alpha") is None
    rebuilt_alpha_projection, rebuilt_alpha_statuses = (
        service.refresh_bridge_governance_projection("interaction-alpha")
    )
    service.clear_bridge_governance_namespace()
    projections_after_namespace_clear = service.list_bridge_governance_projections()

    return {
        "summary": {
            "history_namespace": HISTORY_NAMESPACE,
            "projection_namespace": PROJECTION_NAMESPACE,
            "history_event_count": len(recorded_history),
            "projection_keys_before_clear": [
                item["key"] for item in projections_before_clear
            ],
            "rebuilt_matches_before_clear": (
                rebuilt_alpha_projection["payload"] == alpha_projection["payload"]
            ),
            "namespace_clear_removed_all": projections_after_namespace_clear == [],
            "data_dir": str(root),
        },
        "details": {
            "recorded_history": recorded_history,
            "interaction_alpha": {
                "projection": alpha_projection,
                "status_transitions": alpha_statuses,
                "rebuilt_projection": rebuilt_alpha_projection,
                "rebuilt_status_transitions": rebuilt_alpha_statuses,
            },
            "interaction_beta": {
                "projection": beta_projection,
                "status_transitions": beta_statuses,
            },
            "projections_before_clear": projections_before_clear,
            "projections_after_namespace_clear": projections_after_namespace_clear,
        },
    }
