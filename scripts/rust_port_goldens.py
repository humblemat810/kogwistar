from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from pydantic_extension.model_slicing import use_mode

from kogwistar.engine_core.models import Document, Edge, Grounding, Node, Span
from kogwistar.runtime.models import (
    RunFailure,
    RunSuccess,
    RunSuspended,
    StepRunResult,
    WorkflowEdge,
    WorkflowNode,
)


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_ROOT = ROOT / "contracts" / "golden"

MODEL_TYPES = {
    "Node": Node,
    "Edge": Edge,
    "Document": Document,
    "Grounding": Grounding,
    "Span": Span,
    "WorkflowNode": WorkflowNode,
    "WorkflowEdge": WorkflowEdge,
    "StepRunResult": StepRunResult,
    "RunSuccess": RunSuccess,
    "RunFailure": RunFailure,
    "RunSuspended": RunSuspended,
}


def _model_schemas() -> dict[str, Any]:
    schemas: dict[str, Any] = {
        "default": {
            name: (
                model.model_json_schema()
                if hasattr(model, "model_json_schema")
                else TypeAdapter(model).json_schema()
            )
            for name, model in sorted(MODEL_TYPES.items())
        }
    }
    for mode in ("backend", "dto", "frontend", "llm", "llm_in"):
        context = use_mode(mode)
        with context:
            schemas[mode] = {
                name: model.model_json_schema()
                for name, model in sorted(MODEL_TYPES.items())
                if hasattr(model, "modes") and mode in model.modes()
            }
    return schemas


def _openapi() -> dict[str, Any]:
    from kogwistar.server_mcp_with_admin import app

    app.openapi_schema = None
    return app.openapi()


def _event_replay() -> dict[str, Any]:
    events = [
        {
            "type": "entity.upsert",
            "event_seq": 1,
            "version": 1,
            "entity": {
                "id": "node:alpha",
                "label": "Alpha",
                "type": "entity",
                "summary": "initial",
                "metadata": {"workspace_id": "ws:golden", "graph_space": "base_kg"},
                "source_ids": [],
                "target_ids": ["node:beta"],
            },
        },
        {
            "type": "entity.upsert",
            "event_seq": 2,
            "version": 2,
            "entity": {
                "id": "node:alpha",
                "label": "Alpha",
                "type": "entity",
                "summary": "replacement",
                "metadata": {"workspace_id": "ws:golden", "graph_space": "base_kg"},
                "source_ids": [],
                "target_ids": [],
            },
        },
        {
            "type": "entity.tombstone",
            "event_seq": 3,
            "version": 3,
            "entity": {"id": "node:alpha", "type": "entity"},
        },
    ]
    return {
        "namespace": "ws:golden:base_kg",
        "events": events,
        "accepted_tombstone_aliases": ["entity.delete", "entity.remove"],
        "expected_replay": {
            "cursor": 3,
            "active_entities": [],
            "tombstoned_entities": ["node:alpha"],
            "replacement_is_observable_at_sequence": 2,
        },
    }


def _errors() -> dict[str, Any]:
    return {
        "taxonomy_version": 1,
        "errors": [
            {"code": "invalid_json", "python_exception": "ValueError", "http_status": 400},
            {
                "code": "evidence_digest_must_be_object",
                "python_exception": "ValueError",
                "http_status": 400,
            },
            {
                "code": "rust_extension_unavailable",
                "python_exception": "RustExtensionUnavailableError",
                "http_status": 503,
            },
            {
                "code": "rust_parity_mismatch",
                "python_exception": "RustParityError",
                "http_status": 500,
            },
            {
                "code": "budget_exhausted",
                "python_exception": "BudgetExhaustedError",
                "http_status": 409,
            },
            {
                "code": "projection_conflict",
                "python_exception": "ProjectionConflictError",
                "http_status": 409,
            },
            {
                "code": "durable_queue_unavailable",
                "python_exception": "DurableQueueUnavailableError",
                "http_status": 503,
            },
            {
                "code": "retry_exhausted",
                "python_exception": "RetryExhaustedError",
                "http_status": 409,
            },
            {
                "code": "KOGWISTAR_CONTRACT_SHORT_ID_INVALID",
                "python_exception": "ValueError",
                "http_status": 400,
            },
            {
                "code": "KOGWISTAR_CONTRACT_STATE_UPDATE_CONFLICT",
                "python_exception": "Exception",
                "http_status": 400,
            },
            {
                "code": "KOGWISTAR_CONTRACT_EVENT_TYPE_UNSUPPORTED",
                "python_exception": "ValueError",
                "http_status": 400,
            },
        ],
    }


def _database_contract() -> dict[str, Any]:
    metadata_tables = [
        "global_seq",
        "user_seq",
        "index_jobs",
        "projected_lane_messages",
        "index_applied_state",
        "namespace_seq",
        "entity_events",
        "replay_cursors",
        "named_projections",
        "workflow_design_snapshots",
        "workflow_design_version_deltas",
        "server_runs",
        "server_run_events",
    ]
    return {
        "contract_version": 1,
        "sqlite": {"metadata_tables": metadata_tables},
        "postgresql": {
            "metadata_tables": metadata_tables,
            "pgvector_same_transaction": True,
        },
        "invariants": [
            "namespace-local-monotonic-sequence",
            "event-id-idempotency",
            "compare-and-swap-projections",
            "lease-expiry-reclaim",
            "forward-only-backward-readable-migrations",
            "no-automatic-startup-data-mutation",
        ],
    }


def _runtime_wire_contract() -> dict[str, Any]:
    success = RunSuccess(
        conversation_node_id="node-result",
        state_update=[("u", {"answer": 42}), ("a", {"items": "one"})],
        update=None,
        _route_next=["go"],
    )
    effect = {
        "contract_version": 1,
        "effect_id": "effect-1",
        "status": "success",
        "state_update": [list(item) for item in success.state_update],
        "update": success.update,
        "state_schema": {"items": "a"},
        "successors": [{"node_id": "done", "join_mask": 0}],
        "route_next": list(success.next_step_names),
        "result": {"workflow_status": "succeeded", "final_state": {"answer": 42}},
        "errors": [],
        "wait_reason": None,
        "resume_payload": None,
        "usage": {"input_tokens": 3, "output_tokens": 2},
        "trace_events": [{"type": "step_completed"}],
    }
    handoff = {
        "message_id": "lane-1",
        "claimed_by": "worker-1",
        "run_id": "run-1",
        "step_id": "entry",
        "correlation_id": "run-1",
    }
    transition = {
        "contract_version": 1,
        "transition_id": "transition-1",
        "expected_event_seq": 7,
        "kind": "recorded_step_success",
        "run_id": "run-1",
        "workflow_id": "wf-1",
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "user_turn_node_id": None,
        "step_seq": 1,
        "node_id": "entry",
        "token_id": "token-1",
        "parent_token_id": None,
        "initial_state": None,
        "state_update": [["u", {"answer": 42}]],
        "update": None,
        "state_schema": {},
        "frontier": {
            "pending": [],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {},
        },
        "result": {"workflow_status": "succeeded"},
        "wait_reason": None,
        "resume_payload": None,
        "errors": [],
    }
    claimed_work = {
        "message_id": "lane-1",
        "claimed_by": "worker-1",
        "run_id": "run-1",
        "step_id": "entry",
        "correlation_id": "run-1",
        "payload": {
            "contract_version": 1,
            "kind": "workflow.step.execute",
            "run_id": "run-1",
            "workflow_id": "wf-1",
            "conversation_id": "conv-1",
            "turn_node_id": None,
            "node_id": "entry",
            "op": "start",
            "step_seq": 0,
            "token_id": "token-1",
            "parent_token_id": None,
            "state": {"seed": 1},
            "runtime_routes": [],
        },
        "expected_event_seq": 7,
        "lease_until": 1784430000,
    }
    step_execute = {
        "contract_version": 1,
        "kind": "workflow.step.execute",
        "run_id": "run-1",
        "workflow_id": "wf-1",
        "conversation_id": "conv-1",
        "turn_node_id": "turn-1",
        "node_id": "done",
        "op": "finish",
        "runtime_routes": [],
        "join_mask": 0,
        "token_id": "token-1",
        "parent_token_id": None,
        "step_seq": 1,
        "expected_event_seq": 7,
        "state": {"seed": 1},
        "resume_effect": effect,
    }
    return {
        "contract_version": 1,
        "submit": {
            "run_id": "run-1",
            "workflow_id": "wf-1",
            "conversation_id": "conv-1",
            "turn_node_id": None,
            "user_id": "user-1",
            "initial_state": {"seed": 1},
            "priority_class": "foreground",
            "token_budget": 100,
            "time_budget_ms": 5000,
            "runtime_kind": "sync",
            "join_node_ids": ["join"],
            "start_join_mask": 1,
            "runtime_routes": [
                {
                    "edge_id": "edge-1",
                    "source_node_id": "entry",
                    "target_node_id": "done",
                    "aliases": ["go", "done"],
                    "join_mask": 0,
                    "predicate": None,
                    "multiplicity": "one",
                    "is_default": False,
                    "priority": 7,
                    "source_fanout": False,
                }
            ],
            "start_node_id": "entry",
            "node_ops": {"entry": "start", "done": "finish"},
        },
        "claim": {
            "claimed_by": "worker-1",
            "limit": 2,
            "lease_seconds": 60,
            "run_id": "run-1",
        },
        "result_effect": {
            "handoff": handoff,
            "transition": None,
            "effect": effect,
        },
        "result_transition": {
            "handoff": handoff,
            "transition": transition,
            "effect": None,
        },
        "claimed_work_sqlite": claimed_work,
        "claimed_work_postgres": {
            **claimed_work,
            "lease_until": "2026-07-19T03:00:00Z",
        },
        "step_execute": step_execute,
        "resume": {
            "suspended_node_id": "entry",
            "suspended_token_id": "token-1",
            "client_result": effect,
            "workflow_id": "wf-1",
            "conversation_id": "conv-1",
            "turn_node_id": "turn-1",
        },
    }


def golden_payloads(*, include_openapi: bool = True) -> dict[str, Any]:
    payloads = {
        "python-model-schemas.json": _model_schemas(),
        "event-replay.json": _event_replay(),
        "errors.json": _errors(),
        "database-contract.json": _database_contract(),
        "adr015-runtime-wire-v1.json": _runtime_wire_contract(),
    }
    if include_openapi:
        payloads["openapi.json"] = _openapi()
    return payloads


def _encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ADR-015 golden contracts.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--skip-openapi", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    failures: list[str] = []
    GOLDEN_ROOT.mkdir(parents=True, exist_ok=True)
    for name, payload in golden_payloads(include_openapi=not args.skip_openapi).items():
        path = GOLDEN_ROOT / name
        encoded = _encoded(payload)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != encoded:
                failures.append(name)
        else:
            path.write_text(encoded, encoding="utf-8")
    if failures:
        print("golden contract drift: " + ", ".join(failures))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
