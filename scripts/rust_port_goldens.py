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


def golden_payloads(*, include_openapi: bool = True) -> dict[str, Any]:
    payloads = {
        "python-model-schemas.json": _model_schemas(),
        "event-replay.json": _event_replay(),
        "errors.json": _errors(),
        "database-contract.json": _database_contract(),
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
