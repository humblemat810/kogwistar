from __future__ import annotations

"""Named-projection keys for workflow runtime serving surfaces."""

WORKFLOW_RUNTIME_PROJECTION_SCHEMA_VERSION = 1


def workflow_checkpoint_latest_projection_namespace(conversation_id: str) -> str:
    return f"{conversation_id}:workflow_checkpoint_latest"


def workflow_run_status_projection_namespace(conversation_id: str) -> str:
    return f"{conversation_id}:workflow_run_status"
