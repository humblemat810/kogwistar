"""Typed decoding for persisted workflow checkpoint state."""

from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

from kogwistar.json_types import JsonObject

from .models import WorkflowState

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def _json_int(value: object, *, default: int) -> int:
    """Decode an integer-like scalar from persisted JSON metadata."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else default
    if isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def decode_checkpoint_state(value: object, *, run_id: str) -> WorkflowState:
    """Decode persisted state and fail closed unless it is a JSON object."""
    try:
        if isinstance(value, str):
            decoded = _JSON_OBJECT_ADAPTER.validate_json(value)
        else:
            decoded = _JSON_OBJECT_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise ValueError(
            f"Cannot resume workflow run {run_id}: checkpoint state is not a JSON object."
        ) from exc

    # Workflow state is intentionally open to application-defined values at
    # runtime; checkpoint decoding has already constrained persisted values
    # to JSON-compatible data.
    state: WorkflowState = {key: item for key, item in decoded.items()}
    return state
