from __future__ import annotations

from typing import Any


def build_structured_output_runnable(
    model: Any,
    schema: Any,
    *,
    include_raw: bool = True,
    prefer_json_schema: bool = True,
):
    """Build a structured-output runnable with strict-schema-first fallback."""
    attempts: list[dict[str, Any]] = []
    if prefer_json_schema:
        attempts.append({"include_raw": include_raw, "method": "json_schema"})
    attempts.append({"include_raw": include_raw, "method": "function_calling"})
    attempts.append({"include_raw": include_raw})

    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return model.with_structured_output(schema, **kwargs)
        except (TypeError, ValueError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise TypeError("with_structured_output is unavailable on this model")
