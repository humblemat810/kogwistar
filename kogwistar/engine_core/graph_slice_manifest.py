"""Export pydantic-extension field visibility as a Rust-consumable manifest.

The extension remains authoritative for Python model slicing.  This adapter only
serializes its resolved field policy; it does not replace ``model_dump`` and it
never changes the model or canonical storage payload.
"""

from __future__ import annotations

from typing import Annotated, Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel
from pydantic_extension.model_slicing.mixin import ExcludeMode, ModeSlicingMixin

SLICE_MODES = ("backend", "frontend", "dto", "llm", "llm_in")


def _metadata(annotation: Any) -> tuple[Any, ...]:
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        return (*_metadata(args[0]), *args[1:])
    return ()


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    if get_origin(annotation) is Annotated:
        return _nested_model(get_args(annotation)[0])
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if isinstance(annotation, type) and issubclass(annotation, ModeSlicingMixin):
            return annotation
        return None
    for arg in get_args(annotation):
        nested = _nested_model(arg)
        if nested is not None:
            return nested
    return None


def _mode_names(model: type[ModeSlicingMixin], annotation: Any) -> set[str]:
    markers = getattr(model, "_mode_markers", {})
    metadata = _metadata(annotation)
    return {
        mode
        for mode, marker in markers.items()
        if any(isinstance(item, marker) for item in metadata)
    }


def _excluded_modes(annotation: Any) -> set[str]:
    return {
        mode
        for item in _metadata(annotation)
        if isinstance(item, ExcludeMode)
        for mode in item.modes
    }


def build_slice_manifest(
    model: type[ModeSlicingMixin],
    *,
    model_name: str | None = None,
    _seen: tuple[type[ModeSlicingMixin], ...] = (),
) -> dict[str, Any]:
    """Return a deterministic version-1 field-slice manifest.

    A recursive cycle is represented without a child manifest.  Cyclic graph
    *data* remains valid; the manifest only describes the finite model schema.
    """

    if not issubclass(model, ModeSlicingMixin):
        raise TypeError("slice manifests require a ModeSlicingMixin model")
    if model in _seen:
        return {
            "schema_version": 1,
            "model_name": model_name or model.__name__,
            "retain_unlisted_modes": ["backend"],
            "fields": {},
        }

    hints = get_type_hints(model, include_extras=True)
    include_unmarked: set[str] = set(
        getattr(model, "include_unmarked_for_modes", set())
    )
    fields: dict[str, dict[str, Any]] = {}
    for field_name in sorted(model.model_fields):
        annotation = hints.get(field_name, model.model_fields[field_name].annotation)
        marked = _mode_names(model, annotation)
        excluded = _excluded_modes(annotation)
        modes = {
            mode
            for mode in SLICE_MODES
            if ((mode in marked) or (not marked and mode in include_unmarked))
            and mode not in excluded
        }
        rule: dict[str, Any] = {"modes": sorted(modes)}
        nested = _nested_model(annotation)
        if nested is not None and nested not in (*_seen, model):
            rule["child"] = build_slice_manifest(
                nested,
                model_name=nested.__name__,
                _seen=(*_seen, model),
            )
        fields[field_name] = rule

    return {
        "schema_version": 1,
        "model_name": model_name or model.__name__,
        "retain_unlisted_modes": ["backend"],
        "fields": fields,
    }


def project_with_manifest(
    model: type[ModeSlicingMixin],
    value: BaseModel,
    mode: str,
) -> dict[str, Any]:
    """Return the Python projection described by the exported manifest.

    This is a differential-test helper and a reference implementation for a
    Rust consumer; normal application code should continue using
    ``model_dump(field_mode=mode)``.
    """

    if mode not in SLICE_MODES:
        raise ValueError(f"unsupported graph slice mode: {mode}")
    manifest = build_slice_manifest(model)
    # The Rust consumer receives the canonical/backend payload and applies a
    # requested public or LLM projection after crossing the language seam.
    return _project_value(
        value.model_dump(dump_format="json", field_mode="backend"), manifest, mode
    )


def _project_value(value: Any, manifest: dict[str, Any], mode: str) -> Any:
    if isinstance(value, list):
        return [_project_value(item, manifest, mode) for item in value]
    if not isinstance(value, dict):
        return value
    fields = manifest["fields"]
    retain_unlisted = mode in manifest.get("retain_unlisted_modes", [])
    projected: dict[str, Any] = {}
    for name, item in value.items():
        rule = fields.get(name)
        if rule is None:
            if retain_unlisted:
                projected[name] = item
            continue
        if mode not in rule.get("modes", []):
            continue
        child = rule.get("child")
        projected[name] = _project_value(item, child, mode) if child else item
    return projected


__all__ = ["SLICE_MODES", "build_slice_manifest", "project_with_manifest"]
