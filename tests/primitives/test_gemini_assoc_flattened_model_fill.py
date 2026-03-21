from __future__ import annotations

import hashlib
import os
from typing import Any, Literal

import pytest
from joblib import Memory

from kogwistar.engine_core.models import (
    AssocFlattenedLLMGraphExtraction,
    LLMGraphExtraction,
)

pytestmark = [pytest.mark.ci_full]

SchemaMode = Literal["full", "lean", "flattened_lean", "flattened_full"]
DEFAULT_SCHEMA_MODES: list[SchemaMode] = ["flattened_lean"]

DEFAULT_GEMINI_MODELS = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
]


def _parse_model_list() -> list[str]:
    raw = os.getenv("TEST_GEMINI_MODELS")
    if not raw:
        return list(DEFAULT_GEMINI_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]


def _parse_schema_modes() -> list[SchemaMode]:
    raw = os.getenv("TEST_GEMINI_SCHEMA_MODES")
    if not raw:
        return list(DEFAULT_SCHEMA_MODES)

    allowed: set[str] = {"full", "lean", "flattened_lean", "flattened_full"}
    parsed = [m.strip() for m in raw.split(",") if m.strip()]
    invalid = sorted({m for m in parsed if m not in allowed})
    if invalid:
        raise ValueError(
            f"TEST_GEMINI_SCHEMA_MODES has unsupported modes: {invalid}. "
            f"Expected subset of {sorted(allowed)}"
        )
    # Preserve order and de-duplicate.
    out: list[SchemaMode] = []
    seen: set[str] = set()
    for mode in parsed:
        if mode not in seen:
            out.append(mode)  # type: ignore[arg-type]
            seen.add(mode)
    return out


def _schema_for_mode(mode: SchemaMode):
    if mode == "full":
        return LLMGraphExtraction["llm"], "v3_full"
    if mode == "lean":
        return LLMGraphExtraction["llm_in"], "v4_llm_in_lean_non_flat"
    if mode == "flattened_lean":
        return AssocFlattenedLLMGraphExtraction["llm_in"], "v4_llm_in_lean_assoc"
    if mode == "flattened_full":
        return AssocFlattenedLLMGraphExtraction["llm"], "v3_full_assoc"
    raise ValueError(f"Unsupported schema mode: {mode!r}")


def _prompt_for_mode(mode: SchemaMode) -> str:
    if mode in {"flattened_lean", "flattened_full"}:
        return (
            "Return a minimal valid graph extraction payload following the schema exactly. "
            "Must include at least one span, one node, one grounding, one node_groundings link, "
            "and one grounding_spans link. Edge is optional. "
            "Use ids like sp:1 and gr:1."
        )
    if mode == "lean":
        return (
            "Return a minimal valid graph extraction payload following the schema exactly. "
            "Include at least one node with one mention and one span. Edge is optional. "
            "For each span include page_number, start_char, end_char, and excerpt."
        )
    return (
        "Return a minimal valid graph extraction payload following the schema exactly. "
        "Include at least one node with one mention and one full span object including "
        "collection_page_url, document_page_url, doc_id, page_number, start_char, end_char, "
        "excerpt, context_before, and context_after. Edge is optional."
    )


def _validate_parsed_payload(mode: SchemaMode, parsed: Any) -> dict[str, Any]:
    if mode == "flattened_lean":
        parsed_model = AssocFlattenedLLMGraphExtraction["llm_in"].model_validate(parsed)
        if len(parsed_model.spans) < 1:
            return {"ok": False, "reason": "spans is empty"}
        if len(parsed_model.nodes) < 1:
            return {"ok": False, "reason": "nodes is empty"}
        if len(parsed_model.groundings) < 1:
            return {"ok": False, "reason": "groundings is empty"}
        if len(parsed_model.grounding_spans) < 1:
            return {"ok": False, "reason": "grounding_spans is empty"}
        if len(parsed_model.node_groundings) + len(parsed_model.edge_groundings) < 1:
            return {
                "ok": False,
                "reason": "both node_groundings and edge_groundings are empty",
            }
        return {
            "ok": True,
            "reason": None,
            "counts": {
                "spans": len(parsed_model.spans),
                "nodes": len(parsed_model.nodes),
                "edges": len(parsed_model.edges),
                "groundings": len(parsed_model.groundings),
                "node_groundings": len(parsed_model.node_groundings),
                "edge_groundings": len(parsed_model.edge_groundings),
                "grounding_spans": len(parsed_model.grounding_spans),
            },
        }

    if mode == "flattened_full":
        parsed_model = (
            parsed
            if isinstance(parsed, AssocFlattenedLLMGraphExtraction)
            else AssocFlattenedLLMGraphExtraction.model_validate(
                parsed, context={"insertion_method": "llm"}
            )
        )
        if len(parsed_model.spans) < 1:
            return {"ok": False, "reason": "spans is empty"}
        if len(parsed_model.nodes) < 1:
            return {"ok": False, "reason": "nodes is empty"}
        if len(parsed_model.groundings) < 1:
            return {"ok": False, "reason": "groundings is empty"}
        if len(parsed_model.grounding_spans) < 1:
            return {"ok": False, "reason": "grounding_spans is empty"}
        if len(parsed_model.node_groundings) + len(parsed_model.edge_groundings) < 1:
            return {
                "ok": False,
                "reason": "both node_groundings and edge_groundings are empty",
            }
        return {
            "ok": True,
            "reason": None,
            "counts": {
                "spans": len(parsed_model.spans),
                "nodes": len(parsed_model.nodes),
                "edges": len(parsed_model.edges),
                "groundings": len(parsed_model.groundings),
                "node_groundings": len(parsed_model.node_groundings),
                "edge_groundings": len(parsed_model.edge_groundings),
                "grounding_spans": len(parsed_model.grounding_spans),
            },
        }

    if mode == "lean":
        parsed_model = LLMGraphExtraction["llm_in"].model_validate(parsed)
        if len(parsed_model.nodes) < 1:
            return {"ok": False, "reason": "nodes is empty"}
        first_mentions = parsed_model.nodes[0].mentions
        if len(first_mentions) < 1:
            return {"ok": False, "reason": "first node has no mentions"}
        if len(first_mentions[0].spans) < 1:
            return {"ok": False, "reason": "first grounding has no spans"}
        span = first_mentions[0].spans[0]
        if span.end_char <= span.start_char:
            return {"ok": False, "reason": "span has invalid offsets"}
        if not span.excerpt:
            return {"ok": False, "reason": "span excerpt is empty"}
        return {
            "ok": True,
            "reason": None,
            "counts": {
                "nodes": len(parsed_model.nodes),
                "edges": len(parsed_model.edges),
            },
        }

    parsed_model = (
        parsed
        if isinstance(parsed, LLMGraphExtraction)
        else LLMGraphExtraction.model_validate(
            parsed, context={"insertion_method": "llm"}
        )
    )
    if len(parsed_model.nodes) < 1:
        return {"ok": False, "reason": "nodes is empty"}
    first_mentions = parsed_model.nodes[0].mentions
    if len(first_mentions) < 1:
        return {"ok": False, "reason": "first node has no mentions"}
    if len(first_mentions[0].spans) < 1:
        return {"ok": False, "reason": "first grounding has no spans"}
    span = first_mentions[0].spans[0]
    if span.end_char <= span.start_char:
        return {"ok": False, "reason": "span has invalid offsets"}
    if not span.excerpt:
        return {"ok": False, "reason": "span excerpt is empty"}
    return {
        "ok": True,
        "reason": None,
        "counts": {
            "nodes": len(parsed_model.nodes),
            "edges": len(parsed_model.edges),
        },
    }


def _invoke_schema_live(
    *,
    model_name: str,
    schema_mode: SchemaMode,
    prompt: str,
    api_key_fingerprint: str,
    schema_version: str,
) -> dict[str, Any]:
    _ = api_key_fingerprint
    _ = schema_version
    genai = pytest.importorskip("langchain_google_genai")
    ChatGoogleGenerativeAI = genai.ChatGoogleGenerativeAI

    schema, _ = _schema_for_mode(schema_mode)
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    try:
        runnable = llm.with_structured_output(
            schema, method="json_schema", include_raw=True
        )
    except TypeError:
        runnable = llm.with_structured_output(schema, include_raw=True)

    try:
        result = runnable.invoke(prompt)
    except Exception as exc:
        return {"ok": False, "reason": f"invoke exception: {type(exc).__name__}: {exc}"}

    if isinstance(result, dict):
        parsed = result.get("parsed")
        parsing_error = result.get("parsing_error")
        if parsing_error is not None:
            return {"ok": False, "reason": f"parsing_error: {parsing_error}"}
    else:
        parsed = result

    if parsed is None:
        return {"ok": False, "reason": "parsed is None"}

    try:
        return _validate_parsed_payload(schema_mode, parsed)
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"schema validation failed: {type(exc).__name__}: {exc}",
        }


@pytest.mark.manual
def test_gemini_models_can_fill_assoc_flattened_schema():
    """
    pytest tests/primitives/test_gemini_assoc_flattened_model_fill.py::test_gemini_models_can_fill_assoc_flattened_schema
    Diagnostic integration test:
    - Uses real extraction schemas selected by TEST_GEMINI_SCHEMA_MODES
    - Probes multiple Gemini models
    - Passes if at least one (mode, model) combination succeeds
    - Caches calls to avoid repeated cost
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY (or GEMINI_API_KEY) not set")
    os.environ.setdefault("GOOGLE_API_KEY", api_key)
    pytest.importorskip("langchain_google_genai")

    models = _parse_model_list()
    if not models:
        pytest.skip("No Gemini models configured for probing")
    schema_modes = _parse_schema_modes()
    if not schema_modes:
        pytest.skip("No schema modes configured for probing")

    cache_dir = os.getenv("GEMINI_ASSOC_FILL_CACHE_DIR", ".joblib/gemini_assoc_fill")
    memory = Memory(location=cache_dir, verbose=0)
    cached_invoke = memory.cache(_invoke_schema_live)
    api_key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]

    outcomes: dict[tuple[str, str], dict[str, Any]] = {}
    for schema_mode in schema_modes:
        prompt = _prompt_for_mode(schema_mode)
        _, schema_version = _schema_for_mode(schema_mode)
        for model_name in models:
            key = (schema_mode, model_name)
            outcomes[key] = cached_invoke(
                model_name=model_name,
                schema_mode=schema_mode,
                prompt=prompt,
                api_key_fingerprint=api_key_fingerprint,
                schema_version=schema_version,
            )

    for (schema_mode, model_name), outcome in outcomes.items():
        print(
            f"[schema-fill] mode={schema_mode} model={model_name} "
            f"ok={outcome.get('ok')} reason={outcome.get('reason')}"
        )

    success_pairs = [
        (schema_mode, model_name)
        for (schema_mode, model_name), outcome in outcomes.items()
        if bool(outcome.get("ok"))
    ]
    assert success_pairs, (
        f"No probed Gemini (mode, model) pair filled the selected schemas successfully: {outcomes}"
    )
