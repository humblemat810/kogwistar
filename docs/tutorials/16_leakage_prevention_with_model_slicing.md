# 16 Leakage Prevention with Model Slicing

Audience: Advanced / Security-conscious Developers
Time: 15-20 minutes

## What You Will Build

You will understand how the repo prevents sensitive data leakage to LLMs using `pydantic-extension` and `ModeSlicingMixin`. You'll see how to mark fields as internal-only and how the system enforces these boundaries.

## Why This Matters

When building RAG systems, your data models often contain a mix of content intended for the LLM (summaries, labels) and internal metadata (database IDs, insertion methods, scores). Accidentally sending internal metadata to the LLM (leakage) can:
1. Waste expensive tokens.
2. Confuse the LLM with irrelevant data.
3. Expose sensitive system internals or implementation details.

## Core Concepts

### Model Slicing

Instead of maintaining multiple similar models (e.g., `Node`, `LLMNode`, `NodeDTO`), we use a single base model and "slice" it for different purposes.

- **`llm` slice**: Used for structured output from the LLM. Only includes fields the LLM should provide.
- **`llm_in` slice**: Used for data sent TO the LLM as context. Excludes internal identifiers or backend-only status fields.
- **`dto` slice**: Used for API responses to the frontend.

### Automatic Detection

The library includes a heuristic that detects if it's being called from within a LangChain `with_structured_output` stack. If it is, it automatically switches to the `llm` slice, ensuring that even if you pass the base model, the LLM only sees the safe schema.

## Demo Walkthrough

The companion script `scripts/tutorial_sections/16_leakage_prevention_with_model_slicing.py` demonstrates:

1. **Explicit Exclusion**: Using `Annotated[T, ExcludeMode("llm")]` to hide fields.
2. **Schema Comparison**: Printing the JSON schema of the base model vs. the `llm` slice.
3. **Validation Guards**: Showing how trying to force LLM data back into a base model raises a `ValidationError` because required internal fields are missing.
4. **Heuristic Protection**: How the library protects you even if you forget to slice explicitly.

## Run or Inspect

Open `scripts/tutorial_sections/16_leakage_prevention_with_model_slicing.py` to inspect the slice definitions and schema comparison logic, or run it to see the model slicing output locally.

## Inspect The Result

The result should make it obvious which fields are omitted from the `llm` slice and which internal fields remain available only to the engine.

## Invariant Demonstrated

Sensitive engine metadata stays out of the LLM-facing schema unless it is explicitly marked for inclusion.

## Next Tutorial

Continue with `17_custom_llm_provider.md` to see how provider selection and model routing fit into the same pattern.

## Next Steps

Explore `kogwistar/engine_core/models.py` to see how `Node` and `Edge` use these decorators to keep your knowledge graph secure.
