# %% [markdown]
# # 16 Leakage Prevention with Model Slicing
# This tutorial demonstrates how to use `ModeSlicingMixin` and `ExcludeMode`
# to prevent internal system metadata from leaking into LLM prompts.

# %%
from typing import Annotated
from pydantic import BaseModel, Field, ValidationError
from pydantic_extension.model_slicing import ModeSlicingMixin, use_mode, LLMField
from pydantic_extension.model_slicing.mixin import ExcludeMode
import json

# Define a model that represents a Knowledge Graph Node,
# but contains sensitive internal fields that should NEVER be sent to an LLM.
class SecureNode(ModeSlicingMixin, BaseModel):
    # Public fields: intended for the LLM. 
    # We use Annotated with LLMField() or include "llm" in include_unmarked_for_modes.
    label: Annotated[str, LLMField()] = Field(..., description="The name of the entity")
    summary: Annotated[str, LLMField()] = Field(..., description="A brief description")

    # Internal fields: intended for the database or backend only.
    # We use ExcludeMode("llm") to ensure these are removed when slicing for LLM.
    db_id: Annotated[int, ExcludeMode("llm")] = Field(..., description="Internal DB primary key")
    auth_token: Annotated[str, ExcludeMode("llm")] = Field(..., description="Sensitive access token")

def show_schema(title, model_cls):
    print(f"\n=== {title} ===")
    print(json.dumps(model_cls.model_json_schema(), indent=2))

# %% [markdown]
# ## 1. Comparing Schemas
# Notice how the base model includes all fields, while the `llm` slice automatically removes the protected ones.

# %%
# The original model contains everything.
show_schema("Base SecureNode Schema (Full)", SecureNode)

# The "llm" slice is a dynamic class that only contains permitted fields.
show_schema("SecureNode['llm'] Schema (Protected)", SecureNode["llm"])

# %% [markdown]
# ## 2. Validation Errors (Leakage Prevention in Practice)
# If an LLM returns data, we should validate it against the `llm` slice.
# If we try to validate it against the base model, it will FAIL because the internal fields are missing.
# This "fails safe" by preventing us from accidentally creating partially-initialized base models from untrusted data.

# %%
llm_data = {
    "label": "Apple Inc.",
    "summary": "A technology company."
}

print("\n--- Validation Test ---")
try:
    # This works! The LLM slice expects exactly what the LLM provides.
    node_slice = SecureNode["llm"].model_validate(llm_data)
    print("✅ Successfully validated against SecureNode['llm']")
except ValidationError as e:
    print(f"❌ Unexpected error: {e}")

try:
    # This FAILS! The base model requires 'db_id' and 'auth_token', which the LLM doesn't know about.
    print("\nAttempting to validate LLM data against base SecureNode...")
    SecureNode.model_validate(llm_data)
except ValidationError as e:
    print("✅ Correctly raised ValidationError when using base model:")
    for error in e.errors():
        print(f"   - Missing field: {error['loc'][0]}")

# %% [markdown]
# ## 3. Automatic Stack Detection
# `pydantic-extension` can detect if it's being called from within a LangChain stack.
# We can simulate this by using the `use_mode` context manager, which is what the library 
# uses internally when it sniffs a LangChain function on the execution stack.

# %%
def mock_langchain_with_structured_output(model_class):
    """Simulates a LangChain call that requests a JSON schema."""
    # When this is called, pydantic-extension detects the context (simulated here with use_mode)
    with use_mode("llm"):
        schema = model_class.model_json_schema()
        return schema

print("\n--- Heuristic Protection Test ---")
schema = mock_langchain_with_structured_output(SecureNode)
properties = schema.get("properties", {}).keys()
print(f"Fields found in schema: {list(properties)}")

if "db_id" not in properties and "auth_token" not in properties:
    print("✅ Success: Leakage prevented via automatic mode detection!")
else:
    print("❌ Failure: Internal fields leaked into schema.")

# %% [markdown]
# ## Summary
# By using `ModeSlicingMixin`, you define your security boundaries once in the model definition.
# 1. `ExcludeMode("llm")` ensures sensitive fields never reach the LLM.
# 2. Sliced models (`Model["llm"]`) provide a frozen, safe interface for LangChain.
# 3. Validation failures on the base model act as a secondary guardrail against "half-baked" objects.
