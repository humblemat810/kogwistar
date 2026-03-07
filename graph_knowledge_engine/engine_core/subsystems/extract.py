from __future__ import annotations

from .base import NamespaceProxy


class ExtractSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(
            engine,
            aliases={
                "resolve_extraction_schema_mode": "_resolve_extraction_schema_mode",
                "schema_prompt_rules": "_schema_prompt_rules",
                "build_structured_output_for_mode": "_build_structured_output_for_mode",
                "to_canonical_extraction_for_mode": "_to_canonical_extraction_for_mode",
                "extract_graph_with_llm_aliases": "_extract_graph_with_llm_aliases",
                "de_alias_ids_in_result": "_de_alias_ids_in_result",
                "aliasify_for_prompt": "_aliasify_for_prompt",
                "repair_lean_offsets_for_mode": "_repair_lean_offsets_for_mode",
                "coerce_pages": "_coerce_pages",
                "fetch_document_text": "_fetch_document_text",
            },
        )
