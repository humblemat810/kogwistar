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
                "cached_extract_graph_with_llm": "_cached_extract_graph_with_llm",
            },
        )

    def resolve_extraction_schema_mode(self, *args, **kwargs):
        return self._call("resolve_extraction_schema_mode", *args, **kwargs)

    def schema_prompt_rules(self, *args, **kwargs):
        return self._call("schema_prompt_rules", *args, **kwargs)

    def build_structured_output_for_mode(self, *args, **kwargs):
        return self._call("build_structured_output_for_mode", *args, **kwargs)

    def to_canonical_extraction_for_mode(self, *args, **kwargs):
        return self._call("to_canonical_extraction_for_mode", *args, **kwargs)

    def extract_graph_with_llm_aliases(self, *args, **kwargs):
        return self._call("extract_graph_with_llm_aliases", *args, **kwargs)

    def de_alias_ids_in_result(self, *args, **kwargs):
        return self._call("de_alias_ids_in_result", *args, **kwargs)

    def aliasify_for_prompt(self, *args, **kwargs):
        return self._call("aliasify_for_prompt", *args, **kwargs)

    def repair_lean_offsets_for_mode(self, *args, **kwargs):
        return self._call("repair_lean_offsets_for_mode", *args, **kwargs)

    def coerce_pages(self, *args, **kwargs):
        return self._call("coerce_pages", *args, **kwargs)

    def fetch_document_text(self, document_id: str) -> str:
        return self._call("fetch_document_text", document_id)

    def cached_extract_graph_with_llm(self, *args, **kwargs):
        return self._call("cached_extract_graph_with_llm", *args, **kwargs)
