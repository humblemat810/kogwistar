from __future__ import annotations

from .base import NamespaceProxy


class ExtractSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    def resolve_extraction_schema_mode(self, *args, **kwargs):
        return self._e._resolve_extraction_schema_mode(*args, **kwargs)

    def schema_prompt_rules(self, *args, **kwargs):
        return self._e._schema_prompt_rules(*args, **kwargs)

    def build_structured_output_for_mode(self, *args, **kwargs):
        return self._e._build_structured_output_for_mode(*args, **kwargs)

    def to_canonical_extraction_for_mode(self, *args, **kwargs):
        return self._e._to_canonical_extraction_for_mode(*args, **kwargs)

    def extract_graph_with_llm_aliases(self, *args, **kwargs):
        return self._e._extract_graph_with_llm_aliases(*args, **kwargs)

    def de_alias_ids_in_result(self, *args, **kwargs):
        return self._e._de_alias_ids_in_result(*args, **kwargs)

    def aliasify_for_prompt(self, *args, **kwargs):
        return self._e._aliasify_for_prompt(*args, **kwargs)

    def repair_lean_offsets_for_mode(self, *args, **kwargs):
        return self._e._repair_lean_offsets_for_mode(*args, **kwargs)

    def coerce_pages(self, *args, **kwargs):
        return self._e._coerce_pages(*args, **kwargs)

    def fetch_document_text(self, document_id: str) -> str:
        return self._e._fetch_document_text(document_id)

    def cached_extract_graph_with_llm(self, *args, **kwargs):
        return self._e._cached_extract_graph_with_llm(*args, **kwargs)
