from __future__ import annotations

from .base import NamespaceProxy


class PersistSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(
            engine,
            aliases={
                "preflight_validate": "_preflight_validate",
                "resolve_llm_ids": "_resolve_llm_ids",
                "build_deps": "_build_deps",
                "assert_endpoints_exist": "_assert_endpoints_exist",
                "exists_node": "_exists_node",
                "exists_edge": "_exists_edge",
                "exists_any": "_exists_any",
                "dealias_span": "_dealias_span",
                "select_doc_context": "_select_doc_context",
            },
        )
