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

    def preflight_validate(self, *args, **kwargs):
        return self._call("preflight_validate", *args, **kwargs)

    def resolve_llm_ids(self, *args, **kwargs):
        return self._call("resolve_llm_ids", *args, **kwargs)

    def build_deps(self, *args, **kwargs):
        return self._call("build_deps", *args, **kwargs)

    def assert_endpoints_exist(self, *args, **kwargs):
        return self._call("assert_endpoints_exist", *args, **kwargs)

    def exists_node(self, rid: str) -> bool:
        return self._call("exists_node", rid)

    def exists_edge(self, rid: str) -> bool:
        return self._call("exists_edge", rid)

    def exists_any(self, rid: str) -> bool:
        return self._call("exists_any", rid)

    def dealias_span(self, *args, **kwargs):
        return self._call("dealias_span", *args, **kwargs)

    def select_doc_context(self, *args, **kwargs):
        return self._call("select_doc_context", *args, **kwargs)

    def persist_graph_extraction(self, *args, **kwargs):
        return self._call("persist_graph_extraction", *args, **kwargs)
