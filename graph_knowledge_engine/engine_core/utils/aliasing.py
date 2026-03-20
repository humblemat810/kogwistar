"""Graph ID Aliasing Utilities.

### Rationale:
In GraphRAG systems, raw entity IDs (typically UUIDs) are often long and token-heavy.
When including multiple nodes and edges in an LLM prompt, using raw IDs can quickly
exhaust the token budget and lead to LLM confusion.

This module provides utilities to:
1. **Shorten IDs**: Map long UUIDs to short, stable aliases like `N1`, `E2` (session-based)
   or deterministic base62 strings (e.g., `N~...`).
2. **Ensure Stability**: Maintain consistent mappings within a conversation session
   or a single document extraction run, allowing the LLM to refer back to previously
   mentioned entities reliably.
3. **De-alias**: Reconstruct the original graph structure by mapping LLM-generated
   aliases back to their canonical UUIDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_UUID_RE = re.compile(r"^[0-9a-fA-F\-]{36}$")


def uuid_to_base62(u: str) -> str:
    n = int(u.replace("-", ""), 16)
    if n == 0:
        return "0"
    out: list[str] = []
    while n:
        n, r = divmod(n, 62)
        out.append(ALPHABET[r])
    return "".join(reversed(out))


def base62_to_uuid(s: str) -> str:
    n = 0
    for ch in s:
        n = n * 62 + ALPHABET.index(ch)
    hex128 = f"{n:032x}"
    return f"{hex128[0:8]}-{hex128[8:12]}-{hex128[12:16]}-{hex128[16:20]}-{hex128[20:]}"


def _is_uuid(x: str | None) -> bool:
    return bool(x and _UUID_RE.match(x))


def _is_alias(x: str | None) -> bool:
    # Accept session aliases N\d+, E\d+ and base62 N~..., E~...
    return bool(x) and (x.startswith("N") or x.startswith("E"))


def _is_new_node(x: str | None) -> bool:
    return bool(x) and x.startswith("nn:")


def _is_new_edge(x: str | None) -> bool:
    return bool(x) and x.startswith("ne:")


@dataclass
class AliasBook:
    """Stable per-session alias book. Append-only for cache friendliness."""

    next_n: int = 1
    next_e: int = 1
    real_to_alias: dict = field(default_factory=dict)  # real_id -> alias "N#"/"E#"
    alias_to_real: dict = field(default_factory=dict)  # alias -> real_id

    def alias_for_node(self, real_id: str) -> str:
        a = self.real_to_alias.get(real_id)
        if a:
            return a
        a = f"N{self.next_n}"
        self.next_n += 1
        self.real_to_alias[real_id] = a
        self.alias_to_real[a] = real_id
        return a

    def alias_for_edge(self, real_id: str) -> str:
        a = self.real_to_alias.get(real_id)
        if a:
            return a
        a = f"E{self.next_e}"
        self.next_e += 1
        self.real_to_alias[real_id] = a
        self.alias_to_real[a] = real_id
        return a

    def assign_for_sets(self, node_ids: list[str], edge_ids: list[str]) -> None:
        for rid in node_ids:
            self.alias_for_node(rid)
        for rid in edge_ids:
            self.alias_for_edge(rid)

    def legend_delta(
        self, node_ids: list[str], edge_ids: list[str]
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Return only (real_id, alias) pairs that are NEW since last turn."""
        new_nodes: list[tuple[str, str]] = []
        new_edges: list[tuple[str, str]] = []
        for rid in node_ids:
            if rid not in self.real_to_alias:
                new_nodes.append((rid, self.alias_for_node(rid)))
        for rid in edge_ids:
            if rid not in self.real_to_alias:
                new_edges.append((rid, self.alias_for_edge(rid)))
        return new_nodes, new_edges


@dataclass
class AliasBookStore:
    """Small keyed store for per-session/per-document alias books."""

    books: dict[str, AliasBook] = field(default_factory=dict)

    def get(self, key: str) -> AliasBook:
        if key not in self.books:
            self.books[key] = AliasBook()
        return self.books[key]


def build_aliases(node_ids, edge_ids):
    node_aliases = {rid: f"N{i}" for i, rid in enumerate(node_ids, start=1)}
    edge_aliases = {rid: f"E{i}" for i, rid in enumerate(edge_ids, start=1)}
    alias_for_real = {**node_aliases, **edge_aliases}
    real_for_alias = {v: k for k, v in alias_for_real.items()}
    return alias_for_real, real_for_alias


def aliasify_graph(nodes, edges, alias_for_real):
    """Return shallow copies with ids replaced by aliases for prompt."""

    def a(rid):
        return alias_for_real.get(rid, rid)

    aliased_nodes = [
        {
            "id": a(n["id"]),
            "label": n["label"],
            "type": n["type"],
            "summary": n.get("summary", ""),
        }
        for n in nodes
    ]
    aliased_edges = [
        {
            "id": a(e["id"]),
            "relation": e["relation"],
            "source_ids": [a(s) for s in e.get("source_ids", [])],
            "target_ids": [a(t) for t in e.get("target_ids", [])],
        }
        for e in edges
    ]
    return aliased_nodes, aliased_edges


def de_alias_ids(llm_result, real_for_alias):
    """Translate LLM aliases back to real UUIDs in-place."""

    def r(a):
        return real_for_alias.get(a, a)

    for n in llm_result.nodes:
        if n.id:
            n.id = r(n.id)
    for e in llm_result.edges:
        if e.id:
            e.id = r(e.id)
        e.source_ids = [r(x) for x in e.source_ids]
        e.target_ids = [r(x) for x in e.target_ids]
    return llm_result
