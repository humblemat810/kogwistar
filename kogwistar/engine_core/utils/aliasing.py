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

import re
from dataclasses import dataclass, field
from threading import RLock
from typing import Literal

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_UUID_RE = re.compile(r"^[0-9a-fA-F\-]{36}$")
_NODE_ALIAS_RE = re.compile(r"^N[1-9][0-9]*$")
_EDGE_ALIAS_RE = re.compile(r"^E[1-9][0-9]*$")
_NODE_BASE62_RE = re.compile(r"^N~[0-9A-Za-z]+$")
_EDGE_BASE62_RE = re.compile(r"^E~[0-9A-Za-z]+$")

AliasKind = Literal["node", "edge"]


class UnknownAliasError(ValueError):
    """Raised when a model returns an alias absent from the request legend."""


class AliasKindMismatchError(ValueError):
    """Raised when a node reference is used where an edge is required, or vice versa."""


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
    return bool(x) and bool(
        _NODE_ALIAS_RE.fullmatch(x)
        or _EDGE_ALIAS_RE.fullmatch(x)
        or _NODE_BASE62_RE.fullmatch(x)
        or _EDGE_BASE62_RE.fullmatch(x)
    )


def _alias_kind(x: str) -> AliasKind | None:
    if _NODE_ALIAS_RE.fullmatch(x) or _NODE_BASE62_RE.fullmatch(x):
        return "node"
    if _EDGE_ALIAS_RE.fullmatch(x) or _EDGE_BASE62_RE.fullmatch(x):
        return "edge"
    return None


def _is_new_node(x: str | None) -> bool:
    return bool(x) and x.startswith("nn:")


def _is_new_edge(x: str | None) -> bool:
    return bool(x) and x.startswith("ne:")


@dataclass
class AliasBook:
    """Stable per-session alias book. Append-only for cache friendliness."""

    next_n: int = 1
    next_e: int = 1
    # These two mappings remain as compatibility views for existing callers.
    # Typed methods below are authoritative and handle a node/edge ID collision.
    real_to_alias: dict[str, str] = field(default_factory=dict)
    alias_to_real: dict[str, str] = field(default_factory=dict)
    _node_real_to_alias: dict[str, str] = field(default_factory=dict, repr=False)
    _edge_real_to_alias: dict[str, str] = field(default_factory=dict, repr=False)
    _node_alias_to_real: dict[str, str] = field(default_factory=dict, repr=False)
    _edge_alias_to_real: dict[str, str] = field(default_factory=dict, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def alias_for_node(self, real_id: str) -> str:
        with self._lock:
            a = self._node_real_to_alias.get(real_id)
            if a:
                return a
            a = f"N{self.next_n}"
            self.next_n += 1
            self._node_real_to_alias[real_id] = a
            self._node_alias_to_real[a] = real_id
            self.real_to_alias.setdefault(real_id, a)
            self.alias_to_real[a] = real_id
            return a

    def alias_for_edge(self, real_id: str) -> str:
        with self._lock:
            a = self._edge_real_to_alias.get(real_id)
            if a:
                return a
            a = f"E{self.next_e}"
            self.next_e += 1
            self._edge_real_to_alias[real_id] = a
            self._edge_alias_to_real[a] = real_id
            self.real_to_alias.setdefault(real_id, a)
            self.alias_to_real[a] = real_id
            return a

    def resolve(self, alias_or_id: str, *, kind: AliasKind) -> str:
        """Resolve one typed reference, rejecting unknown or wrong-kind aliases."""
        if not alias_or_id:
            raise ValueError("alias_or_id must not be empty")
        alias_kind = _alias_kind(alias_or_id)
        if alias_kind is None:
            return alias_or_id
        if alias_kind != kind:
            raise AliasKindMismatchError(
                f"{alias_or_id!r} is an {alias_kind} alias, expected a {kind} alias"
            )
        with self._lock:
            mapping = self._node_alias_to_real if kind == "node" else self._edge_alias_to_real
            try:
                return mapping[alias_or_id]
            except KeyError as exc:
                raise UnknownAliasError(f"unknown {kind} alias: {alias_or_id}") from exc

    def resolve_node(self, alias_or_id: str) -> str:
        return self.resolve(alias_or_id, kind="node")

    def resolve_edge(self, alias_or_id: str) -> str:
        return self.resolve(alias_or_id, kind="edge")

    @classmethod
    def deterministic(cls, node_ids: list[str], edge_ids: list[str]) -> AliasBook:
        """Create a restart-stable book for one immutable prompt projection."""
        book = cls()
        book.assign_for_sets(sorted(set(node_ids)), sorted(set(edge_ids)))
        return book

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
        with self._lock:
            new_nodes = [rid for rid in node_ids if rid not in self._node_real_to_alias]
            new_edges = [rid for rid in edge_ids if rid not in self._edge_real_to_alias]
        return (
            [(rid, self.alias_for_node(rid)) for rid in new_nodes],
            [(rid, self.alias_for_edge(rid)) for rid in new_edges],
        )


@dataclass
class AliasBookStore:
    """Small keyed store for per-session/per-document alias books."""

    books: dict[str, AliasBook] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def get(self, key: str) -> AliasBook:
        with self._lock:
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
    """Translate LLM aliases back to real IDs, rejecting unknown aliases."""

    def r(a):
        if not a:
            raise ValueError("ID references must not be empty")
        if _is_alias(a) and a not in real_for_alias:
            raise UnknownAliasError(f"unknown alias: {a}")
        return real_for_alias.get(a, a)

    for n in llm_result.nodes:
        if n.id:
            n.id = r(n.id)
    for e in llm_result.edges:
        if e.id:
            e.id = r(e.id)
        e.source_ids = [r(x) for x in e.source_ids]
        e.target_ids = [r(x) for x in e.target_ids]
        if e.source_edge_ids is not None:
            e.source_edge_ids = [r(x) for x in e.source_edge_ids]
        if e.target_edge_ids is not None:
            e.target_edge_ids = [r(x) for x in e.target_edge_ids]
    return llm_result
