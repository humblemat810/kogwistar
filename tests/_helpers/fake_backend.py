from __future__ import annotations

"""Test backend used by tests that need Chroma-shaped behavior without Chroma.

This backend is intentionally small but not simplistic:
- it exposes the same collection verbs as the engine backend contract
- it supports Chroma-style `where` expressions, including:
  - single-field dicts like `{"doc_id": "X"}`
  - logical operators `{"$and": [...]}` and `{"$or": [...]}`
  - comparison operators `{"$in"}`, `{"$ne"}`, `{"$gt"}`, `{"$gte"}`, `{"$lt"}`, `{"$lte"}`
- it returns Chroma-shaped `get()` and `query()` payloads
- it uses the real SQLite metastore on a temp path, so engine/runtime code
  exercises the same meta-store contract as the normal SQLite backend

Usage:

```python
from tests._helpers.fake_backend import build_fake_backend

engine = GraphKnowledgeEngine(
    persist_directory=str(tmp_path / "conv"),
    kg_graph_type="conversation",
    embedding_function=ConstantEmbeddingFunction(dim=384),
    backend_factory=build_fake_backend,
)
```

The same pattern can be parameterized in pytest fixtures so a test can opt into:
- fake backend + fake embeddings for CI
- real backend + real/provider embeddings for fuller coverage
"""

import copy
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

try:
    from kogwistar.engine_core.engine_sqlite import EngineSQLite
except Exception:  # pragma: no cover - optional in lightweight environments
    EngineSQLite = None  # type: ignore[assignment]
from kogwistar.engine_core.storage_backend import NoopUnitOfWork


def _is_operator_dict(value: Any) -> bool:
    return isinstance(value, dict) and any(
        isinstance(k, str) and k.startswith("$") for k in value.keys()
    )


def _safe_cmp(left: Any, right: Any, op: str) -> bool:
    try:
        if op == "$eq":
            return left == right
        if op == "$ne":
            return left != right
        if op == "$gt":
            return left > right
        if op == "$gte":
            return left >= right
        if op == "$lt":
            return left < right
        if op == "$lte":
            return left <= right
    except Exception:
        return False
    return False


def _contains_any(left: Any, expected: Any) -> bool:
    if isinstance(left, (list, tuple, set, frozenset)):
        if isinstance(expected, (list, tuple, set, frozenset)):
            return any(item in expected for item in left)
        return expected in left
    if isinstance(expected, (list, tuple, set, frozenset)):
        return left in expected
    return left == expected


_MISSING = object()


def _matches_field(value: Any, condition: Any) -> bool:
    if value is _MISSING:
        return False
    if _is_operator_dict(condition):
        for op, expected in condition.items():
            if op == "$in":
                if isinstance(expected, (list, tuple, set, frozenset)):
                    if isinstance(value, (list, tuple, set, frozenset)):
                        if not any(item in expected for item in value):
                            return False
                    elif value not in expected:
                        return False
                else:
                    if isinstance(value, (list, tuple, set, frozenset)):
                        if expected not in value:
                            return False
                    elif value != expected:
                        return False
            elif op == "$nin":
                if isinstance(expected, (list, tuple, set, frozenset)):
                    if isinstance(value, (list, tuple, set, frozenset)):
                        if any(item in expected for item in value):
                            return False
                    elif value in expected:
                        return False
                else:
                    if isinstance(value, (list, tuple, set, frozenset)):
                        if expected in value:
                            return False
                    elif value == expected:
                        return False
            elif op == "$contains":
                if isinstance(value, str):
                    if str(expected) not in value:
                        return False
                elif isinstance(value, (list, tuple, set, frozenset)):
                    if expected not in value:
                        return False
                else:
                    return False
            elif op in {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte"}:
                if not _safe_cmp(value, expected, op):
                    return False
            else:
                raise ValueError(f"Unsupported where operator: {op!r}")
        return True

    return value == condition


def _matches_where(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    if not isinstance(where, dict):
        raise TypeError(f"where must be a dict or None, got {type(where)!r}")

    # Chroma-style logic keys are explicit; plain field keys are combined with AND.
    for key, condition in where.items():
        if key == "$and":
            clauses = condition or []
            if not all(_matches_where(metadata, clause) for clause in clauses):
                return False
            continue
        if key == "$or":
            clauses = condition or []
            if not any(_matches_where(metadata, clause) for clause in clauses):
                return False
            continue

        if key not in metadata and _is_operator_dict(condition):
            # still let the operator logic decide against a missing field
            if not _matches_field(_MISSING, condition):
                return False
            continue

        if not _matches_field(metadata.get(key), condition):
            return False
    return True


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return -1.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return dot / (left_norm * right_norm)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@dataclass
class _StoredRow:
    id: str
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


class _InMemoryCollection:
    def __init__(self, *, name: str, backend: "InMemoryBackend"):
        self.name = name
        self._backend = backend
        self._rows: dict[str, _StoredRow] = {}
        self._order: list[str] = []

    def _clone_row(self, row: _StoredRow) -> _StoredRow:
        return _StoredRow(
            id=row.id,
            document=row.document,
            metadata=copy.deepcopy(row.metadata),
            embedding=list(row.embedding) if row.embedding is not None else None,
        )

    def _ensure_order(self, row_id: str) -> None:
        if row_id not in self._order:
            self._order.append(row_id)

    def _store(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
        replace: bool = False,
    ) -> None:
        ids_list = [str(i) for i in ids]
        docs_list = list(documents or [])
        metas_list = list(metadatas or [])
        embs_list = list(embeddings or [])

        if docs_list and len(docs_list) != len(ids_list):
            raise ValueError("documents length must match ids length")
        if metas_list and len(metas_list) != len(ids_list):
            raise ValueError("metadatas length must match ids length")
        if embs_list and len(embs_list) != len(ids_list):
            raise ValueError("embeddings length must match ids length")

        for index, row_id in enumerate(ids_list):
            existing = self._rows.get(row_id)
            if existing is None:
                existing = _StoredRow(id=row_id, document="", metadata={})
            if replace and row_id in self._rows:
                existing = _StoredRow(id=row_id, document="", metadata={})

            document = docs_list[index] if index < len(docs_list) else existing.document
            metadata = (
                copy.deepcopy(metas_list[index])
                if index < len(metas_list)
                else copy.deepcopy(existing.metadata)
            )
            embedding = (
                list(embs_list[index])
                if index < len(embs_list) and embs_list[index] is not None
                else existing.embedding
            )

            self._rows[row_id] = _StoredRow(
                id=row_id,
                document=str(document) if document is not None else "",
                metadata=dict(metadata or {}),
                embedding=list(embedding) if embedding is not None else None,
            )
            self._ensure_order(row_id)

    def add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
        **_: Any,
    ) -> None:
        self._store(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def upsert(self, **kwargs: Any) -> None:
        self.add(**kwargs)

    def update(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str | None] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
        **_: Any,
    ) -> None:
        ids_list = [str(i) for i in ids]
        docs_list = list(documents or [])
        metas_list = list(metadatas or [])
        embs_list = list(embeddings or [])

        for index, row_id in enumerate(ids_list):
            row = self._rows.get(row_id)
            if row is None:
                row = _StoredRow(id=row_id, document="", metadata={})
                self._rows[row_id] = row
                self._ensure_order(row_id)

            if index < len(docs_list) and docs_list[index] is not None:
                row.document = str(docs_list[index])
            if index < len(metas_list) and metas_list[index] is not None:
                row.metadata.update(copy.deepcopy(metas_list[index]))
            if index < len(embs_list) and embs_list[index] is not None:
                row.embedding = list(embs_list[index])

    def _select_rows(
        self,
        *,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[_StoredRow]:
        if ids is not None:
            id_order = [str(i) for i in ids]
            rows = [self._rows[i] for i in id_order if i in self._rows]
        else:
            rows = [self._rows[i] for i in self._order if i in self._rows]
        filtered = [row for row in rows if _matches_where(row.metadata, where)]
        if limit is not None:
            filtered = filtered[: max(int(limit), 0)]
        return [self._clone_row(row) for row in filtered]

    def _format_get(
        self,
        rows: list[_StoredRow],
        *,
        include: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        include_set = set(include or ["documents", "metadatas", "embeddings"])
        out: dict[str, Any] = {"ids": [row.id for row in rows]}
        out["documents"] = [row.document for row in rows]
        out["metadatas"] = [copy.deepcopy(row.metadata) for row in rows]
        out["embeddings"] = [
            list(row.embedding) if row.embedding is not None else None for row in rows
        ]
        if "distances" in include_set:
            out["distances"] = [0.0 for _ in rows]
        return out

    def get(
        self,
        *,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
        include: Sequence[str] | None = None,
        limit: int | None = 200,
        **_: Any,
    ) -> dict[str, Any]:
        rows = self._select_rows(ids=ids, where=where, limit=limit)
        return self._format_get(rows, include=include)

    def query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]] | None = None,
        query_texts: Sequence[str] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        include: Sequence[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        include_set = set(include or ["documents", "metadatas", "embeddings"])
        if query_embeddings is None and query_texts is not None:
            query_embeddings = self._backend.embed(query_texts)
        query_embeddings = list(query_embeddings or [])
        rows = self._select_rows(where=where, limit=None)

        if not query_embeddings:
            batch_rows = rows[: max(int(n_results), 0)]
            nested = self._format_get(batch_rows, include=include)
            out = {k: [v] if k in {"ids", "documents", "metadatas", "embeddings"} else v for k, v in nested.items()}
            out["ids"] = [nested["ids"]]
            out["documents"] = [nested["documents"]]
            out["metadatas"] = [nested["metadatas"]]
            out["embeddings"] = [nested["embeddings"]]
            out["distances"] = [[0.0 for _ in batch_rows]]
            return out

        ids_batches: list[list[str]] = []
        docs_batches: list[list[str]] = []
        metas_batches: list[list[dict[str, Any]]] = []
        embs_batches: list[list[list[float] | None]] = []
        dist_batches: list[list[float]] = []

        for q_emb in query_embeddings:
            ranked: list[tuple[float, _StoredRow]] = []
            for row in rows:
                if row.embedding is None:
                    score = -1.0
                else:
                    score = _cosine_similarity(q_emb, row.embedding)
                ranked.append((score, row))
            ranked.sort(key=lambda pair: pair[0], reverse=True)
            picked = [row for _, row in ranked[: max(int(n_results), 0)]]
            ids_batches.append([row.id for row in picked])
            docs_batches.append([row.document for row in picked])
            metas_batches.append([copy.deepcopy(row.metadata) for row in picked])
            embs_batches.append(
                [list(row.embedding) if row.embedding is not None else None for row in picked]
            )
            dist_batches.append(
                [1.0 - max(score, -1.0) for score, _ in ranked[: max(int(n_results), 0)]]
            )

        out: dict[str, Any] = {"ids": ids_batches}
        out["documents"] = docs_batches
        out["metadatas"] = metas_batches
        out["embeddings"] = embs_batches
        out["distances"] = dist_batches
        if "documents" not in include_set:
            out["documents"] = docs_batches
        if "metadatas" not in include_set:
            out["metadatas"] = metas_batches
        if "embeddings" not in include_set:
            out["embeddings"] = embs_batches
        return out

    def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        if ids is not None:
            for row_id in [str(i) for i in ids]:
                self._rows.pop(row_id, None)
            self._order = [row_id for row_id in self._order if row_id in self._rows]
            return
        if where:
            doomed = [row.id for row in self._select_rows(where=where, limit=None)]
            for row_id in doomed:
                self._rows.pop(row_id, None)
            self._order = [row_id for row_id in self._order if row_id in self._rows]


class InMemoryBackend:
    def __init__(self, engine: Any):
        self._engine = engine
        self.unit_of_work = NoopUnitOfWork()
        self.node_index = _InMemoryCollection(name="node_index", backend=self)
        self.node = _InMemoryCollection(name="node", backend=self)
        self.edge = _InMemoryCollection(name="edge", backend=self)
        self.edge_endpoints = _InMemoryCollection(name="edge_endpoints", backend=self)
        self.document = _InMemoryCollection(name="document", backend=self)
        self.domain = _InMemoryCollection(name="domain", backend=self)
        self.node_docs = _InMemoryCollection(name="node_docs", backend=self)
        self.node_refs = _InMemoryCollection(name="node_refs", backend=self)
        self.edge_refs = _InMemoryCollection(name="edge_refs", backend=self)

    def _c(self, key: str) -> _InMemoryCollection:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(f"Unknown collection_key={key!r}") from exc

    def call(self, collection_key: str, method: str, **kwargs) -> Any:
        coll = self._c(collection_key)
        fn = getattr(coll, method)
        return fn(**kwargs)

    def node_index_get(self, **kwargs) -> Any:
        return self.call("node_index", "get", **kwargs)

    def node_index_query(self, **kwargs) -> Any:
        return self.call("node_index", "query", **kwargs)

    def node_index_add(self, **kwargs) -> Any:
        return self.call("node_index", "add", **kwargs)

    def node_index_upsert(self, **kwargs) -> Any:
        return self.call("node_index", "upsert", **kwargs)

    def node_index_update(self, **kwargs) -> Any:
        return self.call("node_index", "update", **kwargs)

    def node_index_delete(self, **kwargs) -> Any:
        return self.call("node_index", "delete", **kwargs)

    def node_get(self, **kwargs) -> Any:
        return self.call("node", "get", **kwargs)

    def node_query(self, **kwargs) -> Any:
        return self.call("node", "query", **kwargs)

    def node_add(self, **kwargs) -> Any:
        return self.call("node", "add", **kwargs)

    def node_upsert(self, **kwargs) -> Any:
        return self.call("node", "upsert", **kwargs)

    def node_update(self, **kwargs) -> Any:
        return self.call("node", "update", **kwargs)

    def node_delete(self, **kwargs) -> Any:
        return self.call("node", "delete", **kwargs)

    def edge_get(self, **kwargs) -> Any:
        return self.call("edge", "get", **kwargs)

    def edge_query(self, **kwargs) -> Any:
        return self.call("edge", "query", **kwargs)

    def edge_add(self, **kwargs) -> Any:
        return self.call("edge", "add", **kwargs)

    def edge_upsert(self, **kwargs) -> Any:
        return self.call("edge", "upsert", **kwargs)

    def edge_update(self, **kwargs) -> Any:
        return self.call("edge", "update", **kwargs)

    def edge_delete(self, **kwargs) -> Any:
        return self.call("edge", "delete", **kwargs)

    def edge_endpoints_get(self, **kwargs) -> Any:
        return self.call("edge_endpoints", "get", **kwargs)

    def edge_endpoints_query(self, **kwargs) -> Any:
        return self.call("edge_endpoints", "query", **kwargs)

    def edge_endpoints_add(self, **kwargs) -> Any:
        return self.call("edge_endpoints", "add", **kwargs)

    def edge_endpoints_upsert(self, **kwargs) -> Any:
        return self.call("edge_endpoints", "upsert", **kwargs)

    def edge_endpoints_update(self, **kwargs) -> Any:
        return self.call("edge_endpoints", "update", **kwargs)

    def edge_endpoints_delete(self, **kwargs) -> Any:
        return self.call("edge_endpoints", "delete", **kwargs)

    def document_get(self, **kwargs) -> Any:
        return self.call("document", "get", **kwargs)

    def document_query(self, **kwargs) -> Any:
        return self.call("document", "query", **kwargs)

    def document_add(self, **kwargs) -> Any:
        return self.call("document", "add", **kwargs)

    def document_upsert(self, **kwargs) -> Any:
        return self.call("document", "upsert", **kwargs)

    def document_update(self, **kwargs) -> Any:
        return self.call("document", "update", **kwargs)

    def document_delete(self, **kwargs) -> Any:
        return self.call("document", "delete", **kwargs)

    def domain_get(self, **kwargs) -> Any:
        return self.call("domain", "get", **kwargs)

    def domain_query(self, **kwargs) -> Any:
        return self.call("domain", "query", **kwargs)

    def domain_add(self, **kwargs) -> Any:
        return self.call("domain", "add", **kwargs)

    def domain_upsert(self, **kwargs) -> Any:
        return self.call("domain", "upsert", **kwargs)

    def domain_update(self, **kwargs) -> Any:
        return self.call("domain", "update", **kwargs)

    def domain_delete(self, **kwargs) -> Any:
        return self.call("domain", "delete", **kwargs)

    def node_docs_get(self, **kwargs) -> Any:
        return self.call("node_docs", "get", **kwargs)

    def node_docs_query(self, **kwargs) -> Any:
        return self.call("node_docs", "query", **kwargs)

    def node_docs_add(self, **kwargs) -> Any:
        return self.call("node_docs", "add", **kwargs)

    def node_docs_upsert(self, **kwargs) -> Any:
        return self.call("node_docs", "upsert", **kwargs)

    def node_docs_update(self, **kwargs) -> Any:
        return self.call("node_docs", "update", **kwargs)

    def node_docs_delete(self, **kwargs) -> Any:
        return self.call("node_docs", "delete", **kwargs)

    def node_refs_get(self, **kwargs) -> Any:
        return self.call("node_refs", "get", **kwargs)

    def node_refs_query(self, **kwargs) -> Any:
        return self.call("node_refs", "query", **kwargs)

    def node_refs_add(self, **kwargs) -> Any:
        return self.call("node_refs", "add", **kwargs)

    def node_refs_upsert(self, **kwargs) -> Any:
        return self.call("node_refs", "upsert", **kwargs)

    def node_refs_update(self, **kwargs) -> Any:
        return self.call("node_refs", "update", **kwargs)

    def node_refs_delete(self, **kwargs) -> Any:
        return self.call("node_refs", "delete", **kwargs)

    def edge_refs_get(self, **kwargs) -> Any:
        return self.call("edge_refs", "get", **kwargs)

    def edge_refs_query(self, **kwargs) -> Any:
        return self.call("edge_refs", "query", **kwargs)

    def edge_refs_add(self, **kwargs) -> Any:
        return self.call("edge_refs", "add", **kwargs)

    def edge_refs_upsert(self, **kwargs) -> Any:
        return self.call("edge_refs", "upsert", **kwargs)

    def edge_refs_update(self, **kwargs) -> Any:
        return self.call("edge_refs", "update", **kwargs)

    def edge_refs_delete(self, **kwargs) -> Any:
        return self.call("edge_refs", "delete", **kwargs)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ef = getattr(self._engine, "_ef", None)
        if callable(ef):
            return list(ef(list(texts)))
        return [[0.0] for _ in texts]


class _FakeMetaStore:
    """Legacy pure-Python meta-store stub kept for reference.

    The active fake backend now uses EngineSQLite so tests exercise the same
    metastore API as the real sqlite path.
    """

    def __init__(self) -> None:
        self._user_seq: dict[str, int] = {}
        self._global_seq = 0

    def ensure_initialized(self) -> None:
        return None

    @contextmanager
    def transaction(self):
        yield None

    def next_user_seq(self, user_id: str) -> int:
        next_value = self._user_seq.get(user_id, 0) + 1
        self._user_seq[user_id] = next_value
        return next_value

    def next_scoped_seq(self, scope_id: str) -> int:
        return self.next_user_seq(scope_id)

    def current_user_seq(self, user_id: str) -> int:
        return self._user_seq.get(user_id, 0)

    def current_scoped_seq(self, scope_id: str) -> int:
        return self.current_user_seq(scope_id)

    def set_user_seq(self, user_id: str, value: int) -> None:
        self._user_seq[user_id] = int(value)

    def set_scoped_seq(self, scope_id: str, value: int) -> None:
        self.set_user_seq(scope_id, value)

    def next_global_seq(self) -> int:
        self._global_seq += 1
        return self._global_seq

    def current_global_seq(self) -> int:
        return self._global_seq


class _DummyLock:
    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def release(self) -> None:
        return None

    def __enter__(self) -> "_DummyLock":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def build_fake_backend(engine: Any) -> InMemoryBackend:
    backend = InMemoryBackend(engine)
    engine.backend_kind = "fake"
    backend.backend_kind = "fake"
    if EngineSQLite is None:
        engine.meta_sqlite = _FakeMetaStore()
    else:
        meta_root = Path(engine.persist_directory or ".").resolve() / "_fake_meta"
        engine.meta_sqlite = EngineSQLite(meta_root, "meta.sqlite")
        engine.meta_sqlite.ensure_initialized()
    engine.collection_lock = {
        "node": _DummyLock(),
        "edge": _DummyLock(),
        "node_index": _DummyLock(),
        "edge_endpoints": _DummyLock(),
        "document": _DummyLock(),
        "domain": _DummyLock(),
        "node_docs": _DummyLock(),
        "node_refs": _DummyLock(),
        "edge_refs": _DummyLock(),
    }
    engine.node_index_collection = backend.node_index
    engine.node_collection = backend.node
    engine.edge_collection = backend.edge
    engine.edge_endpoints_collection = backend.edge_endpoints
    engine.document_collection = backend.document
    engine.domain_collection = backend.domain
    engine.node_docs_collection = backend.node_docs
    engine.node_refs_collection = backend.node_refs
    engine.edge_refs_collection = backend.edge_refs
    return backend
