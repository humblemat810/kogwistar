from __future__ import annotations

import json
import pathlib
import sqlite3
from typing import Any

from ...cdc.change_event import EntityRefModel
from ..subsystems.base import NamespaceProxy
from .models import (
    IndexingItem,
    build_embedding_text,
    make_index_key,
    make_index_key_for_item,
)
from .storage_sqlite import ensure_index_tables


class SearchIndexService(NamespaceProxy):
    def __init__(self, engine: Any, index_db_path: str) -> None:
        super().__init__(engine)
        self.index_db_path = index_db_path
        self.ensure_initialized()

    def _connect(self) -> sqlite3.Connection:
        if self.index_db_path not in ("", ":memory:"):
            pathlib.Path(self.index_db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.index_db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def ensure_initialized(self) -> None:
        conn = self._connect()
        try:
            ensure_index_tables(conn)
        finally:
            conn.close()

    def _defensive_embedding(self, text: str) -> list[float]:
        """Resolve the engine's defensive embedding hook in compatibility order."""
        if hasattr(self._e, "iterative_defensive_emb"):
            return self._e.iterative_defensive_emb(text)
        if hasattr(self._e, "_iterative_defensive_emb"):
            return self._e._iterative_defensive_emb(text)
        if hasattr(self._e, "embed") and hasattr(self._e.embed, "iterative_defensive_emb"):
            return self._e.embed.iterative_defensive_emb(text)
        raise AttributeError("engine has no iterative defensive embedding API")

    def upsert_entries(self, items: list[IndexingItem]) -> None:

        conn = self._connect()
        try:
            cur = conn.cursor()
            for item in items:
                kw = " ".join(item.keywords) if item.keywords else ""
                al = " ".join(item.aliases) if item.aliases else ""
                index_key = make_index_key_for_item(item)
                embedding_text = build_embedding_text(item)

                cur.execute(
                    """
                    INSERT INTO semantic_index
                        (index_key, node_id, canonical_title, keywords, aliases, provision, document_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(index_key) DO UPDATE SET
                        node_id          = excluded.node_id,
                        canonical_title  = excluded.canonical_title,
                        keywords         = excluded.keywords,
                        aliases          = excluded.aliases,
                        provision        = excluded.provision,
                        document_id      = excluded.document_id,
                        updated_at       = CURRENT_TIMESTAMP
                    """,
                    (
                        index_key,
                        item.node_id,
                        item.canonical_title,
                        kw,
                        al,
                        item.provision,
                        item.doc_id,
                    ),
                )

                self._e.backend.node_index_upsert(
                    ids=[f"idx:{index_key}"],
                    metadatas=[
                        {
                            "index_key": index_key,
                            "target_node_id": item.node_id,
                            "canonical_title": item.canonical_title,
                            "provision": item.provision,
                            "keywords": json.dumps(item.keywords),
                            "aliases": json.dumps(item.aliases),
                            "doc_id": item.doc_id,
                        }
                    ],
                    documents=[embedding_text],
                    embeddings=[self._defensive_embedding(embedding_text)],
                )

                payload = {
                    "node_id": item.node_id,
                    "canonical_title": item.canonical_title,
                    "keywords": item.keywords,
                    "aliases": item.aliases,
                    "provision": item.provision,
                    "doc_id": item.doc_id,
                }
                self._e._append_event_for_entity(
                    namespace=self._e.namespace,
                    entity_kind="search_index",
                    entity_id=index_key,
                    op="search_index.upsert",
                    payload=payload,
                )
                self._e._emit_change(
                    op="search_index.upsert",
                    entity=EntityRefModel(
                        kind="search_index",
                        id=index_key,
                        kg_graph_type=self._e.kg_graph_type,
                        url=None,
                    ),
                    payload=payload,
                )

            conn.commit()
        finally:
            conn.close()

    def search_hybrid(
        self, q: str, limit: int = 10, resolve_node: bool = False
    ) -> dict[str, Any]:

        conn = self._connect()
        try:
            cur = conn.cursor()

            cur.execute(
                """
                SELECT
                    s.index_key,
                    s.node_id,
                    s.canonical_title,
                    s.provision,
                    s.document_id,
                    bm25(semantic_index_fts) AS fts_score
                FROM semantic_index AS s
                JOIN semantic_index_fts
                  ON semantic_index_fts.rowid = s.id
                WHERE semantic_index_fts MATCH ?
                ORDER BY fts_score
                LIMIT ?;
                """,
                (q, limit),
            )
            fts_rows = cur.fetchall()

            vector_results = (
                self._e.backend.node_index_query(
                    query_texts=[q],
                    n_results=limit,
                )
                or {}
            )

            fts_norm = self._normalize_fts_rows(fts_rows)
            vec_norm = self._normalize_vector_results(vector_results)

            combined: dict[str, dict[str, Any]] = {}

            for r in fts_rows:
                key = str(r["index_key"])
                combined[key] = {
                    "index_key": key,
                    "node_id": str(r["node_id"]),
                    "canonical_title": str(r["canonical_title"]),
                    "provision": str(r["provision"]),
                    "document_id": r["document_id"],
                    "fts_score": fts_norm.get(key, 0.0),
                    "vec_score": 0.0,
                }

            vec_ids = vector_results.get("ids") or [[]]
            vec_metas = vector_results.get("metadatas") or [[]]

            if vec_ids and vec_ids[0] and vec_metas and vec_metas[0]:
                for idx, _ in enumerate(vec_ids[0]):
                    meta = vec_metas[0][idx] or {}
                    key = str(meta.get("index_key") or "")
                    if not key:
                        node_id = str(meta.get("target_node_id") or "")
                        canonical_title = str(meta.get("canonical_title") or "")
                        provision = str(meta.get("provision") or "")
                        if not node_id or not canonical_title:
                            continue
                        key = make_index_key(node_id, canonical_title, provision)

                    node_id = str(meta.get("target_node_id") or "")
                    canonical_title = str(meta.get("canonical_title") or "")
                    provision = str(meta.get("provision") or "")

                    if key not in combined:
                        combined[key] = {
                            "index_key": key,
                            "node_id": node_id,
                            "canonical_title": canonical_title,
                            "provision": provision,
                            "document_id": meta.get("doc_id"),
                            "fts_score": 0.0,
                            "vec_score": vec_norm.get(key, 0.0),
                        }
                    else:
                        combined[key]["vec_score"] = vec_norm.get(key, 0.0)

            for row in combined.values():
                row["hybrid_score"] = 0.6 * row["fts_score"] + 0.4 * row["vec_score"]

            ranked = sorted(
                combined.values(), key=lambda x: x["hybrid_score"], reverse=True
            )

            if resolve_node:
                return self._resolve_nodes(ranked[:limit], q)

            return {"query": q, "results": ranked[:limit]}
        finally:
            conn.close()

    def _normalize_fts_rows(self, rows: list[sqlite3.Row]) -> dict[str, float]:
        if not rows:
            return {}

        raw = [float(r["fts_score"]) for r in rows]
        min_s = min(raw)
        max_s = max(raw)

        out: dict[str, float] = {}
        for r in rows:
            key = str(r["index_key"])
            score = float(r["fts_score"])
            if max_s == min_s:
                norm = 1.0
            else:
                # lower bm25 is better
                norm = (max_s - score) / (max_s - min_s)
            out[key] = max(0.0, min(1.0, norm))
        return out

    def _normalize_vector_results(self, vr: dict[str, Any]) -> dict[str, float]:
        ids = vr.get("ids") or [[]]
        metas = vr.get("metadatas") or [[]]
        distances = vr.get("distances") or [[]]

        if (
            not ids
            or not ids[0]
            or not metas
            or not metas[0]
            or not distances
            or not distances[0]
        ):
            return {}

        raw = [float(d) for d in distances[0]]
        min_d = min(raw)
        max_d = max(raw)

        out: dict[str, float] = {}
        for idx, _ in enumerate(ids[0]):
            meta = metas[0][idx] or {}
            key = str(meta.get("index_key") or "")
            if not key:
                node_id = str(meta.get("target_node_id") or "")
                canonical_title = str(meta.get("canonical_title") or "")
                provision = str(meta.get("provision") or "")
                if not node_id or not canonical_title:
                    continue
                key = make_index_key(node_id, canonical_title, provision)

            dist = float(distances[0][idx])
            if max_d == min_d:
                norm = 1.0
            else:
                # lower distance is better
                norm = (max_d - dist) / (max_d - min_d)

            out[key] = max(0.0, min(1.0, norm))

        return out

    def _resolve_nodes(
        self, ranked_rows: list[dict[str, Any]], q: str
    ) -> dict[str, Any]:
        unique_node_ids: list[str] = []
        seen: set[str] = set()

        for row in ranked_rows:
            nid = row["node_id"]
            if nid and nid not in seen:
                seen.add(nid)
                unique_node_ids.append(nid)

        res = (
            self._e.backend.node_get(
                ids=unique_node_ids,
                include=["documents", "metadatas"],
            )
            or {}
        )

        rows_by_node_id: dict[str, dict[str, Any]] = {}
        res_ids = res.get("ids") or []
        res_docs = res.get("documents") or []
        res_metas = res.get("metadatas") or []

        for i, nid in enumerate(res_ids):
            if not nid:
                continue
            rows_by_node_id[str(nid)] = {
                "documents": res_docs[i] if i < len(res_docs) else None,
                "metadatas": res_metas[i] if i < len(res_metas) else None,
            }

        output = []
        for row in ranked_rows:
            enriched = dict(row)
            enriched.update(rows_by_node_id.get(row["node_id"], {}))
            output.append(enriched)

        return {"query": q, "results": output}
