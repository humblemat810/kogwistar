from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from .models import Node, Edge, Span, Grounding

if TYPE_CHECKING:
    # Avoid runtime import cycles; we only need this for typing.
    from .engine import GraphKnowledgeEngine


def _is_tombstoned(meta: dict | None) -> bool:
    meta = meta or {}
    return str(meta.get("lifecycle_status") or "active") == "tombstoned"


def _extract_doc_ids_from_refs(refs: list[Span] | list[Grounding]) -> list[str]:
    out: list[str] = []
    for r in refs or []:
        if type(r) is Grounding:
            for sp in r.spans:
                did = getattr(sp, "doc_id", None)
                if did:
                    out.append(did)
        elif type(r) is Span:
            did = getattr(r, "doc_id", None)
            if did:
                out.append(did)
        else:
            raise ValueError(f"unsupported ref type: {type(r)}")
    # unique + stable order
    return sorted(dict.fromkeys(out))


@dataclass
class IndexingSubsystem:
    """
    Owns durable index-jobs + join-index reconciliation logic.

    Engine remains the facade/entrypoint, but indexing logic is isolated here to:
    - improve locality
    - allow focused tests
    - reduce "god object" feel in engine.py
    """
    engine: "GraphKnowledgeEngine"

    _PHASE1_JOIN_INDEX_KINDS = ("node_docs", "node_refs", "edge_refs", "edge_endpoints")

    def enqueue_index_job(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        index_kind: str,
        op: str,
        payload_json: str | None = None,
        namespace: str | None = None,
    ) -> str:
        if not getattr(self.engine, "_phase1_enable_index_jobs", False):
            return ""
        job_id = str(uuid.uuid4())

        enqueue = getattr(self.engine.meta_sqlite, "enqueue_index_job", None)
        if enqueue is None:
            return ""

        job_id2 = self.engine.meta_sqlite.enqueue_index_job(
            job_id=job_id,
            namespace=(self.engine.namespace if namespace is None else namespace),
            entity_kind=entity_kind,
            entity_id=entity_id,
            index_kind=index_kind,
            op=op,
            payload_json=payload_json,
        )
        return str(job_id2) if job_id2 else job_id

    def enqueue_index_jobs_for_node(self, node_id: str, *, op: str) -> None:
        for idx in ("node_docs", "node_refs"):
            self.enqueue_index_job(entity_kind="node", entity_id=node_id, index_kind=idx, op=op)

    def enqueue_index_jobs_for_edge(self, edge_id: str, *, op: str) -> None:
        for idx in ("edge_refs", "edge_endpoints"):
            self.enqueue_index_job(entity_kind="edge", entity_id=edge_id, index_kind=idx, op=op)

    def reconcile_indexes(
        self,
        *,
        max_jobs: int = 100,
        lease_seconds: int = 60,
        namespace: str | None = None,
    ) -> int:
        if not getattr(self.engine, "_phase1_enable_index_jobs", False):
            return 0
        claim = getattr(self.engine.meta_sqlite, "claim_index_jobs", None)
        if claim is None:
            return 0

        ns = self.engine.namespace if namespace is None else namespace
        jobs = claim(limit=max_jobs, lease_seconds=lease_seconds, namespace=ns)

        applied = 0
        for job in jobs:
            # EngineSQLite returns IndexJobRow; PG meta might return dict.
            job_id = getattr(job, "job_id", None) or (job.get("job_id") if isinstance(job, dict) else None)
            entity_kind = getattr(job, "entity_kind", None) or (job.get("entity_kind") if isinstance(job, dict) else None)
            entity_id = getattr(job, "entity_id", None) or (job.get("entity_id") if isinstance(job, dict) else None)
            index_kind = getattr(job, "index_kind", None) or (job.get("index_kind") if isinstance(job, dict) else None)
            op = getattr(job, "op", None) or (job.get("op") if isinstance(job, dict) else None)

            retry_count = getattr(job, "retry_count", None) if not isinstance(job, dict) else job.get("retry_count")
            max_retries = getattr(job, "max_retries", None) if not isinstance(job, dict) else job.get("max_retries")
            try_rc = int(retry_count or 0)
            try_mr = int(max_retries or 10)

            try:
                self.apply_index_job(
                    job_id=str(job_id),
                    entity_kind=str(entity_kind),
                    entity_id=str(entity_id),
                    index_kind=str(index_kind),
                    op=str(op),
                    namespace=ns,
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                bump = getattr(self.engine.meta_sqlite, "bump_retry_and_requeue", None)
                mark_failed = getattr(self.engine.meta_sqlite, "mark_index_job_failed", None)

                if job_id:
                    next_retry = try_rc + 1
                    if bump is not None and next_retry < try_mr:
                        delay = min(300, 2 ** min(next_retry - 1, 8))
                        bump(str(job_id), err, next_run_at_seconds=int(delay))
                    elif mark_failed is not None:
                        mark_failed(str(job_id), err, final=True)
                continue

            mark_done = getattr(self.engine.meta_sqlite, "mark_index_job_done", None)
            if mark_done is not None and job_id:
                mark_done(str(job_id))
            applied += 1

        return applied

    def make_index_job_worker(
        self,
        *,
        max_inflight: int = 1,
        batch_size: int = 50,
        lease_seconds: int = 60,
        max_jobs_per_tick: int = 200,
        namespace: str | None = None,
    ):
        # local import to avoid subsystem importing workers at module import time if you want
        from .workers.index_job_worker import IndexJobWorker

        return IndexJobWorker(
            engine=self.engine,
            max_inflight=max_inflight,
            batch_size=batch_size,
            lease_seconds=lease_seconds,
            max_jobs_per_tick=max_jobs_per_tick,
            namespace=namespace,
        )

    def apply_index_job(
        self,
        *,
        job_id: str,
        entity_kind: str,
        entity_id: str,
        index_kind: str,
        op: str,
        namespace: str,
    ) -> None:
        """
        Moved from engine.indexing.apply_index_job, but still uses engine's lower-level primitives:
        - engine.backend.* join index collections
        - engine._index_* implementations
        - engine._fanout_endpoints_rows
        - engine._iterative_defensive_emb
        - engine._delete_*_ref_rows
        """
        if index_kind not in self._PHASE1_JOIN_INDEX_KINDS:
            return

        # Phase 2: fingerprints + drift detection
        coalesce_key = f"{entity_kind}:{entity_id}:{index_kind}"
        get_applied = getattr(self.engine.meta_sqlite, "get_index_applied_fingerprint", None)
        set_applied = getattr(self.engine.meta_sqlite, "set_index_applied_fingerprint", None)

        def _fp(obj: object) -> str:
            blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
            return hashlib.blake2b(blob, digest_size=16).hexdigest()

        def _actual_fp() -> str:
            if entity_kind == "node":
                if index_kind == "node_docs":
                    got = self.engine.backend.node_docs_get(where={"node_id": entity_id})
                elif index_kind == "node_refs":
                    got = self.engine.backend.node_refs_get(where={"node_id": entity_id})
                else:
                    got = {"ids": []}
            elif entity_kind == "edge":
                if index_kind == "edge_refs":
                    got = self.engine.backend.edge_refs_get(where={"edge_id": entity_id})
                elif index_kind == "edge_endpoints":
                    got = self.engine.backend.edge_endpoints_get(where={"edge_id": entity_id})
                else:
                    got = {"ids": []}
            else:
                got = {"ids": []}
            ids = sorted(list(got.get("ids") or []))
            return _fp(ids)

        applied_fp = (
            self.engine.meta_sqlite.get_index_applied_fingerprint(namespace=namespace, coalesce_key=coalesce_key)
            if callable(get_applied)
            else None
        )

        if entity_kind == "node":
            if index_kind == "node_docs":
                if op == "DELETE":
                    self.engine.backend.node_docs_delete(where={"node_id": entity_id})
                    if callable(set_applied):
                        set_applied(namespace=namespace, coalesce_key=coalesce_key, applied_fingerprint=None, last_job_id=job_id)
                    return

                got = self.engine.backend.node_get(ids=[entity_id], include=["documents", "metadatas"])
                docs = got.get("documents") or []
                if not docs or not docs[0]:
                    raise Exception("document not found")
                n = Node.model_validate_json(docs[0])

                meta0 = (got.get("metadatas") or [None])[0] or {}
                if _is_tombstoned(meta0):
                    self.engine.backend.node_docs_delete(where={"node_id": entity_id})
                    if callable(set_applied):
                        set_applied(namespace=namespace, coalesce_key=coalesce_key, applied_fingerprint=None, last_job_id=job_id)
                    return

                desired_fp = _fp(_extract_doc_ids_from_refs(n.mentions))
                if op != "DELETE" and applied_fp is not None and applied_fp == desired_fp:
                    if _actual_fp() == desired_fp:
                        return

                self.engine.write.index_node_docs(n)
                if callable(set_applied):
                    set_applied(namespace=namespace, coalesce_key=coalesce_key, applied_fingerprint=desired_fp, last_job_id=job_id)
                return

            if index_kind == "node_refs":
                if op == "DELETE":
                    self.engine.write.delete_node_ref_rows(entity_id)
                    if callable(set_applied):
                        set_applied(namespace=namespace, coalesce_key=coalesce_key, applied_fingerprint=None, last_job_id=job_id)
                    return

                got = self.engine.backend.node_get(ids=[entity_id], include=["documents", "metadatas"])
                docs = got.get("documents") or []
                if not docs or not docs[0]:
                    raise Exception("document not found")
                n = Node.model_validate_json(docs[0])

                meta0 = (got.get("metadatas") or [None])[0] or {}
                if _is_tombstoned(meta0):
                    self.engine.write.delete_node_ref_rows(entity_id)
                    if callable(set_applied):
                        set_applied(namespace=namespace, coalesce_key=coalesce_key, applied_fingerprint=None, last_job_id=job_id)
                    return

                spans_payload: list[dict[str, Any]] = []
                for g in (n.mentions or []):
                    for sp in getattr(g, "spans", []) or []:
                        ver = getattr(sp, "verification", None)
                        spans_payload.append(
                            {
                                "doc_id": getattr(sp, "doc_id", None),
                                "insertion_method": getattr(sp, "insertion_method", None),
                                "page": getattr(sp, "page_number", None),
                                "sc": getattr(sp, "start_char", None),
                                "ec": getattr(sp, "end_char", None),
                                "method": getattr(ver, "method", None),
                                "is_verified": getattr(ver, "is_verified", None),
                                "score": getattr(ver, "score", None),
                            }
                        )

                desired_fp = _fp(spans_payload)
                if op != "DELETE" and applied_fp is not None and applied_fp == desired_fp:
                    if _actual_fp() == desired_fp:
                        return

                self.engine.write.index_node_refs(n)
                if callable(set_applied):
                    set_applied(namespace=namespace, coalesce_key=coalesce_key, applied_fingerprint=desired_fp, last_job_id=job_id)
                return

            return

        if entity_kind == "edge":
            if index_kind == "edge_refs":
                if op == "DELETE":
                    self.engine.write.delete_edge_ref_rows(entity_id)
                    if callable(set_applied):
                        set_applied(namespace=namespace, coalesce_key=coalesce_key, applied_fingerprint=None, last_job_id=job_id)
                    return

                got = self.engine.backend.edge_get(ids=[entity_id], include=["documents", "metadatas"])
                docs = got.get("documents") or []
                if not docs or not docs[0]:
                    raise Exception("document not found")
                e = Edge.model_validate_json(docs[0])

                meta0 = (got.get("metadatas") or [None])[0] or {}
                if _is_tombstoned(meta0):
                    self.engine.write.delete_edge_ref_rows(entity_id)
                    if callable(set_applied):
                        set_applied(namespace=namespace, coalesce_key=coalesce_key, applied_fingerprint=None, last_job_id=job_id)
                    return

                spans_payload: list[dict[str, Any]] = []
                for g in (e.mentions or []):
                    for sp in getattr(g, "spans", []) or []:
                        ver = getattr(sp, "verification", None)
                        spans_payload.append(
                            {
                                "doc_id": getattr(sp, "doc_id", None),
                                "insertion_method": getattr(sp, "insertion_method", None),
                                "page": getattr(sp, "page_number", None),
                                "sc": getattr(sp, "start_char", None),
                                "ec": getattr(sp, "end_char", None),
                                "method": getattr(ver, "method", None),
                                "is_verified": getattr(ver, "is_verified", None),
                                "score": getattr(ver, "score", None),
                            }
                        )

                desired_fp = _fp(spans_payload)
                if op != "DELETE" and applied_fp is not None and applied_fp == desired_fp:
                    if _actual_fp() == desired_fp:
                        return

                self.engine.write.index_edge_refs(e)
                if callable(set_applied):
                    set_applied(namespace=namespace, coalesce_key=coalesce_key, applied_fingerprint=desired_fp, last_job_id=job_id)
                return

            if index_kind == "edge_endpoints":
                if op == "DELETE":
                    got = self.engine.backend.edge_endpoints_get(where={"edge_id": entity_id}, include=[])
                    ids = got.get("ids") or []
                    if ids:
                        self.engine.backend.edge_endpoints_delete(ids=ids)
                    if callable(set_applied):
                        set_applied(namespace=namespace, coalesce_key=coalesce_key, applied_fingerprint=None, last_job_id=job_id)
                    return

                got = self.engine.backend.edge_get(ids=[entity_id], include=["documents", "metadatas"])
                docs = got.get("documents") or []
                if not docs or not docs[0]:
                    raise Exception("document not found")
                e = Edge.model_validate_json(docs[0])

                meta = (got.get("metadatas") or [None])[0] or {}
                doc_id = meta.get("doc_id") if isinstance(meta, dict) else None

                # delete existing
                existing = self.engine.backend.edge_endpoints_get(where={"edge_id": entity_id}, include=[])
                ex_ids = existing.get("ids") or []
                if ex_ids:
                    self.engine.backend.edge_endpoints_delete(ids=ex_ids)

                rows = self.engine.write.fanout_endpoints_rows(e, doc_id)
                if not rows:
                    raise Exception("endpoints not found")
                desired_fp = _fp(sorted(rows, key=lambda r: r.get("id") or ""))

                meta0 = (got.get("metadatas") or [None])[0] or {}
                if _is_tombstoned(meta0):
                    got2 = self.engine.backend.edge_endpoints_get(where={"edge_id": entity_id}, include=[])
                    ids2 = got2.get("ids") or []
                    if ids2:
                        self.engine.backend.edge_endpoints_delete(ids=ids2)
                    if callable(set_applied):
                        set_applied(namespace=namespace, coalesce_key=coalesce_key, applied_fingerprint=None, last_job_id=job_id)
                    return

                if op != "DELETE" and applied_fp is not None and applied_fp == desired_fp:
                    if _actual_fp() == desired_fp:
                        return

                ep_ids = [r["id"] for r in rows]
                ep_docs = [json.dumps(r) for r in rows]
                ep_metas = rows
                self.engine.backend.edge_endpoints_add(
                    ids=ep_ids,
                    documents=ep_docs,
                    metadatas=ep_metas,
                    embeddings=[self.engine.embed.iterative_defensive_emb(str(d)) for d in ep_docs],
                )
                if callable(set_applied):
                    set_applied(namespace=namespace, coalesce_key=coalesce_key, applied_fingerprint=desired_fp, last_job_id=job_id)
                return
