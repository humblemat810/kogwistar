from __future__ import annotations

import hashlib
import json
import uuid
import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from .async_compat import run_awaitable_blocking
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

    def _profile_step(self, label: str, started_s: float) -> None:
        hook = getattr(self, "_profile_hook", None)
        if callable(hook):
            hook(str(label), time.perf_counter() - float(started_s))

    def enqueue_index_jobs_for_node(self, node_id: str, *, op: str) -> None:
        for idx in ("node_docs", "node_refs"):
            self.enqueue_index_job(
                entity_kind="node", entity_id=node_id, index_kind=idx, op=op
            )

    def enqueue_index_jobs_for_edge(self, edge_id: str, *, op: str) -> None:
        for idx in ("edge_refs", "edge_endpoints"):
            self.enqueue_index_job(
                entity_kind="edge", entity_id=edge_id, index_kind=idx, op=op
            )

    def reconcile_indexes(
        self,
        *,
        max_jobs: int = 100,
        lease_seconds: int = 60,
        namespace: str | None = None,
        use_validation_cache: bool | None = None,
    ) -> int:
        """Claim runnable index jobs, apply them, and record terminal state.

        Successful jobs are marked DONE after apply_index_job returns. Failures are
        delayed and requeued through bump_retry_and_requeue until max_retries is
        exhausted; only then do rows become terminal FAILED. This reconciler repairs
        derived indexes after base entities exist and does not make invalid ingest
        ordering succeed on its own.
        """
        if not getattr(self.engine, "_phase1_enable_index_jobs", False):
            return 0
        claim = getattr(self.engine.meta_sqlite, "claim_index_jobs", None)
        if claim is None:
            return 0

        ns = self.engine.namespace if namespace is None else namespace
        claim_started = time.perf_counter()
        jobs = claim(limit=max_jobs, lease_seconds=lease_seconds, namespace=ns)
        self._profile_step("reconcile.claim_index_jobs", claim_started)

        applied = 0
        effective_cache = (
            getattr(self.engine, "_phase1_enable_validation_cache", True)
            if use_validation_cache is None
            else bool(use_validation_cache)
        )
        entity_cache: dict[tuple[str, str, str], Any] | None = {} if effective_cache else None
        for job in jobs:
            # EngineSQLite returns IndexJobRow; PG meta might return dict.
            job_id = getattr(job, "job_id", None) or (
                job.get("job_id") if isinstance(job, dict) else None
            )
            entity_kind = getattr(job, "entity_kind", None) or (
                job.get("entity_kind") if isinstance(job, dict) else None
            )
            entity_id = getattr(job, "entity_id", None) or (
                job.get("entity_id") if isinstance(job, dict) else None
            )
            index_kind = getattr(job, "index_kind", None) or (
                job.get("index_kind") if isinstance(job, dict) else None
            )
            op = getattr(job, "op", None) or (
                job.get("op") if isinstance(job, dict) else None
            )

            retry_count = (
                getattr(job, "retry_count", None)
                if not isinstance(job, dict)
                else job.get("retry_count")
            )
            max_retries = (
                getattr(job, "max_retries", None)
                if not isinstance(job, dict)
                else job.get("max_retries")
            )
            try_rc = int(retry_count or 0)
            try_mr = int(max_retries or 10)

            job_started = time.perf_counter()
            try:
                self.apply_index_job(
                    job_id=str(job_id),
                    entity_kind=str(entity_kind),
                    entity_id=str(entity_id),
                    index_kind=str(index_kind),
                    op=str(op),
                    namespace=ns,
                    validated_entity_cache=entity_cache,
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                bump = getattr(self.engine.meta_sqlite, "bump_retry_and_requeue", None)
                mark_failed = getattr(
                    self.engine.meta_sqlite, "mark_index_job_failed", None
                )

                if job_id:
                    next_retry = try_rc + 1
                    if bump is not None and next_retry < try_mr:
                        delay = min(300, 2 ** min(next_retry - 1, 8))
                        bump_started = time.perf_counter()
                        bump(str(job_id), err, next_run_at_seconds=int(delay))
                        self._profile_step("reconcile.bump_retry_and_requeue", bump_started)
                    elif mark_failed is not None:
                        fail_started = time.perf_counter()
                        mark_failed(str(job_id), err, final=True)
                        self._profile_step("reconcile.mark_index_job_failed", fail_started)
                self._profile_step("reconcile.job_total", job_started)
                continue

            mark_done = getattr(self.engine.meta_sqlite, "mark_index_job_done", None)
            if mark_done is not None and job_id:
                done_started = time.perf_counter()
                mark_done(str(job_id))
                self._profile_step("reconcile.mark_index_job_done", done_started)
            self._profile_step("reconcile.job_total", job_started)
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
        from ..workers.index_job_worker import IndexJobWorker

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
        validated_entity_cache: dict[tuple[str, str, str], Any] | None = None,
    ) -> None:
        """
        Bring one derived index projection into sync with the authoritative entity.

        Jobs operate only on phase1 join indexes and are idempotent: applied-state
        fingerprints plus actual-row fingerprints let the method skip already-synced
        work. DELETE removes derived rows, tombstoned entities clear their
        projections, and missing base rows raise so the caller can retry with
        backoff instead of synthesizing state for entities that do not exist yet.
        """
        if index_kind not in self._PHASE1_JOIN_INDEX_KINDS:
            return

        # Phase 2: fingerprints + drift detection
        coalesce_key = f"{entity_kind}:{entity_id}:{index_kind}"
        get_applied = getattr(
            self.engine.meta_sqlite, "get_index_applied_fingerprint", None
        )
        set_applied = getattr(
            self.engine.meta_sqlite, "set_index_applied_fingerprint", None
        )

        def _fp(obj: object) -> str:
            blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            return hashlib.blake2b(blob, digest_size=16).hexdigest()

        def _stable_sort_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return sorted(
                rows,
                key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")),
            )

        def _as_dict(meta: Any) -> dict[str, Any]:
            return meta if isinstance(meta, dict) else {}

        def _drop_none_values(row: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in row.items() if v is not None}

        def _emit(label: str, started_s: float) -> None:
            self._profile_step(label, started_s)

        def _validated_entity(
            *,
            raw_json: str,
            cache_key: tuple[str, str, str],
            parser: Any,
            timing_label: str,
        ) -> Any:
            cache = validated_entity_cache
            if isinstance(cache, dict):
                cached = cache.get(cache_key)
                if cached is not None:
                    if hasattr(cached, "model_copy"):
                        return cached.model_copy(deep=True)
                    return cached

            started = time.perf_counter()
            obj = parser(raw_json)
            _emit(timing_label, started)
            if isinstance(cache, dict):
                cache[cache_key] = obj
            if hasattr(obj, "model_copy"):
                return obj.model_copy(deep=True)
            return obj

        def _backend_call(fn, *args, **kwargs):
            return run_awaitable_blocking(fn(*args, **kwargs))

        def _actual_payload() -> list[Any]:
            if entity_kind == "node":
                if index_kind == "node_docs":
                    started = time.perf_counter()
                    got = _backend_call(
                        self.engine.backend.node_docs_get,
                        where={"node_id": entity_id},
                        include=["metadatas"],
                    )
                    _emit("apply.node_docs.actual_payload_get", started)
                    doc_ids = [
                        str(md.get("doc_id"))
                        for md in (got.get("metadatas") or [])
                        if isinstance(md, dict) and md.get("doc_id") is not None
                    ]
                    return sorted(doc_ids)
                if index_kind == "node_refs":
                    started = time.perf_counter()
                    got = _backend_call(
                        self.engine.backend.node_refs_get,
                        where={"node_id": entity_id},
                        include=["metadatas"],
                    )
                    _emit("apply.node_refs.actual_payload_get", started)
                    payload = []
                    for md in got.get("metadatas") or []:
                        meta = _as_dict(md)
                        payload.append(
                            {
                                "doc_id": meta.get("doc_id"),
                                "insertion_method": meta.get("insertion_method"),
                                "page": meta.get("page_number"),
                                "sc": meta.get("start_char"),
                                "ec": meta.get("end_char"),
                                "method": meta.get("verification_method"),
                                "is_verified": meta.get("is_verified"),
                                "score": meta.get("verificication_score"),
                            }
                        )
                    return _stable_sort_dicts(payload)
            elif entity_kind == "edge":
                if index_kind == "edge_refs":
                    started = time.perf_counter()
                    got = _backend_call(
                        self.engine.backend.edge_refs_get,
                        where={"edge_id": entity_id},
                        include=["metadatas"],
                    )
                    _emit("apply.edge_refs.actual_payload_get", started)
                    payload = []
                    for md in got.get("metadatas") or []:
                        meta = _as_dict(md)
                        payload.append(
                            {
                                "doc_id": meta.get("doc_id"),
                                "insertion_method": meta.get("insertion_method"),
                                "page": meta.get("page_number"),
                                "sc": meta.get("start_char"),
                                "ec": meta.get("end_char"),
                                "method": meta.get("verification_method"),
                                "is_verified": meta.get("is_verified"),
                                "score": meta.get("verificication_score"),
                            }
                        )
                    return _stable_sort_dicts(payload)
                if index_kind == "edge_endpoints":
                    started = time.perf_counter()
                    got = _backend_call(
                        self.engine.backend.edge_endpoints_get,
                        where={"edge_id": entity_id},
                        include=["metadatas"],
                    )
                    _emit("apply.edge_endpoints.actual_payload_get", started)
                    payload = []
                    for md in got.get("metadatas") or []:
                        meta = _as_dict(md)
                        payload.append(
                            _drop_none_values(
                                {
                                    "id": meta.get("id"),
                                    "edge_id": meta.get("edge_id"),
                                    "endpoint_id": meta.get("endpoint_id"),
                                    "endpoint_type": meta.get("endpoint_type"),
                                    "role": meta.get("role"),
                                    "causal_type": meta.get("causal_type"),
                                    "relation": meta.get("relation"),
                                    "doc_id": meta.get("doc_id"),
                                }
                            )
                        )
                    return _stable_sort_dicts(payload)
            return []

        def _actual_fp() -> str:
            return _fp(_actual_payload())

        started = time.perf_counter()
        applied_fp = (
            self.engine.meta_sqlite.get_index_applied_fingerprint(
                namespace=namespace, coalesce_key=coalesce_key
            )
            if callable(get_applied)
            else None
        )
        _emit("apply.applied_fp_get", started)

        if entity_kind == "node":
            if index_kind == "node_docs":
                if op == "DELETE":
                    started = time.perf_counter()
                    _backend_call(
                        self.engine.backend.node_docs_delete,
                        where={"node_id": entity_id},
                    )
                    _emit("apply.node_docs.delete", started)
                    if callable(set_applied):
                        started = time.perf_counter()
                        set_applied(
                            namespace=namespace,
                            coalesce_key=coalesce_key,
                            applied_fingerprint=None,
                            last_job_id=job_id,
                        )
                        _emit("apply.applied_fp_set", started)
                    return

                started = time.perf_counter()
                got = _backend_call(
                    self.engine.backend.node_get,
                    ids=[entity_id],
                    include=["documents", "metadatas"],
                )
                _emit("apply.node_docs.base_get", started)
                docs = got.get("documents") or []
                if not docs or not docs[0]:
                    raise Exception("document not found")
                n = _validated_entity(
                    raw_json=docs[0],
                    cache_key=("node", entity_id, docs[0]),
                    parser=Node.model_validate_json,
                    timing_label="apply.node_docs.model_validate",
                )

                meta0 = (got.get("metadatas") or [None])[0] or {}
                started = time.perf_counter()
                if _is_tombstoned(meta0):
                    _emit("apply.node_docs.tombstone_check", started)
                    _backend_call(
                        self.engine.backend.node_docs_delete,
                        where={"node_id": entity_id},
                    )
                    if callable(set_applied):
                        started = time.perf_counter()
                        set_applied(
                            namespace=namespace,
                            coalesce_key=coalesce_key,
                            applied_fingerprint=None,
                            last_job_id=job_id,
                        )
                        _emit("apply.applied_fp_set", started)
                    return
                _emit("apply.node_docs.tombstone_check", started)

                started = time.perf_counter()
                desired_fp = _fp(_extract_doc_ids_from_refs(n.mentions))
                _emit("apply.node_docs.desired_fp", started)
                if (
                    op != "DELETE"
                    and applied_fp is not None
                    and applied_fp == desired_fp
                ):
                    started = time.perf_counter()
                    if _actual_fp() == desired_fp:
                        _emit("apply.node_docs.actual_fp_check", started)
                        return
                    _emit("apply.node_docs.actual_fp_check", started)

                started = time.perf_counter()
                self.engine.write.index_node_docs(n)
                _emit("apply.node_docs.write", started)
                if callable(set_applied):
                    started = time.perf_counter()
                    set_applied(
                        namespace=namespace,
                        coalesce_key=coalesce_key,
                        applied_fingerprint=desired_fp,
                        last_job_id=job_id,
                    )
                    _emit("apply.applied_fp_set", started)
                return

            if index_kind == "node_refs":
                if op == "DELETE":
                    started = time.perf_counter()
                    self.engine.write.delete_node_ref_rows(entity_id)
                    _emit("apply.node_refs.delete", started)
                    if callable(set_applied):
                        started = time.perf_counter()
                        set_applied(
                            namespace=namespace,
                            coalesce_key=coalesce_key,
                            applied_fingerprint=None,
                            last_job_id=job_id,
                        )
                        _emit("apply.applied_fp_set", started)
                    return

                started = time.perf_counter()
                got = _backend_call(
                    self.engine.backend.node_get,
                    ids=[entity_id],
                    include=["documents", "metadatas"],
                )
                _emit("apply.node_refs.base_get", started)
                docs = got.get("documents") or []
                if not docs or not docs[0]:
                    raise Exception("document not found")
                n = _validated_entity(
                    raw_json=docs[0],
                    cache_key=("node", entity_id, docs[0]),
                    parser=Node.model_validate_json,
                    timing_label="apply.node_refs.model_validate",
                )

                meta0 = (got.get("metadatas") or [None])[0] or {}
                started = time.perf_counter()
                if _is_tombstoned(meta0):
                    _emit("apply.node_refs.tombstone_check", started)
                    self.engine.write.delete_node_ref_rows(entity_id)
                    if callable(set_applied):
                        started = time.perf_counter()
                        set_applied(
                            namespace=namespace,
                            coalesce_key=coalesce_key,
                            applied_fingerprint=None,
                            last_job_id=job_id,
                        )
                        _emit("apply.applied_fp_set", started)
                    return
                _emit("apply.node_refs.tombstone_check", started)

                spans_payload: list[dict[str, Any]] = []
                started = time.perf_counter()
                for g in n.mentions or []:
                    for sp in getattr(g, "spans", []) or []:
                        ver = getattr(sp, "verification", None)
                        spans_payload.append(
                            {
                                "doc_id": getattr(sp, "doc_id", None),
                                "insertion_method": getattr(
                                    sp, "insertion_method", None
                                ),
                                "page": getattr(sp, "page_number", None),
                                "sc": getattr(sp, "start_char", None),
                                "ec": getattr(sp, "end_char", None),
                                "method": getattr(ver, "method", None),
                                "is_verified": getattr(ver, "is_verified", None),
                                "score": getattr(ver, "score", None),
                            }
                        )

                spans_payload = _stable_sort_dicts(spans_payload)
                desired_fp = _fp(spans_payload)
                _emit("apply.node_refs.desired_fp", started)
                if (
                    op != "DELETE"
                    and applied_fp is not None
                    and applied_fp == desired_fp
                ):
                    started = time.perf_counter()
                    if _actual_fp() == desired_fp:
                        _emit("apply.node_refs.actual_fp_check", started)
                        return
                    _emit("apply.node_refs.actual_fp_check", started)

                started = time.perf_counter()
                self.engine.write.index_node_refs(n)
                _emit("apply.node_refs.write", started)
                if callable(set_applied):
                    started = time.perf_counter()
                    set_applied(
                        namespace=namespace,
                        coalesce_key=coalesce_key,
                        applied_fingerprint=desired_fp,
                        last_job_id=job_id,
                    )
                    _emit("apply.applied_fp_set", started)
                return

            return

        if entity_kind == "edge":
            if index_kind == "edge_refs":
                if op == "DELETE":
                    started = time.perf_counter()
                    self.engine.write.delete_edge_ref_rows(entity_id)
                    _emit("apply.edge_refs.delete", started)
                    if callable(set_applied):
                        started = time.perf_counter()
                        set_applied(
                            namespace=namespace,
                            coalesce_key=coalesce_key,
                            applied_fingerprint=None,
                            last_job_id=job_id,
                        )
                        _emit("apply.applied_fp_set", started)
                    return

                started = time.perf_counter()
                got = _backend_call(
                    self.engine.backend.edge_get,
                    ids=[entity_id],
                    include=["documents", "metadatas"],
                )
                _emit("apply.edge_refs.base_get", started)
                docs = got.get("documents") or []
                if not docs or not docs[0]:
                    raise Exception("document not found")
                e = _validated_entity(
                    raw_json=docs[0],
                    cache_key=("edge", entity_id, docs[0]),
                    parser=Edge.model_validate_json,
                    timing_label="apply.edge_refs.model_validate",
                )

                meta0 = (got.get("metadatas") or [None])[0] or {}
                started = time.perf_counter()
                if _is_tombstoned(meta0):
                    _emit("apply.edge_refs.tombstone_check", started)
                    self.engine.write.delete_edge_ref_rows(entity_id)
                    if callable(set_applied):
                        started = time.perf_counter()
                        set_applied(
                            namespace=namespace,
                            coalesce_key=coalesce_key,
                            applied_fingerprint=None,
                            last_job_id=job_id,
                        )
                        _emit("apply.applied_fp_set", started)
                    return
                _emit("apply.edge_refs.tombstone_check", started)

                spans_payload: list[dict[str, Any]] = []
                started = time.perf_counter()
                for g in e.mentions or []:
                    for sp in getattr(g, "spans", []) or []:
                        ver = getattr(sp, "verification", None)
                        spans_payload.append(
                            {
                                "doc_id": getattr(sp, "doc_id", None),
                                "insertion_method": getattr(
                                    sp, "insertion_method", None
                                ),
                                "page": getattr(sp, "page_number", None),
                                "sc": getattr(sp, "start_char", None),
                                "ec": getattr(sp, "end_char", None),
                                "method": getattr(ver, "method", None),
                                "is_verified": getattr(ver, "is_verified", None),
                                "score": getattr(ver, "score", None),
                            }
                        )

                spans_payload = _stable_sort_dicts(spans_payload)
                desired_fp = _fp(spans_payload)
                _emit("apply.edge_refs.desired_fp", started)
                if (
                    op != "DELETE"
                    and applied_fp is not None
                    and applied_fp == desired_fp
                ):
                    started = time.perf_counter()
                    if _actual_fp() == desired_fp:
                        _emit("apply.edge_refs.actual_fp_check", started)
                        return
                    _emit("apply.edge_refs.actual_fp_check", started)

                started = time.perf_counter()
                self.engine.write.index_edge_refs(e)
                _emit("apply.edge_refs.write", started)
                if callable(set_applied):
                    started = time.perf_counter()
                    set_applied(
                        namespace=namespace,
                        coalesce_key=coalesce_key,
                        applied_fingerprint=desired_fp,
                        last_job_id=job_id,
                    )
                    _emit("apply.applied_fp_set", started)
                return

            if index_kind == "edge_endpoints":
                if op == "DELETE":
                    started = time.perf_counter()
                    got = _backend_call(
                        self.engine.backend.edge_endpoints_get,
                        where={"edge_id": entity_id},
                        include=[],
                    )
                    _emit("apply.edge_endpoints.delete_lookup", started)
                    ids = got.get("ids") or []
                    if ids:
                        started = time.perf_counter()
                        _backend_call(self.engine.backend.edge_endpoints_delete, ids=ids)
                        _emit("apply.edge_endpoints.delete", started)
                    if callable(set_applied):
                        started = time.perf_counter()
                        set_applied(
                            namespace=namespace,
                            coalesce_key=coalesce_key,
                            applied_fingerprint=None,
                            last_job_id=job_id,
                        )
                        _emit("apply.applied_fp_set", started)
                    return

                started = time.perf_counter()
                got = _backend_call(
                    self.engine.backend.edge_get,
                    ids=[entity_id],
                    include=["documents", "metadatas"],
                )
                _emit("apply.edge_endpoints.base_get", started)
                docs = got.get("documents") or []
                if not docs or not docs[0]:
                    raise Exception("document not found")
                e = _validated_entity(
                    raw_json=docs[0],
                    cache_key=("edge", entity_id, docs[0]),
                    parser=Edge.model_validate_json,
                    timing_label="apply.edge_endpoints.model_validate",
                )

                meta = (got.get("metadatas") or [None])[0] or {}
                doc_id = meta.get("doc_id") if isinstance(meta, dict) else None

                meta0 = (got.get("metadatas") or [None])[0] or {}
                started = time.perf_counter()
                if _is_tombstoned(meta0):
                    _emit("apply.edge_endpoints.tombstone_check", started)
                    got2 = _backend_call(
                        self.engine.backend.edge_endpoints_get,
                        where={"edge_id": entity_id},
                        include=[],
                    )
                    delete_lookup_started = time.perf_counter()
                    ids2 = got2.get("ids") or []
                    if ids2:
                        _backend_call(self.engine.backend.edge_endpoints_delete, ids=ids2)
                    _emit("apply.edge_endpoints.delete_existing", delete_lookup_started)
                    if callable(set_applied):
                        started = time.perf_counter()
                        set_applied(
                            namespace=namespace,
                            coalesce_key=coalesce_key,
                            applied_fingerprint=None,
                            last_job_id=job_id,
                        )
                        _emit("apply.applied_fp_set", started)
                    return
                _emit("apply.edge_endpoints.tombstone_check", started)

                started = time.perf_counter()
                rows = self.engine.write.fanout_endpoints_rows(e, doc_id)
                _emit("apply.edge_endpoints.fanout_rows", started)
                if not rows:
                    raise Exception("endpoints not found")
                rows = _stable_sort_dicts(rows)
                desired_fp = _fp(rows)
                _emit("apply.edge_endpoints.desired_fp", started)

                if (
                    op != "DELETE"
                    and applied_fp is not None
                    and applied_fp == desired_fp
                ):
                    started = time.perf_counter()
                    if _actual_fp() == desired_fp:
                        _emit("apply.edge_endpoints.actual_fp_check", started)
                        return
                    _emit("apply.edge_endpoints.actual_fp_check", started)

                started = time.perf_counter()
                existing = _backend_call(
                    self.engine.backend.edge_endpoints_get,
                    where={"edge_id": entity_id},
                    include=["metadatas"],
                )
                _emit("apply.edge_endpoints.existing_get", started)
                existing_ids = {
                    str(i) for i in (existing.get("ids") or []) if i is not None
                }
                desired_ids = {str(r["id"]) for r in rows}
                stale_ids = sorted(existing_ids - desired_ids)
                if stale_ids:
                    started = time.perf_counter()
                    _backend_call(self.engine.backend.edge_endpoints_delete, ids=stale_ids)
                    _emit("apply.edge_endpoints.delete_stale", started)

                ep_ids = [r["id"] for r in rows]
                ep_docs = [json.dumps(r) for r in rows]
                ep_metas = rows
                started = time.perf_counter()
                _backend_call(
                    self.engine.backend.edge_endpoints_upsert,
                    ids=ep_ids,
                    documents=ep_docs,
                    metadatas=ep_metas,
                    embeddings=[
                        self.engine.embed.iterative_defensive_emb(str(d))
                        for d in ep_docs
                    ],
                )
                _emit("apply.edge_endpoints.upsert", started)
                if callable(set_applied):
                    started = time.perf_counter()
                    set_applied(
                        namespace=namespace,
                        coalesce_key=coalesce_key,
                        applied_fingerprint=desired_fp,
                        last_job_id=job_id,
                    )
                    _emit("apply.applied_fp_set", started)
                return
