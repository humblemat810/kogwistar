"""SQLite staging plus Chroma semantic projection for ADR-018.

The adapter is intentionally narrow: SQLite owns only short-lived Stage-1
rows, while Chroma remains the Stage-2 vector projection. Canonical events and
their revision fingerprint decide whether a promotion is still current.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, cast

from .async_compat import run_awaitable_blocking
from .canonical_events import canonical_revision_job_payload
from .edge_endpoint_rows import edge_endpoint_rows
from .engine import GraphKnowledgeEngine
from .models import Edge, Node
from .storage_backend import TwoStageProjectionCapability
from .two_stage_namespace import active_graph_namespace


class _Stage1MetaStore(Protocol):
    def replace_stage1_node_projection(self, **kwargs: object) -> object: ...

    def query_stage1_node_projections(
        self, *args: object, **kwargs: object
    ) -> Iterable[Mapping[str, object]]: ...

    def clear_stage1_node_projection(self, *args: object, **kwargs: object) -> object: ...

    def get_stage1_node_projection(
        self, *args: object, **kwargs: object
    ) -> Mapping[str, object] | None: ...


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _objects(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _embedding(value: object) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if not all(isinstance(item, (int, float)) for item in value):
        return None
    return [float(item) for item in value]


def _sequence_number(value: object) -> int:
    return value if isinstance(value, int) else 0


def _job_value(job: object, name: str) -> object:
    if isinstance(job, Mapping):
        return job.get(name)
    return getattr(job, name, None)


def chroma_two_stage_capability() -> TwoStageProjectionCapability:
    return TwoStageProjectionCapability(
        supports_two_stage=True,
        canonical_event_replay=True,
        canonical_read=True,
        stage1_strategy="transient_projection",
        stage1_metadata_query=True,
        stage1_cleanup=True,
        stage2_semantic_projection=True,
        revision_gated_promotion=True,
        semantic_readiness_gate=True,
        delete_reconciliation=True,
        atomic_promotion="eventual_reconcile",
        reason="SQLite transient Stage-1 with Chroma Stage-2 projection",
    )


class SQLiteChromaTwoStageProjectionAdapter:
    """Concrete Chroma arrangement using the existing SQLite Stage-1 table."""

    def __init__(self, engine: GraphKnowledgeEngine) -> None:
        self.engine = engine

    def _meta(self) -> _Stage1MetaStore:
        return cast(_Stage1MetaStore, self.engine.metadata)

    def _namespace(self) -> str:
        return active_graph_namespace(self.engine)

    @staticmethod
    def _stage1_key(entity_kind: str, entity_id: str) -> str:
        return f"{entity_kind}:{entity_id}"

    def _stage1_payload(
        self, *, entity_kind: str, entity: Node | Edge
    ) -> dict[str, Any]:
        if entity_kind == "node":
            document, metadata = self.engine.write.node_doc_and_meta(cast(Node, entity))
        else:
            edge = cast(Edge, entity)
            document = edge.model_dump_json(field_mode="backend", exclude=["embedding"])
            metadata = self.engine.write.enrich_edge_meta(edge)
        revision_payload = self.engine.indexing.canonical_revision_payload(
            entity_kind=entity_kind, entity_id=entity.safe_get_id()
        )
        fingerprint = json.loads(revision_payload).get("source_fingerprint")
        return {
            "id": entity.safe_get_id(),
            "entity_kind": entity_kind,
            "document": str(document),
            "metadata": dict(metadata or {}),
            "source_fingerprint": str(fingerprint or ""),
        }

    def _stage1_upsert(self, *, entity_kind: str, entity: Node | Edge) -> None:
        payload = self._stage1_payload(entity_kind=entity_kind, entity=entity)
        revision = self.engine.indexing.canonical_entity_revision(
            entity_kind=entity_kind, entity_id=entity.safe_get_id()
        )
        seq = int(getattr(revision, "revision", 0) if revision is not None else 0)
        replace = getattr(self._meta(), "replace_stage1_node_projection", None)
        if not callable(replace):
            raise RuntimeError("Chroma two-stage arrangement requires SQLite Stage-1 storage")
        replace(
            self._namespace(),
            self._stage1_key(entity_kind, entity.safe_get_id()),
            payload,
            last_authoritative_seq=seq,
            last_materialized_seq=seq,
            projection_schema_version=1,
            materialization_status="pending",
        )

    def restore_stage1_from_canonical_event(
        self, *, entity_kind: str, entity_id: str
    ) -> bool:
        """Rebuild missing staging from one revalidated canonical event snapshot."""
        revision = self.engine.indexing.canonical_entity_revision(
            entity_kind=entity_kind, entity_id=entity_id
        )
        if revision is None or revision.state != "active":
            return False
        if entity_kind == "node":
            entity: Node | Edge = Node.model_validate(revision.payload)
            document, metadata = self.engine.write.node_doc_and_meta(entity)
        elif entity_kind == "edge":
            entity = Edge.model_validate(revision.payload)
            document = entity.model_dump_json(
                field_mode="backend", exclude=["embedding"]
            )
            metadata = self.engine.write.enrich_edge_meta(entity)
        else:
            raise ValueError(f"unsupported two-stage entity kind: {entity_kind!r}")

        revision_payload = json.loads(canonical_revision_job_payload(revision))
        source_fingerprint = str(revision_payload["source_fingerprint"])
        payload = {
            "id": entity_id,
            "entity_kind": entity_kind,
            "document": str(document),
            "metadata": dict(metadata or {}),
            "source_fingerprint": source_fingerprint,
        }
        replace = getattr(self._meta(), "replace_stage1_node_projection", None)
        if not callable(replace):
            raise RuntimeError("Chroma two-stage arrangement requires SQLite Stage-1 storage")
        replace(
            self._namespace(),
            self._stage1_key(entity_kind, entity_id),
            payload,
            last_authoritative_seq=int(revision.revision),
            last_materialized_seq=int(revision.revision),
            projection_schema_version=1,
            materialization_status="pending",
        )

        latest = self.engine.indexing.canonical_entity_revision(
            entity_kind=entity_kind, entity_id=entity_id
        )
        if latest is None or latest.revision != revision.revision:
            self.remove_stage1(
                entity_kind=entity_kind,
                entity_id=entity_id,
                expected_last_authoritative_seq=int(revision.revision),
            )
            return False
        return True

    def _enqueue(self, *, entity_kind: str, entity_id: str, op: str) -> None:
        self.engine.indexing.enqueue_index_job(
            entity_kind=entity_kind,
            entity_id=entity_id,
            index_kind="node_embedding",
            op=op,
            payload_json=self.engine.indexing.canonical_revision_payload(
                entity_kind=entity_kind, entity_id=entity_id
            ),
        )

    def _restore_current_stage1(self, *, entity_kind: str, entity_id: str) -> bool:
        """Make stale-job replay schedule the current canonical projection."""
        revision = self.engine.indexing.canonical_entity_revision(
            entity_kind=entity_kind, entity_id=entity_id
        )
        if revision is None or revision.state != "active":
            self._delete_projection_and_restore_if_active(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return False
        if not self.restore_stage1_from_canonical_event(
            entity_kind=entity_kind, entity_id=entity_id
        ):
            raise RuntimeError("current Chroma revision changed during Stage-1 recovery")
        self._enqueue(entity_kind=entity_kind, entity_id=entity_id, op="UPSERT")
        return True

    enqueue_embedding_job = _enqueue

    def add_node(self, node: Node, *, doc_id: str | None = None) -> None:
        if doc_id is not None:
            node.doc_id = doc_id
        # Remove any older semantic projection before exposing the new
        # transient revision. Promotion is a handoff, never dual residency.
        self.remove_stage2_or_invalidate(entity_kind="node", entity_id=node.safe_get_id())
        self._stage1_upsert(entity_kind="node", entity=node)

    def add_edge(self, edge: Edge, *, doc_id: str | None = None) -> None:
        if doc_id is not None:
            edge.doc_id = doc_id
        self.remove_stage2_or_invalidate(entity_kind="edge", entity_id=edge.safe_get_id())
        self._stage1_upsert(entity_kind="edge", entity=edge)

    def stage1_query(self, **kwargs: object) -> list[dict[str, Any]]:
        query = getattr(self._meta(), "query_stage1_node_projections", None)
        if not callable(query):
            raise RuntimeError("Chroma two-stage arrangement lacks Stage-1 query")
        return list(
            cast(
                Iterable[dict[str, Any]],
                query(self._namespace(), **kwargs),
            )
        )

    def remove_stage1(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        expected_last_authoritative_seq: int | None = None,
        **_: object,
    ) -> bool | None:
        if entity_kind not in {"node", "edge"}:
            raise ValueError(f"unsupported Stage-1 entity kind: {entity_kind!r}")
        meta = self._meta()
        if expected_last_authoritative_seq is not None:
            compare_clear = getattr(
                meta, "clear_stage1_node_projection_if_current", None
            )
            if not callable(compare_clear):
                raise RuntimeError(
                    "Chroma two-stage promotion requires revision-conditional Stage-1 cleanup"
                )
            return bool(
                compare_clear(
                    self._namespace(),
                    self._stage1_key(entity_kind, entity_id),
                    int(expected_last_authoritative_seq),
                )
            )
        clear = getattr(meta, "clear_stage1_node_projection", None)
        if not callable(clear):
            raise RuntimeError("Chroma two-stage arrangement lacks Stage-1 cleanup")
        clear(self._namespace(), self._stage1_key(entity_kind, entity_id))

    def _collection_call(
        self, entity_kind: str, method: str, **kwargs: object
    ) -> Mapping[str, object]:
        fn = getattr(self.engine.backend, f"{entity_kind}_{method}")
        return cast(Mapping[str, object], run_awaitable_blocking(fn(**kwargs)))

    def _edge_endpoint_rows(self, edge: Edge) -> list[dict[str, Any]]:
        return edge_endpoint_rows(edge)

    def _promote_edge_endpoints(
        self, edge: Edge, embedding: list[float] | None = None
    ) -> None:
        rows = self._edge_endpoint_rows(edge)
        if not rows:
            return
        if embedding is None:
            raise RuntimeError(
                "edge endpoint promotion requires the current edge embedding"
            )
        documents = [json.dumps(row) for row in rows]
        self._collection_call(
            "edge_endpoints",
            "upsert",
            ids=[row["id"] for row in rows],
            documents=documents,
            metadatas=rows,
            # Chroma requires vectors here; structural rows reuse the already
            # computed edge vector and never call the provider.
            embeddings=[list(embedding) for _ in rows],
        )

    def remove_stage2_or_invalidate(
        self, *, entity_kind: str, entity_id: str, **_: object
    ) -> None:
        self._collection_call(entity_kind, "delete", ids=[entity_id])
        if entity_kind == "edge":
            self._collection_call("edge_endpoints", "delete", where={"edge_id": entity_id})

    def _current_fingerprint(self, *, entity_kind: str, entity_id: str) -> str:
        revision_payload = self.engine.indexing.canonical_revision_payload(
            entity_kind=entity_kind, entity_id=entity_id
        )
        return str(json.loads(revision_payload).get("source_fingerprint") or "")

    def _delete_projection_and_restore_if_active(
        self, *, entity_kind: str, entity_id: str
    ) -> None:
        row = self._meta().get_stage1_node_projection(
            self._namespace(), self._stage1_key(entity_kind, entity_id)
        )
        if row is not None:
            self.remove_stage1(
                entity_kind=entity_kind,
                entity_id=entity_id,
                expected_last_authoritative_seq=_sequence_number(
                    row.get("last_authoritative_seq")
                ),
            )
        self.remove_stage2_or_invalidate(entity_kind=entity_kind, entity_id=entity_id)
        latest = self.engine.indexing.canonical_entity_revision(
            entity_kind=entity_kind, entity_id=entity_id
        )
        if latest is not None and latest.state == "active":
            if not self.restore_stage1_from_canonical_event(
                entity_kind=entity_kind, entity_id=entity_id
            ):
                raise RuntimeError("Chroma delete raced with resurrection and could not restore Stage-1")
            self._enqueue(entity_kind=entity_kind, entity_id=entity_id, op="UPSERT")

    def apply_embedding_job(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        op: str,
        payload_json: str | None,
    ) -> None:
        if entity_kind not in {"node", "edge"}:
            raise ValueError(f"unsupported two-stage entity kind: {entity_kind!r}")
        expected = str(json.loads(payload_json or "{}").get("source_fingerprint") or "")
        current = self.engine.indexing.canonical_entity_revision(
            entity_kind=entity_kind, entity_id=entity_id
        )
        if op.upper() == "DELETE":
            if (
                expected
                and current is not None
                and current.state == "active"
                and expected != self._current_fingerprint(
                    entity_kind=entity_kind, entity_id=entity_id
                )
            ):
                self._restore_current_stage1(
                    entity_kind=entity_kind, entity_id=entity_id
                )
                return
            self._delete_projection_and_restore_if_active(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return
        if current is None or current.state != "active":
            self._delete_projection_and_restore_if_active(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return
        if expected and expected != self._current_fingerprint(
            entity_kind=entity_kind, entity_id=entity_id
        ):
            self._restore_current_stage1(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return
        row = self._meta().get_stage1_node_projection(
            self._namespace(), self._stage1_key(entity_kind, entity_id)
        )
        if not row:
            self._restore_current_stage1(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return
        staged = _mapping(row.get("payload"))
        staged_seq = _sequence_number(row.get("last_authoritative_seq"))
        staged_fingerprint = str(staged.get("source_fingerprint") or "")
        if not staged_fingerprint:
            raise RuntimeError("Chroma Stage-1 projection has no source fingerprint")
        if (expected and staged_fingerprint != expected) or (
            staged_fingerprint != self._current_fingerprint(
                entity_kind=entity_kind, entity_id=entity_id
            )
        ):
            self._restore_current_stage1(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return
        document = str(staged.get("document") or "")
        embedding = self.engine.embed.iterative_defensive_emb(document)
        metadata = cast(dict[str, Any], staged.get("metadata") or {})
        metadata["_kogwistar_stage2_ready"] = True
        metadata["_kogwistar_source_fingerprint"] = staged_fingerprint
        metadata["_kogwistar_source_revision"] = staged_seq
        self._collection_call(
            entity_kind,
            "upsert",
            ids=[entity_id],
            documents=[document],
            metadatas=[metadata],
            embeddings=[embedding],
        )
        if entity_kind == "edge":
            from .models import Edge

            self._promote_edge_endpoints(Edge.model_validate_json(document), embedding)
        latest = self.engine.indexing.canonical_entity_revision(
            entity_kind=entity_kind, entity_id=entity_id
        )
        if latest is None or latest.state != "active":
            self.remove_stage1(
                entity_kind=entity_kind,
                entity_id=entity_id,
                expected_last_authoritative_seq=staged_seq,
            )
            self.remove_stage2_or_invalidate(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return
        if self._current_fingerprint(
            entity_kind=entity_kind, entity_id=entity_id
        ) != str(staged.get("source_fingerprint") or ""):
            self._restore_current_stage1(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return
        self.remove_stage1(
            entity_kind=entity_kind,
            entity_id=entity_id,
            expected_last_authoritative_seq=staged_seq,
        )

    def apply_embedding_jobs_batch(
        self, jobs: Sequence[object]
    ) -> dict[str, BaseException | None]:
        """Best-effort batch embedding with per-job promotion and failures."""
        prepared: list[tuple[str, str, str, str, str | None, dict[str, object]]] = []
        outcomes: dict[str, BaseException | None] = {}

        for job in jobs:
            job_id = str(_job_value(job, "job_id") or "")
            entity_kind = str(_job_value(job, "entity_kind") or "")
            entity_id = str(_job_value(job, "entity_id") or "")
            op = str(_job_value(job, "op") or "UPSERT")
            payload_json = cast(str | None, _job_value(job, "payload_json"))
            try:
                if op.upper() == "DELETE":
                    self.apply_embedding_job(
                        entity_kind=entity_kind,
                        entity_id=entity_id,
                        op=op,
                        payload_json=payload_json,
                    )
                    outcomes[job_id] = None
                    continue
                expected = str(
                    json.loads(payload_json or "{}").get("source_fingerprint") or ""
                )
                current = self.engine.indexing.canonical_entity_revision(
                    entity_kind=entity_kind, entity_id=entity_id
                )
                if current is None or current.state != "active":
                    self.apply_embedding_job(
                        entity_kind=entity_kind,
                        entity_id=entity_id,
                        op=op,
                        payload_json=payload_json,
                    )
                    outcomes[job_id] = None
                    continue
                if expected and expected != self._current_fingerprint(
                    entity_kind=entity_kind, entity_id=entity_id
                ):
                    self._restore_current_stage1(
                        entity_kind=entity_kind, entity_id=entity_id
                    )
                    outcomes[job_id] = None
                    continue
                row = self._meta().get_stage1_node_projection(
                    self._namespace(), self._stage1_key(entity_kind, entity_id)
                )
                if not row:
                    self._restore_current_stage1(
                        entity_kind=entity_kind, entity_id=entity_id
                    )
                    outcomes[job_id] = None
                    continue
                staged = _mapping(row.get("payload"))
                staged_fingerprint = str(staged.get("source_fingerprint") or "")
                if not staged_fingerprint:
                    raise RuntimeError("Chroma Stage-1 projection has no source fingerprint")
                if (expected and staged_fingerprint != expected) or (
                    staged_fingerprint != self._current_fingerprint(
                        entity_kind=entity_kind, entity_id=entity_id
                    )
                ):
                    self._restore_current_stage1(
                        entity_kind=entity_kind, entity_id=entity_id
                    )
                    outcomes[job_id] = None
                    continue
                staged["_stage1_authoritative_seq"] = int(
                    _sequence_number(row.get("last_authoritative_seq"))
                )
                prepared.append(
                    (job_id, entity_kind, entity_id, expected, payload_json, staged)
                )
            except BaseException as exc:
                outcomes[job_id] = exc

        if not prepared:
            return outcomes

        documents = [str(item[5].get("document") or "") for item in prepared]
        try:
            provider = getattr(self.engine, "_ef", None)
            if not callable(provider):
                raise RuntimeError("embedding provider has no batch interface")
            raw_embeddings = run_awaitable_blocking(provider(documents))
            embeddings = list(cast(Sequence[list[float]], raw_embeddings))
            if len(embeddings) != len(prepared):
                raise RuntimeError("embedding provider returned wrong batch length")
        except BaseException:
            # Provider batch failure must degrade to isolated jobs, not lose all.
            for job_id, entity_kind, entity_id, _, payload_json, _ in prepared:
                try:
                    self.apply_embedding_job(
                        entity_kind=entity_kind,
                        entity_id=entity_id,
                        op="UPSERT",
                        payload_json=payload_json,
                    )
                    outcomes[job_id] = None
                except BaseException as exc:
                    outcomes[job_id] = exc
            return outcomes

        for (job_id, entity_kind, entity_id, _, _, staged), embedding in zip(
            prepared, embeddings
        ):
            try:
                current = self.engine.indexing.canonical_entity_revision(
                    entity_kind=entity_kind, entity_id=entity_id
                )
                if current is None or current.state != "active":
                    continue
                staged_fingerprint = str(staged.get("source_fingerprint") or "")
                if staged_fingerprint != self._current_fingerprint(
                    entity_kind=entity_kind, entity_id=entity_id
                ):
                    outcomes[job_id] = None
                    continue
                metadata = cast(dict[str, Any], staged.get("metadata") or {})
                metadata["_kogwistar_stage2_ready"] = True
                metadata["_kogwistar_source_fingerprint"] = staged_fingerprint
                metadata["_kogwistar_source_revision"] = _sequence_number(
                    staged.get("_stage1_authoritative_seq")
                )
                self._collection_call(
                    entity_kind,
                    "upsert",
                    ids=[entity_id],
                    documents=[str(staged.get("document") or "")],
                    metadatas=[metadata],
                    embeddings=[embedding],
                )
                if entity_kind == "edge":
                    from .models import Edge

                    self._promote_edge_endpoints(
                        Edge.model_validate_json(str(staged["document"])), embedding
                    )
                latest = self.engine.indexing.canonical_entity_revision(
                    entity_kind=entity_kind, entity_id=entity_id
                )
                if latest is None or latest.state != "active":
                    self.remove_stage1(
                        entity_kind=entity_kind,
                        entity_id=entity_id,
                        expected_last_authoritative_seq=_sequence_number(
                            staged.get("_stage1_authoritative_seq")
                        ),
                    )
                    self.remove_stage2_or_invalidate(
                        entity_kind=entity_kind, entity_id=entity_id
                    )
                    outcomes[job_id] = None
                    continue
                if self._current_fingerprint(
                    entity_kind=entity_kind, entity_id=entity_id
                ) != str(staged.get("source_fingerprint") or ""):
                    if self.restore_stage1_from_canonical_event(
                        entity_kind=entity_kind, entity_id=entity_id
                    ):
                        self._enqueue(
                            entity_kind=entity_kind, entity_id=entity_id, op="UPSERT"
                        )
                        outcomes[job_id] = None
                        continue
                    raise RuntimeError(
                        "Chroma batch promotion became stale and could not restore Stage-1"
                    )
                self.remove_stage1(
                    entity_kind=entity_kind,
                    entity_id=entity_id,
                    expected_last_authoritative_seq=_sequence_number(
                        staged.get("_stage1_authoritative_seq")
                    ),
                )
                outcomes[job_id] = None
            except BaseException as exc:
                outcomes[job_id] = exc
        return outcomes

    def promote_stage2(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        op: str,
        payload_json: str | None,
    ) -> None:
        self.apply_embedding_job(
            entity_kind=entity_kind,
            entity_id=entity_id,
            op=op,
            payload_json=payload_json,
        )

    def reconcile_projection(self, **_: object) -> int:
        removed = 0
        for row in self.stage1_query():
            payload = _mapping(row.get("payload"))
            entity_id = str(payload.get("id") or row.get("key") or "")
            entity_kind = str(payload.get("entity_kind") or "node")
            staged_seq = _sequence_number(row.get("last_authoritative_seq"))
            current = self.engine.indexing.canonical_entity_revision(
                entity_kind=entity_kind, entity_id=entity_id
            )
            if current is None or current.state != "active":
                if self.remove_stage1(
                    entity_kind=entity_kind, entity_id=entity_id,
                    expected_last_authoritative_seq=staged_seq,
                ):
                    self.remove_stage2_or_invalidate(entity_kind=entity_kind, entity_id=entity_id)
                    removed += 1
                continue
            current_fp = self._current_fingerprint(
                entity_kind=entity_kind, entity_id=entity_id
            )
            staged_fp = str(payload.get("source_fingerprint") or "")
            if staged_fp and staged_fp != current_fp:
                if self.remove_stage1(
                    entity_kind=entity_kind, entity_id=entity_id,
                    expected_last_authoritative_seq=staged_seq,
                ):
                    removed += 1
                continue
            got = self._collection_call(
                entity_kind, "get", ids=[entity_id],
                include=["documents", "metadatas", "embeddings"]
            )
            metadata_values = _objects(got.get("metadatas"))
            metadata = _mapping(metadata_values[0]) if metadata_values else {}
            if metadata.get("_kogwistar_source_fingerprint") == current_fp:
                if entity_kind == "edge":
                    from .models import Edge

                    documents = _objects(got.get("documents"))
                    document = documents[0] if documents else None
                    if isinstance(document, str):
                        embeddings = _objects(got.get("embeddings"))
                        self._promote_edge_endpoints(
                            Edge.model_validate_json(document),
                            _embedding(embeddings[0]) if embeddings else None,
                        )
                if self.remove_stage1(
                    entity_kind=entity_kind, entity_id=entity_id,
                    expected_last_authoritative_seq=staged_seq,
                ):
                    removed += 1
        return removed


__all__ = [
    "SQLiteChromaTwoStageProjectionAdapter",
    "chroma_two_stage_capability",
]
