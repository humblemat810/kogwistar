"""PostgreSQL Stage-1 to pgvector Stage-2 projection adapter."""

from __future__ import annotations

import json
from typing import Any

from .async_compat import run_awaitable_blocking
from .storage_backend import TwoStageProjectionCapability
from ..utils.embedding_vectors import normalize_embedding_vector


def postgres_two_stage_capability() -> TwoStageProjectionCapability:
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
        atomic_promotion="same_store",
        reason="PostgreSQL transient Stage-1 and pgvector Stage-2 share one SQL transaction",
    )


class PostgresTwoStageProjectionAdapter:
    """Keep transient metadata rows separate from PostgreSQL vector serving rows."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        if getattr(engine.backend, "_is_async_engine", False):
            raise ValueError(
                "PostgreSQL two-stage adapter currently requires a synchronous engine"
            )

    def _namespace(self) -> str:
        return str(getattr(self.engine, "namespace", "default"))

    def _backend(self) -> Any:
        return self.engine.backend

    def _payload(self, *, entity_kind: str, entity: Any) -> tuple[str, dict[str, Any], str, int]:
        if entity_kind == "node":
            document, metadata = self.engine.write.node_doc_and_meta(entity)
        elif entity_kind == "edge":
            document = entity.model_dump_json(field_mode="backend", exclude=["embedding"])
            metadata = self.engine.write.enrich_edge_meta(entity)
        else:
            raise ValueError(f"unsupported Stage-1 entity kind: {entity_kind!r}")
        revision_payload = self.engine.indexing.canonical_revision_payload(
            entity_kind=entity_kind, entity_id=entity.safe_get_id()
        )
        revision = self.engine.indexing.canonical_entity_revision(
            entity_kind=entity_kind, entity_id=entity.safe_get_id()
        )
        data = json.loads(revision_payload)
        return (
            str(document),
            dict(metadata or {}),
            str(data.get("source_fingerprint") or ""),
            int(getattr(revision, "revision", 0) if revision is not None else 0),
        )

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

    enqueue_embedding_job = _enqueue

    def add_node(self, node: Any, *, doc_id: str | None = None) -> None:
        if doc_id is not None:
            node.doc_id = doc_id
        self._add(entity_kind="node", entity=node)

    def add_edge(self, edge: Any, *, doc_id: str | None = None) -> None:
        if doc_id is not None:
            edge.doc_id = doc_id
        self._add(entity_kind="edge", entity=edge)

    def _add(self, *, entity_kind: str, entity: Any) -> None:
        entity_id = entity.safe_get_id()
        document, metadata, fingerprint, revision = self._payload(
            entity_kind=entity_kind, entity=entity
        )
        self.remove_stage2_or_invalidate(entity_kind=entity_kind, entity_id=entity_id)
        self._backend().stage1_projection_upsert(
            namespace=self._namespace(),
            entity_kind=entity_kind,
            entity_id=entity_id,
            document=document,
            metadata=metadata,
            source_fingerprint=fingerprint,
            revision=revision,
        )
        self._enqueue(entity_kind=entity_kind, entity_id=entity_id, op="UPSERT")

    def stage1_query(self, **kwargs: Any) -> list[dict[str, Any]]:
        rows = self._backend().stage1_projection_query(
            namespace=self._namespace(),
            entity_kind=str(kwargs.get("entity_kind") or "node"),
            ids=kwargs.get("ids"),
            metadata=kwargs.get("metadata"),
            limit=kwargs.get("limit", 200),
        )
        return [
            {
                "key": f"{row['entity_kind']}:{row['entity_id']}",
                "payload": {
                    "id": row["entity_id"],
                    "entity_kind": row["entity_kind"],
                    "document": row["document"],
                    "metadata": dict(row.get("metadata") or {}),
                    "source_fingerprint": row["source_fingerprint"],
                },
            }
            for row in rows
        ]

    def remove_stage1(self, *, entity_kind: str, entity_id: str, **_: Any) -> None:
        self._backend().stage1_projection_delete(
            namespace=self._namespace(), entity_kind=entity_kind, entity_id=entity_id
        )

    def remove_stage2_or_invalidate(self, *, entity_kind: str, entity_id: str, **_: Any) -> None:
        getattr(self._backend(), f"{entity_kind}_delete")(ids=[entity_id])

    def _current(self, *, entity_kind: str, entity_id: str) -> Any:
        return self.engine.indexing.canonical_entity_revision(
            entity_kind=entity_kind, entity_id=entity_id
        )

    def _current_fingerprint(self, *, entity_kind: str, entity_id: str) -> str:
        return str(
            json.loads(
                self.engine.indexing.canonical_revision_payload(
                    entity_kind=entity_kind, entity_id=entity_id
                )
            ).get("source_fingerprint")
            or ""
        )

    def apply_embedding_job(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        op: str,
        payload_json: str | None,
    ) -> None:
        expected = str(json.loads(payload_json or "{}").get("source_fingerprint") or "")
        current = self._current(entity_kind=entity_kind, entity_id=entity_id)
        if op.upper() == "DELETE" or current is None or current.state != "active":
            with self.engine.uow():
                self.remove_stage1(entity_kind=entity_kind, entity_id=entity_id)
                self.remove_stage2_or_invalidate(entity_kind=entity_kind, entity_id=entity_id)
            return
        if expected and expected != self._current_fingerprint(
            entity_kind=entity_kind, entity_id=entity_id
        ):
            return
        row = self._backend().stage1_projection_get(
            namespace=self._namespace(), entity_kind=entity_kind, entity_id=entity_id
        )
        if row is None:
            raise RuntimeError("current PostgreSQL Stage-1 projection is missing")
        document = str(row["document"])
        embedding = self.engine.embed.iterative_defensive_emb(document)
        metadata = dict(row.get("metadata") or {})
        metadata["_kogwistar_stage2_ready"] = True
        metadata["_kogwistar_source_fingerprint"] = expected or str(
            row.get("source_fingerprint") or ""
        )
        # Embedding happens outside SQL transaction. Recheck identity, then make
        # vector upsert + Stage-1 deletion one local PostgreSQL handoff.
        with self.engine.uow():
            current = self._current(entity_kind=entity_kind, entity_id=entity_id)
            if current is None or current.state != "active":
                self.remove_stage1(entity_kind=entity_kind, entity_id=entity_id)
                self.remove_stage2_or_invalidate(entity_kind=entity_kind, entity_id=entity_id)
                return
            if expected and expected != self._current_fingerprint(
                entity_kind=entity_kind, entity_id=entity_id
            ):
                return
            getattr(self._backend(), f"{entity_kind}_upsert")(
                ids=[entity_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[embedding],
            )
            self.remove_stage1(entity_kind=entity_kind, entity_id=entity_id)

    def apply_embedding_jobs_batch(self, jobs: list[Any]) -> dict[str, BaseException | None]:
        """Embed compatible pending rows in one provider call when possible.

        The worker still owns leases and acknowledgements. A provider batch
        failure is isolated by retrying members individually, so one bad
        document cannot discard successful members of the claimed batch.
        """
        prepared: list[tuple[str, str, str, str, dict[str, Any]]] = []
        outcomes: dict[str, BaseException | None] = {}
        for job in jobs:
            value = lambda name: getattr(job, name, None) if not isinstance(job, dict) else job.get(name)
            job_id = str(value("job_id") or "")
            entity_kind = str(value("entity_kind") or "")
            entity_id = str(value("entity_id") or "")
            op = str(value("op") or "UPSERT")
            payload_json = value("payload_json")
            try:
                expected = str(json.loads(payload_json or "{}").get("source_fingerprint") or "")
                current = self._current(entity_kind=entity_kind, entity_id=entity_id)
                if op.upper() == "DELETE" or current is None or current.state != "active":
                    self.apply_embedding_job(
                        entity_kind=entity_kind, entity_id=entity_id,
                        op=op, payload_json=payload_json
                    )
                    outcomes[job_id] = None
                    continue
                if expected and expected != self._current_fingerprint(
                    entity_kind=entity_kind, entity_id=entity_id
                ):
                    outcomes[job_id] = None
                    continue
                row = self._backend().stage1_projection_get(
                    namespace=self._namespace(), entity_kind=entity_kind, entity_id=entity_id
                )
                if row is None:
                    raise RuntimeError("current PostgreSQL Stage-1 projection is missing")
                prepared.append((job_id, entity_kind, entity_id, expected, row))
            except BaseException as exc:
                outcomes[job_id] = exc

        if not prepared:
            return outcomes
        documents = [str(row[4]["document"]) for row in prepared]
        try:
            raw_embeddings = run_awaitable_blocking(self.engine._ef(documents))
            embeddings = [normalize_embedding_vector(value, allow_none=False) for value in raw_embeddings]
            if len(embeddings) != len(prepared):
                raise RuntimeError("embedding provider returned wrong batch length")
        except BaseException:
            # Preserve partial-success semantics when provider rejects one item.
            embeddings = []
            for job_id, _, _, _, row in prepared:
                try:
                    embeddings.append(
                        self.engine.embed.iterative_defensive_emb(str(row["document"]))
                    )
                except BaseException as exc:
                    outcomes[job_id] = exc
                    embeddings.append(None)

        with self.engine.uow():
            for (job_id, entity_kind, entity_id, expected, row), embedding in zip(
                prepared, embeddings
            ):
                if embedding is None or job_id in outcomes:
                    continue
                current = self._current(entity_kind=entity_kind, entity_id=entity_id)
                if current is None or current.state != "active" or (
                    expected and expected != self._current_fingerprint(
                        entity_kind=entity_kind, entity_id=entity_id
                    )
                ):
                    continue
                metadata = dict(row.get("metadata") or {})
                metadata["_kogwistar_stage2_ready"] = True
                metadata["_kogwistar_source_fingerprint"] = expected or str(
                    row.get("source_fingerprint") or ""
                )
                getattr(self._backend(), f"{entity_kind}_upsert")(
                    ids=[entity_id], documents=[str(row["document"])],
                    metadatas=[metadata], embeddings=[embedding]
                )
                self.remove_stage1(entity_kind=entity_kind, entity_id=entity_id)
                outcomes[job_id] = None
        return outcomes

    def promote_stage2(self, **kwargs: Any) -> None:
        self.apply_embedding_job(**kwargs)

    def reconcile_projection(self, **_: Any) -> int:
        removed = 0
        for row in self._backend().stage1_projection_query(
            namespace=self._namespace(), entity_kind="node", limit=1000
        ) + self._backend().stage1_projection_query(
            namespace=self._namespace(), entity_kind="edge", limit=1000
        ):
            kind, entity_id = row["entity_kind"], row["entity_id"]
            current = self._current(entity_kind=kind, entity_id=entity_id)
            if current is None or current.state != "active":
                with self.engine.uow():
                    self.remove_stage1(entity_kind=kind, entity_id=entity_id)
                    self.remove_stage2_or_invalidate(entity_kind=kind, entity_id=entity_id)
                removed += 1
                continue
            if self._current_fingerprint(entity_kind=kind, entity_id=entity_id) != str(
                row.get("source_fingerprint") or ""
            ):
                self.remove_stage1(entity_kind=kind, entity_id=entity_id)
                removed += 1
                continue
            current_stage2 = getattr(self._backend(), f"{kind}_get")(
                ids=[entity_id], include=["metadatas"]
            )
            stage2_metadata = (current_stage2.get("metadatas") or [None])[0]
            if (
                current_stage2.get("ids")
                and isinstance(stage2_metadata, dict)
                and stage2_metadata.get("_kogwistar_source_fingerprint")
                == row.get("source_fingerprint")
            ):
                self.remove_stage1(entity_kind=kind, entity_id=entity_id)
                removed += 1
        return removed


__all__ = ["PostgresTwoStageProjectionAdapter", "postgres_two_stage_capability"]
