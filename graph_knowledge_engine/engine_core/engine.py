from __future__ import annotations
from contextlib import contextmanager
import contextvars

import pathlib
import uuid


import time

from .utils import AliasBook

from .chroma_backend import ChromaBackend


from .engine_sqlite import EngineSQLite
from .storage_backend import NoopUnitOfWork, StorageBackend
from ..workers.index_job_worker import IndexJobWorker
from ..utils.log import bind_log_context
from .indexing import IndexingSubsystem
from .subsystems import (
    AdjudicateSubsystem,
    EmbedSubsystem,
    ExtractSubsystem,
    IngestSubsystem,
    PersistSubsystem,
    ReadSubsystem,
    RollbackSubsystem,
    WriteSubsystem,
)
from .search_index.service import SearchIndexService
from .types import (
    EngineType,
    ExtractionSchemaMode,
    OffsetMismatchPolicy,
    OffsetRepairScorer,
    ResolvedExtractionSchemaMode,
)
from ..graph_kinds import normalize_graph_kind
from .utils.aliasing import AliasBookStore

if True:
    """_summary_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
        
    sample usage:
    from graph_knowledge_engine.strategies import candidate_proposals as CP
    from graph_knowledge_engine.strategies import adjudicators as AJ
    from graph_knowledge_engine.strategies import merge_policies as MP
    from graph_knowledge_engine.strategies import verifiers as VF
    from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine

    engine = GraphKnowledgeEngine(
        persist_directory="./chroma_db",
        candidate_generator=CP.hybrid,         # or CP.by_vector_similarity
        adjudicator=AJ.llm_pair,               # or AJ.rule_first_token / AJ.llm_batch in your batch path
        merge_policy=MP.prefer_existing_canonical,
        verifier=VF.ensemble_default,          # or VF.coverage_only / VF.strict_with_min_span
    )
    """
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Tuple, cast

try:
    from typing import Self, TypeAlias
except ImportError:  # pragma: no cover - py<3.11 compatibility
    from typing_extensions import TypeAlias
from ..typing_interfaces import EmbeddingFunctionLike
from ..graph_query import GraphQuery
from graph_knowledge_engine.extraction import BaseDocValidator
from .models import (
    Node,
    Edge,
    Document,
    Domain,
    Grounding,
    PureChromaNode,
    PureChromaEdge,
    PureGraph,
    Span,
    LLMGraphExtraction,
    AdjudicationTarget,
    AdjudicationCandidate,
    AdjudicationQuestionCode,
    AdjudicationVerdict,
)
from ..cdc.change_event import Op, EntityRefModel
from ..llm_tasks import (
    DefaultTaskProviderConfig,
    LLMTaskSet,
    build_default_llm_tasks,
    validate_llm_task_set,
)
from ..integrations.openai_embeddings import build_azure_embedding_fn_from_env
import json
import os
from dotenv import load_dotenv
from joblib import Memory
from functools import wraps
import warnings
from .models import (
    GraphExtractionWithIDs,
)
from typing import (
    Callable,
    Iterable,
    Sequence,
    Literal,
    Type,
    TypeVar,
    Union,
)
import math

# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# from chromadb.utils import embedding_functions
from datetime import datetime
from graph_knowledge_engine.cdc.change_bus import ChangeBus, FastAPIChangeSink
from graph_knowledge_engine.cdc.change_event import ChangeEvent
from graph_knowledge_engine.cdc.oplog import OplogWriter

from pydantic import BaseModel

# Optional: RapidFuzz
try:

    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

PageLike = Union[str, Dict[str, Any]]
NodeOrEdge: TypeAlias = Node | Edge

if TYPE_CHECKING:
    from .engine_postgres_meta import EnginePostgresMetaStore
    from .postgres_backend import PgVectorBackend

    T = TypeVar("T", Node, Edge)
    # TT= TypeVar("TT", Type[Node], Type[Edge])
    TNode = TypeVar("TNode", bound=Node)
    TEdge = TypeVar("TEdge", bound=Edge)


from typing import TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


def cached(memory: Memory, fn: Callable[P, R], *args, **kwargs) -> Callable[P, R]:
    return cast(Callable[P, R], memory.cache(fn, *args, **kwargs))


def _optional_dependency_error(*, extra: str, detail: str) -> RuntimeError:
    return RuntimeError(
        f"{detail}. Install optional dependency group '{extra}' with: pip install 'kogwistar[{extra}]'"
    )


def _import_chroma_client():
    try:
        from chromadb import Client  # type: ignore
        from chromadb.config import Settings  # type: ignore
    except Exception as e:  # pragma: no cover - import errors depend on env
        raise _optional_dependency_error(
            extra="chroma",
            detail="Chroma backend requires the 'chromadb' package",
        ) from e
    return Client, Settings


def _is_pgvector_backend_instance(backend: object) -> bool:
    try:
        from .postgres_backend import PgVectorBackend  # type: ignore
    except Exception:
        return False
    return isinstance(backend, PgVectorBackend)


def _build_postgres_uow_if_needed(backend: StorageBackend):
    from .postgres_backend import PgVectorBackend

    if not _is_pgvector_backend_instance(backend):
        return NoopUnitOfWork()
    try:
        from graph_knowledge_engine.engine_core.postgres_backend import (
            PostgresUnitOfWork,
        )  # type: ignore
    except Exception as e:  # pragma: no cover - import errors depend on env
        raise _optional_dependency_error(
            extra="pgvector",
            detail="PgVector backend requires optional PostgreSQL dependencies",
        ) from e
    pg_backend = cast(PgVectorBackend, backend)
    return PostgresUnitOfWork(engine=pg_backend.engine)


def _safe_json_dict(doc: Any) -> dict:
    if isinstance(doc, dict):
        return doc
    if not isinstance(doc, str):
        return {}
    try:
        x = json.loads(doc)
        return x if isinstance(x, dict) else {}
    except Exception:
        return {}


def _merge_meta(base_meta: dict | None, patch: dict) -> dict:
    base_meta = base_meta or {}
    # flat merge
    return {**base_meta, **patch}


F = TypeVar("F", bound=Callable[..., Any])


def _strip_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _json_or_none(v):
    return None if v is None else json.dumps(v)


def _node_doc_and_meta(n: Union["Node", "PureChromaNode"]) -> tuple[str, dict]:
    """Return (documents_string, metadata_dict) for Chroma. helper when inserting to backend db,"""
    """Extract and flatten certain fields that can be searched via collection """
    doc = n.model_dump_json(field_mode="backend", exclude=["embedding", "metadata"])
    meta = n.metadata  # user custom metadata will be overwritten by system metadata
    meta.update(
        {
            "doc_id": getattr(n, "doc_id", None),
            "label": n.label,
            "type": n.type,
            "summary": n.summary,
            "domain_id": n.domain_id,
            "canonical_entity_id": getattr(n, "canonical_entity_id", None),
            "properties": _json_or_none(getattr(n, "properties", None)),
        }
    )
    meta.update(n.get_extra_update())

    mentions = getattr(n, "mentions", None)
    if mentions is not None:
        meta["mentions"] = _json_or_none(
            [r.model_dump(field_mode="backend") for r in mentions]
        )
    meta = _strip_none(meta)
    return doc, meta


def _edge_doc_and_meta(e: Union["Edge", "PureChromaEdge"]) -> tuple[str, dict]:
    """Return (documents_string, metadata_dict) for Chroma."""
    doc = e.model_dump_json(field_mode="backend")
    meta = _strip_none(
        {
            "doc_id": getattr(e, "doc_id", None),
            "relation": e.relation,
            "source_ids": _json_or_none(e.source_ids),
            "target_ids": _json_or_none(e.target_ids),
            "type": e.type,
            "summary": e.summary,
            "domain_id": e.domain_id,
            "canonical_entity_id": getattr(e, "canonical_entity_id", None),
            "properties": _json_or_none(getattr(e, "properties", None)),
        }
    )
    # if hasattr(e,"mentions"):
    #     meta['mentions'] = _json_or_none([r.model_dump(field_mode = 'backend') for r in (e.mentions or [])])
    mentions = getattr(e, "mentions", None)
    if mentions is not None:
        meta["mentions"] = _json_or_none(
            [r.model_dump(field_mode="backend") for r in mentions]
        )
    meta = _strip_none(meta)
    return doc, meta


def _backend_update_record_lifecycle(
    *,
    backend: StorageBackend,
    kind: str,
    record_id: str,
    lifecycle_patch: dict,
) -> bool:
    """Update one record's lifecycle metadata via backend get/update methods."""
    get_fn = getattr(backend, f"{kind}_get", None)
    upd_fn = getattr(backend, f"{kind}_update", None)
    if get_fn is None or upd_fn is None:
        raise AttributeError(f"backend missing {kind}_get/{kind}_update")
    got = get_fn(ids=[record_id], include=["documents", "metadatas", "embeddings"])
    ids = got.get("ids") or []
    if not ids:
        return False

    doc = (got.get("documents") or [None])[0]
    meta = (got.get("metadatas") or [None])[0]
    emb = got.get("embeddings")

    embedding = (emb if emb is not None else [None])[0]
    base = _safe_json_dict(doc)

    base_meta = base.get("metadata") if isinstance(base.get("metadata"), dict) else {}
    base["metadata"] = _merge_meta(base_meta, lifecycle_patch)

    new_meta = _merge_meta(meta if isinstance(meta, dict) else {}, lifecycle_patch)
    doc = json.dumps(base, ensure_ascii=False)
    upd_fn(
        ids=[record_id],
        documents=[json.dumps(base, ensure_ascii=False)],
        metadatas=[new_meta],
        embeddings=[embedding],
    )
    return True


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def uuid_to_base62(u: str) -> str:
    n = int(u.replace("-", ""), 16)
    if n == 0:
        return "0"
    out = []
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


# strategy toggle
# "session_alias" -> N#/E# with session-stable AliasBook (+ delta legend)
# "base62"        -> N~<22ch> / E~<22ch> (no legend, fully deterministic)
ID_STRATEGY = "session_alias"  # or "base62"
_DOC_ALIAS = "::DOC::"  # short, token-friendly

try:
    from chromadb.api.types import EmbeddingFunction, Embeddings  # type: ignore
except Exception:

    class EmbeddingFunction:  # pragma: no cover - only used when chromadb is absent
        @staticmethod
        def name() -> str:
            return "default"

    Embeddings = list[list[float]]  # type: ignore

from typing import Any, Callable


# ---------------------------------------------------------------------------
# Embedding providers — now in embedding_factory.py
# ---------------------------------------------------------------------------
from graph_knowledge_engine.engine_core.embedding_factory import (
    get_embedding_function,  # backwards compat alias → OllamaEmbeddingFunction
)


def engine_context(fn):
    """
    Decorator for engine boundary methods.
    Automatically binds engine_type and common engine attributes.
    Assumes `self` has:
        - self.engine_id (optional)
    And that method may accept:
        - conversation_id / conv_id
        - workflow_run_id / run_id
        - step_id
    """

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        ctx_kwargs = {
            "engine_type": getattr(self, "kg_graph_type", None),
            "engine_id": getattr(self, "engine_id", None),
            "workflow_run_id": kwargs.get("workflow_run_id") or kwargs.get("run_id"),
            "step_id": kwargs.get("step_id"),
            "conversation_id": kwargs.get("conv_id"),
        }

        with bind_log_context(**ctx_kwargs):
            return fn(self, *args, **kwargs)

    return wrapper


class GraphKnowledgeEngine:
    """
    The **Base Abstraction** for the knowledge graph/graph database.

    This engine manages the lifecycle of **provenance-heavy primitives** (`Node`, `Edge`).
    Unlike typical graph databases, every primitive here carries rich metadata about its origin
    (source document, span, verification status).

    Key responsibilities:
    - Persisting nodes and edges with full provenance.
    - Managing extensions like chat/workflow node variants.
    - Providing low-level to high-level APIs for extraction, storage, and adjudication.

    Methods are generally arranged from low-level generic helpers to task-specific calls.
    High-level orchestration for extracting, storing, and adjudicating knowledge graph data.
    """

    # --------------------
    # Puhlic Interface
    # --------------------
    def _filter_items_by_resolve_mode(
        self, items: list[T], resolve_mode: str
    ) -> list[T]:
        return self.lifecycle.filter_items(items, resolve_mode)

    def _resolve_redirect_chain(
        self,
        *,
        initial_items: list[T],
        fetch_by_ids: Callable[[Sequence[str]], list[T]],
        resolve_mode: Literal["active_only", "redirect", "include_tombstones"],
    ) -> list[T]:
        return self.lifecycle.resolve_redirect_chain(
            initial_items=initial_items,
            fetch_by_ids=fetch_by_ids,
            resolve_mode=resolve_mode,
        )

    def tombstone_node(self, node_id: str, **kw) -> bool:
        return self.lifecycle.tombstone_node(node_id, **kw)

    def redirect_node(self, from_id: str, to_id: str, **kw) -> bool:
        return self.lifecycle.redirect_node(from_id, to_id, **kw)

    @engine_context
    def tombstone_edge(self, edge_id: str, **kw) -> bool:
        return self.lifecycle.tombstone_edge(edge_id, **kw)

    def redirect_edge(self, from_id: str, to_id: str, **kw) -> bool:
        return self.lifecycle.redirect_edge(from_id, to_id, **kw)

    def node_ids_by_doc(self, doc_id: str) -> List[str]:

        return self._nodes_by_doc(doc_id)

    def edge_ids_by_doc(self, doc_id: str) -> List[str]:

        return self._edge_ids_by_doc(doc_id)

    @property
    def embedding_function(self):
        return self._ef

    @embedding_function.setter
    def embedding_function(self, val):
        self._ef = val

    def _infer_doc_id_from_ref(self, ref: Span) -> Optional[str]:
        """Best-effort: prefer explicit ref.doc_id; else try to parse document_page_url like 'document/<id>'."""
        did = getattr(ref, "doc_id", None)
        if did:
            return did
        url = getattr(ref, "document_page_url", None) or ""
        # simple heuristic: last path token if present
        try:
            tail = url.strip("/").split("/")[-1]
            return tail or None
        except Exception:
            return None

    def extract_reference_contexts(
        self,
        node_or_id: Union[
            Node | Edge, str
        ],  # also works if you pass an Edge or edge id
        *,
        window_chars: int = 120,
        max_contexts: Optional[int] = None,
        prefer_label_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        return self.read.extract_reference_contexts(
            node_or_id=node_or_id,
            window_chars=window_chars,
            max_contexts=max_contexts,
            prefer_label_fallback=prefer_label_fallback,
        )

    def replay_namespace(
        self,
        *,
        namespace: str = "default",
        from_seq: int = 1,
        to_seq: int | None = None,
        apply_indexes: bool = False,
        repair_backend: bool = False,
    ) -> int:
        return self.persist.replay_namespace(
            namespace=namespace,
            from_seq=from_seq,
            to_seq=to_seq,
            apply_indexes=apply_indexes,
            repair_backend=repair_backend,
        )

    def replay_repair_namespace(
        self,
        *,
        namespace: str = "default",
        from_seq: int = 1,
        to_seq: int | None = None,
        apply_indexes: bool = False,
    ) -> int:
        return self.persist.replay_repair_namespace(
            namespace=namespace,
            from_seq=from_seq,
            to_seq=to_seq,
            apply_indexes=apply_indexes,
        )

    ## get
    def get_nodes(
        self,
        ids: Sequence[str] | None = None,
        node_type: Type[Node] | None = None,
        include: None | list[str] = None,
        where=None,
        limit: None | int = 200,
        resolve_mode: Literal[
            "active_only", "redirect", "include_tombstones"
        ] = "active_only",
    ) -> List[Node]:
        return self.read.get_nodes(
            ids=ids,
            node_type=node_type,
            include=include,
            where=where,
            limit=limit,
            resolve_mode=resolve_mode,
        )

    def query_nodes(
        self,
        *args,
        query=None,
        query_embeddings=None,
        include=["documents", "embeddings", "metadatas"],
        node_type: Type[Node] = Node,
        **kwargs,
    ):
        return self.read.query_nodes(
            *args,
            query=query,
            query_embeddings=query_embeddings,
            include=include,
            node_type=node_type,
            **kwargs,
        )

    def search_nodes_as_of(
        self,
        *,
        query: str | None = None,
        query_embeddings: list[list[float]] | None = None,
        as_of_ts: datetime | str,
        where: dict[str, Any] | None = None,
        n_results: int = 20,
        follow_redirects: bool = True,
        node_type: Type[Node] = Node,
        include: list[str] | None = None,
        max_redirect_hops: int = 16,
        **kwargs,
    ) -> list[Node]:
        return self.read.search_nodes_as_of(
            query=query,
            query_embeddings=query_embeddings,
            as_of_ts=as_of_ts,
            where=where,
            n_results=n_results,
            follow_redirects=follow_redirects,
            node_type=node_type,
            include=include,
            max_redirect_hops=max_redirect_hops,
            **kwargs,
        )

    def query_edges(
        self,
        *args,
        query=None,
        query_embeddings=None,
        include=["documents", "embeddings", "metadatas"],
        edge_type: Type[Edge] = Edge,
        **kwargs,
    ):
        return self.read.query_edges(
            *args,
            query=query,
            query_embeddings=query_embeddings,
            include=include,
            edge_type=edge_type,
            **kwargs,
        )

    def nodes_from_single_or_id_query_result(
        self,
        got,
        node_type: Type[TNode] = Node,
    ) -> list[TNode]:
        return self.read.nodes_from_single_or_id_query_result(got, node_type=node_type)

    def edges_from_single_or_id_query_result(
        self, got, edge_type: Type[Edge] = Edge, include=None
    ):
        return self.read.edges_from_single_or_id_query_result(
            got,
            edge_type=edge_type,
            include=include,
        )

    def nodes_from_query_result(self, gots, node_type: Type[Node] = Node):
        return self.read.nodes_from_query_result(gots, node_type=node_type)

    def edges_from_query_result(self, gots, edge_type: Type[Edge] = Edge):
        return self.read.edges_from_query_result(gots, edge_type=edge_type)

    def _where_update_from_resolve_mode(
        self, resolve_mode: Literal["active_only", "redirect", "include_tombstones"]
    ):
        return self.read.where_update_from_resolve_mode(resolve_mode)

    def get_edges(
        self,
        ids: Sequence[str] | None = None,
        edge_type: Type[Edge] | None = None,
        where=None,
        limit: int | None = 400,
        include: None | list[str] = None,
        resolve_mode: Literal[
            "active_only", "redirect", "include_tombstones"
        ] = "active_only",
    ) -> List[Edge]:
        return self.read.get_edges(
            ids=ids,
            edge_type=edge_type,
            where=where,
            limit=limit,
            include=include,
            resolve_mode=resolve_mode,
        )

    def all_nodes_for_doc(self, doc_id: str) -> List[Node]:
        return self.get_nodes(self._nodes_by_doc(doc_id))

    def all_edges_for_doc(self, doc_id: str) -> List[Edge]:
        return self.get_edges(self._edge_ids_by_doc(doc_id))

    def _delete_edge_ref_rows(self, edge_id: str) -> None:
        return self.write.delete_edge_ref_rows(edge_id)

    def _delete_node_ref_rows(self, node_id: str) -> None:
        return self.write.delete_node_ref_rows(node_id)

    # ----------------------------
    # Phase 1: durable index jobs + reconciler
    # ----------------------------

    _PHASE1_JOIN_INDEX_KINDS = ("node_docs", "node_refs", "edge_refs", "edge_endpoints")

    # ------------------------------------------------------------------
    # Phase 2b: event log foundation
    # ------------------------------------------------------------------

    def _append_event_for_entity(
        self,
        *,
        namespace: str,
        entity_kind: str,
        entity_id: str,
        op: str,
        payload: dict,
    ) -> None:
        """
        Best-effort append to the meta outbox. Must never block the primary write path.
        Replay suppresses this via self._disable_event_log.
        """
        """
        ### Replay vs Repair Replay (Event Sourcing / Projections)

        The entity event log (`entity_events`) is the source of truth. Storage backends (Chroma / pgvector)
        are projections derived from the event stream.

        - `replay_namespace(...)` replays events to rebuild projection state.
        - For create-only backends like Chroma (`collection.add`), normal replay can rebuild missing ids,
            but cannot overwrite ids that already exist with corrupted/tampered content.

        - `replay_namespace(..., repair_backend=True)` (or `replay_repair_namespace(...)`) performs a
        "repair replay" that best-effort overwrites projection state to match the event log.
        - Use this when you suspect backend drift/tampering.

        Invariant: replay never emits new `entity_events` (guarded by `_disable_event_log`).
        """
        if getattr(self, "_disable_event_log", False):
            return
        append = getattr(self.meta_sqlite, "append_entity_event", None)
        if append is None:
            return

        event_id = str(uuid.uuid4())
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        append(
            namespace=namespace,
            event_id=event_id,
            entity_kind=entity_kind,
            entity_id=entity_id,
            op=op,
            payload_json=payload_json,
        )

    def enqueue_index_job(self, *args, **kwargs) -> str:
        return self.indexing.enqueue_index_job(*args, **kwargs)

    def enqueue_index_jobs_for_node(self, node_id: str, *, op: str) -> None:
        return self.indexing.enqueue_index_jobs_for_node(node_id, op=op)

    def enqueue_index_jobs_for_edge(self, edge_id: str, *, op: str) -> None:
        return self.indexing.enqueue_index_jobs_for_edge(edge_id, op=op)

    def reconcile_indexes(
        self,
        *,
        max_jobs: int = 100,
        lease_seconds: int = 60,
        namespace: str | None = None,
    ) -> int:
        return self.indexing.reconcile_indexes(
            max_jobs=max_jobs, lease_seconds=lease_seconds, namespace=namespace
        )

    def make_index_job_worker(
        self,
        *,
        max_inflight: int = 1,
        batch_size: int = 50,
        lease_seconds: int = 60,
        max_jobs_per_tick: int = 200,
        namespace: str | None = None,
    ) -> "IndexJobWorker":
        return self.indexing.make_index_job_worker(
            max_inflight=max_inflight,
            batch_size=batch_size,
            lease_seconds=lease_seconds,
            max_jobs_per_tick=max_jobs_per_tick,
            namespace=namespace,
        )

    # ----------------------------
    # Utilities
    # ----------------------------
    # @staticmethod
    # def _default_ref(doc_id: str, excerpt: Optional[str] = None) -> Span:
    #     return _default_ref(doc_id, excerpt)
    #     pass
    @staticmethod
    def chroma_sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Drop keys whose values are None. ChromaDB metadata rejects None values."""
        return _strip_none(
            metadata
        )  # {k: v for k, v in metadata.items() if v is not None}

    @staticmethod
    def _strip_none(d: dict):
        return _strip_none(d)

    @staticmethod
    def _json_or_none(obj: Any) -> Optional[str]:
        return json.dumps(obj) if obj is not None else None

    def _exists_node(self, rid: str) -> bool:
        return self.persist.exists_node(rid)

    def _exists_edge(self, rid: str) -> bool:
        return self.persist.exists_edge(rid)

    def _exists_any(self, rid: str) -> bool:
        return self.persist.exists_any(rid)

    def _select_doc_context(
        self, doc_id: str, max_nodes: int = 200, max_edges: int = 400
    ):
        return self.persist.select_doc_context(
            doc_id, max_nodes=max_nodes, max_edges=max_edges
        )

    def _preflight_validate(
        self,
        parsed: LLMGraphExtraction | PureGraph | GraphExtractionWithIDs,
        alias_key: str,
        alias_book: AliasBook | None = None,
    ):
        return self.persist.preflight_validate(parsed, alias_key, alias_book=alias_book)

    def _assert_endpoints_exist(self, edge: Edge | PureChromaEdge):
        return self.persist.assert_endpoints_exist(edge)

    def _build_deps(self, parsed):
        return self.persist.build_deps(parsed)

    def ingest_with_toposort(self, parsed, *, doc_id: str):
        return self.persist.ingest_with_toposort(parsed, doc_id=doc_id)

    def _resolve_llm_ids(
        self,
        doc_id: str,
        parsed: LLMGraphExtraction | PureGraph | GraphExtractionWithIDs,
        alias_book: AliasBook | None = None,
    ) -> None:
        return self.persist.resolve_llm_ids(doc_id, parsed, alias_book=alias_book)

    def _aliasify_for_prompt(
        self, doc_id: str, ctx_nodes: list[dict], ctx_edges: list[dict]
    ):
        return self.extract.aliasify_for_prompt(doc_id, ctx_nodes, ctx_edges)

    def _resolve_extraction_schema_mode(
        self,
        extraction_schema_mode: ExtractionSchemaMode | None = None,
    ) -> ResolvedExtractionSchemaMode:
        return self.extract.resolve_extraction_schema_mode(extraction_schema_mode)

    def _schema_prompt_rules(self, mode: ResolvedExtractionSchemaMode) -> str:
        return self.extract.schema_prompt_rules(mode)

    def _structured_schema_for_mode(self, mode: ResolvedExtractionSchemaMode):
        return self.extract.structured_schema_for_mode(mode)

    def _build_structured_output_for_mode(self, mode: ResolvedExtractionSchemaMode):
        # Back-compat shim: structured-output construction moved into llm task providers.
        return self.extract.structured_schema_for_mode(mode)

    @staticmethod
    def _offset_repair_threshold(excerpt_len: int) -> float:
        return ExtractSubsystem._offset_repair_threshold(excerpt_len)

    @staticmethod
    def _clip_offset_excerpt(text: str, *, max_chars: int = 80) -> str:
        return ExtractSubsystem._clip_offset_excerpt(text, max_chars=max_chars)

    @staticmethod
    def _find_all_exact_occurrences(content: str, excerpt: str) -> list[int]:
        return ExtractSubsystem._find_all_exact_occurrences(content, excerpt)

    @staticmethod
    def _coerce_offset_score(raw_score: Any) -> float:
        return ExtractSubsystem._coerce_offset_score(raw_score)

    def _default_offset_repair_scorer(self, candidate: str, excerpt: str) -> float:
        return self.extract.default_offset_repair_scorer(candidate, excerpt)

    def _resolve_offset_repair_scorer(
        self,
        override: OffsetRepairScorer | None,
    ) -> OffsetRepairScorer:
        return self.extract.resolve_offset_repair_scorer(override)

    def _iter_lean_spans_for_mode(
        self,
        *,
        mode: ResolvedExtractionSchemaMode,
        payload: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        return self.extract._iter_lean_spans_for_mode(mode=mode, payload=payload)

    def _build_offset_failure_detail(
        self,
        *,
        path: str,
        content: str,
        excerpt: str,
        start_char: int,
        end_char: int,
        exact_hits: int,
        best_fuzzy_score: float | None,
    ) -> str:
        return self.extract._build_offset_failure_detail(
            path=path,
            content=content,
            excerpt=excerpt,
            start_char=start_char,
            end_char=end_char,
            exact_hits=exact_hits,
            best_fuzzy_score=best_fuzzy_score,
        )

    def _find_best_fuzzy_span(
        self,
        *,
        content: str,
        excerpt: str,
        origin_start: int,
        scorer: OffsetRepairScorer,
    ) -> tuple[int, int, float] | None:
        return self.extract._find_best_fuzzy_span(
            content=content,
            excerpt=excerpt,
            origin_start=origin_start,
            scorer=scorer,
        )

    def _repair_lean_offsets_for_mode(
        self,
        *,
        mode: ResolvedExtractionSchemaMode,
        payload: dict[str, Any],
        content: str,
        policy: OffsetMismatchPolicy,
        offset_repair_scorer: OffsetRepairScorer | None,
    ) -> dict[str, Any]:
        return self.extract.repair_lean_offsets_for_mode(
            mode=mode,
            payload=payload,
            content=content,
            policy=policy,
            offset_repair_scorer=offset_repair_scorer,
        )

    def _to_canonical_extraction_for_mode(
        self,
        *,
        mode: ResolvedExtractionSchemaMode,
        parsed: Any,
        content: str,
        offset_mismatch_policy: OffsetMismatchPolicy = "exact_fuzzy",
        offset_repair_scorer: OffsetRepairScorer | None = None,
    ) -> LLMGraphExtraction:
        return self.extract.to_canonical_extraction_for_mode(
            mode=mode,
            parsed=parsed,
            content=content,
            offset_mismatch_policy=offset_mismatch_policy,
            offset_repair_scorer=offset_repair_scorer,
        )

    def _extract_graph_with_llm_aliases(
        self,
        content: str,
        alias_nodes_str: str,
        alias_edges_str: str,
        instruction_for_node_edge_contents_parsing_inclusion: None | str = None,
        last_iteration_result: dict | None = None,
        extraction_schema_mode: ExtractionSchemaMode | None = None,
        offset_mismatch_policy: OffsetMismatchPolicy = "exact_fuzzy",
        offset_repair_scorer: OffsetRepairScorer | None = None,
    ):
        return self.extract.extract_graph_with_llm_aliases(
            content=content,
            alias_nodes_str=alias_nodes_str,
            alias_edges_str=alias_edges_str,
            instruction_for_node_edge_contents_parsing_inclusion=instruction_for_node_edge_contents_parsing_inclusion,
            last_iteration_result=last_iteration_result,
            extraction_schema_mode=extraction_schema_mode,
            offset_mismatch_policy=offset_mismatch_policy,
            offset_repair_scorer=offset_repair_scorer,
        )

    def _de_alias_ids_in_result(
        self, doc_id: str, parsed: LLMGraphExtraction
    ) -> LLMGraphExtraction:
        return self.extract.de_alias_ids_in_result(doc_id, parsed)

    def _alias_book(self, key: str) -> AliasBook:
        return self.alias_books.get(key)

    def _coerce_pages(
        self, content_or_pages
    ):  # -> list[tuple[int, str]] | list[Any] | Any:
        return self.extract.coerce_pages(content_or_pages)

    # ----------------------------
    # Init
    # ----------------------------

    def __init__(
        self,
        persist_directory: str | None = None,
        embedding_function: EmbeddingFunctionLike | None = None,
        embedding_cache_path: str | None = None,
        proposer=None,  # callable(pairs) -> List[LLMMergeAdjudication]
        adjudicator=None,  # callable(left: Node, right: Node) -> AdjudicationVerdict
        merge_policy=None,  # callable(left, right, verdict) -> str (canonical_id)
        verifier=None,  # callable(extracted, full_text, ref, **kw) -> ReferenceSession
        kg_graph_type: EngineType = "knowledge",
        debug_dir: pathlib.Path | None = None,
        backend: str | StorageBackend | None = None,
        namespace: str = "default",
        extraction_schema_mode: ExtractionSchemaMode = "auto",
        offset_repair_scorer: OffsetRepairScorer | None = None,
        llm_tasks: LLMTaskSet | None = None,
        default_task_provider_config: DefaultTaskProviderConfig | None = None,
        cache_dir: os.PathLike|str|None = None
    ):
        """
        embedding_function: callable(texts: List[str]) -> List[List[float]].
          If None, defaults to SentenceTransformerEmbeddingFunction with model:
          - default_st_model argument, or
          - ENV SENTENCE_TRANSFORMERS_MODEL, or
          - "all-MiniLM-L6-v2".
        """
        load_dotenv()
        self.kg_graph_type = normalize_graph_kind(kg_graph_type)
        self.persist_directory = persist_directory
        self.namespace = namespace
        self.extraction_schema_mode: ExtractionSchemaMode = extraction_schema_mode
        self.offset_repair_scorer: OffsetRepairScorer | None = offset_repair_scorer
        self.cache_dir = cache_dir or self.persist_directory
        self.indexing = IndexingSubsystem(self)

        self._uow_ctx_conn: contextvars.ContextVar[object | None] = (
            contextvars.ContextVar("gke_uow_conn", default=None)
        )
        self._uow_ctx_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
            "gke_uow_depth", default=0
        )

        self.changes = ChangeBus()
        if cdc_publish_endpoint := os.environ.get("CDC_PUBLISH_ENDPOINT"):
            self.changes.add_sink(
                FastAPIChangeSink(
                    cdc_publish_endpoint, name=f"{self.kg_graph_type}-fastapi-sink"
                )
            )

        # from .debug_producer import DebugEventProducer
        # self._debug_producer = DebugEventProducer("http://127.0.0.1:8000")
        self._oplog = None
        if debug_dir is not None:
            self._oplog = OplogWriter(debug_dir / "changes.jsonl", fsync=False)
        self.tool_call_id_factory: Callable[[], str] | None = None

        self._disable_event_log = False
        self.query = GraphQuery(self)

        self.allow_cross_kind_adjudication = True  # can be set by user
        self.cross_kind_strategy = (
            "reifies"  # "reifies" | "equivalent" (default "reifies")
        )
        # to do- refractor via composition. protocol template in strategies.py, strategies helper in ./strategies/
        # strategies now are function objects
        from ..strategies import (
            VectorProposer,
            DefaultVerifier,
            PreferExistingCanonical,
            Adjudicator,
        )

        # from .strategies.adjudicators import LLMPairAdjudicatorImpl, LLMBatchAdjudicatorImpl
        from ..strategies.verifiers import DefaultVerifier, VerifierConfig
        from ..strategies.types import Verifier
        from graph_knowledge_engine.strategies import IAdjudicator

        self.proposer = proposer or VectorProposer(self)
        self.adjudicator: IAdjudicator = adjudicator or Adjudicator(self)
        self.verifier: Verifier = verifier or DefaultVerifier(
            self, VerifierConfig(use_embeddings=False)
        )
        self.merge_policy = merge_policy or PreferExistingCanonical(self)
        self.meta_sqlite: EngineSQLite | EnginePostgresMetaStore

        self.alias_books = AliasBookStore()
        self.pre_add_node_hooks: list[Callable[[Node], None]] = []
        self.pre_add_edge_hooks: list[Callable[[Edge], bool]] = []
        self.pre_add_pure_edge_hooks: list[Callable[[Edge], bool]] = []
        self.allow_missing_doc_id_on_endpoint_rows_hooks: list[
            Callable[[Edge], bool]
        ] = []
        ef = embedding_function or get_embedding_function()
        self.embedding_length_limit = 512
        self._ef: EmbeddingFunctionLike = ef  # embedding_function or ef #embedding_functions.DefaultEmbeddingFunction()

        # Keep a 1-string convenience to reuse in cosine checks
        # convenience: single-string embed for verifiers
        def _embed_one(text: str):
            vecs = self._ef(
                [text]
            )  # DefaultEmbeddingFunction is callable(texts: List[str]) -> List[List[float]]
            return vecs[0] if vecs else None

        self._embed_one = _embed_one
        if backend is None or (type(backend) is str and backend == "chroma"):
            # 2) Chroma client + collections; inject embedder on vectorized collections
            ChromaClient, ChromaSettings = _import_chroma_client()
            self.chroma_client = ChromaClient(
                ChromaSettings(
                    is_persistent=True,
                    persist_directory=persist_directory or "./chroma_db",
                    anonymized_telemetry=False,
                )
            )
            # IMPORTANT: pass embedding_function to vector collections
            from threading import Lock

            self.collection_lock = {"node": Lock(), "edge": Lock()}

            self.node_index_collection = self.chroma_client.get_or_create_collection(
                "nodes_index",
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
            self.node_collection = self.chroma_client.get_or_create_collection(
                "nodes", embedding_function=self._ef, metadata={"hnsw:space": "cosine"}
            )
            self.edge_collection = self.chroma_client.get_or_create_collection(
                "edges", embedding_function=self._ef, metadata={"hnsw:space": "cosine"}
            )
            self.edge_endpoints_collection = (
                self.chroma_client.get_or_create_collection(
                    "edge_endpoints",
                    embedding_function=self._ef,
                    metadata={"hnsw:space": "cosine"},
                )
            )
            self.document_collection = self.chroma_client.get_or_create_collection(
                "documents",
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
            self.domain_collection = self.chroma_client.get_or_create_collection(
                "domains",
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
            self.node_docs_collection = self.chroma_client.get_or_create_collection(
                "node_docs",
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
            self.node_refs_collection = self.chroma_client.get_or_create_collection(
                "node_refs"
            )
            self.edge_refs_collection = self.chroma_client.get_or_create_collection(
                "edge_refs"
            )
            # Backend adapter (Phase 1: Chroma)
            self.backend: StorageBackend = ChromaBackend(
                node_index_collection=self.node_index_collection,
                node_collection=self.node_collection,
                edge_collection=self.edge_collection,
                edge_endpoints_collection=self.edge_endpoints_collection,
                document_collection=self.document_collection,
                domain_collection=self.domain_collection,
                node_docs_collection=self.node_docs_collection,
                node_refs_collection=self.node_refs_collection,
                edge_refs_collection=self.edge_refs_collection,
            )
            self.meta_sqlite = EngineSQLite(
                pathlib.Path(persist_directory or "./chroma_db"), "meta.sqlite"
            )
            self.meta_sqlite.ensure_initialized()
        elif _is_pgvector_backend_instance(backend):
            from .engine_postgres_meta import EnginePostgresMetaStore

            if type(backend) is str:
                raise Exception("unreacheable")
            else:
                backend2: PgVectorBackend = backend  # let static checker happy
                meta_postgre = EnginePostgresMetaStore(
                    engine=backend2.engine, schema=backend2.schema
                )
            self.backend: StorageBackend = backend

            meta_postgre.ensure_initialized()
            self.meta_sqlite = meta_postgre
        else:
            if isinstance(backend, str):
                raise ValueError(
                    f"Unsupported backend string {backend!r}. "
                    "Use backend='chroma' (or None), or pass a PgVectorBackend instance for Postgres."
                )
            raise ValueError(
                "Unrecognised argument for backend. "
                "Expected None/'chroma' or a PgVectorBackend instance."
            )
        # Backend UoW: in Postgres mode this becomes a real SQL transaction.
        self._backend_uow = _build_postgres_uow_if_needed(
            getattr(self, "backend", None)
        )

        from .lifecycle import LifecycleSubsystem

        self._backend_update_record_lifecycle = _backend_update_record_lifecycle
        self.lifecycle = LifecycleSubsystem(self)

        provider_cfg = default_task_provider_config or DefaultTaskProviderConfig()
        self.llm_tasks: LLMTaskSet = validate_llm_task_set(
            llm_tasks or build_default_llm_tasks(provider_cfg)
        )

        # Phase 1 resilience: derived "join indexes" (node_docs/node_refs/edge_refs/edge_endpoints)
        # are converged via a durable outbox-style index job queue (EngineSQLite or EnginePostgresMetaStore).
        # Fast path may attempt to apply immediately; correctness relies on reconciliation.
        self._phase1_enable_index_jobs: bool = True
        self.embeddings: Optional[Callable[[str], Optional[List[float]]]] = (
            build_azure_embedding_fn_from_env()
        )

        self.memory = Memory(location=self.cache_dir or os.path.join(".", ".kg_cache"))
        self._cached_extract_graph_with_llm = self.memory.cache( # engine owned, use engine cache
            self.extract_graph_with_llm, ignore=["self"]
        )
        self.cached_embed: Optional[Callable[[str], Iterable[float]]] = None
        if embedding_cache_path:
            self.embedding_cache = Memory(location=embedding_cache_path)

            def cached_embed(query, model_name):

                return self._iterative_defensive_emb(query)

            from functools import partial

            cached_embed = partial(
                cached(self.embedding_cache, cached_embed), model_name=self._ef.name()
            )
            # MethodType
            # a.foo = MethodType(foo, a)
            self.cached_embed = cast(Callable[[str], Iterable[float]], cached_embed)

        # Namespaced subsystem APIs (new source-of-truth surface).
        self.read = ReadSubsystem(self)
        self.write = WriteSubsystem(self)
        self.extract = ExtractSubsystem(self)
        self.persist = PersistSubsystem(self)
        self.rollback = RollbackSubsystem(self)
        self.adjudicate = AdjudicateSubsystem(self)
        self.ingest = IngestSubsystem(self)
        self.embed = EmbedSubsystem(self)

        # Initialize the search index subsystem if persistence directory is set.
        # For pure in-memory or alternative setups you might handle pathing differently,
        # but defaulting to persist_directory/index.db
        idx_db_path = ":memory:"
        if persist_directory:
            idx_db_path = str(pathlib.Path(persist_directory) / "index.db")
        elif (
            hasattr(self, "meta_sqlite")
            and hasattr(self.meta_sqlite, "conn_str")
            and self.meta_sqlite.conn_str != ":memory:"
        ):
            # PG setup might not have persist_directory in kwargs but might store sqlite locally
            pass  # fall back to memory or consider handling PG specifically if index.db isn't used there

        self.search_index = SearchIndexService(self, index_db_path=idx_db_path)

    def _emit_change(
        self,
        *,
        op: Op,
        entity: EntityRefModel,
        payload: object,
        run_id: str | None = None,
        step_id: str | None = None,
    ) -> None:
        seq = self.changes.next_seq()
        ev = ChangeEvent(
            seq=seq,
            op=op,
            ts_unix_ms=int(time.time() * 1000),
            entity=entity.model_dump_entity_ref(),
            payload=payload,
            run_id=run_id,
            step_id=step_id,
        )
        self.changes.emit(ev)
        if self._oplog:
            self._oplog.append(ev)

    def iterative_defensive_emb(self, emb_text0):
        return self.embed.iterative_defensive_emb(emb_text0)

    # ... existing methods ...
    @staticmethod
    def _node_doc_and_meta(n: "Node") -> tuple[str, dict]:
        return _node_doc_and_meta(n)

    @staticmethod
    def _edge_doc_and_meta(e: "Edge") -> tuple[str, dict]:
        return _edge_doc_and_meta(e)

    def _maybe_reindex_edge_refs(self, edge: Edge, *, force: bool = False) -> None:
        return self.write.maybe_reindex_edge_refs(edge, force=force)

    def _maybe_reindex_node_refs(self, node: Node, *, force: bool = False) -> None:
        return self.write.maybe_reindex_node_refs(node, force=force)

    def _index_edge_refs(self, edge: Edge) -> list[str]:
        return self.write.index_edge_refs(edge)

    def _index_node_refs(self, node: Node) -> list[str]:
        return self.write.index_node_refs(node)

    def _alias_doc_in_prompt(self) -> str:
        return self.extract.alias_doc_in_prompt()

    def _delias_one_span(self, span: Span, real_doc_id: str) -> Span:
        return self.extract.delias_one_span(span, real_doc_id)

    def _dealias_one_grounding(
        self, grounding: Grounding, real_doc_id: str
    ) -> Grounding:
        return self.extract.dealias_one_grounding(grounding, real_doc_id)

    def _dealias_span(self, mentions: List[Grounding] | None, real_doc_id: str):
        return self.extract.dealias_span(mentions, real_doc_id)

    def _target_from_node(self, n: "Node") -> "AdjudicationTarget":
        return self.adjudicate.target_from_node(n)

    def _target_from_edge(self, e: "Edge") -> "AdjudicationTarget":
        return self.adjudicate.target_from_edge(e)

    def check_document_exist(self, document_id: str | list[str]):
        doc_ids = [document_id] if type(document_id) is str else document_id
        got = self.backend.document_get(ids=doc_ids, include=[])
        return set(got["ids"]).union(set(document_id))

    def _fetch_document_text(self, document_id: str) -> str:
        return self.extract.fetch_document_text(document_id)

    @staticmethod
    def _cosine(u: List[float], v: List[float]) -> Optional[float]:
        if not u or not v or len(u) != len(v):
            return None
        dot = sum(a * b for a, b in zip(u, v))
        nu = math.sqrt(sum(a * a for a in u))
        nv = math.sqrt(sum(b * b for b in v))
        if nu == 0 or nv == 0:
            return None
        return dot / (nu * nv)

    # ----------------------------
    # Chroma-style api adders
    # ----------------------------
    def add_pure_node(self, node: PureChromaNode):
        return self.write.add_pure_node(node)

    def add_pure_edge(self, edge: PureChromaEdge):
        return self.write.add_pure_edge(edge)

    # ---- Unit of Work (meta-store transaction boundary) ----
    def _ensure_uow_ctxvars(self):

        if not hasattr(self, "_uow_ctx_conn"):
            self._uow_ctx_conn = contextvars.ContextVar("gke_uow_conn", default=None)
            self._uow_ctx_depth = contextvars.ContextVar("gke_uow_depth", default=0)

    @contextmanager
    def uow(self):
        """Nest-safe Unit of Work for meta-store writes.

        Nested calls join the outer transaction; only the outermost commits/rolls back.
        """
        self._ensure_uow_ctxvars()

        depth = self._uow_ctx_depth.get()
        if depth > 0:
            self._uow_ctx_depth.set(depth + 1)
            try:
                yield self._uow_ctx_conn.get()
            finally:
                self._uow_ctx_depth.set(self._uow_ctx_depth.get() - 1)
            return

        self._uow_ctx_depth.set(1)
        try:
            with self.meta_sqlite.transaction() as conn:
                token = self._uow_ctx_conn.set(conn)
                try:
                    with self._backend_uow.transaction():
                        yield conn
                finally:
                    self._uow_ctx_conn.reset(token)
        finally:
            self._uow_ctx_depth.set(0)

    def get_collection_lock(self, collection_name):
        """Return the lock for a given collection name.

        Note: callers should not depend on the raw Chroma collection object; use backend methods instead.
        """
        if collection_name not in self.collection_lock:
            raise ValueError(f"{collection_name} has not implemented lock")
        return self.collection_lock[collection_name]

    @engine_context
    def add_node(self, node: Node, doc_id: Optional[str] = None):
        return self.write._add_node_impl(node, doc_id=doc_id)

    def _fanout_endpoints_rows(self, edge: Edge, doc_id: str | None):
        return self.write.fanout_endpoints_rows(edge, doc_id)

    def enrich_edge_meta(self, edge):
        return self.write.enrich_edge_meta(edge)

    @engine_context
    def add_edge(self, edge: Edge, doc_id: Optional[str] = None):
        return self.write._add_edge_impl(edge, doc_id=doc_id)

    @engine_context
    def add_document(self, document: Document):
        return self.write.add_document(document)

    def add_domain(self, domain: Domain):
        return self.write.add_domain(domain)

    def _index_node_docs(self, node: Node) -> list[str]:
        return self.write.index_node_docs(node)

    def _nodes_by_doc(
        self, doc_id: str, insertion_method: Optional[str] = None
    ) -> list[str]:
        return self.read.node_ids_by_doc(doc_id, insertion_method=insertion_method)

    def _edge_ids_by_doc(
        self, doc_id: str, insertion_method: Optional[str] = None
    ) -> list[str]:
        return self.read.edge_ids_by_doc(doc_id, insertion_method=insertion_method)

    def _prune_node_refs_for_doc(self, node_id: str, doc_id: str) -> bool:
        return self.write.prune_node_refs_for_doc(node_id, doc_id)

    def rebuild_edge_refs_for_doc(self, doc_id: str) -> int:
        return self.write.rebuild_edge_refs_for_doc(doc_id)

    def rebuild_all_edge_refs(self) -> int:
        return self.write.rebuild_all_edge_refs()

    def edges_by_doc(self, doc_id: str, where: Optional[dict] = None) -> list[str]:
        return self.read.edges_by_doc(doc_id, where=where)

    def list_edges_with_ref_filter(
        self, doc_id: str, where: dict | None = None
    ) -> list[Edge]:
        return self.read.list_edges_with_ref_filter(doc_id, where=where)

    def nodes_by_ids(self, node_ids):
        return self.backend.node_get(ids=node_ids)

    def edges_by_ids(self, edge_ids):
        return self.backend.edge_get(ids=edge_ids)

    def nodes_by_doc(self, doc_id: str, *, where: Optional[dict] = None) -> list[str]:
        return self.read.nodes_by_doc(doc_id, where=where)

    def list_nodes_with_ref_filter(
        self, doc_id: str, *, where: Optional[dict] = None
    ) -> list[Node]:
        return self.read.list_nodes_with_ref_filter(doc_id, where=where)

    def rebuild_node_refs_for_doc(self, doc_id: str) -> int:
        return self.write.rebuild_node_refs_for_doc(doc_id)

    def rebuild_all_node_refs(self) -> int:
        return self.write.rebuild_all_node_refs()

    # ----------------------------
    # helpers for rollback
    # ----------------------------

    def _delete_edges_by_ids(self, edge_ids: list[str]):
        return self.write.delete_edges_by_ids(edge_ids)

    # ----------------------------
    # Vector queries
    # ----------------------------
    def vector_search_nodes(self, embedding: List[float], top_k: int = 5):
        return self.backend.node_query(query_embeddings=[embedding], n_results=top_k)

    def vector_search_edges(self, embedding: List[float], top_k: int = 5):
        return self.backend.edge_query(query_embeddings=[embedding], n_results=top_k)

    def _choose_anchor(self, node_ids: list[str]) -> str:
        return self.adjudicate.choose_anchor(node_ids)

    def _rebalance_same_as_edge(
        self, e: Edge, removed_node_id: str
    ) -> tuple[bool, Edge | None]:
        return self.adjudicate.rebalance_same_as_edge(e, removed_node_id)

    def persist_graph(self, *, parsed: PureGraph, session_id: str, mode=None):
        return self.persist.persist_graph(
            parsed=parsed,
            session_id=session_id,
            mode=mode,
        )

    def persist_graph_extraction(
        self,
        *,
        document: Document,
        parsed: LLMGraphExtraction,
        mode: str = "append",  # "replace" | "append" | "skip-if-exists"
        assign_real_id_in_place=True,
    ) -> dict:
        return self.persist.persist_graph_extraction(
            document=document,
            parsed=parsed,
            mode=mode,
            assign_real_id_in_place=assign_real_id_in_place,
        )

    def extract_graph_with_llm(
        self,
        *,
        content: str,
        doc_type: str,
        alias_nodes_str="[Empty]",
        alias_edges_str="[Empty]",
        with_parsed=True,
        instruction_for_node_edge_contents_parsing_inclusion: None | str = None,
        validate=True,
        autofix: bool | str = True,
        last_iteration_result=None,
        extraction_schema_mode: ExtractionSchemaMode | None = None,
        offset_mismatch_policy: OffsetMismatchPolicy = "exact_fuzzy",
        offset_repair_scorer: OffsetRepairScorer | None = None,
    ):
        return self.extract.extract_graph_with_llm(
            content=content,
            doc_type=doc_type,
            alias_nodes_str=alias_nodes_str,
            alias_edges_str=alias_edges_str,
            with_parsed=with_parsed,
            instruction_for_node_edge_contents_parsing_inclusion=instruction_for_node_edge_contents_parsing_inclusion,
            validate=validate,
            autofix=autofix,
            last_iteration_result=last_iteration_result,
            extraction_schema_mode=extraction_schema_mode,
            offset_mismatch_policy=offset_mismatch_policy,
            offset_repair_scorer=offset_repair_scorer,
        )

    def get_document(self, doc_id: str):
        return self.read.get_document(doc_id)

    def get_span_validator_of_doc_type(
        self,
        *,
        doc_id: str | None = None,
        doc_type: Literal["text", "ocr_document"] | str | None = None,
        document: Document | None = None,
    ) -> BaseDocValidator:
        return self.extract.get_span_validator_of_doc_type(
            doc_id=doc_id,
            doc_type=doc_type,
            document=document,
        )

    def persist_document_graph_extraction(
        self,
        *,
        doc_id,
        parsed: GraphExtractionWithIDs | LLMGraphExtraction,
        mode: str = "append",  # "replace" | "append" | "skip-if-exists"
    ) -> dict:
        return self.persist.persist_document_graph_extraction(
            doc_id=doc_id,
            parsed=parsed,
            mode=mode,
        )

    def rollback_document_extraction(
        self,
        doc_id: str,
        extraction_method: Literal["llm_graph_extraction", "document_ingestion"],
    ) -> dict:
        return self.rollback.rollback_document_extraction(doc_id, extraction_method)

    def ingest_document_with_llm(
        self,
        document: Document,
        *,
        mode: str = "append",
        instruction_for_node_edge_contents_parsing_inclusion=None,
        raw_with_parsed=None,
        extraction_schema_mode: ExtractionSchemaMode | None = None,
        offset_mismatch_policy: OffsetMismatchPolicy = "exact_fuzzy",
        offset_repair_scorer: OffsetRepairScorer | None = None,
    ):
        return self.ingest.ingest_document_with_llm(
            document,
            mode=mode,
            instruction_for_node_edge_contents_parsing_inclusion=instruction_for_node_edge_contents_parsing_inclusion,
            raw_with_parsed=raw_with_parsed,
            extraction_schema_mode=extraction_schema_mode,
            offset_mismatch_policy=offset_mismatch_policy,
            offset_repair_scorer=offset_repair_scorer,
        )

    def _extract_graph_with_llm(
        self, content: str, doc: Document
    ) -> Tuple[Any, Optional[LLMGraphExtraction], Optional[str]]:
        return self.ingest.extract_graph_with_llm_internal(content, doc)

    def _ingest_text_with_llm(
        self,
        *,
        doc_id: str,
        content: str,
        auto_adjudicate: bool = False,
        extraction_schema_mode: ExtractionSchemaMode | None = None,
        offset_mismatch_policy: OffsetMismatchPolicy = "exact_fuzzy",
        offset_repair_scorer: OffsetRepairScorer | None = None,
    ):
        return self.ingest.ingest_text_with_llm(
            doc_id=doc_id,
            content=content,
            auto_adjudicate=auto_adjudicate,
            extraction_schema_mode=extraction_schema_mode,
            offset_mismatch_policy=offset_mismatch_policy,
            offset_repair_scorer=offset_repair_scorer,
        )

    def prune_node_from_edges(self, node_id: str):
        return self.rollback.prune_node_from_edges(node_id)

    def rollback_document(self, document_id: str):
        return self.rollback.rollback_document(document_id)

    def rollback_many_documents(self, document_ids: list[str]):
        return self.rollback.rollback_many_documents(document_ids)

    # ----------------------------
    # Adjudication (LLM-assisted merge decision)
    # ----------------------------
    def _fetch_target(self, t: AdjudicationTarget) -> Node | Edge:
        return self.adjudicate.fetch_target(t)

    def commit_merge_target(
        self,
        left: AdjudicationTarget,
        right: AdjudicationTarget,
        verdict: AdjudicationVerdict,
    ) -> str:
        canonical_id = self.merge_policy.commit_merge_target(left, right, verdict)
        return canonical_id

    def add_edge_with_endpoint_docs(
        self, edge: Edge, endpoint_doc_ids: dict[str, str | None]
    ):
        # Add the main edge row (neutral doc_id)
        doc = edge.model_dump_json(field_mode="backend")
        self.backend.edge_add(
            ids=[edge.id],
            documents=[],
            embeddings=[edge.embedding]
            if edge.embedding
            else [self._iterative_defensive_emb(doc)],
            metadatas=[
                self.chroma_sanitize_metadata(
                    {
                        "doc_id": getattr(edge, "doc_id", None),
                        "relation": edge.relation,
                        "source_ids": json.dumps(edge.source_ids),
                        "target_ids": json.dumps(edge.target_ids),
                        "type": edge.type,
                        "summary": edge.summary,
                        "domain_id": edge.domain_id,
                        "canonical_entity_id": edge.canonical_entity_id,
                        "properties": json.dumps(edge.properties)
                        if edge.properties is not None
                        else None,
                        "references": json.dumps(
                            [
                                ref.model_dump(field_mode="backend")
                                for ref in (edge.mentions or [])
                            ]
                        ),
                    }
                )
            ],
        )

        # Fan-out edge_endpoints; each endpoint gets the *node's* doc_id for rollback
        ep_ids, ep_docs, ep_metas = [], [], []
        for role, node_ids in (
            ("src", edge.source_ids or []),
            ("tgt", edge.target_ids or []),
        ):
            for nid in node_ids:
                eid = f"{edge.id}::{role}::{nid}"
                doc_id = endpoint_doc_ids.get(nid)
                meta_ep = self.chroma_sanitize_metadata(
                    {
                        "id": eid,
                        "edge_id": edge.id,
                        "node_id": nid,
                        "role": role,
                        "relation": edge.relation,
                        "doc_id": doc_id,  # <-- specific to that node's document
                    }
                )
                ep_ids.append(eid)
                ep_docs.append(json.dumps(meta_ep))
                ep_metas.append(meta_ep)

        if ep_ids:
            self.backend.edge_endpoints_add(
                ids=ep_ids,
                documents=ep_docs,
                metadatas=ep_metas,
                embeddings=[self._iterative_defensive_emb(d) for d in ep_docs],
            )

    def _classify_endpoint_id(self, rid: str) -> str:
        return self.adjudicate.classify_endpoint_id(rid)

    def _split_endpoints(
        self, src_ids: list[str] | None, tgt_ids: list[str] | None
    ) -> tuple[
        list[Any], list[Any], list[Any], list[Any]
    ]:  # -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        return self.adjudicate.split_endpoints(src_ids, tgt_ids)

    def commit_merge(
        self,
        left: Node,
        right: Node,
        verdict: AdjudicationVerdict,
        method: str = "unspecified",
    ) -> str:
        canonical_id = self.merge_policy.commit_merge(left, right, verdict, method)
        return canonical_id

    def commit_any_kind(
        self,
        node_or_edge_l: AdjudicationTarget,
        node_or_edge_r: AdjudicationTarget,
        verdict: AdjudicationVerdict,
    ) -> str:
        return self.merge_policy.commit_any_kind(
            node_or_edge_l, node_or_edge_r, verdict
        )

    def generate_merge_candidates_doc_brute_force(
        self,
        kind: str = "node",
        scope_doc_id: Optional[str] = None,
        top_k: int = 200,
        *,
        # NEW optional knobs:
        allowed_docs: Optional[List[str]] = None,
        anchor_doc_id: Optional[str] = None,
        cross_doc_only: bool = False,
        anchor_only: bool = True,
    ):
        """
        Back-compat:
        - If no new knobs are provided and scope_doc_id is set, behave exactly like before (same-doc only).
        - Otherwise, use the unified proposer with richer scoping.
        """
        if (
            allowed_docs is None
            and anchor_doc_id is None
            and not cross_doc_only
            and scope_doc_id
        ):
            # legacy behavior (same doc only)
            return self.proposer.same_kind_in_doc(
                engine=self,
                doc_id=scope_doc_id,
                kind="node" if kind == "node" else "edge",
            )

        pair_kind = "node_node" if kind == "node" else "edge_edge"
        return self.proposer.propose_any_kind_any_doc(
            engine=self,
            pair_kind=pair_kind,
            allowed_docs=allowed_docs,
            anchor_doc_id=anchor_doc_id or None,
            cross_doc_only=cross_doc_only,
            anchor_only=anchor_only,
            limit_per_bucket=top_k,
        )

    def generate_cross_kind_candidates(
        self,
        scope_doc_id: Optional[str] = None,
        limit_per_bucket: int = 200,
        *,
        # NEW optional knobs:
        allowed_docs: Optional[List[str]] = None,
        anchor_doc_id: Optional[str] = None,
        cross_doc_only: bool = False,
        anchor_only: bool = True,
    ) -> List[AdjudicationCandidate]:
        """
        Back-compat:
        - Without new knobs and with scope_doc_id set, behave as before (same-doc only).
        - Otherwise use unified proposer in nodeâ†”edge mode.
        """
        if not self.allow_cross_kind_adjudication:
            raise ValueError("Configuration disallow cross kind adjudication.")

        if (
            allowed_docs is None
            and anchor_doc_id is None
            and not cross_doc_only
            and scope_doc_id
        ):
            pairs = self.proposer.cross_kind_in_doc(
                engine=self, doc_id=scope_doc_id, limit_per_bucket=limit_per_bucket
            )
        else:
            pairs = self.proposer.propose_any_kind_any_doc(
                engine=self,
                pair_kind="node_edge",
                allowed_docs=allowed_docs,
                anchor_doc_id=anchor_doc_id,
                cross_doc_only=cross_doc_only,
                anchor_only=anchor_only,
                limit_per_bucket=limit_per_bucket,
            )

        return [
            AdjudicationCandidate(
                left=self.adjudicate.target_from_node(left),
                right=self.adjudicate.target_from_edge(right),
                question="node_edge_equivalence",
            )
            for left, right in pairs
        ]

    def generate_merge_candidates(
        self,
        new_node: Union[Node, str, Sequence[Union[Node, str]]],
        top_k: int = 10,
        *,
        # NEW optional knobs (post-filtering on vector hits):
        allowed_docs: Optional[List[str]] = None,
        anchor_doc_id: Optional[str] = None,
        cross_doc_only: bool = False,
        anchor_only: bool = True,
    ):
        out = self.proposer.generate_merge_candidates(
            engine=self,
            new_node=new_node,
            top_k=top_k,
            allowed_docs=allowed_docs,
            anchor_doc_id=anchor_doc_id,
            cross_doc_only=cross_doc_only,
            anchor_only=anchor_only,
            new_edge=[],
        )
        return out

    def adjudicate_pair(
        self, left: AdjudicationTarget, right: AdjudicationTarget, question: str
    ):
        """deligate to adjudicator to decide if any nodes/ edges meaning the same"""
        return self.adjudicator.adjudicate_pair(left, right, question)

    def adjudicate_pair_trace(
        self,
        left: AdjudicationTarget,
        right: AdjudicationTarget,
        question: str,
        *,
        cache_dir=None,
    ):
        """Return the validated adjudication result plus raw/parsing diagnostics."""
        return self.adjudicator.adjudicate_pair_trace(
            left, right, question, cache_dir=cache_dir
        )

    def adjudicate_merge(
        self, left_node: Node | Edge, right_node: Node | Edge
    ) -> Dict[Any, Any] | BaseModel:
        """deligate to adjudicator to commit merge if any nodes/ edges was (supposedly) earlier decided meaning the same.
        The api only join them with reconsidering."""
        return self.adjudicator.adjudicate_merge(left_node, right_node)

    def batch_adjudicate_merges(
        self,
        pairs: List[Tuple["Node", "Node"]],
        question_code: "AdjudicationQuestionCode" = AdjudicationQuestionCode.SAME_ENTITY,
    ):
        if not pairs:
            return []
        return self.adjudicator.batch_adjudicate_merges(pairs, question_code)

    def add_page(
        self,
        *,
        document_id: str,
        page_text: str | List[str] | Dict[str, Any],
        page_number: int | None = None,
        auto_adjudicate: bool = True,
        extraction_schema_mode: ExtractionSchemaMode | None = None,
        offset_mismatch_policy: OffsetMismatchPolicy = "exact_fuzzy",
        offset_repair_scorer: OffsetRepairScorer | None = None,
    ):
        return self.ingest.add_page(
            document_id=document_id,
            page_text=page_text,
            page_number=page_number,
            auto_adjudicate=auto_adjudicate,
            extraction_schema_mode=extraction_schema_mode,
            offset_mismatch_policy=offset_mismatch_policy,
            offset_repair_scorer=offset_repair_scorer,
        )

    def verify_mentions_for_doc(
        self,
        document_id: str,
        *,
        source_text: Optional[str] = None,
        min_ngram: int = 5,
        threshold: float = 0.70,
        weights: Dict[str, float] = {
            "rapidfuzz": 0.5,
            "coverage": 0.3,
            "embedding": 0.2,
        },
        update_edges: bool = True,
    ) -> Dict[str, int]:
        to_return = self.verifier.verify_mentions_for_doc(
            document_id,
            source_text=source_text,
            min_ngram=min_ngram,
            threshold=threshold,
            weights=weights,
            update_edges=update_edges,
        )
        return to_return

    def ids_with_insertion_method(
        self,
        *,
        kind: str,  # "node" | "edge"
        insertion_method: str,
        ids: Optional[Iterable[str]] = None,  # optionally restrict to this set
        doc_id: Optional[str] = None,  # optionally restrict to a document
    ) -> list[str]:
        return self.read.ids_with_insertion_method(
            kind=kind,
            insertion_method=insertion_method,
            ids=list(ids) if ids is not None else None,
            doc_id=doc_id,
        )

    def _verify_one_reference(
        self,
        extracted_text: str,
        full_text: str,
        ref: Span,
        *,
        min_ngram: int = 5,
        weights: Dict[str, float] = {
            "rapidfuzz": 0.5,
            "coverage": 0.3,
            "embedding": 0.2,
        },
        threshold: float = 0.70,
    ) -> Span:
        return self.verifier._verify_one_reference(
            extracted_text=extracted_text,
            full_text=full_text,
            ref=ref,
            min_ngram=min_ngram,
            weights=weights,
            threshold=threshold,
        )

    def verify_mentions_for_items(
        self,
        items: List[Tuple[str, str]],  # list of ("node"|"edge", id)
        *,
        source_text_by_doc: Optional[Dict[str, str]] = None,
        min_ngram: int = 5,
        threshold: float = 0.70,
        weights: Dict[str, float] = {
            "rapidfuzz": 0.5,
            "coverage": 0.3,
            "embedding": 0.2,
        },
    ) -> Dict[str, int]:
        """
        Targeted verification for a mixed set of nodes/edges.
        source_text_by_doc lets you pass pre-fetched doc text keyed by doc_id.
        """

        to_return = self.verifier.verify_mentions_for_items(
            items,
            source_text_by_doc=source_text_by_doc,
            min_ngram=min_ngram,
            threshold=threshold,
            weights=weights,
        )
        return to_return

    def _iterative_defensive_emb(self, emb_text0):
        return self.embed.iterative_defensive_emb_internal(emb_text0)


class ApproxTokenizer:
    def count_tokens(self, text: str) -> int:
        # very cheap approximation; stable for budget enforcement
        return max(1, len(text) // 4)


_SHIM_METHOD_MAP: dict[str, tuple[str, str]] = {
    # write
    "add_node": ("write", "add_node"),
    "add_edge": ("write", "add_edge"),
    "add_document": ("write", "add_document"),
    "add_domain": ("write", "add_domain"),
    # read
    "get_nodes": ("read", "get_nodes"),
    "get_edges": ("read", "get_edges"),
    "query_nodes": ("read", "query_nodes"),
    "query_edges": ("read", "query_edges"),
    # lifecycle
    "tombstone_node": ("lifecycle", "tombstone_node"),
    "redirect_node": ("lifecycle", "redirect_node"),
    "tombstone_edge": ("lifecycle", "tombstone_edge"),
    "redirect_edge": ("lifecycle", "redirect_edge"),
    # persist / ingest / rollback
    "persist_graph_extraction": ("persist", "persist_graph_extraction"),
    "ingest_document_with_llm": ("ingest", "ingest_document_with_llm"),
    "rollback_document": ("rollback", "rollback_document"),
}


def _install_legacy_shims() -> None:
    for method_name, (namespace_name, namespace_method) in _SHIM_METHOD_MAP.items():
        original = getattr(GraphKnowledgeEngine, method_name, None)
        if original is None:
            continue
        impl_name = f"_impl_{method_name}"
        if not hasattr(GraphKnowledgeEngine, impl_name):
            setattr(GraphKnowledgeEngine, impl_name, original)

        def _make_shim(
            *,
            original_method: Callable[..., Any],
            old_name: str,
            ns_name: str,
            ns_method: str,
        ):
            @wraps(original_method)
            def _shim(self, *args, **kwargs):
                warnings.warn(
                    (
                        f"GraphKnowledgeEngine.{old_name} is deprecated; "
                        f"use engine.{ns_name}.{ns_method} instead."
                    ),
                    DeprecationWarning,
                    stacklevel=2,
                )
                namespace_obj = getattr(self, ns_name)
                return getattr(namespace_obj, ns_method)(*args, **kwargs)

            return _shim

        shim = _make_shim(
            original_method=original,
            old_name=method_name,
            ns_name=namespace_name,
            ns_method=namespace_method,
        )
        setattr(GraphKnowledgeEngine, method_name, shim)


_install_legacy_shims()
