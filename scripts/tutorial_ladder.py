from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence
if True:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from graph_knowledge_engine.conversation.knowledge_retriever import KnowledgeRetriever
from graph_knowledge_engine.conversation.memory_retriever import MemoryRetriever
from graph_knowledge_engine.conversation.models import (
    ConversationAIResponse,
    ConversationEdge,
    ConversationNode,
    FilteringResult,
    MetaFromLastSummary,
    RetrievalResult,
)
from graph_knowledge_engine.conversation.service import ConversationService
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.models import (
    Edge,
    Grounding,
    MentionVerification,
    Node,
    Span,
)
from graph_knowledge_engine.llm_tasks import (
    AnswerWithCitationsTaskRequest,
    AnswerWithCitationsTaskResult,
    DefaultTaskProviderConfig,
    LLMTaskProviderHints,
    LLMTaskSet,
    RepairCitationsTaskRequest,
    RepairCitationsTaskResult,
    SummarizeContextTaskRequest,
    SummarizeContextTaskResult,
    build_default_llm_tasks,
)

# replace with your embedding functions if you want real semantic embeddings
from scripts.tutorial_sections._helpers import LexicalHashEmbeddingFunction 


def _now_id(prefix: str) -> str:
    return f"{prefix}-{int(time.time() * 1000)}"


def _span(doc_id: str, excerpt: str, *, insertion_method: str = "tutorial") -> Span:
    return Span(
        collection_page_url=f"tutorial/{doc_id}",
        document_page_url=f"tutorial/{doc_id}",
        doc_id=doc_id,
        insertion_method=insertion_method,
        page_number=1,
        start_char=0,
        end_char=max(1, len(excerpt)),
        excerpt=excerpt[:512],
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="system", is_verified=True, score=1.0, notes="tutorial"
        ),
    )


def _grounding(
    doc_id: str, excerpt: str, *, insertion_method: str = "tutorial"
) -> Grounding:
    return Grounding(spans=[_span(doc_id, excerpt, insertion_method=insertion_method)])


def _print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _configure_tutorial_cache_env(data_dir: Path) -> None:
    os.environ["GKE_JOBLIB_CACHE_DIR"] = str((data_dir / ".joblib").resolve())


def _context_messages_to_text(messages: Sequence[Any]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = str(getattr(msg, "role", "") or "system")
        content = str(getattr(msg, "content", "") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _emit_ollama_pull_note(model_name: str | None) -> None:
    model = str(model_name or "qwen3:4b")
    print(
        (
            f"[tutorial] Ollama provider selected with model '{model}'. "
            "If this is the first run, Ollama may need to download the model and "
            "that can take several minutes. Native Ollama pull progress will appear "
            "in the terminal if a download is required."
        ),
        file=sys.stderr,
    )


def _extract_ollama_model_names(response: Any) -> list[str]:
    models = None
    if isinstance(response, dict):
        models = response.get("models")
    else:
        models = getattr(response, "models", None)

    names: list[str] = []
    for item in models or []:
        name = None
        if isinstance(item, dict):
            name = item.get("model") or item.get("name")
        else:
            name = getattr(item, "model", None) or getattr(item, "name", None)
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _ensure_local_ollama_model(model_name: str | None) -> None:
    model = str(model_name or "qwen3:4b")
    try:
        import ollama  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Ollama tutorial mode requires the 'ollama' Python package."
        ) from exc

    try:
        response = ollama.Client().list()
    except Exception as exc:
        raise RuntimeError(
            "Failed to query the local Ollama server. Ensure Ollama is running."
        ) from exc

    names = _extract_ollama_model_names(response)
    if any(model == name or name.startswith(f"{model}:") for name in names):
        return
    available = ", ".join(names) if names else "(none)"
    raise RuntimeError(
        f"Ollama model '{model}' is not available locally. "
        f"Run `ollama pull {model}` first. Available local models: {available}"
    )


def _invoke_structured_model(
    *, model: Any, schema: type[Any], prompt: str
) -> tuple[object | None, dict[str, Any] | None, str | None]:
    try:
        structured = model.with_structured_output(schema, include_raw=True)
    except TypeError:
        structured = model.with_structured_output(schema)
    result = structured.invoke(prompt)
    if isinstance(result, dict):
        raw = result.get("raw")
        parsed = result.get("parsed")
        parsing_error = result.get("parsing_error")
    else:
        raw = None
        parsed = result
        parsing_error = None
    if parsing_error is not None:
        return raw, None, str(parsing_error)
    try:
        validated = schema.model_validate(parsed)
    except Exception as exc:
        return raw, None, str(exc)
    payload = validated.model_dump(mode="python")
    return raw, payload if isinstance(payload, dict) else {"value": payload}, None


def _build_live_tutorial_model(*, provider: str, model_name: str | None) -> Any:
    provider_key = str(provider or "deterministic").strip().lower()
    if provider_key == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Gemini tutorial mode requires 'langchain-google-genai'. "
                "Install the Gemini extra and set GOOGLE_API_KEY."
            ) from exc
        return ChatGoogleGenerativeAI(
            model=model_name or os.getenv("GKE_TUTORIAL_GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=0.1,
            max_retries=2,
        )
    if provider_key == "openai":
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "OpenAI tutorial mode requires 'langchain-openai'. "
                "Install the OpenAI extra and set OPENAI_API_KEY."
            ) from exc
        return ChatOpenAI(
            model=model_name or os.getenv("GKE_TUTORIAL_OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=0.1,
            max_retries=2,
        )
    if provider_key == "ollama":
        _emit_ollama_pull_note(model_name)
        _ensure_local_ollama_model(model_name)
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Ollama tutorial mode requires 'langchain-ollama'. "
                "Install it and ensure the Ollama server is running."
            ) from exc
        return ChatOllama(
            model=model_name or os.getenv("GKE_TUTORIAL_OLLAMA_MODEL", "qwen3:4b"),
            temperature=0.1,
        )
    raise ValueError(f"Unsupported tutorial provider: {provider!r}")


def _build_provider_task_set(
    *,
    base_tasks: LLMTaskSet,
    provider: str,
    model_name: str | None,
) -> tuple[LLMTaskSet, str]:
    provider_key = str(provider or "deterministic").strip().lower()
    model = _build_live_tutorial_model(provider=provider_key, model_name=model_name)
    resolved_model = str(
        getattr(model, "model", "") or getattr(model, "model_name", "") or model_name or ""
    )

    def _answer_with_citations(
        request: AnswerWithCitationsTaskRequest,
    ) -> AnswerWithCitationsTaskResult:
        prompt = (
            f"{request.system_prompt}\n\n"
            "Answer the question using only the supplied evidence.\n"
            "Return structured output matching the requested schema.\n\n"
            f"Question:\n{request.question}\n\n"
            f"Evidence:\n{request.evidence}"
        )
        raw, payload, parsing_error = _invoke_structured_model(
            model=model,
            schema=request.response_model,
            prompt=prompt,
        )
        return AnswerWithCitationsTaskResult(
            answer_payload=payload,
            raw=raw,
            parsing_error=parsing_error,
        )

    def _repair_citations(
        request: RepairCitationsTaskRequest,
    ) -> RepairCitationsTaskResult:
        prompt = (
            f"{request.system_prompt}\n\n"
            "Repair or simplify the answer so it conforms to the requested schema.\n"
            "Use only the supplied evidence and avoid unsupported citations.\n\n"
            f"Question:\n{request.question}\n\n"
            f"Current answer text:\n{request.answer_text}\n\n"
            f"Evidence:\n{request.evidence}"
        )
        raw, payload, parsing_error = _invoke_structured_model(
            model=model,
            schema=request.response_model,
            prompt=prompt,
        )
        return RepairCitationsTaskResult(
            answer_payload=payload,
            raw=raw,
            parsing_error=parsing_error,
        )

    answer_hint = provider_key if provider_key in {"gemini", "openai", "ollama"} else "custom"
    hints = LLMTaskProviderHints(
        extract_graph_provider=base_tasks.provider_hints.extract_graph_provider,
        adjudicate_pair_provider=base_tasks.provider_hints.adjudicate_pair_provider,
        adjudicate_batch_provider=base_tasks.provider_hints.adjudicate_batch_provider,
        filter_candidates_provider=base_tasks.provider_hints.filter_candidates_provider,
        summarize_context_provider=base_tasks.provider_hints.summarize_context_provider,
        answer_with_citations_provider=answer_hint,
        repair_citations_provider=answer_hint,
    )
    return (
        LLMTaskSet(
            extract_graph=base_tasks.extract_graph,
            adjudicate_pair=base_tasks.adjudicate_pair,
            adjudicate_batch=base_tasks.adjudicate_batch,
            filter_candidates=base_tasks.filter_candidates,
            summarize_context=base_tasks.summarize_context,
            answer_with_citations=_answer_with_citations,
            repair_citations=_repair_citations,
            provider_hints=hints,
        ),
        resolved_model,
    )


def _build_engines(data_dir: Path) -> tuple[GraphKnowledgeEngine, GraphKnowledgeEngine]:
    data_dir.mkdir(parents=True, exist_ok=True)
    ef = LexicalHashEmbeddingFunction()
    kg_engine = GraphKnowledgeEngine(
        persist_directory=str(data_dir / "kg"),
        kg_graph_type="knowledge",
        embedding_function=ef,
    )
    conv_engine = GraphKnowledgeEngine(
        persist_directory=str(data_dir / "conv"),
        kg_graph_type="conversation",
        embedding_function=ef,
    )
    return kg_engine, conv_engine


def _build_workflow_engine(data_dir: Path) -> GraphKnowledgeEngine:
    ef = LexicalHashEmbeddingFunction()
    return GraphKnowledgeEngine(
        persist_directory=str(data_dir / "wf"),
        kg_graph_type="workflow",
        embedding_function=ef,
    )


def _has_seed_data(kg_engine: GraphKnowledgeEngine) -> bool:
    got = kg_engine.backend.node_get(ids=["K:architecture"], include=[])
    return bool(got.get("ids"))


def _upsert_knowledge_graph(kg_engine: GraphKnowledgeEngine) -> None:
    rows = [
        (
            "K:architecture",
            "Repo architecture for graph RAG workflows and retrieval orchestration.",
            "Architecture",
            0,
        ),
        (
            "K:rag_basics",
            "Simple RAG retrieves relevant nodes before answer generation.",
            "RAG basics",
            0,
        ),
        (
            "K:retrieval_orchestration",
            "Memory retrieval and KG retrieval can be orchestrated with bounded depth.",
            "Retrieval orchestration",
            1,
        ),
        (
            "K:provenance",
            "Grounding and span provenance make evidence inspectable and auditable.",
            "Provenance",
            1,
        ),
        (
            "K:ttl_guardrail",
            "TTL loop budgets limit recursive self-routing and prevent infinite loops.",
            "TTL guardrails",
            1,
        ),
        (
            "K:cdc",
            "CDC bridge streams graph change events for live debugging views.",
            "CDC bridge",
            1,
        ),
        (
            "K:hidden_guardrail",
            "Budget stopper halts runaway recursion in unattended agents.",
            "Budget stopper",
            2,
        ),
    ]

    for node_id, summary, label, level in rows:
        kg_engine.add_node(
            Node(
                id=node_id,
                label=label,
                type="entity",
                summary=summary,
                doc_id=f"doc:{node_id}",
                mentions=[_grounding(f"doc:{node_id}", summary)],
                properties={},
                metadata={"level_from_root": level},
                domain_id=None,
                canonical_entity_id=None,
                level_from_root=level,
                embedding=None,
            )
        )

    edge_rows = [
        ("E:arch->rag", "K:architecture", "K:rag_basics", "supports"),
        (
            "E:arch->retrieval",
            "K:architecture",
            "K:retrieval_orchestration",
            "supports",
        ),
        ("E:retrieval->prov", "K:retrieval_orchestration", "K:provenance", "uses"),
        ("E:arch->hidden", "K:architecture", "K:hidden_guardrail", "enforces"),
        ("E:ttl->hidden", "K:ttl_guardrail", "K:hidden_guardrail", "guards"),
        ("E:ttl->cdc", "K:ttl_guardrail", "K:cdc", "observed_by"),
    ]
    for edge_id, src, dst, rel in edge_rows:
        summary = f"{src} {rel} {dst}"
        kg_engine.add_edge(
            Edge(
                id=edge_id,
                source_ids=[src],
                target_ids=[dst],
                relation=rel,
                label=rel,
                type="relationship",
                summary=summary,
                doc_id=f"doc:{edge_id}",
                mentions=[_grounding(f"doc:{edge_id}", summary)],
                properties={},
                metadata={"level_from_root": 1},
                source_edge_ids=[],
                target_edge_ids=[],
                domain_id=None,
                canonical_entity_id=None,
                embedding=None,
            )
        )


def _upsert_memory_history(conv_engine: GraphKnowledgeEngine) -> None:
    svc = ConversationService.from_engine(conv_engine, knowledge_engine=conv_engine)
    svc.create_conversation("demo-user", "conv-history", "conv-history-start")

    mem_summary = ConversationNode(
        id="hist-summary-architecture",
        label="History summary",
        type="entity",
        doc_id="conv-history-summary",
        summary="Earlier investigation said architecture drives retrieval quality.",
        role="system",
        turn_index=1,
        conversation_id="conv-history",
        user_id="demo-user",
        mentions=[
            _grounding(
                "conv:history",
                "architecture drives retrieval quality",
                insertion_method="tutorial_memory",
            )
        ],
        properties={},
        metadata={
            "entity_type": "conversation_summary",
            "level_from_root": 0,
            "in_conversation_chain": True,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )
    conv_engine.add_node(mem_summary)

    ptr = ConversationNode(
        id="hist-ref-architecture",
        label="Ref: Architecture",
        type="reference_pointer",
        doc_id="conv-history-ref",
        summary="Pinned reference to architecture node from prior run.",
        role="system",
        turn_index=2,
        conversation_id="conv-history",
        user_id="demo-user",
        mentions=[
            _grounding(
                "conv:history",
                "reference architecture",
                insertion_method="tutorial_memory",
            )
        ],
        properties={
            "target_namespace": "kg",
            "refers_to_collection": "nodes",
            "refers_to_id": "K:architecture",
            "entity_type": "knowledge_reference",
        },
        metadata={
            "entity_type": "knowledge_reference",
            "level_from_root": 0,
            "in_conversation_chain": False,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )
    conv_engine.add_node(ptr)
    conv_engine.add_edge(
        ConversationEdge(
            id="hist-edge-summary-ref",
            source_ids=[mem_summary.safe_get_id()],
            target_ids=[ptr.safe_get_id()],
            relation="references",
            label="references",
            type="relationship",
            summary="History summary references architecture pointer.",
            doc_id="conv-history-ref-edge",
            mentions=[
                _grounding(
                    "conv:history",
                    "references architecture",
                    insertion_method="tutorial_memory",
                )
            ],
            properties={"entity_type": "conversation_edge"},
            metadata={
                "entity_type": "conversation_edge",
                "char_distance_from_last_summary": 0,
                "turn_distance_from_last_summary": 0,
                "tail_turn_index": 2,
                "causal_type": "reference",
            },
            source_edge_ids=[],
            target_edge_ids=[],
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
    )


def reset_data(data_dir: Path) -> dict[str, Any]:
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "data_dir": str(data_dir), "action": "reset"}


def seed_data(data_dir: Path) -> dict[str, Any]:
    kg_engine, conv_engine = _build_engines(data_dir)
    _upsert_knowledge_graph(kg_engine)
    _upsert_memory_history(conv_engine)
    return {
        "ok": True,
        "data_dir": str(data_dir),
        "action": "seed",
        "knowledge_seed_marker": _has_seed_data(kg_engine),
    }


def _ensure_seed(data_dir: Path) -> tuple[GraphKnowledgeEngine, GraphKnowledgeEngine]:
    kg_engine, conv_engine = _build_engines(data_dir)
    if not _has_seed_data(kg_engine):
        _upsert_knowledge_graph(kg_engine)
        _upsert_memory_history(conv_engine)
    return kg_engine, conv_engine


def _ensure_seed_with_workflow(
    data_dir: Path,
) -> tuple[GraphKnowledgeEngine, GraphKnowledgeEngine, GraphKnowledgeEngine]:
    kg_engine, conv_engine = _ensure_seed(data_dir)
    workflow_engine = _build_workflow_engine(data_dir)
    return kg_engine, conv_engine, workflow_engine


def _parse_node_lines(cand_node_list_str: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in str(cand_node_list_str or "").splitlines():
        if "| Label:" not in line or "Node ID:" not in line:
            continue
        head = line.split("|", 1)[0].replace("-Node ID:", "").strip()
        out[head] = line
    return out


def _parse_edge_lines(cand_edge_list_str: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in str(cand_edge_list_str or "").splitlines():
        if "| Label:" not in line or "Edge ID:" not in line:
            continue
        head = line.split("|", 1)[0].replace("-Edge ID:", "").strip()
        out[head] = line
    return out


def deterministic_filter_callback(
    _llm_tasks: Any,
    conversation_content: str,
    cand_node_list_str: str,
    cand_edge_list_str: str,
    candidate_node_ids: list[str],
    candidate_edge_ids: list[str],
    _context_text: str,
) -> tuple[FilteringResult, str]:
    q = str(conversation_content or "").lower()
    keywords = [k for k in re.findall(r"[a-z0-9_]+", q) if len(k) > 3]
    node_lines = _parse_node_lines(cand_node_list_str)
    edge_lines = _parse_edge_lines(cand_edge_list_str)

    node_scores: list[tuple[int, str]] = []
    for nid in candidate_node_ids:
        line = str(node_lines.get(nid, "")).lower()
        score = sum(1 for kw in keywords if kw in line)
        if "ref: architecture" in line:
            score += 3
        if "knowledge_reference" in line:
            score += 2
        node_scores.append((score, nid))
    node_scores.sort(key=lambda x: (-x[0], x[1]))

    selected_nodes = [nid for score, nid in node_scores if score > 0][:3]
    if not selected_nodes:
        selected_nodes = list(candidate_node_ids[:2])

    edge_scores: list[tuple[int, str]] = []
    for eid in candidate_edge_ids:
        line = str(edge_lines.get(eid, "")).lower()
        score = sum(1 for kw in keywords if kw in line)
        edge_scores.append((score, eid))
    edge_scores.sort(key=lambda x: (-x[0], x[1]))
    selected_edges = [eid for score, eid in edge_scores if score > 0][:2]

    return (
        FilteringResult(node_ids=selected_nodes, edge_ids=selected_edges),
        "deterministic lexical selection",
    )


def run_level0(data_dir: Path, question: str) -> dict[str, Any]:
    kg_engine, _ = _ensure_seed(data_dir)
    q_emb = kg_engine._iterative_defensive_emb(question)
    retrieved = kg_engine.query_nodes(
        query_embeddings=[q_emb],
        n_results=5,
        where={"level_from_root": {"$lte": 3}},
        include=["metadatas", "documents", "embeddings"],
    )[0]
    evidence = [{"id": n.id, "summary": n.summary} for n in retrieved[:3]]
    answer = None
    ##### Optional: real LLM call (uncomment if you have a local model running)
    #####
    ##### Requires: pip install langchain-ollama
    
    # from langchain_ollama import ChatOllama
    
    # ollama_model = ChatOllama(
    #     model="qwen3:4b",
    #     temperature=0.1,
    #     base_url="http://localhost:11434"
    # )
    
    # context = "\n".join(e["summary"] for e in evidence)
    # prompt = f"""
    # Answer the question using the provided evidence paraphrased.
    
    # Question:
    # {question}
    
    # Evidence:
    # {context}
    # """
    
    # answer = ollama_model.invoke(prompt).content
    
    if answer is None:
        answer = (
            "Answer (tutorial baseline): "
            + " ".join([str(e["summary"]) for e in evidence[:2]])
            + " This response is grounded in retrieved graph evidence."
        ).strip()
    return {
        "level": 0,
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "checkpoint_pass": bool(evidence),
        "checkpoint": "non-empty evidence context for generated answer",
    }


def run_level1(
    data_dir: Path, question: str, max_retrieval_level: int
) -> dict[str, Any]:
    kg_engine, conv_engine = _ensure_seed(data_dir)

    q_emb = conv_engine._iterative_defensive_emb(question)
    mem_retriever = MemoryRetriever(
        conversation_engine=conv_engine,
        llm_tasks=conv_engine.llm_tasks,
        filtering_callback=deterministic_filter_callback,
    )
    mem = mem_retriever.retrieve(
        user_id="demo-user",
        current_conversation_id="conv-current",
        query_embedding=q_emb,
        user_text=question,
        context_text="",
        n_results=12,
    )

    fallback_seed_ids: list[str] = []
    try:
        ptr = conv_engine.get_nodes(["hist-ref-architecture"], resolve_mode="redirect")
        if ptr:
            rid = (ptr[0].properties or {}).get("refers_to_id")
            if isinstance(rid, str) and rid:
                fallback_seed_ids.append(rid)
    except Exception:
        fallback_seed_ids = []

    seed_ids = list(mem.seed_kg_node_ids or [])
    if not seed_ids:
        seed_ids = fallback_seed_ids
    seed_ids = [sid for sid in seed_ids if str(sid).startswith("K:")]

    shallow_nodes = kg_engine.query_nodes(
        query_embeddings=[q_emb],
        n_results=2,
        where={"level_from_root": {"$lte": max_retrieval_level}},
        include=["metadatas", "documents", "embeddings"],
    )[0]
    without_ids = {n.id for n in shallow_nodes}

    expanded_node_ids: set[str] = set()
    if seed_ids:
        layers = kg_engine.query.k_hop(seed_ids[:5], k=max_retrieval_level)
        for layer in layers:
            for nid in layer.get("nodes", []):
                expanded_node_ids.add(str(nid))
    with_ids = set(without_ids).union(expanded_node_ids)
    added_by_seed = sorted(with_ids.difference(without_ids))
    return {
        "level": 1,
        "question": question,
        "max_retrieval_level": max_retrieval_level,
        "seed_kg_node_ids": seed_ids,
        "candidate_count_without_seed": len(without_ids),
        "candidate_count_with_seed": len(with_ids),
        "added_by_seed": added_by_seed,
        "checkpoint_pass": bool(seed_ids) and bool(added_by_seed),
        "checkpoint": "seeded expansion changes retrieved candidate set",
    }


def run_level2(
    data_dir: Path, question: str, max_retrieval_level: int
) -> dict[str, Any]:
    kg_engine, conv_engine = _ensure_seed(data_dir)
    svc = ConversationService.from_engine(conv_engine, knowledge_engine=kg_engine)
    conversation_id, _ = svc.create_conversation(
        "demo-user", "conv-demo", "conv-demo-start"
    )

    add_result = svc.add_conversation_turn(
        user_id="demo-user",
        conversation_id=conversation_id,
        turn_id=_now_id("turn"),
        mem_id="mem-demo",
        role="user",
        content=question,
        ref_knowledge_engine=kg_engine,
        filtering_callback=deterministic_filter_callback,
        max_retrieval_level=max_retrieval_level,
        add_turn_only=True,
    )

    q_emb = conv_engine._iterative_defensive_emb(question)
    mem_retriever = MemoryRetriever(
        conversation_engine=conv_engine,
        llm_tasks=conv_engine.llm_tasks,
        filtering_callback=deterministic_filter_callback,
    )
    kg_retriever = KnowledgeRetriever(
        conversation_engine=conv_engine,
        ref_knowledge_engine=kg_engine,
        llm_tasks=conv_engine.llm_tasks,
        filtering_callback=deterministic_filter_callback,
        max_retrieval_level=max_retrieval_level,
    )

    prev_meta = add_result.prev_turn_meta_summary or MetaFromLastSummary(0, 0, 0)
    mem = mem_retriever.retrieve(
        user_id="demo-user",
        current_conversation_id=conversation_id,
        query_embedding=q_emb,
        user_text=question,
        context_text="",
        n_results=12,
    )
    # Keep level2 deterministic and resilient: use explicit selected evidence ids,
    # then materialize pinning via the same production pinner.
    selected_nodes = kg_engine.get_nodes(
        ["K:architecture", "K:provenance"], resolve_mode="redirect"
    )
    selected_edges = kg_engine.get_edges(
        ["E:arch->retrieval", "E:retrieval->prov"], resolve_mode="redirect"
    )
    selected_filter = FilteringResult(
        node_ids=[n.safe_get_id() for n in selected_nodes],
        edge_ids=[e.id for e in selected_edges],
    )

    self_span = _span(
        f"conv:{conversation_id}", question, insertion_method="tutorial_level2"
    )
    memory_context_node_id = None
    if mem.selected and mem.memory_context_text:
        pin = mem_retriever.pin_selected(
            user_id="demo-user",
            current_conversation_id=conversation_id,
            mem_id="mem-demo",
            turn_node_id=add_result.user_turn_node_id,
            turn_index=add_result.turn_index,
            self_span=self_span,
            selected_memory=mem.selected,
            memory_context_text=mem.memory_context_text,
            prev_turn_meta_summary=prev_meta,
        )
        if pin:
            memory_context_node_id = pin.memory_context_node.safe_get_id()

    pinned_ptrs: list[str] = []
    pinned_edges: list[str] = []
    if selected_nodes or selected_edges:
        pinned_ptrs, pinned_edges = kg_retriever.pin_selected(
            user_id="demo-user",
            conversation_id=conversation_id,
            turn_node_id=add_result.user_turn_node_id,
            turn_index=add_result.turn_index,
            self_span=self_span,
            selected_knowledge=selected_filter,
            selected_knowledge_nodes=RetrievalResult(
                nodes=selected_nodes, edges=selected_edges
            ),
            prev_turn_meta_summary=prev_meta,
        )

    node_check = (
        bool(conv_engine.backend.node_get(ids=pinned_ptrs, include=[]).get("ids"))
        if pinned_ptrs
        else False
    )
    return {
        "level": 2,
        "question": question,
        "conversation_id": conversation_id,
        "turn_node_id": add_result.user_turn_node_id,
        "memory_context_node_id": memory_context_node_id,
        "pinned_kg_pointer_node_ids": pinned_ptrs,
        "pinned_kg_edge_ids": pinned_edges,
        "checkpoint_pass": bool(pinned_ptrs) and bool(pinned_edges) and node_check,
        "checkpoint": "reference_pointer nodes and references edges materialized for this turn",
    }


def run_level2b(
    data_dir: Path,
    question: str,
    max_retrieval_level: int,
    llm_provider: str = "deterministic",
    llm_model: str | None = None,
) -> dict[str, Any]:
    kg_engine, conv_engine, workflow_engine = _ensure_seed_with_workflow(data_dir)
    provider_key = str(llm_provider or "deterministic").strip().lower()
    resolved_model_name = ""
    
    
    
    if provider_key != "deterministic":

        if provider_key == "ollama":
            config = DefaultTaskProviderConfig(summarize_context_provider=provider_key,
                                    answer_with_citations_provider=provider_key,
                                    repair_citations_provider=provider_key)
            from dataclasses import replace
            config = replace(config, ollama_model_name="qwen3:4b")
            llm_tasks = build_default_llm_tasks(config=config)
            conv_engine.llm_tasks = llm_tasks
        llm_tasks, resolved_model_name = _build_provider_task_set(
            base_tasks=conv_engine.llm_tasks,
            provider=provider_key,
            model_name=llm_model,
        )
    # else:
        

    svc = ConversationService(
        conversation_engine=conv_engine,
        knowledge_engine=kg_engine,
        workflow_engine=workflow_engine,
        llm_tasks=llm_tasks,
    )
    conversation_id, _ = svc.create_conversation(
        "demo-user", "conv-demo-v2", "conv-demo-v2-start"
    )

    def answer_only_harness(
        *, conversation_id: str, prev_turn_meta_summary: MetaFromLastSummary, **_
    ) -> ConversationAIResponse:
        _ = conversation_id, prev_turn_meta_summary
        # This harness intentionally exercises the explicit KG/context-backed
        # answer_only primitive rather than the full provider-backed workflow.
        answer_text = "Tutorial v2 reply grounded in architecture and provenance."
        return ConversationAIResponse(
            text=answer_text,
            llm_decision_need_summary=False,
            used_kg_node_ids=["K:architecture", "K:provenance"],
            projected_conversation_node_ids=[],
            meta={
                "source": "tutorial_level2b_deterministic",
                "provider": "deterministic",
                "model_name": "",
            },
            response_node_id=None,
        )

    if provider_key == "deterministic":
        svc.orchestrator.answer_only = answer_only_harness
    add_result = svc.add_turn_workflow_v2(
        run_id="run-demo-v2",
        user_id="demo-user",
        conversation_id=conversation_id,
        turn_id="turn-demo-v2",
        mem_id="mem-demo-v2",
        role="user",
        content=question,
        filtering_callback=deterministic_filter_callback,
        workflow_id="conversation.add_turn.v2.tutorial",
        max_retrieval_level=max_retrieval_level,
        in_conv=True,
        add_turn_only=False,
        max_workers=1,
        strict_answer_failure=(provider_key != "deterministic"),
        force_answer_only=(provider_key == "deterministic"),
        cache_dir=data_dir / "run_level_2b"
    )

    transcript = svc.get_conversation_view(
        conversation_id=conversation_id,
        user_id="demo-user",
    ).messages
    visible_turn_roles = [
        str(role)
        for role in (getattr(m, "role", "") for m in transcript)
        if role in {"user", "assistant"}
    ]
    assistant_node_id = add_result.response_turn_node_id
    assistant_exists = bool(
        assistant_node_id
        and conv_engine.backend.node_get(ids=[assistant_node_id], include=[]).get("ids")
    )
    pinned_ptrs = list(add_result.pinned_kg_pointer_node_ids or [])
    pinned_edges = list(add_result.pinned_kg_edge_ids or [])
    assistant_text = ""
    if assistant_node_id:
        try:
            assistant_nodes = conv_engine.get_nodes([assistant_node_id], resolve_mode="redirect")
            if assistant_nodes:
                assistant_text = str(getattr(assistant_nodes[0], "summary", "") or "")
        except Exception:
            assistant_text = ""
    if not assistant_text:
        assistant_messages = [m for m in transcript if getattr(m, "role", "") == "assistant"]
        if assistant_messages:
            assistant_text = str(getattr(assistant_messages[-1], "content", "") or "")

    return {
        "level": "2b",
        "question": question,
        "conversation_id": conversation_id,
        "turn_node_id": add_result.user_turn_node_id,
        "assistant_turn_node_id": assistant_node_id,
        "pinned_kg_pointer_node_ids": pinned_ptrs,
        "pinned_kg_edge_ids": pinned_edges,
        "transcript_roles": visible_turn_roles,
        "assistant_text": assistant_text,
        "llm_provider": provider_key,
        "llm_model": resolved_model_name or llm_model or "",
        "checkpoint_pass": bool(assistant_exists and pinned_ptrs and pinned_edges),
        "checkpoint": "workflow-driven v2 conversation path materializes assistant output and the same inspectable evidence pointers",
    }


def level3_command_hints(data_dir: Path) -> dict[str, Any]:
    claw_data_dir = str(data_dir.parent / "claw-loop")
    cmds = [
        f"python scripts/claw_runtime_loop.py init --data-dir {claw_data_dir}",
        f'python scripts/claw_runtime_loop.py enqueue --data-dir {claw_data_dir} --conversation-id conv-demo --event-type user.message --payload \'{{"text":"hello claw","ttl":2}}\'',
        f"python scripts/claw_runtime_loop.py run-once --data-dir {claw_data_dir}",
        f"python scripts/claw_runtime_loop.py list-events --data-dir {claw_data_dir} --direction in --limit 10",
        f"python scripts/claw_runtime_loop.py list-events --data-dir {claw_data_dir} --direction out --limit 10",
    ]
    return {"level": 3, "commands": cmds}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG ladder helper: simple RAG to reinforced RAG patterns."
    )
    parser.add_argument("--data-dir", default=".gke-data/tutorial-ladder")
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default=".gke-data/tutorial-ladder")

    sub.add_parser("reset", parents=[common])
    sub.add_parser("seed", parents=[common])

    p0 = sub.add_parser("level0", parents=[common])
    p0.add_argument("--question", default="How does this repo implement simple RAG?")

    p1 = sub.add_parser("level1", parents=[common])
    p1.add_argument("--question", default="How does architecture reinforce retrieval?")
    p1.add_argument("--max-retrieval-level", type=int, default=2)

    p2 = sub.add_parser("level2", parents=[common])
    p2.add_argument(
        "--question", default="Show evidence and provenance for retrieval decisions."
    )
    p2.add_argument("--max-retrieval-level", type=int, default=2)

    p2b = sub.add_parser("level2b", parents=[common])
    p2b.add_argument(
        "--question",
        default="Show the equivalent provenance flow through add_turn_workflow_v2.",
    )
    p2b.add_argument("--max-retrieval-level", type=int, default=2)
    p2b.add_argument(
        "--llm-provider",
        choices=["deterministic", "gemini", "openai", "ollama"],
        default="deterministic",
    )
    p2b.add_argument("--llm-model", default=None)

    sub.add_parser("level3-hints", parents=[common])

    p_all = sub.add_parser("run-all", parents=[common])
    p_all.add_argument(
        "--question0", default="How does this repo implement simple RAG?"
    )
    p_all.add_argument(
        "--question1", default="How does architecture reinforce retrieval?"
    )
    p_all.add_argument(
        "--question2", default="Show evidence and provenance for retrieval decisions."
    )
    p_all.add_argument("--max-retrieval-level", type=int, default=2)

    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    _configure_tutorial_cache_env(data_dir)

    if args.cmd == "reset":
        _print_json(reset_data(data_dir))
        return
    if args.cmd == "seed":
        _print_json(seed_data(data_dir))
        return
    if args.cmd == "level0":
        _print_json(run_level0(data_dir, question=str(args.question)))
        return
    if args.cmd == "level1":
        _print_json(
            run_level1(
                data_dir,
                question=str(args.question),
                max_retrieval_level=int(args.max_retrieval_level),
            )
        )
        return
    if args.cmd == "level2":
        _print_json(
            run_level2(
                data_dir,
                question=str(args.question),
                max_retrieval_level=int(args.max_retrieval_level),
            )
        )
        return
    if args.cmd == "level2b":
        _print_json(
            run_level2b(
                data_dir,
                question=str(args.question),
                max_retrieval_level=int(args.max_retrieval_level),
                llm_provider=str(args.llm_provider),
                llm_model=(
                    None if args.llm_model in (None, "", "None") else str(args.llm_model)
                ),
            )
        )
        return
    if args.cmd == "level3-hints":
        _print_json(level3_command_hints(data_dir))
        return
    if args.cmd == "run-all":
        out = {
            "reset": reset_data(data_dir),
            "seed": seed_data(data_dir),
            "level0": run_level0(data_dir, question=str(args.question0)),
            "level1": run_level1(
                data_dir,
                question=str(args.question1),
                max_retrieval_level=int(args.max_retrieval_level),
            ),
            "level2": run_level2(
                data_dir,
                question=str(args.question2),
                max_retrieval_level=int(args.max_retrieval_level),
            ),
            "level2b": run_level2b(
                data_dir,
                question="Show the equivalent provenance flow through add_turn_workflow_v2.",
                max_retrieval_level=int(args.max_retrieval_level),
                llm_provider="deterministic",
                llm_model=None,
            ),
            "level3_hints": level3_command_hints(data_dir),
        }
        _print_json(out)
        return
    raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

# view cdc: python -m graph_knowledge_engine.cdc.change_bridge --host 127.0.0.1 --port 8787 --oplog-file .cdc_debug/data/cdc_oplog.jsonl --reset-oplog
