from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import webbrowser
from pathlib import Path
from typing import Any, Sequence

from kogwistar.conversation.models import (
    ConversationAIResponse,
    ConversationEdge,
    ConversationNode,
    FilteringResult,
    MetaFromLastSummary,
)
from kogwistar.conversation.service import ConversationService
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import (
    Edge,
    Grounding,
    MentionVerification,
    Node,
    Span,
)
from kogwistar.runtime.replay import replay_to
from kogwistar.utils.kge_debug_dump import dump_paired_bundles


DEFAULT_QUESTION = "How does Kogwistar make AI workflows replayable and auditable?"


class DeterministicLexicalEmbeddingFunction:
    """Small deterministic embedder used by the packaged quickstart."""

    def __init__(self, dim: int = 96) -> None:
        self._dim = dim

    @staticmethod
    def name() -> str:
        return "kogwistar-quickstart-lexical-v1"

    def is_legacy(self) -> bool:
        return False

    @staticmethod
    def supported_spaces() -> list[str]:
        return ["cosine"]

    @staticmethod
    def get_config() -> dict[str, object]:
        return {}

    @classmethod
    def build_from_config(
        cls, config: dict[str, object] | None = None
    ) -> "DeterministicLexicalEmbeddingFunction":
        _ = config
        return cls()

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in input:
            vec = [0.0] * self._dim
            tokens = re.findall(r"[a-z0-9_]+", str(text or "").lower())
            for token in tokens:
                idx = (
                    int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
                    % self._dim
                )
                vec[idx] += 1.0
            norm = math.sqrt(sum(value * value for value in vec)) or 1.0
            vectors.append([value / norm for value in vec])
        return vectors


def _span(doc_id: str, excerpt: str, *, insertion_method: str) -> Span:
    return Span(
        collection_page_url=f"quickstart/{doc_id}",
        document_page_url=f"quickstart/{doc_id}",
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
            method="system", is_verified=True, score=1.0, notes=insertion_method
        ),
    )


def _grounding(doc_id: str, excerpt: str, *, insertion_method: str) -> Grounding:
    return Grounding(spans=[_span(doc_id, excerpt, insertion_method=insertion_method)])


def _build_engines(
    base_dir: Path,
) -> tuple[GraphKnowledgeEngine, GraphKnowledgeEngine, GraphKnowledgeEngine]:
    embedding = DeterministicLexicalEmbeddingFunction()
    return (
        GraphKnowledgeEngine(
            persist_directory=str(base_dir / "knowledge"),
            kg_graph_type="knowledge",
            embedding_function=embedding,
        ),
        GraphKnowledgeEngine(
            persist_directory=str(base_dir / "conversation"),
            kg_graph_type="conversation",
            embedding_function=embedding,
        ),
        GraphKnowledgeEngine(
            persist_directory=str(base_dir / "workflow"),
            kg_graph_type="workflow",
            embedding_function=embedding,
        ),
    )


def _seed_knowledge(kg_engine: GraphKnowledgeEngine) -> None:
    rows = [
        (
            "K:workflow_provenance",
            "Workflow execution is persisted as graph data with stable run ids and step state.",
            "Workflow provenance",
            0,
        ),
        (
            "K:replay",
            "Replay reconstructs execution from persisted checkpoints and step updates.",
            "Replay",
            0,
        ),
        (
            "K:evidence_pointers",
            "Answering pins reference_pointer nodes and edges so evidence remains inspectable.",
            "Evidence pointers",
            1,
        ),
        (
            "K:context_snapshots",
            "Context snapshots preserve what the model saw at each answering stage.",
            "Context snapshots",
            1,
        ),
        (
            "K:auditability",
            "Auditability comes from linked workflow, conversation, provenance, and evidence artifacts.",
            "Auditability",
            1,
        ),
    ]
    for node_id, summary, label, level in rows:
        kg_engine.write.add_node(
            Node(
                id=node_id,
                label=label,
                type="entity",
                summary=summary,
                doc_id=f"doc:{node_id}",
                mentions=[
                    _grounding(
                        f"doc:{node_id}", summary, insertion_method="quickstart_seed"
                    )
                ],
                properties={},
                metadata={"level_from_root": level},
                domain_id=None,
                canonical_entity_id=None,
                level_from_root=level,
                embedding=None,
            )
        )
    edge_rows = [
        (
            "E:workflow->replay",
            "K:workflow_provenance",
            "K:replay",
            "supports",
        ),
        (
            "E:workflow->snapshots",
            "K:workflow_provenance",
            "K:context_snapshots",
            "materializes",
        ),
        (
            "E:replay->auditability",
            "K:replay",
            "K:auditability",
            "proves",
        ),
        (
            "E:pointers->auditability",
            "K:evidence_pointers",
            "K:auditability",
            "strengthens",
        ),
    ]
    for edge_id, src, dst, relation in edge_rows:
        summary = f"{src} {relation} {dst}"
        kg_engine.write.add_edge(
            Edge(
                id=edge_id,
                source_ids=[src],
                target_ids=[dst],
                relation=relation,
                label=relation,
                type="relationship",
                summary=summary,
                doc_id=f"doc:{edge_id}",
                mentions=[
                    _grounding(
                        f"doc:{edge_id}", summary, insertion_method="quickstart_seed"
                    )
                ],
                properties={},
                metadata={"level_from_root": 1},
                source_edge_ids=[],
                target_edge_ids=[],
                domain_id=None,
                canonical_entity_id=None,
                embedding=None,
            )
        )


def _seed_conversation_memory(conversation_engine: GraphKnowledgeEngine) -> None:
    service = ConversationService.from_engine(
        conversation_engine, knowledge_engine=conversation_engine
    )
    service.create_conversation("demo-user", "conv-history", "conv-history-start")

    summary = ConversationNode(
        id="hist-summary-provenance",
        label="History summary",
        type="entity",
        doc_id="conv-history-summary",
        summary="Previous run concluded that replay and evidence pointers are the core audit proof.",
        role="system",
        turn_index=1,
        conversation_id="conv-history",
        user_id="demo-user",
        mentions=[
            _grounding(
                "conv:history",
                "replay and evidence pointers are the core audit proof",
                insertion_method="quickstart_memory",
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
    pointer = ConversationNode(
        id="hist-ref-workflow",
        label="Ref: Workflow provenance",
        type="reference_pointer",
        doc_id="conv-history-ref",
        summary="Pinned pointer to workflow provenance guidance from a previous run.",
        role="system",
        turn_index=2,
        conversation_id="conv-history",
        user_id="demo-user",
        mentions=[
            _grounding(
                "conv:history",
                "workflow provenance pointer",
                insertion_method="quickstart_memory",
            )
        ],
        properties={
            "target_namespace": "kg",
            "refers_to_collection": "nodes",
            "refers_to_id": "K:workflow_provenance",
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
    conversation_engine.write.add_node(summary)
    conversation_engine.write.add_node(pointer)
    conversation_engine.write.add_edge(
        ConversationEdge(
            id="hist-edge-summary-ref",
            source_ids=[summary.safe_get_id()],
            target_ids=[pointer.safe_get_id()],
            relation="references",
            label="references",
            type="relationship",
            summary="History summary references workflow provenance pointer.",
            doc_id="conv-history-ref-edge",
            mentions=[
                _grounding(
                    "conv:history",
                    "references workflow provenance",
                    insertion_method="quickstart_memory",
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


def _deterministic_filter(
    _llm_tasks: Any,
    conversation_content: str,
    cand_node_list_str: str,
    cand_edge_list_str: str,
    candidate_node_ids: list[str],
    candidate_edge_ids: list[str],
    _context_text: str,
) -> tuple[FilteringResult, str]:
    question = str(conversation_content or "").lower()
    keywords = [token for token in re.findall(r"[a-z0-9_]+", question) if len(token) > 3]

    node_scores: list[tuple[int, str]] = []
    for node_id in candidate_node_ids:
        line = cand_node_list_str.lower()
        score = sum(1 for keyword in keywords if keyword in line)
        if node_id == "hist-ref-workflow":
            score += 3
        node_scores.append((score, node_id))
    node_scores.sort(key=lambda item: (-item[0], item[1]))
    selected_nodes = [node_id for score, node_id in node_scores if score > 0][:3]
    if not selected_nodes:
        selected_nodes = list(candidate_node_ids[:2])

    edge_scores: list[tuple[int, str]] = []
    for edge_id in candidate_edge_ids:
        line = cand_edge_list_str.lower()
        score = sum(1 for keyword in keywords if keyword in line)
        edge_scores.append((score, edge_id))
    edge_scores.sort(key=lambda item: (-item[0], item[1]))
    selected_edges = [edge_id for score, edge_id in edge_scores if score > 0][:2]
    return (
        FilteringResult(node_ids=selected_nodes, edge_ids=selected_edges),
        "deterministic quickstart lexical selection",
    )


def _latest_checkpoint_step(conversation_engine: GraphKnowledgeEngine, run_id: str) -> int:
    checkpoints = conversation_engine.read.get_nodes(
        where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]},
        limit=10_000,
    )
    if not checkpoints:
        raise RuntimeError(f"No workflow checkpoints found for run_id={run_id!r}")
    return max(int((node.metadata or {}).get("step_seq", 0)) for node in checkpoints)


def _extract_answer_text(
    conversation_engine: GraphKnowledgeEngine,
    assistant_turn_node_id: str | None,
    fallback_messages: Sequence[Any],
) -> str:
    if assistant_turn_node_id:
        try:
            assistant_nodes = conversation_engine.read.get_nodes(
                [assistant_turn_node_id], resolve_mode="redirect"
            )
            if assistant_nodes:
                return str(getattr(assistant_nodes[0], "summary", "") or "")
        except Exception:
            pass
    for message in reversed(list(fallback_messages)):
        if getattr(message, "role", "") == "assistant":
            return str(getattr(message, "content", "") or "")
    return ""


def run_provenance_quickstart(
    *,
    data_dir: str | Path = ".gke-data/quickstart",
    question: str = DEFAULT_QUESTION,
    open_browser: bool = False,
) -> dict[str, Any]:
    base_dir = Path(data_dir).resolve()
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    knowledge_engine, conversation_engine, workflow_engine = _build_engines(base_dir)
    _seed_knowledge(knowledge_engine)
    _seed_conversation_memory(conversation_engine)

    service = ConversationService(
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        workflow_engine=workflow_engine,
        llm_tasks=conversation_engine.llm_tasks,
    )

    conversation_id, _ = service.create_conversation(
        "demo-user", "conv-quickstart", "conv-quickstart-start"
    )
    turn_id = "turn-quickstart"
    def answer_only_harness(
        *,
        conversation_id: str,
        prev_turn_meta_summary: MetaFromLastSummary,
        **_: Any,
    ) -> ConversationAIResponse:
        _ = conversation_id, prev_turn_meta_summary
        answer_text = (
            "Kogwistar stores the run, evidence pointers, and replayable context "
            f"snapshots as graph data. Question: {question}"
        )
        return ConversationAIResponse(
            text=answer_text,
            llm_decision_need_summary=False,
            used_kg_node_ids=[
                "K:workflow_provenance",
                "K:replay",
                "K:evidence_pointers",
            ],
            projected_conversation_node_ids=[],
            meta={
                "source": "quickstart_deterministic",
                "provider": "deterministic",
                "model_name": "",
            },
            response_node_id=None,
        )

    service.orchestrator.answer_only = answer_only_harness
    add_result = service.add_turn_workflow_v2(
        run_id=f"add_turn|{turn_id}",
        user_id="demo-user",
        conversation_id=conversation_id,
        turn_id=turn_id,
        mem_id="mem-quickstart",
        role="user",
        content=question,
        filtering_callback=_deterministic_filter,
        workflow_id="conversation.add_turn.v2.quickstart",
        max_retrieval_level=2,
        in_conv=True,
        add_turn_only=False,
        max_workers=1,
        strict_answer_failure=False,
        force_answer_only=True,
        cache_dir=base_dir / "cache",
    )
    run_id = f"add_turn|{add_result.user_turn_node_id}"

    transcript = service.get_conversation_view(
        conversation_id=conversation_id, user_id="demo-user"
    ).messages
    answer_text = _extract_answer_text(
        conversation_engine, add_result.response_turn_node_id, transcript
    )

    latest_step_seq = _latest_checkpoint_step(conversation_engine, run_id)
    replayed_state = replay_to(
        conversation_engine=conversation_engine,
        run_id=run_id,
        target_step_seq=latest_step_seq,
    )

    replay_report = {
        "run_id": run_id,
        "target_step_seq": latest_step_seq,
        "response_node_id_match": (
            (replayed_state.get("answer") or {}).get("response_node_id")
            == add_result.response_turn_node_id
        ),
        "pinned_pointer_ids_match": list(
            (replayed_state.get("kg_pin") or {}).get("pinned_pointer_node_ids") or []
        )
        == list(add_result.pinned_kg_pointer_node_ids or []),
        "pinned_edge_ids_match": list(
            (replayed_state.get("kg_pin") or {}).get("pinned_edge_ids") or []
        )
        == list(add_result.pinned_kg_edge_ids or []),
    }
    replay_report["pass"] = all(
        [
            replay_report["response_node_id_match"],
            replay_report["pinned_pointer_ids_match"],
            replay_report["pinned_edge_ids_match"],
        ]
    )

    artifacts_dir = base_dir / "artifacts"
    bundles_dir = artifacts_dir / "bundles"
    template_html = (
        Path(__file__).resolve().parents[1] / "templates" / "d3.html"
    ).read_text(encoding="utf-8")
    dump_paired_bundles(
        kg_engine=knowledge_engine,
        conversation_engine=conversation_engine,
        workflow_engine=workflow_engine,
        template_html=template_html,
        out_dir=bundles_dir,
        kg_out="kg.bundle.html",
        conversation_out="conversation.bundle.html",
        work_flow_out="workflow.bundle.html",
        mode="reify",
        insertion_method=None,
    )

    summary = {
        "question": question,
        "answer_text": answer_text,
        "replay_pass": bool(replay_report["pass"]),
        "run_id": run_id,
        "conversation_id": conversation_id,
        "assistant_turn_node_id": add_result.response_turn_node_id,
        "pinned_kg_pointer_node_ids": list(add_result.pinned_kg_pointer_node_ids or []),
        "pinned_kg_edge_ids": list(add_result.pinned_kg_edge_ids or []),
        "artifacts": {
            "summary_json": str((artifacts_dir / "summary.json").resolve()),
            "replay_json": str((artifacts_dir / "replay_report.json").resolve()),
            "graph_html": str((bundles_dir / "kg.bundle.html").resolve()),
            "provenance_html": str((bundles_dir / "conversation.bundle.html").resolve()),
            "workflow_html": str((bundles_dir / "workflow.bundle.html").resolve()),
        },
        "next_command": "python scripts/tutorial_ladder.py run-all --data-dir .gke-data/tutorial-ladder",
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (artifacts_dir / "replay_report.json").write_text(
        json.dumps(replay_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if open_browser:
        webbrowser.open(summary["artifacts"]["graph_html"])
        webbrowser.open(summary["artifacts"]["provenance_html"])

    return summary
