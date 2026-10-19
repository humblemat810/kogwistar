from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, MentionVerification, Span
from kogwistar.runtime.models import RunSuccess, WorkflowEdge, WorkflowNode
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.runtime.runtime import StepContext, WorkflowRuntime


class _DeterministicEmbedding:
    @staticmethod
    def name() -> str:
        return "workflow-recovery-test"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [[float(len(text) % 13 + 1), 1.0, 2.0] for text in input]


def _grounding() -> Grounding:
    return Grounding(
        spans=[
            Span(
                collection_page_url="test",
                document_page_url="test",
                doc_id="workflow-recovery-test",
                insertion_method="test",
                page_number=1,
                start_char=0,
                end_char=1,
                excerpt="x",
                context_before="",
                context_after="",
                chunk_id=None,
                source_cluster_id=None,
                verification=MentionVerification(
                    method="system", is_verified=True, score=1.0, notes=""
                ),
            )
        ]
    )


def _engine(path: Path, graph_type: str) -> GraphKnowledgeEngine:
    return GraphKnowledgeEngine(
        persist_directory=str(path),
        kg_graph_type=graph_type,
        embedding_function=_DeterministicEmbedding(),
    )


def _install_design(engine: GraphKnowledgeEngine) -> None:
    wf_id = "fresh-process-recovery"
    for node_id, op, start, terminal in (
        ("start", "durable_start", True, False),
        ("gate", "recovery_gate", False, False),
        ("end", "durable_end", False, True),
    ):
        engine.write.add_node(
            WorkflowNode(
                id=node_id,
                label=op,
                type="entity",
                doc_id=node_id,
                summary=op,
                properties={},
                mentions=[_grounding()],
                metadata={
                    "entity_type": "workflow_node",
                    "workflow_id": wf_id,
                    "wf_op": op,
                    "wf_start": start,
                    "wf_terminal": terminal,
                    "wf_version": "v1",
                },
                domain_id=None,
                canonical_entity_id=None,
                embedding=None,
                level_from_root=0,
            )
        )
    for source, target in (("start", "gate"), ("gate", "end")):
        engine.write.add_edge(
            WorkflowEdge(
                id=f"{source}->{target}",
                label="wf_next",
                type="entity",
                doc_id=f"{source}->{target}",
                summary="next",
                properties={},
                source_ids=[source],
                target_ids=[target],
                source_edge_ids=[],
                target_edge_ids=[],
                relation="wf_next",
                mentions=[_grounding()],
                metadata={
                    "entity_type": "workflow_edge",
                    "workflow_id": wf_id,
                    "wf_predicate": None,
                    "wf_is_default": True,
                },
                domain_id=None,
                canonical_entity_id=None,
                embedding=None,
            )
        )


def main() -> int:
    root = Path(sys.argv[1])
    mode = sys.argv[2]
    workflow_path = root / "workflow"
    conversation_path = root / "conversation"
    if mode == "start":
        workflow_engine = _engine(workflow_path, "workflow")
        conversation_engine = _engine(conversation_path, "conversation")
        _install_design(workflow_engine)
    else:
        workflow_engine = _engine(workflow_path, "workflow")
        conversation_engine = _engine(conversation_path, "conversation")

    resolver = MappingStepResolver()

    @resolver.register("durable_start")
    def _start(_ctx: StepContext):
        count_path = root / "start-count.txt"
        count = int(count_path.read_text(encoding="utf-8")) if count_path.exists() else 0
        count_path.write_text(str(count + 1), encoding="utf-8")
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"durable_start_completed": True})],
        )

    @resolver.register("recovery_gate")
    def _gate(_ctx: StepContext):
        if mode == "start":
            (root / "gate-entered").write_text("ready", encoding="utf-8")
            while True:
                time.sleep(1)
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"gate_recovered": True})],
        )

    @resolver.register("durable_end")
    def _end(_ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"workflow_completed": True})],
        )

    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
        run_ownership_lease_ms=1_000,
    )
    run_id = "fresh-process-recovery-run"
    if mode == "start":
        result = runtime.run(
            workflow_id="fresh-process-recovery",
            conversation_id="fresh-process-recovery-conversation",
            turn_node_id="turn-1",
            initial_state={"user_id": "test", "role": "user", "user_text": ""},
            run_id=run_id,
        )
        return 0 if result.status == "succeeded" else 2

    result = runtime.resume_from_latest_checkpoint(
        run_id=run_id,
        workflow_id="fresh-process-recovery",
        conversation_id="fresh-process-recovery-conversation",
        turn_node_id="turn-1",
    )
    accepted_state = {
        key: result.final_state.get(key)
        for key in (
            "durable_start_completed",
            "gate_recovered",
            "workflow_completed",
        )
    }
    (root / "resume-result.json").write_text(
        json.dumps({"status": result.status, "state": accepted_state}, sort_keys=True),
        encoding="utf-8",
    )
    return 0 if result.status == "succeeded" else 3


if __name__ == "__main__":
    raise SystemExit(main())
