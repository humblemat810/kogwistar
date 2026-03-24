from ..engine_core.models import Grounding, Span
from ..runtime.design import BaseWorkflowDesigner
from ..runtime.models import WorkflowEdge, WorkflowNode
from .agentic_answering_design import (
    agentic_answering_expected_ops,
    build_agentic_answering_workflow_design,
    materialize_workflow_design_artifact,
)


class AgenticAnsweringWorkflowDesigner(BaseWorkflowDesigner):
    """Agentic-answering workflow designer.

    Documents the current V2 migration: rewrite `AgenticAnsweringAgent.answer()` into
    a workflow executed by `WorkflowRuntime`.
    """

    def ensure_answer_flow(
        self,
        *,
        workflow_id: str,
        mode: str = "full",
    ) -> tuple[WorkflowNode, dict[str, WorkflowNode], dict[str, list[WorkflowEdge]]]:
        """Ensure an agentic-answering workflow exists.

        Modes:
          - mode="backbone": start -> end (op="start")
          - mode="full": agentic answering loop ops with a bounded iterate edge
        """
        if mode not in ("backbone", "full"):
            raise ValueError(f"Unknown mode={mode!r}; expected 'backbone' or 'full'")

        expected_ops = set(agentic_answering_expected_ops())

        def _has_full_answer_flow() -> bool:
            nodes = self.workflow_engine.read.get_nodes(
                where={
                    "$and": [
                        {"entity_type": "workflow_node"},
                        {"workflow_id": workflow_id},
                    ]
                },
                limit=5000,
            )
            ops: set[str] = set()
            for node in nodes or []:
                md = getattr(node, "metadata", {}) or {}
                op = str(md.get("wf_op") or md.get("op") or "")
                if op:
                    ops.add(op)
            return expected_ops.issubset(ops)

        # If the full workflow is already present and valid, return it.
        if mode == "full" and _has_full_answer_flow():
            try:
                return self.validate(workflow_id=workflow_id)
            except Exception:
                pass

        if mode == "backbone":
            from ..engine_core.models import Span, Grounding

            wid = lambda suffix: f"wf:{workflow_id}:{suffix}"
            sp = Span.from_dummy_for_workflow(workflow_id)

            def add_node(
                *,
                node_id: str,
                label: str,
                op: str | None,
                start: bool = False,
                terminal: bool = False,
                fanout: bool = False,
            ):
                n = WorkflowNode(
                    id=node_id,
                    label=label,
                    type="entity",
                    doc_id=node_id,
                    summary=label,
                    properties={},
                    mentions=[Grounding(spans=[sp])],
                    metadata={
                        "entity_type": "workflow_node",
                        "workflow_id": workflow_id,
                        "wf_op": op,
                        "wf_start": start,
                        "wf_terminal": terminal,
                        "wf_fanout": fanout,
                        "wf_version": "v2",
                    },
                )
                self.workflow_engine.write.add_node(n)

            def add_edge(
                *,
                edge_id: str,
                src: str,
                dst: str,
                relation,
                pred: str | None,
                priority: int = 100,
                is_default: bool = False,
                multiplicity: str = "one",
            ):
                e = WorkflowEdge(
                    id=edge_id,
                    label="wf_next",
                    type="entity",
                    doc_id=edge_id,
                    summary="wf_next",
                    properties={},
                    source_ids=[src],
                    target_ids=[dst],
                    source_edge_ids=[],
                    target_edge_ids=[],
                    relation=relation,
                    mentions=[Grounding(spans=[sp])],
                    metadata={
                        "entity_type": "workflow_edge",
                        "workflow_id": workflow_id,
                        "wf_edge_kind": "wf_next",
                        "wf_predicate": pred,
                        "wf_priority": priority,
                        "wf_is_default": is_default,
                        "wf_multiplicity": multiplicity,
                    },
                )
                self.workflow_engine.write.add_edge(e)

            start_id = wid("start")
            end_id = wid("end")
            add_node(
                node_id=start_id, label="Start", op="start", start=True, terminal=False
            )
            add_node(node_id=end_id, label="End", op=None, start=False, terminal=True)
            add_edge(
                edge_id=wid("next_start_end"),
                relation="wf_next",
                src=start_id,
                dst=end_id,
                pred="always",
                priority=100,
                is_default=True,
            )
            return self.validate(workflow_id=workflow_id)

        # For full mode, or when the full flow is missing/partial, rebuild.
        if mode == "full":
            design = build_agentic_answering_workflow_design(workflow_id=workflow_id)
            materialize_workflow_design_artifact(self.workflow_engine, design)
            return self.validate(workflow_id=workflow_id)

        raise AssertionError("unreachable")


class ConversationWorkflowDesigner(BaseWorkflowDesigner):
    """Conversation-specific designer.

    Phase 2D Step 1/2:
    - ensure a minimal backbone design exists (start -> end)
    - validate the design schema + predicate resolution

    Phase 2D Step 3+:
    - will enforce stricter conversation invariants (chain linearity, sidecar rules, etc.)
    """

    def ensure_add_turn_flow(
        self,
        *,
        workflow_id: str,
        mode: str = "full",
        include_context_snapshot: bool = True,
    ) -> tuple[WorkflowNode, dict[str, WorkflowNode], dict[str, list[WorkflowEdge]]]:
        """Ensure the Phase-2D add_turn workflow design exists.

        This is the single source of truth for conversation workflow charts.

        - mode="backbone": creates a minimal start->end workflow (start op="start")
        - mode="full": creates the v1-parity add_turn flow using step ops resolved by the resolver.
        """
        from kogwistar.utils.log import bind_log_context

        with bind_log_context(engine_type="workflow"):
            if mode not in ("backbone", "full"):
                raise ValueError(
                    f"Unknown mode={mode!r}; expected 'backbone' or 'full'"
                )

            # If already present and valid, return it.
            try:
                pass
                # return self.validate(workflow_id=workflow_id)
            except Exception:
                pass

            # Lazy import to avoid circular deps for model classes.
            from ..runtime.models import WorkflowEdge

            def wf_node_id(workflow_id, suffix):
                return f"wf:{workflow_id}:{suffix}"

            import functools

            wid = functools.partial(wf_node_id, workflow_id)

            def add_node(
                *,
                node_id: str,
                label: str,
                op: str | None,
                start: bool = False,
                terminal: bool = False,
                fanout: bool = False,
                metadata=None,
                wf_join=False,
            ):
                if metadata is None:
                    metadata = {}
                metadata_final = {
                    "entity_type": "workflow_node",
                    "workflow_id": workflow_id,
                    "wf_op": op,
                    "wf_start": start,
                    "wf_terminal": terminal,
                    "wf_fanout": fanout,
                    "wf_version": "v2",
                }
                metadata_final.update(metadata)
                n = WorkflowNode(
                    id=node_id,
                    label=label,
                    type="entity",
                    doc_id=node_id,
                    summary=label,
                    properties={},
                    metadata={
                        "entity_type": "workflow_node",
                        "workflow_id": workflow_id,
                        "wf_op": op,
                        "wf_start": start,
                        "wf_terminal": terminal,
                        "wf_fanout": fanout,
                        "wf_version": "v2",
                        "wf_join": wf_join,
                    },
                    mentions=[
                        Grounding(spans=[Span.from_dummy_for_workflow(workflow_id)])
                    ],
                )
                self.workflow_engine.write.add_node(n)

            def add_edge(
                *,
                edge_id: str,
                src: str,
                dst: str,
                pred: str | None,
                priority: int = 100,
                is_default: bool = False,
                multiplicity: str = "one",
                metadata=None,
            ):
                if metadata is None:
                    metadata = {}
                metadata_final = {
                    "entity_type": "workflow_edge",
                    "workflow_id": workflow_id,
                    "wf_edge_kind": "wf_next",
                    "wf_predicate": pred,
                    "wf_priority": priority,
                    "wf_is_default": is_default,
                    "wf_multiplicity": multiplicity,
                }
                metadata_final.update(metadata)
                e = WorkflowEdge(
                    id=edge_id,
                    label="wf_next",
                    relation="wf_next",
                    type="entity",
                    doc_id=edge_id,
                    summary="wf_next",
                    properties={},
                    source_ids=[src],
                    target_ids=[dst],
                    source_edge_ids=[],
                    target_edge_ids=[],
                    metadata=metadata_final,
                    mentions=[
                        Grounding(spans=[Span.from_dummy_for_workflow(workflow_id)])
                    ],
                )
                self.workflow_engine.write.add_edge(e)

            # ----------------------------
            # Backbone: start -> add_user_turn -> end
            # ----------------------------
            if mode == "backbone":
                start_id = wid("start")
                add_id = wid("add_user")
                end_id = wid("end")
                add_node(
                    node_id=start_id,
                    label="Start",
                    op="start",
                    start=True,
                    terminal=False,
                )
                add_node(
                    node_id=add_id,
                    label="Add user turn",
                    op="add_user_turn",
                    start=False,
                    terminal=False,
                )
                add_node(
                    node_id=end_id, label="End", op="end", start=False, terminal=True
                )
                add_edge(
                    edge_id=wid("next_start_add"),
                    src=start_id,
                    dst=add_id,
                    pred="always",
                    priority=0,
                    is_default=True,
                )
                add_edge(
                    edge_id=wid("next_add_end"),
                    src=add_id,
                    dst=end_id,
                    pred="always",
                    priority=0,
                    is_default=True,
                )
                return self.validate(workflow_id=workflow_id)

            # ----------------------------
            # Full v2 add_turn flow (v1-equivalent, single-token linear)
            # ----------------------------
            add_node(
                node_id=wid("start"),
                label="Start",
                op="start",
                start=True,
                terminal=False,
            )
            add_node(node_id=wid("add_user"), label="Add user turn", op="add_user_turn")
            add_node(
                node_id=wid("link_prev"), label="Link prev turn", op="link_prev_turn"
            )
            add_node(node_id=wid("mem"), label="Memory retrieve", op="memory_retrieve")
            add_node(
                node_id=wid("kg"), label="KG retrieve", op="kg_retrieve", fanout=True
            )
            add_node(node_id=wid("pin_mem"), label="Pin memory", op="memory_pin")
            add_node(node_id=wid("pin_kg"), label="Pin knowledge", op="kg_pin")
            add_node(node_id=wid("answer"), label="Answer", op="answer", wf_join=True)
            add_node(
                node_id=wid("link_asst"),
                label="Link assistant turn",
                op="link_assistant_turn",
            )

            if include_context_snapshot:
                add_node(
                    node_id=wid("ctx_snap"),
                    label="Context snapshot",
                    op="context_snapshot",
                )

            add_node(
                node_id=wid("decide_sum"),
                label="Decide summarize",
                op="decide_summarize",
            )
            add_node(node_id=wid("summarize"), label="Summarize", op="summarize")
            add_node(node_id=wid("end"), label="End", op="end", terminal=True)

            # edges
            add_edge(
                edge_id=wid("e0"),
                src=wid("start"),
                dst=wid("add_user"),
                pred="always",
                priority=0,
                is_default=True,
            )

            # Conditionally link into the conversation chain (in_conv=True), else skip.
            add_edge(
                edge_id=wid("e1a"),
                src=wid("add_user"),
                dst=wid("link_prev"),
                pred=None,
                priority=0,
            )
            # add_edge(edge_id=wid("e1b"), src=wid("add_user"), dst=wid("mem"), pred=None, priority=100, is_default=True)
            add_edge(
                edge_id=wid("e1c"),
                src=wid("link_prev"),
                dst=wid("mem"),
                pred=None,
                priority=100,
                is_default=True,
            )

            # v1-equivalent: pins are sequential optional steps
            # If should_pin_memory -> go pin_mem, else skip to pin_kg
            add_edge(
                edge_id=wid("e3a"),
                src=wid("mem"),
                dst=wid("pin_mem"),
                pred="should_pin_memory",
                priority=10,
            )
            add_edge(
                edge_id=wid("e3ab"),
                src=wid("mem"),
                dst=wid("kg"),
                pred="always",
                priority=0,
            )  # skip connection needed when no memory retrieved in new conversation
            add_edge(
                edge_id=wid("e2"),
                src=wid("pin_mem"),
                dst=wid("kg"),
                pred=None,
                is_default=True,
            )
            add_edge(
                edge_id=wid("e3b"),
                src=wid("kg"),
                dst=wid("pin_kg"),
                pred=None,
                priority=100,
                is_default=True,
                metadata={"wf_join": True, "wf_join_is_merge": True},
            )

            # From pin_kg (or skipped into pin_kg), always go to answer
            add_edge(
                edge_id=wid("e5"),
                src=wid("pin_kg"),
                dst=wid("answer"),
                pred=None,
                is_default=True,
            )
            add_edge(
                edge_id=wid("e3b6"),
                src=wid("kg"),
                dst=wid("answer"),
                pred=None,
                priority=100,
                is_default=True,
                metadata={"wf_join": True, "wf_join_is_merge": True},
            )

            # Always link assistant response into chain after answering.
            add_edge(
                edge_id=wid("e6"),
                src=wid("answer"),
                dst=wid("link_asst"),
                pred=None,
                is_default=True,
            )

            if include_context_snapshot:
                add_edge(
                    edge_id=wid("e7a"),
                    src=wid("link_asst"),
                    dst=wid("ctx_snap"),
                    pred=None,
                    is_default=True,
                )
                add_edge(
                    edge_id=wid("e7b"),
                    src=wid("ctx_snap"),
                    dst=wid("decide_sum"),
                    pred=None,
                    is_default=True,
                )
            else:
                add_edge(
                    edge_id=wid("e7"),
                    src=wid("link_asst"),
                    dst=wid("decide_sum"),
                    pred=None,
                    is_default=True,
                )

            add_edge(
                edge_id=wid("e8"),
                src=wid("decide_sum"),
                dst=wid("summarize"),
                pred="should_summarize",
                priority=0,
            )
            add_edge(
                edge_id=wid("e9"),
                src=wid("decide_sum"),
                dst=wid("end"),
                pred=None,
                is_default=True,
            )
            add_edge(
                edge_id=wid("e10"),
                src=wid("summarize"),
                dst=wid("end"),
                pred=None,
                is_default=True,
            )
            if allow_branch := False:
                add_node(
                    node_id=wid("start"),
                    label="Start",
                    op="start",
                    start=True,
                    terminal=False,
                )
                add_node(
                    node_id=wid("add_user"), label="Add user turn", op="add_user_turn"
                )
                add_node(
                    node_id=wid("link_prev"),
                    label="Link prev turn",
                    op="link_prev_turn",
                )
                add_node(
                    node_id=wid("mem"),
                    label="Memory retrieve",
                    op="memory_retrieve",
                    metadata={"wf_join": True, "wf_join_is_merge": True},
                )
                add_node(
                    node_id=wid("kg"),
                    label="KG retrieve",
                    op="kg_retrieve",
                    fanout=True,
                )
                add_node(node_id=wid("pin_mem"), label="Pin memory", op="memory_pin")
                add_node(node_id=wid("pin_kg"), label="Pin knowledge", op="kg_pin")
                add_node(
                    node_id=wid("answer"),
                    label="Answer",
                    op="answer",
                    metadata={"wf_join": True, "wf_join_is_merge": True},
                )
                add_node(
                    node_id=wid("link_asst"),
                    label="Link assistant turn",
                    op="link_assistant_turn",
                )

                if include_context_snapshot:
                    add_node(
                        node_id=wid("ctx_snap"),
                        label="Context snapshot",
                        op="context_snapshot",
                    )

                add_node(
                    node_id=wid("decide_sum"),
                    label="Decide summarize",
                    op="decide_summarize",
                )
                add_node(node_id=wid("summarize"), label="Summarize", op="summarize")
                add_node(
                    node_id=wid("end"),
                    label="End",
                    op="end",
                    terminal=True,
                    metadata={"wf_join": True, "wf_join_is_merge": True},
                )

                # edges
                add_edge(
                    edge_id=wid("e0"),
                    src=wid("start"),
                    dst=wid("add_user"),
                    pred="always",
                    priority=0,
                    is_default=True,
                )

                # Conditionally link into the conversation chain (in_conv=True), else skip.
                add_edge(
                    edge_id=wid("e1a"),
                    src=wid("add_user"),
                    dst=wid("link_prev"),
                    pred=None,
                    priority=0,
                )
                add_edge(
                    edge_id=wid("e1b"),
                    src=wid("add_user"),
                    dst=wid("mem"),
                    pred=None,
                    priority=100,
                    is_default=True,
                )
                add_edge(
                    edge_id=wid("e1c"),
                    src=wid("link_prev"),
                    dst=wid("mem"),
                    pred=None,
                    priority=100,
                    is_default=True,
                )

                add_edge(
                    edge_id=wid("e2"),
                    src=wid("mem"),
                    dst=wid("kg"),
                    pred=None,
                    is_default=True,
                )

                # From KG retrieve, optionally pin both memory and KG (fanout node).
                add_edge(
                    edge_id=wid("e3"),
                    src=wid("kg"),
                    dst=wid("pin_mem"),
                    pred="should_pin_memory",
                    priority=0,
                )
                add_edge(
                    edge_id=wid("e4"),
                    src=wid("kg"),
                    dst=wid("pin_kg"),
                    pred="should_pin_kg",
                    priority=1,
                )

                # If no pins happen, go straight to answer.
                add_edge(
                    edge_id=wid("e5"),
                    src=wid("kg"),
                    dst=wid("answer"),
                    pred=None,
                    is_default=True,
                )
                # After pins, continue to answer.
                add_edge(
                    edge_id=wid("e6"),
                    src=wid("pin_mem"),
                    dst=wid("answer"),
                    pred=None,
                    is_default=True,
                )
                add_edge(
                    edge_id=wid("e7"),
                    src=wid("pin_kg"),
                    dst=wid("answer"),
                    pred=None,
                    is_default=True,
                )

                # Always link assistant response into chain after answering.
                add_edge(
                    edge_id=wid("e8"),
                    src=wid("answer"),
                    dst=wid("link_asst"),
                    pred=None,
                    is_default=True,
                )

                if include_context_snapshot:
                    add_edge(
                        edge_id=wid("e9"),
                        src=wid("link_asst"),
                        dst=wid("ctx_snap"),
                        pred=None,
                        is_default=True,
                    )
                    add_edge(
                        edge_id=wid("e9b"),
                        src=wid("ctx_snap"),
                        dst=wid("decide_sum"),
                        pred=None,
                        is_default=True,
                    )
                else:
                    add_edge(
                        edge_id=wid("e9"),
                        src=wid("link_asst"),
                        dst=wid("decide_sum"),
                        pred=None,
                        is_default=True,
                    )

                add_edge(
                    edge_id=wid("e10"),
                    src=wid("decide_sum"),
                    dst=wid("summarize"),
                    pred="should_summarize",
                    priority=0,
                )
                add_edge(
                    edge_id=wid("e11"),
                    src=wid("decide_sum"),
                    dst=wid("end"),
                    pred=None,
                    is_default=True,
                )
                add_edge(
                    edge_id=wid("e12"),
                    src=wid("summarize"),
                    dst=wid("end"),
                    pred=None,
                    is_default=True,
                )

            return self.validate(workflow_id=workflow_id)

    def ensure_backbone(
        self, *, workflow_id: str
    ) -> tuple[WorkflowNode, dict[str, WorkflowNode], dict[str, list[WorkflowEdge]]]:
        """Backward-compatible alias for backbone design.

        Kept for existing tests/callers; delegates to :meth:`ensure_add_turn_flow`.
        """
        return self.ensure_add_turn_flow(
            workflow_id=workflow_id, mode="backbone", include_context_snapshot=False
        )

    def _print_to_do(self):
        print("""_summary_
        to dos
        1. if multiple node target is another node, the add edge should have an arg to say target change to wait join
        if it change, the target need to be tombstoned and point to a new target with correct join meta semantics.
        """)
