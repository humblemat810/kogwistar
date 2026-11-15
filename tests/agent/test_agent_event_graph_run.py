from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import cast

import pytest

from kogwistar.agent import (
    ControlPointSpec,
    EventAgentControlJournal,
    EventWorkflowCallbacks,
    LaneProviderEventJournal,
    ProviderCapabilities,
    ProviderEvent,
    ProviderFieldCompleted,
    ProviderReasoningSummary,
    ProviderRequest,
    ProviderResponseCompleted,
    ProviderResponseFailed,
    ProviderResponseStarted,
    ProviderToolCallCompleted,
    StreamingModelProvider,
    build_event_workflow,
    register_event_workflow_steps,
)
from kogwistar.conversation.models import ConversationNode
from kogwistar.engine_core.engine import GraphKnowledgeEngine, scoped_namespace
from kogwistar.engine_core.models import Grounding, Span
from kogwistar.messaging.service import LaneMessagingService
from kogwistar.runtime import AsyncMappingStepResolver, AsyncWorkflowRuntime
from kogwistar.runtime.models import WorkflowDesignArtifact
from kogwistar.server.auth_middleware import claims_ctx
from kogwistar.server.chat_service_conversation_queries import _ConversationQueryService
from kogwistar.server.chat_service_shared import ChatRunServiceOwner
from tests._helpers.embeddings import ConstantEmbeddingFunction
from tests._helpers.fake_backend import build_fake_backend

pytestmark = [pytest.mark.ci, pytest.mark.runtime, pytest.mark.runtime_async]


class _FakeProvider(StreamingModelProvider):
    provider_id = "fake.event-graph"
    capabilities = ProviderCapabilities(structured_output=True)

    def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        del request

        async def events() -> AsyncIterator[ProviderEvent]:
            yield ProviderResponseStarted(
                provider_event_id="evt-start", response_id="response-1", provider_sequence=1
            )
            yield ProviderFieldCompleted(
                provider_event_id="evt-field",
                response_id="response-1",
                provider_sequence=2,
                field_name="answer",
                value="hello from graph loop",
            )
            yield ProviderResponseCompleted(
                provider_event_id="evt-done",
                response_id="response-1",
                provider_sequence=3,
                text="hello from graph loop",
                structured_output={"answer": "hello from graph loop"},
            )

        return events()


class _Callbacks(EventWorkflowCallbacks):
    def __init__(self, conversation_engine: GraphKnowledgeEngine) -> None:
        self.conversation_engine = conversation_engine
        self.deltas: list[tuple[str, int]] = []
        self.advisory_events: list[str] = []
        self.assistant_turn_keys: list[str] = []
        self.request_inputs: list[str] = []

    def build_request(self, state: Mapping[str, object]) -> ProviderRequest:
        control = state.get("agent_control_input")
        control_text = str(control.get("text") or "") if isinstance(control, Mapping) else ""
        self.request_inputs.append(control_text)
        return ProviderRequest(
            model="fake",
            input="\n".join(
                value for value in (str(state.get("user_input", "question")), control_text) if value
            ),
        )

    async def before_model(self, context: object) -> object:
        del context
        return {"context_hint": "test"}

    async def publish_delta(self, context: object, delta: str, sequence: int) -> object:
        del context
        self.deltas.append((delta, sequence))
        return None

    async def on_advisory_event(
        self,
        context: object,
        event: ProviderFieldCompleted | ProviderReasoningSummary,
        *,
        idempotency_key: str,
    ) -> object:
        del context
        self.advisory_events.append(f"{event.kind}:{idempotency_key}")
        return {"reviewed": True}

    async def authorize_and_dispatch_tool(
        self,
        context: object,
        call: ProviderToolCallCompleted,
        *,
        idempotency_key: str,
    ) -> object:
        del context, call, idempotency_key
        raise AssertionError("fixture does not emit tool calls")

    async def before_tool_dispatch(self, context: object, call: ProviderToolCallCompleted) -> object:
        del context, call
        return {}

    async def after_tool_outcome(
        self,
        context: object,
        call: ProviderToolCallCompleted,
        outcome: Mapping[str, object],
    ) -> object:
        del context, call, outcome
        return {}

    async def append_assistant_turn(
        self,
        context: object,
        response: ProviderResponseCompleted,
        *,
        idempotency_key: str,
    ) -> object:
        del context
        self.assistant_turn_keys.append(idempotency_key)
        node = ConversationNode(
            id="assistant-turn-1",
            label="Assistant turn",
            type="entity",
            doc_id="assistant-turn-1",
            summary=response.text,
            role="assistant",
            turn_index=2,
            conversation_id="event-loop-conversation",
            user_id="user-1",
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation("event-loop-conversation")])],
            metadata={
                "entity_type": "assistant_turn",
                "in_conversation_chain": True,
                "level_from_root": 0,
            },
            properties={},
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
            level_from_root=0,
        )
        self.conversation_engine.write.add_node(node)
        return {"assistant_turn_id": node.id}

    async def record_failure(
        self,
        context: object,
        failure: ProviderResponseFailed,
        *,
        idempotency_key: str,
    ) -> object:
        del context, failure, idempotency_key
        return None


class _TranscriptOwner:
    def __init__(self, engine: GraphKnowledgeEngine) -> None:
        self.engine = engine

    def _conversation_engine(self) -> GraphKnowledgeEngine:
        return self.engine

    def _conversation_nodes(self, conversation_id: str) -> list[ConversationNode]:
        return cast(
            list[ConversationNode],
            self.engine.read.get_nodes(
                where={"conversation_id": conversation_id},
                node_type=ConversationNode,
                limit=100,
            ),
        )


def _add_user_turn(engine: GraphKnowledgeEngine) -> None:
    engine.write.add_node(
        ConversationNode(
            id="user-turn-1",
            label="User turn",
            type="entity",
            doc_id="user-turn-1",
            summary="What is in the graph?",
            role="user",
            turn_index=1,
            conversation_id="event-loop-conversation",
            user_id="user-1",
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation("event-loop-conversation")])],
            metadata={
                "entity_type": "conversation_turn",
                "in_conversation_chain": True,
                "level_from_root": 0,
            },
            properties={},
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
            level_from_root=0,
        )
    )


def test_fake_provider_runs_graph_workflow_and_keeps_events_out_of_transcript(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    embedding = ConstantEmbeddingFunction(dim=4)
    workflow_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "workflow"),
        kg_graph_type="workflow",
        embedding_function=embedding,
        backend_factory=build_fake_backend,
    )
    conversation_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "conversation"),
        kg_graph_type="conversation",
        embedding_function=embedding,
        backend_factory=build_fake_backend,
    )
    workflow_id = "agent.event.integration.v1"
    run_id = "run-event-integration"
    conversation_id = "event-loop-conversation"
    scope_token = claims_ctx.set({"ns": "conversation", "security_scope": "tenant-event"})
    try:
        with scoped_namespace(workflow_engine, f"wf_design:{workflow_id}"), scoped_namespace(
            conversation_engine, "conversation"
        ):
            _add_user_turn(conversation_engine)
            control_messaging = LaneMessagingService(conversation_engine)
            control_journal = EventAgentControlJournal(control_messaging)
            control_submission = control_journal.submit(
                run_id=run_id,
                conversation_id=conversation_id,
                user_id="user-1",
                recipient_id="agent-1",
                control_id="event-control-1",
                text="include the saved control input",
                effective_mode="queue",
                security_scope="tenant-event",
            )
            assert control_submission.status == "pending"
            design: WorkflowDesignArtifact = build_event_workflow(workflow_id=workflow_id)
            for node in design.nodes:
                workflow_engine.write.add_node(node)
            for edge in design.edges:
                workflow_engine.write.add_edge(edge)

            callbacks = _Callbacks(conversation_engine)
            journal = LaneProviderEventJournal(
                LaneMessagingService(conversation_engine), claimed_by="event-worker"
            )
            resolver = AsyncMappingStepResolver()
            predicates = register_event_workflow_steps(
                resolver,
                provider=_FakeProvider(),
                journal=journal,
                callbacks=callbacks,
                conversation_id=conversation_id,
                sender_id="agent-1",
                security_scope="tenant-event",
                control_messaging=control_messaging,
                control_spec=ControlPointSpec(
                    inbox_id="agent-event-controls",
                    policy="queue",
                    recipient_id="agent-1",
                    run_specific_inbox=True,
                ),
                advisory_field_names=frozenset({"answer"}),
            )
            runtime = AsyncWorkflowRuntime(
                workflow_engine=workflow_engine,
                conversation_engine=conversation_engine,
                step_resolver=resolver,
                predicate_registry=predicates,
                checkpoint_every_n_steps=1,
                trace=False,
            )
            result = asyncio.run(
                runtime.run(
                    workflow_id=workflow_id,
                    conversation_id=conversation_id,
                    turn_node_id="user-turn-1",
                    initial_state={"user_input": "What is in the graph?"},
                    run_id=run_id,
                )
            )

            assert result.status == "succeeded"
            assert result.final_state["agent_event_status"] == "succeeded"
            assert len(callbacks.advisory_events) == 1
            assert callbacks.advisory_events[0].startswith("field_completed:agent-advisory-hook:v1:")
            assert callbacks.request_inputs == ["include the saved control input"]
            assert len(callbacks.assistant_turn_keys) == 1
            assert callbacks.assistant_turn_keys[0].startswith("agent-event-effect:v1:")
            assert callbacks.deltas == []
            messages = conversation_engine.list_projected_lane_messages(
                inbox_id="agent-provider-events"
            )
            assert len(messages) == 3
            assert {message.status for message in messages} == {"completed"}
            control_messages = conversation_engine.list_projected_lane_messages(
                inbox_id=f"agent-event-controls:run:{run_id}", run_id=run_id
            )
            assert len(control_messages) == 1
            assert control_messages[0].status == "completed"
            assert control_messages[0].message_id == control_submission.message_id
            committed_control = control_journal.get_submission(
                run_id=run_id, user_id="user-1", control_id="event-control-1"
            )
            assert committed_control is not None
            assert committed_control.status == "committed"
            run_links = conversation_engine.read.get_edges(where={"relation": "about_run"})
            step_links = conversation_engine.read.get_edges(
                where={"relation": "handled_by_step"}
            )
            producer_links = conversation_engine.read.get_edges(
                where={"relation": "emitted_by_step"}
            )
            assert len(run_links) == 3
            assert len(step_links) == 3
            assert len(producer_links) == 3
            execution_steps = conversation_engine.read.get_nodes(
                where={
                    "$and": [
                        {"entity_type": "workflow_step_exec"},
                        {"run_id": run_id},
                    ]
                },
                limit=100,
            )
            receive_steps = [
                step
                for step in execution_steps
                if step.metadata.get("op") == "agent.event.receive"
            ]
            assert len(receive_steps) == 3

            transcript = object.__new__(_ConversationQueryService)
            transcript._owner = cast(
                ChatRunServiceOwner, _TranscriptOwner(conversation_engine)
            )
            projected = transcript.list_transcript(conversation_id)
            assert [turn["entity_type"] for turn in projected] == [
                "conversation_turn",
                "assistant_turn",
            ]
            assert all("provider_event" not in str(turn["entity_type"]) for turn in projected)
            other_user_control = control_journal.submit(
                run_id=run_id,
                conversation_id=conversation_id,
                user_id="user-2",
                recipient_id="agent-1",
                control_id="event-control-1",
                text="different principal, same client control ID",
                effective_mode="queue",
                security_scope="tenant-event",
            )
            assert other_user_control.message_id != control_submission.message_id
    finally:
        claims_ctx.reset(scope_token)
        workflow_engine.close()
        conversation_engine.close()
