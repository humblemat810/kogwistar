from __future__ import annotations

from typing import Any, Callable

from fastmcp import FastMCP


def build_conversation_mcp(
    *,
    get_service: Callable[[], Any],
    tool_roles: Callable[[Any], Callable[[Callable[..., Any]], Callable[..., Any]]],
    require_ns: Callable[[Any], Callable[[Callable[..., Any]], Callable[..., Any]]],
    role_ro: Any,
    role_rw: Any,
    ns_conversation: Any,
):
    mcp = FastMCP("Conversation MCP")

    @tool_roles({role_rw})
    @require_ns({ns_conversation})
    @mcp.tool(name="conversation.create")
    def conversation_create(user_id: str, conversation_id: str | None = None, start_node_id: str | None = None) -> dict[str, Any]:
        return get_service().create_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            start_node_id=start_node_id,
        )

    @tool_roles({role_ro, role_rw})
    @require_ns({ns_conversation})
    @mcp.tool(name="conversation.get_transcript")
    def conversation_get_transcript(conversation_id: str) -> dict[str, Any]:
        return {
            "conversation_id": conversation_id,
            "turns": get_service().list_transcript(conversation_id),
        }

    @tool_roles({role_rw})
    @require_ns({ns_conversation})
    @mcp.tool(name="conversation.ask")
    def conversation_ask(
        conversation_id: str,
        text: str,
        user_id: str | None = None,
        workflow_id: str = "agentic_answering.v2",
    ) -> dict[str, Any]:
        return get_service().submit_turn_for_answer(
            conversation_id=conversation_id,
            user_id=user_id,
            text=text,
            workflow_id=workflow_id,
        )

    @tool_roles({role_ro, role_rw})
    @require_ns({ns_conversation})
    @mcp.tool(name="conversation.run_status")
    def conversation_run_status(run_id: str) -> dict[str, Any]:
        return get_service().get_run(run_id)

    @tool_roles({role_rw})
    @require_ns({ns_conversation})
    @mcp.tool(name="conversation.cancel_run")
    def conversation_cancel_run(run_id: str) -> dict[str, Any]:
        return get_service().cancel_run(run_id)

    return mcp


def build_workflow_mcp(
    *,
    get_service: Callable[[], Any],
    tool_roles: Callable[[Any], Callable[[Callable[..., Any]], Callable[..., Any]]],
    require_ns: Callable[[Any], Callable[[Callable[..., Any]], Callable[..., Any]]],
    role_ro: Any,
    role_rw: Any,
    ns_workflow: Any,
):
    mcp = FastMCP("Workflow Diagnostics MCP")

    @tool_roles({role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.run_submit")
    def workflow_run_submit(
        workflow_id: str,
        conversation_id: str,
        initial_state: dict[str, Any] | None = None,
        turn_node_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        return get_service().submit_workflow_run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            initial_state=initial_state or {},
            turn_node_id=turn_node_id,
            user_id=user_id,
        )

    @tool_roles({role_ro, role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.run_status")
    def workflow_run_status(run_id: str) -> dict[str, Any]:
        return get_service().get_run(run_id)

    @tool_roles({role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.run_cancel")
    def workflow_run_cancel(run_id: str) -> dict[str, Any]:
        return get_service().cancel_run(run_id)

    @tool_roles({role_ro, role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.run_events")
    def workflow_run_events(run_id: str, after_seq: int = 0, limit: int = 200) -> dict[str, Any]:
        events = get_service().list_run_events(run_id, after_seq=after_seq)
        if limit > 0:
            events = events[: int(limit)]
        return {
            "run_id": run_id,
            "events": events,
        }

    @tool_roles({role_ro, role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.run_checkpoint_get")
    def workflow_run_checkpoint_get(run_id: str, step_seq: int) -> dict[str, Any]:
        return get_service().get_checkpoint(run_id, step_seq)

    @tool_roles({role_ro, role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.run_replay")
    def workflow_run_replay(run_id: str, target_step_seq: int) -> dict[str, Any]:
        return get_service().replay_run(run_id, target_step_seq)

    return mcp
