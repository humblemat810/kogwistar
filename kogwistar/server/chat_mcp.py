from __future__ import annotations

from typing import Any, Callable

try:
    from fastmcp import FastMCP
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Knowledge MCP support requires the optional 'server' extra. "
        "Install with: pip install 'kogwistar[server]'"
    ) from exc


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
    def conversation_create(
        user_id: str,
        conversation_id: str | None = None,
        start_node_id: str | None = None,
    ) -> dict[str, Any]:
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
    get_subject: Callable[[], str | None] | None = None,
    get_user_id: Callable[[], str | None] | None = None,
    require_workflow_access: Callable[[str, str], None] | None = None,
):
    mcp = FastMCP("Workflow Diagnostics MCP")

    def _actor_sub() -> str | None:
        if not callable(get_subject):
            return None
        return get_subject()

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
        if require_workflow_access:
            require_workflow_access(workflow_id, "rw")
        effective_user_id = (
            get_user_id() if callable(get_user_id) else None
        ) or user_id
        return get_service().submit_workflow_run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            initial_state=initial_state or {},
            turn_node_id=turn_node_id,
            user_id=effective_user_id,
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
    def workflow_run_events(
        run_id: str, after_seq: int = 0, limit: int = 200
    ) -> dict[str, Any]:
        events = get_service().list_run_events(run_id, after_seq=after_seq)
        if limit > 0:
            events = events[: int(limit)]
        return {
            "run_id": run_id,
            "events": events,
        }

    @tool_roles({role_ro, role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.process_table")
    def workflow_process_table(
        status: str | None = None,
        workflow_id: str | None = None,
        conversation_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        return {
            "processes": get_service().list_process_table(
                status=status,
                workflow_id=workflow_id,
                conversation_id=conversation_id,
                limit=limit,
            )
        }

    @tool_roles({role_ro, role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.operator_inbox")
    def workflow_operator_inbox(
        inbox_id: str | None = None, status: str | None = None, limit: int = 200
    ) -> dict[str, Any]:
        return {
            "messages": get_service().list_operator_inbox(
                inbox_id=inbox_id, status=status, limit=limit
            )
        }

    @tool_roles({role_ro, role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.blocked_runs")
    def workflow_blocked_runs(limit: int = 100) -> dict[str, Any]:
        return {"processes": get_service().list_blocked_runs(limit=limit)}

    @tool_roles({role_ro, role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.process_timeline")
    def workflow_process_timeline(
        run_id: str, after_seq: int = 0, limit: int = 200
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "events": get_service().list_process_timeline(
                run_id=run_id, after_seq=after_seq, limit=limit
            ),
        }

    @tool_roles({role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.service_declare")
    def workflow_service_declare(
        service_id: str,
        service_kind: str,
        target_kind: str,
        target_ref: str,
        target_config: dict[str, Any] | None = None,
        enabled: bool = True,
        autostart: bool = False,
        restart_policy: dict[str, Any] | None = None,
        heartbeat_ttl_ms: int = 60_000,
        trigger_specs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return get_service().declare_service(
            service_id=service_id,
            service_kind=service_kind,
            target_kind=target_kind,
            target_ref=target_ref,
            target_config=target_config or {},
            enabled=enabled,
            autostart=autostart,
            restart_policy=restart_policy or {},
            heartbeat_ttl_ms=heartbeat_ttl_ms,
            trigger_specs=trigger_specs or [],
        )

    @tool_roles({role_ro, role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.service_list")
    def workflow_service_list(limit: int = 200) -> dict[str, Any]:
        return {"services": get_service().list_services(limit=limit)}

    @tool_roles({role_ro, role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.service_get")
    def workflow_service_get(service_id: str) -> dict[str, Any]:
        return get_service().get_service(service_id)

    @tool_roles({role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.service_enable")
    def workflow_service_enable(service_id: str) -> dict[str, Any]:
        return get_service().enable_service(service_id)

    @tool_roles({role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.service_disable")
    def workflow_service_disable(service_id: str) -> dict[str, Any]:
        return get_service().disable_service(service_id)

    @tool_roles({role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.service_heartbeat")
    def workflow_service_heartbeat(
        service_id: str,
        instance_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return get_service().record_service_heartbeat(
            service_id,
            instance_id=instance_id,
            payload=payload or {},
        )

    @tool_roles({role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.service_trigger")
    def workflow_service_trigger(
        service_id: str,
        trigger_type: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return get_service().trigger_service(
            service_id,
            trigger_type=trigger_type,
            payload=payload or {},
        )

    @tool_roles({role_ro, role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.service_events")
    def workflow_service_events(
        service_id: str, limit: int = 500
    ) -> dict[str, Any]:
        return {
            "service_id": service_id,
            "events": get_service().list_service_events(service_id, limit=limit),
        }

    @tool_roles({role_ro, role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.capabilities_snapshot")
    def workflow_capabilities_snapshot() -> dict[str, Any]:
        return get_service().capability_snapshot()

    @tool_roles({role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.capability_approve")
    def workflow_capability_approve(
        action: str,
        capabilities: list[str] | str,
        subject: str | None = None,
    ) -> dict[str, Any]:
        return get_service().capability_approve(
            action=action, capabilities=capabilities, subject=subject
        )

    @tool_roles({role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.capability_revoke")
    def workflow_capability_revoke(
        capability: str,
        subject: str | None = None,
    ) -> dict[str, Any]:
        return get_service().capability_revoke(
            capability=capability, subject=subject
        )

    @tool_roles({role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.design_node_upsert")
    def workflow_design_node_upsert(
        workflow_id: str,
        designer_id: str,
        label: str,
        node_id: str | None = None,
        op: str | None = None,
        start: bool = False,
        terminal: bool = False,
        fanout: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return get_service().workflow_design_upsert_node(
            workflow_id=workflow_id,
            designer_id=designer_id,
            node_id=node_id,
            label=label,
            op=op,
            start=start,
            terminal=terminal,
            fanout=fanout,
            metadata=metadata or {},
            actor_sub=_actor_sub(),
            source="mcp",
        )

    @tool_roles({role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.design_edge_upsert")
    def workflow_design_edge_upsert(
        workflow_id: str,
        designer_id: str,
        src: str,
        dst: str,
        edge_id: str | None = None,
        relation: str = "wf_next",
        predicate: str | None = None,
        priority: int = 100,
        is_default: bool = False,
        multiplicity: str = "one",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return get_service().workflow_design_upsert_edge(
            workflow_id=workflow_id,
            designer_id=designer_id,
            edge_id=edge_id,
            src=src,
            dst=dst,
            relation=relation,
            predicate=predicate,
            priority=priority,
            is_default=is_default,
            multiplicity=multiplicity,
            metadata=metadata or {},
            actor_sub=_actor_sub(),
            source="mcp",
        )

    @tool_roles({role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.design_node_delete")
    def workflow_design_node_delete(
        workflow_id: str, node_id: str, designer_id: str
    ) -> dict[str, Any]:
        return get_service().workflow_design_delete_node(
            workflow_id=workflow_id,
            node_id=node_id,
            designer_id=designer_id,
            actor_sub=_actor_sub(),
            source="mcp",
        )

    @tool_roles({role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.design_edge_delete")
    def workflow_design_edge_delete(
        workflow_id: str, edge_id: str, designer_id: str
    ) -> dict[str, Any]:
        return get_service().workflow_design_delete_edge(
            workflow_id=workflow_id,
            edge_id=edge_id,
            designer_id=designer_id,
            actor_sub=_actor_sub(),
            source="mcp",
        )

    @tool_roles({role_ro, role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.design_history")
    def workflow_design_history(workflow_id: str) -> dict[str, Any]:
        return get_service().workflow_design_history(workflow_id=workflow_id)

    @tool_roles({role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.design_undo")
    def workflow_design_undo(workflow_id: str, designer_id: str) -> dict[str, Any]:
        return get_service().workflow_design_undo(
            workflow_id=workflow_id,
            designer_id=designer_id,
            actor_sub=_actor_sub(),
            source="mcp",
        )

    @tool_roles({role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.design_redo")
    def workflow_design_redo(workflow_id: str, designer_id: str) -> dict[str, Any]:
        return get_service().workflow_design_redo(
            workflow_id=workflow_id,
            designer_id=designer_id,
            actor_sub=_actor_sub(),
            source="mcp",
        )

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

    @tool_roles({role_ro, role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.run_resume_contract")
    def workflow_run_resume_contract(run_id: str) -> dict[str, Any]:
        return get_service().resume_contract(run_id)

    @tool_roles({role_rw})
    @require_ns({ns_workflow})
    @mcp.tool(name="workflow.run_resume")
    def workflow_run_resume(
        run_id: str,
        suspended_node_id: str,
        suspended_token_id: str,
        client_result: dict[str, Any],
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        return get_service().resume_run(
            run_id=run_id,
            suspended_node_id=suspended_node_id,
            suspended_token_id=suspended_token_id,
            client_result=client_result,
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            user_id=user_id,
        )

    return mcp
