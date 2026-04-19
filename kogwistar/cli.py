from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx

from kogwistar.demo import run_provenance_quickstart
from kogwistar.server_mcp_with_admin import main as serve_main


def _print_quickstart_summary(summary: dict[str, object]) -> None:
    artifacts = dict(summary.get("artifacts") or {})
    print(f"Answer: {summary.get('answer_text', '')}")
    print(f"Replay: {'pass' if summary.get('replay_pass') else 'fail'}")
    print(f"Provenance Artifact: {artifacts.get('provenance_html', '')}")
    print(f"Graph Artifact: {artifacts.get('graph_html', '')}")
    print(f"Replay Report: {artifacts.get('replay_json', '')}")
    print(f"Next: {summary.get('next_command', '')}")


def _base_url(args: argparse.Namespace) -> str:
    return str(
        getattr(args, "base_url", None)
        or os.getenv("KOGWISTAR_BASE_URL")
        or "http://127.0.0.1:8787"
    ).rstrip("/")


def _request_json(
    *,
    method: str,
    url: str,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with httpx.Client(timeout=30.0) as client:
        resp = client.request(method, url, params=params, json=json_body)
        resp.raise_for_status()
        if not resp.content:
            return {}
        return resp.json()


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _add_base_url(target: argparse.ArgumentParser) -> None:
    target.add_argument(
        "--base-url",
        default=None,
        help="Kogwistar server base URL, default from KOGWISTAR_BASE_URL or http://127.0.0.1:8787",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kogwistar",
        description="Provenance-first, replayable AI workflow tooling.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_demo_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--data-dir", default=".gke-data/quickstart")
        target.add_argument(
            "--question",
            default="How does Kogwistar make AI workflows replayable and auditable?",
        )
        target.add_argument(
            "--open-browser",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        target.add_argument("--json", action="store_true")

    quickstart = sub.add_parser(
        "quickstart",
        help="Run the deterministic provenance-first demo.",
    )
    _add_demo_args(quickstart)

    serve = sub.add_parser(
        "serve", help="Start the existing MCP/server surface."
    )
    serve.add_argument(
        "--data-dir",
        default=None,
        help="Reserved for future use; server storage remains env-driven.",
    )
    _add_base_url(serve)

    api = sub.add_parser("api", help="Call HTTP diagnostics / operator APIs.")
    _add_base_url(api)
    api_sub = api.add_subparsers(dest="api_command", required=True)

    def _conversation(ns: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
        create = ns.add_parser("conversation.create")
        _add_base_url(create)
        create.add_argument("--user-id", required=True)
        create.add_argument("--conversation-id", default=None)
        create.add_argument("--start-node-id", default=None)

        transcript = ns.add_parser("conversation.get_transcript")
        _add_base_url(transcript)
        transcript.add_argument("--conversation-id", required=True)

        ask = ns.add_parser("conversation.ask")
        _add_base_url(ask)
        ask.add_argument("--conversation-id", required=True)
        ask.add_argument("--text", required=True)
        ask.add_argument("--user-id", default=None)
        ask.add_argument("--workflow-id", default="agentic_answering.v2")

        status = ns.add_parser("conversation.run_status")
        _add_base_url(status)
        status.add_argument("--run-id", required=True)

        cancel = ns.add_parser("conversation.cancel_run")
        _add_base_url(cancel)
        cancel.add_argument("--run-id", required=True)

    def _workflow(ns: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
        run_submit = ns.add_parser("workflow.run_submit")
        _add_base_url(run_submit)
        run_submit.add_argument("--workflow-id", required=True)
        run_submit.add_argument("--conversation-id", required=True)
        run_submit.add_argument("--turn-node-id", default=None)
        run_submit.add_argument("--user-id", default=None)
        run_submit.add_argument("--initial-state", default="{}")
        run_submit.add_argument("--priority-class", default="foreground")
        run_submit.add_argument("--token-budget", type=int, default=None)
        run_submit.add_argument("--time-budget-ms", type=int, default=None)

        p = ns.add_parser("workflow.run_status")
        _add_base_url(p)
        p.add_argument("--run-id", required=True)
        p = ns.add_parser("workflow.run_cancel")
        _add_base_url(p)
        p.add_argument("--run-id", required=True)
        p = ns.add_parser("workflow.run_checkpoint_get")
        _add_base_url(p)
        p.add_argument("--run-id", required=True)
        p.add_argument("--step-seq", type=int, required=True)

        run_events = ns.add_parser("workflow.run_events")
        _add_base_url(run_events)
        run_events.add_argument("--run-id", required=True)
        run_events.add_argument("--after-seq", type=int, default=0)
        run_events.add_argument("--limit", type=int, default=200)

        for name in ("workflow.process_table", "workflow.blocked_runs", "workflow.operator_dashboard", "workflow.service_list", "workflow.service_events", "workflow.dead_letters", "workflow.capabilities_snapshot"):
            p = ns.add_parser(name)
            _add_base_url(p)
        ns.choices["workflow.process_table"].add_argument("--status", default=None)
        ns.choices["workflow.process_table"].add_argument("--workflow-id", default=None)
        ns.choices["workflow.process_table"].add_argument("--conversation-id", default=None)
        ns.choices["workflow.process_table"].add_argument("--limit", type=int, default=100)
        ns.choices["workflow.blocked_runs"].add_argument("--limit", type=int, default=100)
        ns.choices["workflow.operator_dashboard"].add_argument("--limit", type=int, default=100)
        ns.choices["workflow.service_list"].add_argument("--limit", type=int, default=200)
        ns.choices["workflow.service_events"].add_argument("--service-id", required=True)
        ns.choices["workflow.service_events"].add_argument("--limit", type=int, default=500)
        ns.choices["workflow.dead_letters"].add_argument("--limit", type=int, default=100)

        for name in ("workflow.service_get", "workflow.service_enable", "workflow.service_disable", "workflow.service_repair", "workflow.services_repair", "workflow.message_orphans_repair", "workflow.dead_letter_replay", "workflow.run_resume_contract", "workflow.run_replay", "workflow.run_resume", "workflow.design_history", "workflow.design_undo", "workflow.design_redo", "workflow.capability_approve", "workflow.capability_revoke", "workflow.design_node_upsert", "workflow.design_edge_upsert", "workflow.design_node_delete", "workflow.design_edge_delete"):
            p = ns.add_parser(name)
            _add_base_url(p)
            p.add_argument("--service-id", default=None)
            p.add_argument("--run-id", default=None)
            p.add_argument("--workflow-id", default=None)
        ns.choices["workflow.service_get"].add_argument("--service-id", required=True)
        ns.choices["workflow.service_enable"].add_argument("--service-id", required=True)
        ns.choices["workflow.service_disable"].add_argument("--service-id", required=True)
        ns.choices["workflow.service_repair"].add_argument("--service-id", required=True)
        ns.choices["workflow.services_repair"].add_argument("--limit", type=int, default=10000)
        ns.choices["workflow.message_orphans_repair"].add_argument("--inbox-id", default=None)
        ns.choices["workflow.message_orphans_repair"].add_argument("--limit", type=int, default=100)
        ns.choices["workflow.dead_letter_replay"].add_argument("--run-id", required=True)
        ns.choices["workflow.run_resume_contract"].add_argument("--run-id", required=True)
        ns.choices["workflow.run_replay"].add_argument("--target-step-seq", type=int, required=True)
        ns.choices["workflow.run_resume"].add_argument("--suspended-node-id", required=True)
        ns.choices["workflow.run_resume"].add_argument("--suspended-token-id", required=True)
        ns.choices["workflow.run_resume"].add_argument("--client-result", default="{}")
        ns.choices["workflow.run_resume"].add_argument("--workflow-id", required=True)
        ns.choices["workflow.run_resume"].add_argument("--conversation-id", required=True)
        ns.choices["workflow.run_resume"].add_argument("--turn-node-id", required=True)
        ns.choices["workflow.run_resume"].add_argument("--user-id", default=None)
        ns.choices["workflow.design_history"].add_argument("--workflow-id", required=True)
        ns.choices["workflow.design_undo"].add_argument("--workflow-id", required=True)
        ns.choices["workflow.design_undo"].add_argument("--designer-id", required=True)
        ns.choices["workflow.design_redo"].add_argument("--workflow-id", required=True)
        ns.choices["workflow.design_redo"].add_argument("--designer-id", required=True)
        for nm in ("workflow.design_node_upsert", "workflow.design_edge_upsert"):
            ns.choices[nm].add_argument("--designer-id", required=True)
            ns.choices[nm].add_argument("--workflow-id", required=True)
        ns.choices["workflow.design_node_upsert"].add_argument("--label", required=True)
        ns.choices["workflow.design_node_upsert"].add_argument("--node-id", default=None)
        ns.choices["workflow.design_node_upsert"].add_argument("--op", default=None)
        ns.choices["workflow.design_node_upsert"].add_argument("--start", action=argparse.BooleanOptionalAction, default=False)
        ns.choices["workflow.design_node_upsert"].add_argument("--terminal", action=argparse.BooleanOptionalAction, default=False)
        ns.choices["workflow.design_node_upsert"].add_argument("--fanout", action=argparse.BooleanOptionalAction, default=False)
        ns.choices["workflow.design_node_upsert"].add_argument("--metadata", default="{}")
        ns.choices["workflow.design_edge_upsert"].add_argument("--src", required=True)
        ns.choices["workflow.design_edge_upsert"].add_argument("--dst", required=True)
        ns.choices["workflow.design_edge_upsert"].add_argument("--edge-id", default=None)
        ns.choices["workflow.design_edge_upsert"].add_argument("--relation", default="wf_next")
        ns.choices["workflow.design_edge_upsert"].add_argument("--predicate", default=None)
        ns.choices["workflow.design_edge_upsert"].add_argument("--priority", type=int, default=100)
        ns.choices["workflow.design_edge_upsert"].add_argument("--is-default", action=argparse.BooleanOptionalAction, default=False)
        ns.choices["workflow.design_edge_upsert"].add_argument("--multiplicity", default="one")
        ns.choices["workflow.design_edge_upsert"].add_argument("--metadata", default="{}")
        ns.choices["workflow.design_node_delete"].add_argument("--node-id", required=True)
        ns.choices["workflow.design_node_delete"].add_argument("--designer-id", required=True)
        ns.choices["workflow.design_edge_delete"].add_argument("--edge-id", required=True)
        ns.choices["workflow.design_edge_delete"].add_argument("--designer-id", required=True)
        ns.choices["workflow.capabilities_snapshot"].add_argument("--subject", default=None)
        ns.choices["workflow.capability_approve"].add_argument("--action", required=True)
        ns.choices["workflow.capability_approve"].add_argument("--capabilities", required=True)
        ns.choices["workflow.capability_approve"].add_argument("--subject", default=None)
        ns.choices["workflow.capability_revoke"].add_argument("--capability", required=True)
        ns.choices["workflow.capability_revoke"].add_argument("--subject", default=None)

        heartbeat = ns.add_parser("workflow.service_heartbeat")
        _add_base_url(heartbeat)
        heartbeat.add_argument("--service-id", required=True)
        heartbeat.add_argument("--instance-id", required=True)
        heartbeat.add_argument("--payload", default="{}")

        trigger = ns.add_parser("workflow.service_trigger")
        _add_base_url(trigger)
        trigger.add_argument("--service-id", required=True)
        trigger.add_argument("--trigger-type", required=True)
        trigger.add_argument("--payload", default="{}")

        declare = ns.add_parser("workflow.service_declare")
        _add_base_url(declare)
        declare.add_argument("--service-id", required=True)
        declare.add_argument("--service-kind", required=True)
        declare.add_argument("--target-kind", required=True)
        declare.add_argument("--target-ref", required=True)
        declare.add_argument("--target-config", default="{}")
        declare.add_argument("--enabled", action=argparse.BooleanOptionalAction, default=True)
        declare.add_argument("--autostart", action=argparse.BooleanOptionalAction, default=False)
        declare.add_argument("--restart-policy", default="{}")
        declare.add_argument("--heartbeat-ttl-ms", type=int, default=60000)
        declare.add_argument("--trigger-specs", default="[]")

        resume = ns.add_parser("workflow.run_resume")
        _add_base_url(resume)
        resume.add_argument("--run-id", required=True)
        resume.add_argument("--suspended-node-id", required=True)
        resume.add_argument("--suspended-token-id", required=True)
        resume.add_argument("--client-result", default="{}")
        resume.add_argument("--workflow-id", required=True)
        resume.add_argument("--conversation-id", required=True)
        resume.add_argument("--turn-node-id", required=True)
        resume.add_argument("--user-id", default=None)

        replay = ns.add_parser("workflow.run_replay")
        _add_base_url(replay)
        replay.add_argument("--run-id", required=True)
        replay.add_argument("--target-step-seq", type=int, required=True)

        cap_snap = ns.add_parser("workflow.capabilities_snapshot")
        _add_base_url(cap_snap)
        cap_snap.add_argument("--subject", default=None)

    conversation = api_sub.add_parser("conversation", help="Conversation HTTP wrappers")
    conversation_sub = conversation.add_subparsers(dest="conversation_command", required=True)
    _conversation(conversation_sub)

    workflow = api_sub.add_parser("workflow", help="Workflow HTTP wrappers")
    workflow_sub = workflow.add_subparsers(dest="workflow_command", required=True)
    _workflow(workflow_sub)

    demo = sub.add_parser("demo", help="Run named demos.")
    demo_sub = demo.add_subparsers(dest="demo_command", required=True)
    provenance = demo_sub.add_parser(
        "provenance", help="Run the provenance-first signature demo."
    )
    _add_demo_args(provenance)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        serve_main()
        return 0

    if args.command == "api":
        base = _base_url(args)
        cmd = args.api_command
        if cmd == "conversation":
            c = args.conversation_command
            if c == "conversation.create":
                _print_json(
                    _request_json(
                        method="POST",
                        url=f"{base}/api/conversations",
                        json_body={
                            "user_id": args.user_id,
                            "conversation_id": args.conversation_id,
                            "start_node_id": args.start_node_id,
                        },
                    )
                )
                return 0
            if c == "conversation.get_transcript":
                _print_json(
                    _request_json(method="GET", url=f"{base}/api/conversations/{args.conversation_id}/turns")
                )
                return 0
            if c == "conversation.ask":
                _print_json(
                    _request_json(
                        method="POST",
                        url=f"{base}/api/conversations/{args.conversation_id}/turns:answer",
                        json_body={
                            "text": args.text,
                            "user_id": args.user_id,
                            "workflow_id": args.workflow_id,
                        },
                    )
                )
                return 0
            if c == "conversation.run_status":
                _print_json(_request_json(method="GET", url=f"{base}/api/runs/{args.run_id}"))
                return 0
            if c == "conversation.cancel_run":
                _print_json(_request_json(method="POST", url=f"{base}/api/runs/{args.run_id}/cancel"))
                return 0
        if cmd == "workflow":
            w = args.workflow_command
            if w == "workflow.run_submit":
                _print_json(
                    _request_json(
                        method="POST",
                        url=f"{base}/api/workflow/runs",
                        json_body={
                            "workflow_id": args.workflow_id,
                            "conversation_id": args.conversation_id,
                            "turn_node_id": args.turn_node_id,
                            "user_id": args.user_id,
                            "initial_state": json.loads(args.initial_state),
                            "priority_class": args.priority_class,
                            "token_budget": args.token_budget,
                            "time_budget_ms": args.time_budget_ms,
                        },
                    )
                )
                return 0
            if w == "workflow.run_status":
                _print_json(_request_json(method="GET", url=f"{base}/api/workflow/runs/{args.run_id}"))
                return 0
            if w == "workflow.run_cancel":
                _print_json(_request_json(method="POST", url=f"{base}/api/workflow/runs/{args.run_id}/cancel"))
                return 0
            if w == "workflow.run_events":
                _print_json(_request_json(method="GET", url=f"{base}/api/workflow/runs/{args.run_id}/events", params={"after_seq": args.after_seq}))
                return 0
            if w == "workflow.process_table":
                _print_json(_request_json(method="GET", url=f"{base}/api/workflow/operator/dashboard", params={"limit": args.limit}))
                return 0
            if w == "workflow.operator_dashboard":
                _print_json(_request_json(method="GET", url=f"{base}/api/workflow/operator/dashboard", params={"limit": args.limit}))
                return 0
            if w == "workflow.service_list":
                _print_json(_request_json(method="GET", url=f"{base}/api/workflow/services", params={"limit": args.limit}))
                return 0
            if w == "workflow.service_get":
                _print_json(_request_json(method="GET", url=f"{base}/api/workflow/services/{args.service_id}"))
                return 0
            if w == "workflow.service_enable":
                _print_json(_request_json(method="POST", url=f"{base}/api/workflow/services/{args.service_id}/enable"))
                return 0
            if w == "workflow.service_disable":
                _print_json(_request_json(method="POST", url=f"{base}/api/workflow/services/{args.service_id}/disable"))
                return 0
            if w == "workflow.service_heartbeat":
                _print_json(_request_json(method="POST", url=f"{base}/api/workflow/services/{args.service_id}/heartbeat", json_body={"instance_id": args.instance_id, "payload": json.loads(args.payload)}))
                return 0
            if w == "workflow.service_trigger":
                _print_json(_request_json(method="POST", url=f"{base}/api/workflow/services/{args.service_id}/trigger", json_body={"trigger_type": args.trigger_type, "payload": json.loads(args.payload)}))
                return 0
            if w == "workflow.service_events":
                _print_json(_request_json(method="GET", url=f"{base}/api/workflow/services/{args.service_id}/events", params={"limit": args.limit}))
                return 0
            if w == "workflow.services_repair":
                _print_json(_request_json(method="POST", url=f"{base}/api/workflow/services/repair", params={"limit": args.limit}))
                return 0
            if w == "workflow.message_orphans_repair":
                _print_json(_request_json(method="POST", url=f"{base}/api/workflow/messages/repair-orphans", params={"inbox_id": args.inbox_id, "limit": args.limit}))
                return 0
            if w == "workflow.service_declare":
                _print_json(_request_json(method="POST", url=f"{base}/api/workflow/services", json_body={
                    "service_id": args.service_id,
                    "service_kind": args.service_kind,
                    "target_kind": args.target_kind,
                    "target_ref": args.target_ref,
                    "target_config": json.loads(args.target_config),
                    "enabled": bool(args.enabled),
                    "autostart": bool(args.autostart),
                    "restart_policy": json.loads(args.restart_policy),
                    "heartbeat_ttl_ms": args.heartbeat_ttl_ms,
                    "trigger_specs": json.loads(args.trigger_specs),
                }))
                return 0
            if w == "workflow.dead_letters":
                _print_json(_request_json(method="GET", url=f"{base}/api/workflow/dead-letters", params={"limit": args.limit}))
                return 0
            if w == "workflow.dead_letter_replay":
                _print_json(_request_json(method="POST", url=f"{base}/api/workflow/dead-letters/{args.run_id}/replay"))
                return 0
            if w == "workflow.capabilities_snapshot":
                _print_json(_request_json(method="GET", url=f"{base}/api/workflow/capabilities", params={"subject": args.subject}))
                return 0
            if w == "workflow.capability_approve":
                _print_json(_request_json(method="POST", url=f"{base}/api/workflow/capabilities/approve", json_body={"action": args.action, "capabilities": json.loads(args.capabilities), "subject": args.subject}))
                return 0
            if w == "workflow.capability_revoke":
                _print_json(_request_json(method="POST", url=f"{base}/api/workflow/capabilities/revoke", json_body={"capability": args.capability, "subject": args.subject}))
                return 0
            if w == "workflow.run_checkpoint_get":
                _print_json(_request_json(method="GET", url=f"{base}/api/workflow/runs/{args.run_id}/checkpoints/{args.step_seq}"))
                return 0
            if w == "workflow.run_replay":
                _print_json(_request_json(method="GET", url=f"{base}/api/workflow/runs/{args.run_id}/replay", params={"target_step_seq": args.target_step_seq}))
                return 0
            if w == "workflow.run_resume_contract":
                _print_json(_request_json(method="GET", url=f"{base}/api/workflow/runs/{args.run_id}/resume-contract"))
                return 0
            if w == "workflow.run_resume":
                _print_json(_request_json(method="POST", url=f"{base}/api/workflow/runs/{args.run_id}/resume", json_body={
                    "suspended_node_id": args.suspended_node_id,
                    "suspended_token_id": args.suspended_token_id,
                    "client_result": json.loads(args.client_result),
                    "workflow_id": args.workflow_id,
                    "conversation_id": args.conversation_id,
                    "turn_node_id": args.turn_node_id,
                    "user_id": args.user_id,
                }))
                return 0
            if w == "workflow.design_history":
                _print_json(_request_json(method="GET", url=f"{base}/api/workflow/design/{args.workflow_id}/history"))
                return 0
            if w == "workflow.design_undo":
                _print_json(_request_json(method="POST", url=f"{base}/api/workflow/design/{args.workflow_id}/undo", json_body={"designer_id": args.designer_id}))
                return 0
            if w == "workflow.design_redo":
                _print_json(_request_json(method="POST", url=f"{base}/api/workflow/design/{args.workflow_id}/redo", json_body={"designer_id": args.designer_id}))
                return 0
            if w == "workflow.design_node_upsert":
                _print_json(_request_json(method="POST", url=f"{base}/api/workflow/design/{args.workflow_id}/nodes", json_body={
                    "designer_id": args.designer_id,
                    "node_id": args.node_id,
                    "label": args.label,
                    "op": args.op,
                    "start": bool(args.start),
                    "terminal": bool(args.terminal),
                    "fanout": bool(args.fanout),
                    "metadata": json.loads(args.metadata),
                }))
                return 0
            if w == "workflow.design_edge_upsert":
                _print_json(_request_json(method="POST", url=f"{base}/api/workflow/design/{args.workflow_id}/edges", json_body={
                    "designer_id": args.designer_id,
                    "edge_id": args.edge_id,
                    "src": args.src,
                    "dst": args.dst,
                    "relation": args.relation,
                    "predicate": args.predicate,
                    "priority": args.priority,
                    "is_default": bool(args.is_default),
                    "multiplicity": args.multiplicity,
                    "metadata": json.loads(args.metadata),
                }))
                return 0
            if w == "workflow.design_node_delete":
                _print_json(_request_json(method="DELETE", url=f"{base}/api/workflow/design/{args.workflow_id}/nodes/{args.node_id}", json_body={"designer_id": args.designer_id}))
                return 0
            if w == "workflow.design_edge_delete":
                _print_json(_request_json(method="DELETE", url=f"{base}/api/workflow/design/{args.workflow_id}/edges/{args.edge_id}", json_body={"designer_id": args.designer_id}))
                return 0
        parser.error(f"Unknown api command: {args.api_command} {getattr(args, 'conversation_command', '') or getattr(args, 'workflow_command', '')}")
        return 2

    if args.command == "quickstart" or (
        args.command == "demo" and args.demo_command == "provenance"
    ):
        summary = run_provenance_quickstart(
            data_dir=Path(args.data_dir),
            question=str(args.question),
            open_browser=bool(args.open_browser),
        )
        if args.json:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            _print_quickstart_summary(summary)
        return 0

    parser.error("Unknown command")
    return 2
