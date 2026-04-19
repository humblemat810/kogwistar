from __future__ import annotations

import json

from kogwistar.server.auth_middleware import claims_ctx

from ._service_demo_support import build_demo_service


def run_operator_views_demo() -> dict:
    with build_demo_service() as service:
        token = claims_ctx.set(
            {
                "ns": "workflow",
                "role": "rw",
                "storage_ns": "default",
                "execution_ns": "default",
                "security_scope": "tenant/demo",
                "capabilities": [
                    "service.manage",
                    "service.inspect",
                    "service.heartbeat",
                    "project_view",
                    "spawn_process",
                    "workflow.run.read",
                    "workflow.run.write",
                    "workflow.design.write",
                    "workflow.design.inspect",
                    "read_graph",
                    "write_graph",
                ],
                "sub": "ops-demo",
            }
        )
        try:
            conv = service.create_conversation(user_id="ops-user")
            wf = "wf.operator.demo"
            service.workflow_design_upsert_node(
                workflow_id=wf,
                designer_id="ops-demo",
                node_id="start",
                label="Start",
                op="start",
                start=True,
            )
            service.workflow_design_upsert_node(
                workflow_id=wf,
                designer_id="ops-demo",
                node_id="end",
                label="End",
                op="end",
                terminal=True,
            )
            service.workflow_design_upsert_edge(
                workflow_id=wf,
                designer_id="ops-demo",
                edge_id="edge-start-end",
                src="start",
                dst="end",
                relation="wf_next",
                is_default=True,
            )
            service.declare_service(
                service_id="svc.ops.demo",
                service_kind="daemon",
                target_kind="workflow",
                target_ref=wf,
                target_config={"conversation_id": conv["conversation_id"]},
                enabled=True,
                autostart=True,
            )
            service.record_service_heartbeat(
                "svc.ops.demo", instance_id="inst-ops", payload={"beat": 1}
            )
            return {
                "process_table": service.list_process_table(limit=20),
                "operator_inbox": service.list_operator_inbox(limit=20),
                "blocked_runs": service.list_blocked_runs(limit=20),
                "dashboard": service.operator_dashboard(limit=20),
            }
        finally:
            claims_ctx.reset(token)


def main() -> None:
    print(json.dumps(run_operator_views_demo(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
