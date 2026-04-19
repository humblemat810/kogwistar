from __future__ import annotations

import json

from kogwistar.server.auth_middleware import claims_ctx

from ._service_demo_support import build_demo_service


def run_recovery_repair_demo() -> dict:
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
                "sub": "repair-demo",
            }
        )
        try:
            conv = service.create_conversation(user_id="repair-user")
            wf = "wf.repair.demo"
            service.workflow_design_upsert_node(
                workflow_id=wf,
                designer_id="repair-demo",
                node_id="start",
                label="Start",
                op="start",
                start=True,
            )
            service.workflow_design_upsert_node(
                workflow_id=wf,
                designer_id="repair-demo",
                node_id="end",
                label="End",
                op="end",
                terminal=True,
            )
            service.workflow_design_upsert_edge(
                workflow_id=wf,
                designer_id="repair-demo",
                edge_id="edge-start-end",
                src="start",
                dst="end",
                relation="wf_next",
                is_default=True,
            )
            service.declare_service(
                service_id="svc.repair.demo",
                service_kind="daemon",
                target_kind="workflow",
                target_ref=wf,
                target_config={"conversation_id": conv["conversation_id"]},
                enabled=True,
                autostart=True,
            )
            service.service_supervisor.bootstrap()
            repair = service.repair_service_projection("svc.repair.demo")
            dead = service.dead_letter_snapshot(limit=10)
            return {
                "repair": repair,
                "dead_letters": dead,
                "dashboard": service.operator_dashboard(limit=10),
            }
        finally:
            claims_ctx.reset(token)


def main() -> None:
    print(json.dumps(run_recovery_repair_demo(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
