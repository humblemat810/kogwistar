# %% [markdown]
# # 26 Recovery and Durable Operational State
# This companion walks through bounded startup recovery:
# - lane-message projection rows are rebuildable from authoritative truth
# - service health latest-state rows are rebuildable from sparse lifecycle facts
# - inspect is read-only, while recover_startup performs bounded repair

# %%
from __future__ import annotations

from _helpers import banner, reset_data_dir, show
from kogwistar.engine_core import GraphKnowledgeEngine
from kogwistar.engine_core.engine import scoped_namespace
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend
from kogwistar.server.auth_middleware import claims_ctx


data_dir = reset_data_dir("26_recovery_and_durable_operational_state")
engine = GraphKnowledgeEngine(
    persist_directory=str(data_dir),
    backend_factory=build_in_memory_backend,
    kg_graph_type="conversation",
)
lane_ns = "ws:demo:conv:bg"
queue_ns = "ws:demo:maintenance_jobs"
service_ns = "ws:demo:ops"


# %% [markdown]
# ## Lane-message projection repair
# Create one durable lane message, then delete its serving row to simulate a damaged projection.

# %%
banner("Create a lane message and confirm projected visibility.")
token = claims_ctx.set({"storage_ns": lane_ns})
with scoped_namespace(engine, lane_ns):
    sent = engine.send_lane_message(
        conversation_id="conv-demo",
        inbox_id="inbox:worker:demo",
        sender_id="lane:foreground",
        recipient_id="lane:worker:demo",
        msg_type="request.demo",
        payload={"kind": "recovery-demo"},
    )
claims_ctx.reset(token)

rows_before = engine.meta_sqlite.list_projected_lane_messages(
    namespace=lane_ns,
    inbox_id="inbox:worker:demo",
)
show(
    "lane rows before damage",
    {
        "message_ids": [row.message_id for row in rows_before],
        "statuses": [row.status for row in rows_before],
        "data_dir": str(data_dir),
    },
)


# %%
banner("Delete the projected lane row, inspect, then recover.")
removed = engine.meta_sqlite.clear_projected_lane_messages(lane_ns)
inspected = engine.recovery.inspect(
    workspace_id="demo",
    namespaces=[queue_ns, lane_ns, service_ns],
)
recovered = engine.recovery.recover_startup(
    workspace_id="demo",
    namespaces=[queue_ns, lane_ns, service_ns],
)
rows_after = engine.meta_sqlite.list_projected_lane_messages(
    namespace=lane_ns,
    inbox_id="inbox:worker:demo",
)
show(
    "lane repair summary",
    {
        "removed_rows": removed,
        "inspect_repaired_count": inspected.repaired_count,
        "inspect_lane_rows": [row.message_id for row in inspected.lane_rows],
        "recover_repaired_count": recovered.repaired_count,
        "recover_lane_repairs": [
            {
                "namespace": item.namespace,
                "scanned_count": item.scanned_count,
                "repaired_count": item.repaired_count,
            }
            for item in recovered.repaired_lane_projections
        ],
        "lane_rows_after_recovery": [row.message_id for row in rows_after],
    },
)


# %% [markdown]
# ## Lease-based redelivery
# Recovery does not force completion. Leases make interrupted work claimable again after expiry.

# %%
banner("Create one job and one claimed lane row with expired leases.")
engine.jobs.enqueue(
    job_id="job-expired",
    namespace=queue_ns,
    entity_kind="maintenance_job",
    entity_id="entity-1",
    job_kind="maintenance_job",
)
claimed_job = engine.jobs.claim(namespace=queue_ns, limit=1, lease_seconds=60)[0]
engine.meta_sqlite._state.index_jobs[claimed_job.job_id].lease_until = 0
engine.meta_sqlite.claim_projected_lane_messages(
    namespace=lane_ns,
    inbox_id="inbox:worker:demo",
    claimed_by="worker-1",
    limit=1,
    lease_seconds=-1,
)
lease_report = engine.recovery.inspect(
    workspace_id="demo",
    namespaces=[queue_ns, lane_ns, service_ns],
)
show(
    "expired lease visibility",
    {
        "queue_states": [
            {
                "job_id": item.job_id,
                "status": item.status,
                "expired_lease": item.expired_lease,
            }
            for item in lease_report.queues
            if item.expired_lease or item.job_id == "job-expired"
        ],
        "lane_states": [
            {
                "message_id": item.message_id,
                "status": item.status,
                "expired_lease": item.expired_lease,
            }
            for item in lease_report.lane_rows
        ],
    },
)


# %% [markdown]
# ## Service health latest-state rebuild
# Sparse lifecycle facts live in graph truth; latest service health lives in a durable projection row.

# %%
banner("Declare service health, then delete and rebuild the latest-state row.")
engine.service_health.declare_service(
    service_id="svc.demo",
    service_kind="maintenance_daemon",
    owner_app="tutorial",
    deterministic=False,
    llm_assisted=True,
    workspace_id="demo",
    namespace=service_ns,
    operator_tags=["tutorial", "recovery"],
)
engine.service_health.start_instance(
    service_id="svc.demo",
    workspace_id="demo",
    namespace=service_ns,
    instance_id="inst-1",
    started_at_ms=123,
)
engine.service_health.heartbeat(
    service_id="svc.demo",
    workspace_id="demo",
    namespace=service_ns,
    instance_id="inst-1",
    status="degraded",
    last_error="example failure",
)

projection_key = "demo|ws:demo:ops|svc.demo"
before_service = engine.service_health.get_service(
    "svc.demo",
    workspace_id="demo",
    namespace=service_ns,
)
engine.meta_sqlite.clear_named_projection("service_health", projection_key)
inspect_missing = engine.recovery.inspect(
    workspace_id="demo",
    namespaces=[queue_ns, lane_ns, service_ns],
)
recover_service = engine.recovery.recover_startup(
    workspace_id="demo",
    namespaces=[queue_ns, lane_ns, service_ns],
)
after_service = engine.service_health.get_service(
    "svc.demo",
    workspace_id="demo",
    namespace=service_ns,
)
show(
    "service health rebuild",
    {
        "before_projection_clear": before_service,
        "inspect_daemon_health": [
            {
                "daemon_id": item.daemon_id,
                "observed_state": item.observed_state,
            }
            for item in inspect_missing.daemon_health
        ],
        "recover_daemon_health": [
            {
                "daemon_id": item.daemon_id,
                "observed_state": item.observed_state,
            }
            for item in recover_service.daemon_health
        ],
        "after_recovery": after_service,
    },
)


# %% [markdown]
# ## Invariant
# - inspect reports without mutating
# - recover_startup repairs missing latest-state views
# - leases still control redelivery
# - recovery is bounded repair plus visibility, not orchestration authority
