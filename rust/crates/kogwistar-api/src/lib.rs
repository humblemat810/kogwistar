//! Transport-neutral Phase-5 API contracts.
//!
//! Network transports adapt these typed decisions. They do not own domain
//! policy, authentication role checks, SSE framing, or JSON-RPC envelopes.

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use jsonwebtoken::{
    Algorithm, DecodingKey, EncodingKey, Header, Validation, decode, decode_header, encode,
};
use kogwistar_runtime::{
    RECORDED_RUNTIME_CONTRACT_VERSION, RecordedRuntimeState, RecordedRuntimeTransition,
    RecordedTransitionKind, RecordedWorkerHandoff, RecordedWorkerSuccessEffect, RuntimeFrontier,
    RuntimeStepExecutePayload, RuntimeStepExecuteRequest,
};
use kogwistar_store::{
    AuthIdentityStore, AuthUser, EntityEvent, LaneMessageFilter, NamedProjection,
    NamedProjectionWrite, NewProjectedLaneMessage, ProjectedLaneMessage, ResolveExternalIdentity,
    ServerRun, ServerRunCreate, ServerRunEvent, WorkflowDesignDeltaWrite, WorkflowDesignSnapshot,
    WorkflowDesignSnapshotWrite,
};
use kogwistar_store_postgres::{
    NewRawEntityEvent as PostgresNewRawEntityEvent, PostgresAuthStore, PostgresStore,
};
use kogwistar_store_sqlite::{
    NewRawEntityEvent as SqliteNewRawEntityEvent, SqliteAuthStore, SqliteStore,
};

use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{OriginalUri, Query, State},
    http::{HeaderMap, HeaderValue, Method, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{MethodFilter, get, on, post},
};

include!(concat!(env!("OUT_DIR"), "/frozen_routes.rs"));
pub const FROZEN_OPENAPI_JSON: &str = include_str!(concat!(env!("OUT_DIR"), "/openapi.json"));
pub const FROZEN_MCP_TOOLS_JSON: &str = include_str!(concat!(env!("OUT_DIR"), "/mcp-tools.json"));
const CYTOSCAPE_TEMPLATE: &str = include_str!(concat!(env!("OUT_DIR"), "/cytoscape.html"));
const D3_TEMPLATE: &str = include_str!(concat!(env!("OUT_DIR"), "/d3.html"));
const GO_TEMPLATE: &str = include_str!(concat!(env!("OUT_DIR"), "/go.html"));

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HealthSnapshot {
    pub backend: String,
    pub persist_directory: String,
    pub conversation_persist_directory: String,
    pub workflow_persist_directory: String,
    pub wisdom_persist_directory: String,
    #[serde(default)]
    pub pg_schema_base: Option<String>,
}

pub fn health_response(snapshot: &HealthSnapshot) -> Value {
    json!({
        "ok": true,
        "backend": snapshot.backend,
        "persist_directory": snapshot.persist_directory,
        "conversation_persist_directory": snapshot.conversation_persist_directory,
        "workflow_persist_directory": snapshot.workflow_persist_directory,
        "wisdom_persist_directory": snapshot.wisdom_persist_directory,
        "pg_schema_base": snapshot.pg_schema_base,
    })
}

pub fn health_response_from_str(payload_json: &str) -> Result<String, serde_json::Error> {
    let snapshot: HealthSnapshot = serde_json::from_str(payload_json)?;
    serde_json::to_string(&health_response(&snapshot))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ApiAuthRequest {
    #[serde(default)]
    pub roles: Vec<String>,
    #[serde(default)]
    pub required_roles: Vec<String>,
}

pub fn authorize(request: &ApiAuthRequest) -> bool {
    request.required_roles.is_empty()
        || request
            .required_roles
            .iter()
            .any(|required| request.roles.iter().any(|role| role_allows(role, required)))
}

fn role_allows(role: &str, required: &str) -> bool {
    match (role, required) {
        ("rw", "ro" | "rw") | ("ro", "ro") => true,
        _ => role == required,
    }
}

pub fn authorize_from_str(payload_json: &str) -> Result<String, serde_json::Error> {
    let request: ApiAuthRequest = serde_json::from_str(payload_json)?;
    serde_json::to_string(&json!({"allowed": authorize(&request)}))
}

pub fn sse_frame(event: &str, data: &Value, id: Option<&str>) -> String {
    let mut frame = String::new();
    if let Some(id) = id {
        frame.push_str("id: ");
        frame.push_str(id);
        frame.push('\n');
    }
    frame.push_str("event: ");
    frame.push_str(event);
    frame.push('\n');
    frame.push_str("data: ");
    frame.push_str(&serde_json::to_string(data).expect("JSON value always serializes"));
    frame.push_str("\n\n");
    frame
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SseFrameRequest {
    pub event: String,
    pub data: Value,
    #[serde(default)]
    pub id: Option<String>,
}

pub fn sse_frame_from_str(payload_json: &str) -> Result<String, serde_json::Error> {
    let request: SseFrameRequest = serde_json::from_str(payload_json)?;
    Ok(sse_frame(
        &request.event,
        &request.data,
        request.id.as_deref(),
    ))
}

pub fn mcp_result(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct McpResultRequest {
    pub id: Value,
    pub result: Value,
}

pub fn mcp_result_from_str(payload_json: &str) -> Result<String, serde_json::Error> {
    let request: McpResultRequest = serde_json::from_str(payload_json)?;
    serde_json::to_string(&mcp_result(request.id, request.result))
}

pub fn cli_health(snapshot: &HealthSnapshot) -> String {
    format!("ok backend={}", snapshot.backend)
}

pub fn cli_health_from_str(payload_json: &str) -> Result<String, serde_json::Error> {
    let snapshot: HealthSnapshot = serde_json::from_str(payload_json)?;
    Ok(cli_health(&snapshot))
}

#[derive(Clone)]
pub struct ApiState {
    pub health: HealthSnapshot,
    pub required_roles: Vec<String>,
    pub implementation: ImplementationSnapshot,
    pub auth: AuthConfig,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AuthConfig {
    pub mode: String,
    pub algorithm: Option<String>,
    pub key: Option<String>,
    pub jwks_json: Option<String>,
    pub issuer: Option<String>,
    pub audience: Option<String>,
    pub oidc_providers_json: Option<String>,
    pub auth_db_url: Option<String>,
}

impl AuthConfig {
    pub fn from_environment() -> Self {
        Self {
            mode: std::env::var("AUTH_MODE").unwrap_or_else(|_| "oidc".to_owned()),
            algorithm: std::env::var("JWT_ALG")
                .ok()
                .or_else(|| Some("HS256".to_owned())),
            key: std::env::var("JWT_SECRET").ok(),
            jwks_json: std::env::var("JWT_JWKS_JSON").ok(),
            issuer: std::env::var("JWT_ISS").ok(),
            audience: std::env::var("JWT_AUD").ok(),
            oidc_providers_json: std::env::var("OIDC_PROVIDERS_JSON").ok(),
            auth_db_url: std::env::var("AUTH_DB_URL")
                .ok()
                .or_else(|| Some("sqlite:///auth.sqlite".to_owned())),
        }
    }

    fn configured(&self) -> bool {
        self.key.as_deref().is_some_and(|key| !key.is_empty())
            || self
                .jwks_json
                .as_deref()
                .is_some_and(|jwks| !jwks.is_empty())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ImplementationSnapshot {
    pub mode: String,
    pub contract_version: u32,
    pub schema_version: u32,
    pub frozen_route_operations: usize,
    pub implemented_route_operations: usize,
    pub runtime_cutover_ready: bool,
    pub server_cutover_ready: bool,
}

/// Frozen OpenAPI operations still owned by the Python rollback deployment.
/// Keep this list explicit so health/readiness cannot infer server completion
/// from transport registration alone.
pub const PENDING_SERVER_CUTOVER_ROUTES: &[(&str, &str)] = &[
    ("DELETE", "/admin/doc/{doc_id}"),
    ("GET", "/api/conversations"),
    ("GET", "/api/conversations/{conversation_id}"),
    (
        "GET",
        "/api/conversations/{conversation_id}/snapshots/latest",
    ),
    ("GET", "/api/conversations/{conversation_id}/turns"),
    ("GET", "/api/search_index_hybrid"),
    ("GET", "/api/viz/cytoscape.json"),
    ("GET", "/api/viz/d3.json"),
    ("GET", "/api/workflow/tools/audit"),
    ("GET", "/viz/d3.bundle"),
    ("POST", "/api/add_index_entries"),
    ("POST", "/api/conversations"),
    ("POST", "/api/conversations/{conversation_id}/turns:answer"),
    ("POST", "/api/document"),
    ("POST", "/api/document.upsert_tree"),
    ("POST", "/api/graph/upsert"),
];

/// Syscall sub-operations which still require conversation/tool authority.
/// OpenAPI freezes these behind one parameterized route, so keep sub-route
/// readiness explicit instead of classifying the whole dispatcher as absent.
pub const PENDING_SYSCALL_CUTOVER_OPS: &[&str] = &[
    "send_message",
    "receive_message",
    "mount_memory",
    "project_view",
    "invoke_tool",
];

fn environment(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_owned())
}

/// Run the packaged Rust server using the same environment contract as the
/// standalone binary.  PyO3 calls this function so native wheels need not ship
/// a second platform-specific executable beside the extension module.
pub async fn run_server_from_environment() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let host = environment("KOGWISTAR_SERVER_HOST", "127.0.0.1");
    let port = environment("KOGWISTAR_SERVER_PORT", "8000");
    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}")).await?;
    let required_roles = environment("KOGWISTAR_SERVER_REQUIRED_ROLES", "")
        .split(',')
        .map(str::trim)
        .filter(|role| !role.is_empty())
        .map(str::to_owned)
        .collect();
    let backend = environment("KOGWISTAR_BACKEND", "in_memory");
    let state = ApiState {
        health: HealthSnapshot {
            backend: backend.clone(),
            persist_directory: environment("KOGWISTAR_PERSIST_DIRECTORY", ".kogwistar"),
            conversation_persist_directory: environment(
                "KOGWISTAR_CONVERSATION_PERSIST_DIRECTORY",
                ".kogwistar/conversation",
            ),
            workflow_persist_directory: environment(
                "KOGWISTAR_WORKFLOW_PERSIST_DIRECTORY",
                ".kogwistar/workflow",
            ),
            wisdom_persist_directory: environment(
                "KOGWISTAR_WISDOM_PERSIST_DIRECTORY",
                ".kogwistar/wisdom",
            ),
            pg_schema_base: (backend == "pg")
                .then(|| environment("KOGWISTAR_PG_SCHEMA_BASE", "kogwistar")),
        },
        required_roles,
        implementation: ImplementationSnapshot::default(),
        auth: AuthConfig::from_environment(),
    };
    let application: Arc<dyn ApplicationService> = if backend == "pg" {
        match std::env::var("KOGWISTAR_PG_DSN") {
            Ok(dsn) if !dsn.trim().is_empty() => {
                let service = PostgresRunApplicationService::from_dsn(
                    &dsn,
                    &environment("KOGWISTAR_PG_SCHEMA_BASE", "kogwistar"),
                )
                .map_err(std::io::Error::other)?;
                service
                    .ensure_schema()
                    .await
                    .map_err(std::io::Error::other)?;
                Arc::new(service)
            }
            _ => Arc::new(UnavailableApplicationService),
        }
    } else {
        match std::env::var("KOGWISTAR_META_SQLITE_PATH") {
            Ok(path) if !path.trim().is_empty() => {
                Arc::new(SqliteRunApplicationService::open(path).map_err(std::io::Error::other)?)
            }
            _ => Arc::new(UnavailableApplicationService),
        }
    };
    axum::serve(listener, router_with_application(state, application))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

impl Default for ImplementationSnapshot {
    fn default() -> Self {
        Self {
            mode: "rust".to_owned(),
            contract_version: 1,
            schema_version: 1,
            frozen_route_operations: FROZEN_OPENAPI_ROUTES.len(),
            // health + submit + two aliases each for run GET/events/cancel,
            // resume, resume-contract, steps, checkpoint list/item, and replay;
            // plus poll on the conversation API. Static transport-only
            // /api/events and /mcp are not frozen OpenAPI operations.
            implemented_route_operations: FROZEN_OPENAPI_ROUTES
                .len()
                .saturating_sub(PENDING_SERVER_CUTOVER_ROUTES.len()),
            runtime_cutover_ready: false,
            server_cutover_ready: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ApiEffectRequest {
    pub contract_version: u32,
    pub method: String,
    pub path_and_query: String,
    pub body: Vec<u8>,
    pub principal: Value,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ApiEffectResponse {
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
}

pub trait ApplicationService: Send + Sync + 'static {
    fn execute(
        &self,
        request: ApiEffectRequest,
    ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>>;
}

#[derive(Default)]
pub struct UnavailableApplicationService;

impl ApplicationService for UnavailableApplicationService {
    fn execute(
        &self,
        request: ApiEffectRequest,
    ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
        Box::pin(async move { unavailable_effect(request) })
    }
}

fn unavailable_effect(request: ApiEffectRequest) -> ApiEffectResponse {
    ApiEffectResponse {
        status: StatusCode::NOT_IMPLEMENTED.as_u16(),
        content_type: "application/json".to_owned(),
        body: serde_json::to_vec(&json!({
            "code": "KOGWISTAR_API_CAPABILITY_NOT_CUT_OVER",
            "message": "Rust application capability is not cut over",
            "method": request.method,
            "path": request.path_and_query,
            "contract_version": request.contract_version,
        }))
        .unwrap(),
    }
}

fn decoded_json(raw: Option<String>) -> Value {
    raw.as_deref()
        .filter(|value| !value.is_empty())
        .and_then(|value| serde_json::from_str(value).ok())
        .unwrap_or(Value::Null)
}

fn server_run_value(run: ServerRun) -> Value {
    let terminal = matches!(run.status.as_str(), "succeeded" | "failed" | "cancelled");
    json!({
        "run_id": run.run_id,
        "conversation_id": run.conversation_id,
        "workflow_id": run.workflow_id,
        "user_id": run.user_id,
        "user_turn_node_id": run.user_turn_node_id,
        "assistant_turn_node_id": run.assistant_turn_node_id,
        "status": run.status,
        "cancel_requested": run.cancel_requested,
        "result": decoded_json(run.result_json),
        "error": decoded_json(run.error_json),
        "created_at_ms": run.created_at_ms,
        "updated_at_ms": run.updated_at_ms,
        "started_at_ms": run.started_at_ms,
        "finished_at_ms": run.finished_at_ms,
        "terminal": terminal,
    })
}

fn server_run_event_value(event: ServerRunEvent) -> Value {
    json!({
        "seq": event.seq,
        "run_id": event.run_id,
        "event_type": event.event_type,
        "payload": serde_json::from_str::<Value>(&event.payload_json).unwrap_or_else(|_| json!({})),
        "created_at_ms": event.created_at_ms,
    })
}

#[derive(Clone, Debug)]
struct RecordedInspectionSnapshot {
    event_seq: i64,
    created_at_ms: i64,
    transition_id: String,
    step_seq: i64,
    workflow_id: String,
    workflow_node_id: String,
    state: Value,
    result: Value,
    errors: Vec<Value>,
    server_status: String,
}

fn recorded_inspection_snapshots(events: Vec<ServerRunEvent>) -> Vec<RecordedInspectionSnapshot> {
    events
        .into_iter()
        .filter(|event| event.event_type == "workflow.recorded_transition.v1")
        .filter_map(|event| {
            let payload: Value = serde_json::from_str(&event.payload_json).ok()?;
            let reduced = payload.get("reduced")?;
            let state = reduced.get("state")?;
            Some(RecordedInspectionSnapshot {
                event_seq: event.seq,
                created_at_ms: event.created_at_ms,
                transition_id: payload["transition_id"].as_str()?.to_owned(),
                step_seq: state["last_step_seq"].as_i64()?,
                workflow_id: state["workflow_id"].as_str()?.to_owned(),
                workflow_node_id: state["last_node_id"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                state: state.get("state").cloned().unwrap_or_else(|| json!({})),
                result: reduced.get("result").cloned().unwrap_or(Value::Null),
                errors: reduced["errors"].as_array().cloned().unwrap_or_default(),
                server_status: reduced["server_status"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
            })
        })
        .collect()
}

fn recorded_step_snapshots(events: Vec<ServerRunEvent>) -> Vec<RecordedInspectionSnapshot> {
    let mut steps = Vec::<RecordedInspectionSnapshot>::new();
    for snapshot in recorded_inspection_snapshots(events) {
        if snapshot.step_seq < 0 {
            continue;
        }
        if steps.last().is_some_and(|previous| {
            previous.step_seq == snapshot.step_seq
                && previous.workflow_node_id == snapshot.workflow_node_id
        }) {
            steps.pop();
            steps.push(snapshot);
            continue;
        }
        steps.push(snapshot);
    }
    steps
}

fn runtime_inspection_effect(
    request: &ApiEffectRequest,
    run_id: &str,
    tail: &[&str],
    events: Vec<ServerRunEvent>,
) -> ApiEffectResponse {
    let steps = recorded_step_snapshots(events);
    match tail {
        ["steps"] => json_effect(
            StatusCode::OK,
            json!({
                "run_id": run_id,
                "steps": steps.into_iter().map(|step| json!({
                    "node_id": format!("wf_step|{}|{}", run_id, step.step_seq),
                    "step_seq": step.step_seq,
                    "workflow_id": step.workflow_id,
                    "workflow_node_id": step.workflow_node_id,
                    "op": step.workflow_node_id,
                    "status": if step.errors.is_empty() { "success" } else { "failure" },
                    "duration_ms": 0,
                    "result": step.result,
                    "event_seq": step.event_seq,
                    "created_at_ms": step.created_at_ms,
                    "transition_id": step.transition_id,
                })).collect::<Vec<_>>(),
            }),
        ),
        ["checkpoints"] => json_effect(
            StatusCode::OK,
            json!({
                "run_id": run_id,
                "checkpoints": steps.into_iter().map(|step| json!({
                    "node_id": format!("wf_ckpt|{}|{}", run_id, step.step_seq),
                    "step_seq": step.step_seq,
                    "workflow_id": step.workflow_id,
                    "state": step.state,
                    "event_seq": step.event_seq,
                    "created_at_ms": step.created_at_ms,
                    "status": step.server_status,
                })).collect::<Vec<_>>(),
            }),
        ),
        ["checkpoints", raw_step_seq] => {
            let Ok(step_seq) = raw_step_seq.parse::<i64>() else {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    "step_seq must be an integer",
                );
            };
            match steps.into_iter().find(|step| step.step_seq == step_seq) {
                Some(step) => json_effect(
                    StatusCode::OK,
                    json!({"run_id": run_id, "step_seq": step_seq, "state": step.state}),
                ),
                None => effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_CHECKPOINT_NOT_FOUND",
                    format!("No checkpoint for {run_id} at step {step_seq}"),
                ),
            }
        }
        ["replay"] => {
            let target_step_seq =
                query_integer(&request.path_and_query, "target_step_seq", i64::MIN);
            if target_step_seq == i64::MIN {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    "target_step_seq is required and must be an integer",
                );
            }
            match steps
                .into_iter()
                .find(|step| step.step_seq == target_step_seq)
            {
                Some(step) => json_effect(
                    StatusCode::OK,
                    json!({
                        "run_id": run_id,
                        "target_step_seq": target_step_seq,
                        "state": step.state,
                    }),
                ),
                None => effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_CHECKPOINT_NOT_FOUND",
                    format!("No checkpoint for {run_id} at step {target_step_seq}"),
                ),
            }
        }
        _ => unreachable!("runtime inspection called with unsupported route"),
    }
}

fn is_runtime_inspection_tail(tail: &[&str]) -> bool {
    matches!(
        tail,
        ["steps"] | ["checkpoints"] | ["checkpoints", _] | ["replay"]
    )
}

fn resume_contract_value(state: RecordedRuntimeState) -> Value {
    let node_ops = state
        .state
        .get("_rt_node_ops")
        .cloned()
        .unwrap_or_else(|| json!({}));
    json!({
        "run_id": state.run_id,
        "status": state.status,
        "wait_reason": state.wait_reason,
        "resume_payload": state.resume_payload,
        "suspended": state.frontier.suspended,
        "last_step_seq": state.last_step_seq,
        "runtime_routes": state.static_routes,
        "node_ops": node_ops,
    })
}

fn active_runtime_lane_covers_token(lanes: &[ProjectedLaneMessage], token_id: &str) -> bool {
    lanes.iter().any(|lane| {
        matches!(lane.status.as_str(), "pending" | "claimed")
            && lane
                .payload_json
                .as_deref()
                .and_then(|payload| serde_json::from_str::<Value>(payload).ok())
                .and_then(|payload| payload["token_id"].as_str().map(str::to_owned))
                .is_some_and(|candidate| candidate == token_id)
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RuntimeGraphPlan {
    start_node_id: String,
    join_node_ids: Vec<String>,
    start_join_mask: i64,
    routes: Vec<kogwistar_runtime::RuntimeStaticRoute>,
    node_ops: std::collections::BTreeMap<String, String>,
}

fn submitted_runtime_graph_conflicts(
    graph_plan: &RuntimeGraphPlan,
    input: &RuntimeSubmitRun,
) -> bool {
    let Some(start_node_id) = input
        .start_node_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return false;
    };
    if input.node_ops.is_empty() {
        return false;
    }
    let canonical_routes = |routes: &[kogwistar_runtime::RuntimeStaticRoute]| {
        let mut values = routes
            .iter()
            .cloned()
            .map(|mut route| {
                route.aliases.sort();
                route.aliases.dedup();
                serde_json::to_string(&route).expect("runtime route serializes")
            })
            .collect::<Vec<_>>();
        values.sort();
        values
    };
    graph_plan.start_node_id != start_node_id
        || graph_plan.join_node_ids != input.join_node_ids
        || graph_plan.start_join_mask != input.start_join_mask
        || canonical_routes(&graph_plan.routes) != canonical_routes(&input.runtime_routes)
        || graph_plan.node_ops != input.node_ops
}

fn runtime_join_outstanding(join_count: usize, start_join_mask: i64) -> Vec<i64> {
    (0..join_count)
        .map(|index| i64::from(start_join_mask & (1_i64 << index) != 0))
        .collect()
}

fn default_runtime_priority_class() -> String {
    "foreground".to_owned()
}

fn default_runtime_kind() -> String {
    "sync".to_owned()
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RuntimeSubmitRun {
    #[serde(default)]
    run_id: Option<String>,
    workflow_id: String,
    conversation_id: String,
    #[serde(default)]
    turn_node_id: Option<String>,
    #[serde(default)]
    user_id: Option<String>,
    #[serde(default)]
    initial_state: serde_json::Map<String, Value>,
    #[serde(default = "default_runtime_priority_class")]
    priority_class: String,
    #[serde(default)]
    token_budget: Option<i64>,
    #[serde(default)]
    time_budget_ms: Option<i64>,
    #[serde(default = "default_runtime_kind")]
    runtime_kind: String,
    #[serde(default)]
    join_node_ids: Vec<String>,
    #[serde(default)]
    start_join_mask: i64,
    #[serde(default)]
    runtime_routes: Vec<kogwistar_runtime::RuntimeStaticRoute>,
    #[serde(default)]
    start_node_id: Option<String>,
    #[serde(default)]
    node_ops: BTreeMap<String, String>,
}

fn default_runtime_claim_limit() -> usize {
    1
}

fn default_runtime_claim_lease() -> i64 {
    60
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RuntimeClaimRequest {
    claimed_by: String,
    #[serde(default = "default_runtime_claim_limit")]
    limit: usize,
    #[serde(default = "default_runtime_claim_lease")]
    lease_seconds: i64,
    #[serde(default)]
    run_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RuntimeWorkerResultRequest {
    handoff: RecordedWorkerHandoff,
    #[serde(default)]
    transition: Option<RecordedRuntimeTransition>,
    #[serde(default)]
    effect: Option<RecordedWorkerSuccessEffect>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RuntimeResumeRequest {
    suspended_node_id: String,
    suspended_token_id: String,
    #[serde(default)]
    client_result: Value,
    workflow_id: String,
    conversation_id: String,
    #[serde(default)]
    turn_node_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RuntimeClaimedWork {
    message_id: String,
    claimed_by: String,
    run_id: String,
    step_id: Option<String>,
    correlation_id: Option<String>,
    payload: Value,
    expected_event_seq: i64,
    lease_until: Option<Value>,
}

fn runtime_claimed_work(
    lane: ProjectedLaneMessage,
    claimed_by: &str,
    run_id: String,
    payload: Value,
    expected_event_seq: i64,
) -> RuntimeClaimedWork {
    RuntimeClaimedWork {
        message_id: lane.message_id,
        claimed_by: claimed_by.to_owned(),
        run_id,
        step_id: lane.step_id,
        correlation_id: lane.correlation_id,
        payload,
        expected_event_seq,
        lease_until: lane.lease_until,
    }
}

fn runtime_start_transition(payload: &Value, run_id: &str) -> RecordedRuntimeTransition {
    let start_node_id = payload["start_node_id"].as_str().unwrap_or("start");
    let join_node_ids = string_list(&payload["join_node_ids"]);
    let start_join_mask = payload["start_join_mask"].as_i64().unwrap_or_default();
    RecordedRuntimeTransition {
        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
        transition_id: format!("start-{run_id}"),
        expected_event_seq: payload["expected_event_seq"].as_i64().unwrap_or_default(),
        kind: RecordedTransitionKind::Start,
        run_id: run_id.to_owned(),
        workflow_id: payload["workflow_id"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        conversation_id: payload["conversation_id"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        user_id: payload["user_id"].as_str().map(str::to_owned),
        user_turn_node_id: payload["turn_node_id"].as_str().map(str::to_owned),
        step_seq: 0,
        node_id: Some(start_node_id.to_owned()),
        token_id: Some(run_id.to_owned()),
        parent_token_id: None,
        initial_state: Some(
            payload["initial_state"]
                .as_object()
                .cloned()
                .unwrap_or_default(),
        ),
        state_update: Vec::new(),
        update: None,
        state_schema: serde_json::Map::new(),
        frontier: Some(RuntimeFrontier {
            pending: vec![(
                start_node_id.to_owned(),
                start_join_mask,
                run_id.to_owned(),
                None,
            )],
            join_outstanding: runtime_join_outstanding(join_node_ids.len(), start_join_mask),
            join_waiters: join_node_ids
                .iter()
                .cloned()
                .map(|node_id| (node_id, Vec::new()))
                .collect(),
            join_node_ids,
            ..RuntimeFrontier::default()
        }),
        result: None,
        wait_reason: None,
        resume_payload: None,
        errors: Vec::new(),
    }
}

fn runtime_submit_retry_response(
    existing: &ServerRun,
    lanes: &[ProjectedLaneMessage],
    expected_worker_payload: &Value,
) -> Result<Option<Value>, String> {
    let Some(lane) = lanes.iter().find(|lane| {
        lane.run_id.as_deref() == Some(existing.run_id.as_str())
            && lane.msg_type == "workflow.run.execute"
    }) else {
        return Ok(None);
    };
    let mut persisted: Value =
        serde_json::from_str(lane.payload_json.as_deref().unwrap_or("{}"))
            .map_err(|error| format!("existing runtime admission payload is invalid: {error}"))?;
    if let Some(object) = persisted.as_object_mut() {
        object.remove("expected_event_seq");
        object.remove("runtime_started");
    }
    if &persisted != expected_worker_payload {
        return Err(format!(
            "run_id {:?} already belongs to a different runtime admission",
            existing.run_id
        ));
    }
    Ok(Some(json!({
        "run_id": existing.run_id,
        "conversation_id": existing.conversation_id,
        "workflow_id": existing.workflow_id,
        "turn_node_id": existing.user_turn_node_id,
        "status": existing.status,
        "priority_class": expected_worker_payload["priority_class"],
        "token_budget": expected_worker_payload["token_budget"],
        "time_budget_ms": expected_worker_payload["time_budget_ms"],
        "admission": "accepted",
        "idempotent": true,
        "lane_message_id": lane.message_id,
    })))
}

fn string_list(value: &Value) -> Vec<String> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

fn runtime_graph_plan_from_snapshot(payload_json: &str) -> Option<RuntimeGraphPlan> {
    let payload: Value = serde_json::from_str(payload_json).ok()?;
    let nodes = payload["nodes"].as_array()?;
    let edges = payload["edges"].as_array()?;
    let node_ids = nodes
        .iter()
        .filter_map(|node| node["id"].as_str().map(str::to_owned))
        .collect::<Vec<_>>();
    let start_node_id = nodes
        .iter()
        .find(|node| node["metadata"]["wf_start"].as_bool() == Some(true))?["id"]
        .as_str()?
        .to_owned();
    let node_ops = nodes
        .iter()
        .filter_map(|node| {
            let node_id = node["id"].as_str()?;
            let op = node["metadata"]["wf_op"]
                .as_str()
                .or_else(|| node["metadata"]["op"].as_str())
                .unwrap_or("noop");
            Some((node_id.to_owned(), op.to_owned()))
        })
        .collect();
    let node_aliases = nodes
        .iter()
        .filter_map(|node| {
            let node_id = node["id"].as_str()?;
            let mut aliases = vec![
                node_id.to_owned(),
                node_id.split('|').next_back().unwrap_or(node_id).to_owned(),
            ];
            for alias in [node["label"].as_str(), node["metadata"]["wf_op"].as_str()] {
                if let Some(alias) = alias.filter(|value| !value.is_empty()) {
                    aliases.push(alias.to_owned());
                }
            }
            aliases.sort();
            aliases.dedup();
            Some((node_id.to_owned(), aliases))
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut join_node_ids = nodes
        .iter()
        .filter(|node| node["metadata"]["wf_join"].as_bool() == Some(true))
        .filter_map(|node| node["id"].as_str().map(str::to_owned))
        .collect::<Vec<_>>();
    join_node_ids.sort();
    if join_node_ids.len() >= i64::BITS as usize {
        return None;
    }
    let fanout_nodes = nodes
        .iter()
        .filter(|node| node["metadata"]["wf_fanout"].as_bool() == Some(true))
        .filter_map(|node| node["id"].as_str().map(str::to_owned))
        .collect::<std::collections::BTreeSet<_>>();
    let mut topology_edges = Vec::new();
    let mut route_metadata = Vec::new();
    for edge in edges {
        let edge_id = edge["id"].as_str().unwrap_or_default().to_owned();
        let edge_label = edge["label"]
            .as_str()
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        let predicate = edge["metadata"]["wf_predicate"]
            .as_str()
            .filter(|predicate| !predicate.is_empty())
            .map(str::to_owned);
        let multiplicity = edge["metadata"]["wf_multiplicity"]
            .as_str()
            .unwrap_or("one")
            .to_owned();
        let is_default = edge["metadata"]["wf_is_default"].as_bool().unwrap_or(false);
        let priority = edge["metadata"]["wf_priority"].as_i64().unwrap_or(100);
        for source in string_list(&edge["source_ids"]) {
            for target in string_list(&edge["target_ids"]) {
                topology_edges.push([source.clone(), target.clone()]);
                let mut aliases = node_aliases.get(&target).cloned().unwrap_or_else(|| {
                    vec![
                        target.clone(),
                        target.split('|').next_back().unwrap_or(&target).to_owned(),
                    ]
                });
                if let Some(label) = edge_label.as_ref()
                    && !aliases.contains(label)
                {
                    aliases.push(label.clone());
                }
                aliases.sort();
                aliases.dedup();
                route_metadata.push((
                    edge_id.clone(),
                    source.clone(),
                    target.clone(),
                    aliases,
                    predicate.clone(),
                    multiplicity.clone(),
                    is_default,
                    priority,
                    fanout_nodes.contains(&source),
                ));
            }
        }
    }
    let may_reach_json = kogwistar_contracts::workflow_may_reach_join_from_str(
        &serde_json::to_string(&json!({
            "node_ids": node_ids,
            "edges": topology_edges,
            "join_ids": join_node_ids,
        }))
        .ok()?,
    )
    .ok()?;
    let may_reach: Value = serde_json::from_str(&may_reach_json).ok()?;
    let mask = |node_id: &str| -> i64 {
        may_reach[node_id]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_u64)
            .filter(|bit| *bit < i64::BITS as u64)
            .fold(0_i64, |value, bit| value | (1_i64 << bit))
    };
    let routes = route_metadata
        .into_iter()
        .map(
            |(
                edge_id,
                source_node_id,
                target_node_id,
                target_aliases,
                predicate,
                multiplicity,
                is_default,
                priority,
                source_fanout,
            )| {
                kogwistar_runtime::RuntimeStaticRoute {
                    edge_id,
                    source_node_id,
                    join_mask: mask(&target_node_id),
                    target_node_id,
                    aliases: target_aliases,
                    predicate,
                    multiplicity,
                    is_default,
                    priority,
                    source_fanout,
                }
            },
        )
        .collect();
    Some(RuntimeGraphPlan {
        start_join_mask: mask(&start_node_id),
        start_node_id,
        join_node_ids,
        routes,
        node_ops,
    })
}

fn exact_runtime_graph_plan(
    projection: Option<NamedProjection>,
    snapshot: Option<WorkflowDesignSnapshot>,
) -> Option<RuntimeGraphPlan> {
    let projection = projection?;
    let snapshot = snapshot?;
    if projection.materialization_status != "ready"
        || projection.payload["current_version"].as_i64()? != snapshot.version
        || projection.payload["snapshot_schema_version"]
            .as_i64()
            .is_some_and(|version| version != snapshot.schema_version)
    {
        return None;
    }
    runtime_graph_plan_from_snapshot(&snapshot.payload_json)
}

fn json_effect(status: StatusCode, value: Value) -> ApiEffectResponse {
    ApiEffectResponse {
        status: status.as_u16(),
        content_type: "application/json".to_owned(),
        body: serde_json::to_vec(&value).expect("JSON value always serializes"),
    }
}

fn html_effect(body: String) -> ApiEffectResponse {
    ApiEffectResponse {
        status: StatusCode::OK.as_u16(),
        content_type: "text/html; charset=utf-8".to_owned(),
        body: body.into_bytes(),
    }
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn visualization_shell_effect(request: &ApiEffectRequest) -> Option<ApiEffectResponse> {
    if request.method != "GET" {
        return None;
    }
    let path = request.path_and_query.split('?').next()?;
    let template = match path {
        "/viz/cytoscape" => CYTOSCAPE_TEMPLATE,
        "/viz/d3" => D3_TEMPLATE,
        "/viz/go" => GO_TEMPLATE,
        _ => return None,
    };
    let doc_id = query_string(&request.path_and_query, "doc_id").unwrap_or_default();
    let mode = query_string(&request.path_and_query, "mode").unwrap_or_else(|| "reify".to_owned());
    let insertion_method = query_string(&request.path_and_query, "insertion_method");
    if path == "/viz/go" {
        return Some(html_effect(template.to_owned()));
    }
    let doc_html = html_escape(&doc_id);
    let mode_html = html_escape(&mode);
    let insertion_html = html_escape(insertion_method.as_deref().unwrap_or_default());
    let insertion_json = serde_json::to_string(&insertion_method).expect("query value serializes");
    let mut body = template
        .replace("{{ doc_id or '' }}", &doc_html)
        .replace("{{ mode }}", &mode_html)
        .replace("{{ insertion_method or '' }}", &insertion_html)
        .replace("{{ (insertion_method or none) | tojson }}", &insertion_json)
        .replace(
            "{{ insertion_method | tojson if insertion_method is not none else 'null' }}",
            &insertion_json,
        )
        .replace(
            "{% if is_bundle %}&nbsp;|&nbsp;<b>bundle</b>=<code>true</code>{% endif %}",
            "",
        )
        .replace(
            "{{ embedded_data | safe if embedded_data is defined else \"null\" }}",
            "null",
        )
        .replace(
            "{{ bundle_meta | safe if bundle_meta is defined else \"null\" }}",
            "null",
        )
        .replace(
            "{{ bundle_graph_type | safe if bundle_graph_type is defined else \"null\" }}",
            "null",
        )
        .replace(
            "{{ cdc_enabled | safe if cdc_enabled is defined else \"false\" }}",
            "false",
        )
        .replace(
            "{{ cdc_ws_url | safe if cdc_ws_url is defined else \"null\" }}",
            "null",
        );
    // Fail closed if template evolution introduces an unhandled executable
    // expression; never serve raw Jinja into JavaScript.
    if body.contains("{{") || body.contains("{%") {
        return Some(effect_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "KOGWISTAR_TEMPLATE_DRIFT",
            format!("Unhandled visualization template expression in {path}"),
        ));
    }
    body.shrink_to_fit();
    Some(html_effect(body))
}

fn effect_error(
    status: StatusCode,
    code: &'static str,
    message: impl Into<String>,
) -> ApiEffectResponse {
    json_effect(status, json!({"code": code, "message": message.into()}))
}

fn query_integer(path_and_query: &str, name: &str, default: i64) -> i64 {
    path_and_query
        .split_once('?')
        .map(|(_, query)| query)
        .unwrap_or_default()
        .split('&')
        .find_map(|pair| {
            let (key, value) = pair.split_once('=')?;
            (key == name).then(|| value.parse::<i64>().ok()).flatten()
        })
        .unwrap_or(default)
}

fn query_string(path_and_query: &str, name: &str) -> Option<String> {
    path_and_query
        .split_once('?')
        .map(|(_, query)| query)
        .unwrap_or_default()
        .split('&')
        .find_map(|pair| {
            let (key, value) = pair.split_once('=')?;
            (key == name && !value.is_empty()).then(|| value.to_owned())
        })
}

const CAPABILITY_SPECS: &[(&str, &str, &str)] = &[
    ("read_graph", "Read graph state", "read"),
    ("write_graph", "Write graph state", "write"),
    ("send_message", "Send conversation message", "message"),
    ("spawn_process", "Spawn workflow process", "process"),
    ("invoke_tool", "Invoke tool", "tool"),
    ("read_security_scope", "Read security scope", "read"),
    ("project_view", "Project view", "read"),
    ("approve_action", "Approve blocked action", "approve"),
    ("workflow.design.inspect", "Inspect workflow design", "read"),
    ("workflow.design.write", "Mutate workflow design", "write"),
    ("workflow.run.read", "Read workflow run", "read"),
    (
        "workflow.run.write",
        "Create or mutate workflow run",
        "write",
    ),
    ("service.inspect", "Inspect service state", "read"),
    ("service.manage", "Manage service lifecycle", "write"),
    ("service.heartbeat", "Record service heartbeat", "write"),
];

#[derive(Default)]
struct CapabilityState {
    approvals: BTreeMap<(String, String), BTreeSet<String>>,
    revoked: BTreeMap<String, BTreeSet<String>>,
    audit_log: Vec<Value>,
    syscall_audit: Vec<Value>,
}

type SharedCapabilityState = Arc<Mutex<CapabilityState>>;

fn principal_strings(principal: &Value, name: &str) -> Vec<String> {
    match principal.get(name) {
        Some(Value::String(raw)) => raw
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_ascii_lowercase)
            .collect(),
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_ascii_lowercase)
            .collect(),
        _ => Vec::new(),
    }
}

fn principal_subject(principal: &Value) -> String {
    ["sub", "user_id", "agent_id"]
        .into_iter()
        .find_map(|name| principal.get(name).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("anonymous")
        .to_ascii_lowercase()
}

fn principal_role(principal: &Value) -> String {
    principal
        .get("role")
        .and_then(Value::as_str)
        .map(str::to_ascii_lowercase)
        .filter(|role| matches!(role.as_str(), "ro" | "rw"))
        .unwrap_or_else(|| "ro".to_owned())
}

fn base_capabilities(principal: &Value) -> BTreeSet<String> {
    let explicit = principal
        .get("capabilities")
        .or_else(|| principal.get("caps"));
    if explicit.is_some() {
        return principal_strings(
            principal,
            if principal.get("capabilities").is_some() {
                "capabilities"
            } else {
                "caps"
            },
        )
        .into_iter()
        .filter(|capability| capability != "*")
        .collect();
    }
    if principal_role(principal) == "rw" {
        CAPABILITY_SPECS
            .iter()
            .map(|(name, _, _)| (*name).to_owned())
            .collect()
    } else {
        [
            "read_graph",
            "read_security_scope",
            "project_view",
            "workflow.design.inspect",
            "workflow.run.read",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }
}

fn effective_capabilities(principal: &Value, state: &CapabilityState) -> BTreeSet<String> {
    let subject = principal_subject(principal);
    let mut effective = base_capabilities(principal);
    for ((row_subject, _), capabilities) in &state.approvals {
        if row_subject == &subject {
            effective.extend(capabilities.iter().cloned());
        }
    }
    if let Some(revoked) = state.revoked.get(&subject) {
        effective.retain(|capability| !revoked.contains(capability));
    }
    effective
}

fn capability_snapshot(principal: &Value, state: &CapabilityState) -> Value {
    json!({
        "specs": CAPABILITY_SPECS.iter().map(|(name, description, action_kind)| json!({
            "name": name,
            "description": description,
            "action_kind": action_kind,
            "parent": Value::Null,
        })).collect::<Vec<_>>(),
        "approvals": state.approvals.iter().map(|((subject, action), capabilities)| json!({
            "subject": subject,
            "action": action,
            "capabilities": capabilities,
        })).collect::<Vec<_>>(),
        "revoked": state.revoked.iter().map(|(subject, capabilities)| json!({
            "subject": subject,
            "capabilities": capabilities,
        })).collect::<Vec<_>>(),
        "audit_log": state.audit_log,
        "current_subject": principal_subject(principal),
        "effective_capabilities": effective_capabilities(principal, state),
    })
}

fn audit_capability(
    principal: &Value,
    state: &mut CapabilityState,
    action: &str,
    required: &[&str],
) -> bool {
    let effective = effective_capabilities(principal, state);
    let allowed = required.iter().all(|value| effective.contains(*value));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_millis() as u64)
        .unwrap_or_default();
    let missing = required
        .iter()
        .filter(|value| !effective.contains(**value))
        .copied()
        .collect::<Vec<_>>();
    state.audit_log.push(json!({
        "ts_ms": now,
        "subject": principal_subject(principal),
        "action": action,
        "required": required,
        "granted": effective,
        "outcome": if allowed { "allow" } else { "deny" },
        "reason": if allowed { String::new() } else { format!("missing={}", missing.join(",")) },
        "parent_capabilities": base_capabilities(principal),
    }));
    allowed
}

fn security_scope_value(principal: &Value) -> Value {
    let mut namespaces = principal_strings(principal, "ns");
    if namespaces.is_empty() {
        namespaces.push("docs".to_owned());
    }
    namespaces.sort();
    namespaces.dedup();
    let default_namespace = namespaces
        .first()
        .cloned()
        .unwrap_or_else(|| "default".to_owned());
    let storage_namespace = principal["storage_ns"]
        .as_str()
        .filter(|value| !value.is_empty())
        .unwrap_or(&default_namespace)
        .to_ascii_lowercase();
    let execution_namespace = principal["execution_ns"]
        .as_str()
        .filter(|value| !value.is_empty())
        .unwrap_or(&default_namespace)
        .to_ascii_lowercase();
    let tenant = principal["tenant"]
        .as_str()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let workspace = principal["workspace"]
        .as_str()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let project = principal["project"]
        .as_str()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let scope_path = [tenant.as_str(), workspace.as_str(), project.as_str()]
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join("/");
    let security_scope = ["security_scope", "tenant", "scope"]
        .into_iter()
        .find_map(|name| {
            principal[name]
                .as_str()
                .filter(|value| !value.is_empty() && *value != "*")
        })
        .map(str::to_ascii_lowercase)
        .or_else(|| (!scope_path.is_empty()).then(|| scope_path.clone()))
        .unwrap_or_else(|| default_namespace.clone());
    json!({
        "storage_namespace": storage_namespace,
        "execution_namespace": execution_namespace,
        "security_scope": security_scope,
        "security_scope_path": scope_path,
        "tenant": tenant,
        "workspace": workspace,
        "project": project,
    })
}

fn capability_effect(
    request: &ApiEffectRequest,
    shared: &SharedCapabilityState,
) -> Option<ApiEffectResponse> {
    let path = request
        .path_and_query
        .split('?')
        .next()
        .unwrap_or(&request.path_and_query);
    if !matches!(
        path,
        "/api/workflow/visibility"
            | "/api/workflow/capabilities"
            | "/api/workflow/capabilities/approve"
            | "/api/workflow/capabilities/revoke"
    ) {
        return None;
    }
    let mut state = shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (action, required): (&str, &[&str]) = match path {
        "/api/workflow/visibility" => (
            "read_security_scope",
            &["read_security_scope", "project_view"],
        ),
        "/api/workflow/capabilities" => ("project_view", &["project_view"]),
        _ => ("approve_action", &["approve_action"]),
    };
    if !audit_capability(&request.principal, &mut state, action, required) {
        return Some(effect_error(
            StatusCode::FORBIDDEN,
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
            format!("Forbidden: action '{action}' requires capability {required:?}"),
        ));
    }
    match (request.method.as_str(), path) {
        ("GET", "/api/workflow/visibility") => {
            let mapping = security_scope_value(&request.principal);
            Some(json_effect(
                StatusCode::OK,
                json!({
                    "current_subject": principal_subject(&request.principal),
                    "current_role": principal_role(&request.principal),
                    "current_capabilities": effective_capabilities(&request.principal, &state),
                    "namespaces": {
                        "storage_namespace": mapping["storage_namespace"],
                        "execution_namespace": mapping["execution_namespace"],
                    },
                    "security_scope": mapping["security_scope"],
                    "storage_security_mapping": mapping,
                    "can_access_public": true,
                }),
            ))
        }
        ("GET", "/api/workflow/capabilities") => Some(json_effect(
            StatusCode::OK,
            capability_snapshot(&request.principal, &state),
        )),
        ("POST", "/api/workflow/capabilities/approve") => {
            let input: Value = match serde_json::from_slice(&request.body) {
                Ok(value) => value,
                Err(error) => {
                    return Some(effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_INVALID_REQUEST",
                        error.to_string(),
                    ));
                }
            };
            let Some(action) = input["action"]
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                return Some(effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    "action is required",
                ));
            };
            let subject = input["subject"]
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_ascii_lowercase)
                .unwrap_or_else(|| principal_subject(&request.principal));
            let capabilities = match &input["capabilities"] {
                Value::String(value) => vec![value.to_ascii_lowercase()],
                Value::Array(values) => values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_ascii_lowercase)
                    .collect(),
                _ => Vec::new(),
            };
            if capabilities.is_empty() {
                return Some(effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    "capabilities are required",
                ));
            }
            state
                .approvals
                .entry((subject, action.to_ascii_lowercase()))
                .or_default()
                .extend(capabilities);
            Some(json_effect(
                StatusCode::ACCEPTED,
                capability_snapshot(&request.principal, &state),
            ))
        }
        ("POST", "/api/workflow/capabilities/revoke") => {
            let input: Value = match serde_json::from_slice(&request.body) {
                Ok(value) => value,
                Err(error) => {
                    return Some(effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_INVALID_REQUEST",
                        error.to_string(),
                    ));
                }
            };
            let Some(capability) = input["capability"]
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                return Some(effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    "capability is required",
                ));
            };
            let subject = input["subject"]
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_ascii_lowercase)
                .unwrap_or_else(|| principal_subject(&request.principal));
            state
                .revoked
                .entry(subject)
                .or_default()
                .insert(capability.to_ascii_lowercase());
            Some(json_effect(
                StatusCode::ACCEPTED,
                capability_snapshot(&request.principal, &state),
            ))
        }
        _ => Some(unavailable_effect(request.clone())),
    }
}

const SYSCALL_OPS: &[&str] = &[
    "spawn_process",
    "terminate_process",
    "send_message",
    "receive_message",
    "mount_memory",
    "project_view",
    "invoke_tool",
    "checkpoint",
    "resume",
    "request_approval",
];

#[derive(Clone, Debug, Deserialize)]
struct SyscallInput {
    #[serde(default = "default_syscall_version")]
    version: String,
    #[serde(default)]
    op: String,
    #[serde(default)]
    args: serde_json::Map<String, Value>,
}

fn default_syscall_version() -> String {
    "v1".to_owned()
}

fn syscall_operation(path_and_query: &str) -> Option<String> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    path.strip_prefix("/api/syscall/v1/")
        .filter(|value| !value.is_empty() && !value.contains('/'))
        .map(|value| value.trim().to_ascii_lowercase())
}

fn syscall_input(
    request: &ApiEffectRequest,
    path_op: &str,
) -> Result<SyscallInput, ApiEffectResponse> {
    let mut input: SyscallInput = serde_json::from_slice(&request.body).map_err(|error| {
        effect_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "KOGWISTAR_INVALID_REQUEST",
            error.to_string(),
        )
    })?;
    input.version = input.version.trim().to_owned();
    if input.version.is_empty() {
        input.version = default_syscall_version();
    }
    input.op = path_op.to_owned();
    Ok(input)
}

fn syscall_result(version: &str, op: &str, response: ApiEffectResponse) -> ApiEffectResponse {
    if !(200..300).contains(&response.status) {
        return response;
    }
    let result = serde_json::from_slice::<Value>(&response.body).unwrap_or_else(|_| json!({}));
    json_effect(
        StatusCode::OK,
        json!({"version": version, "op": op, "status": "ok", "result": result}),
    )
}

fn syscall_audit(shared: &SharedCapabilityState, version: &str, op: &str, status: &str) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_millis() as u64)
        .unwrap_or_default();
    shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .syscall_audit
        .push(json!({"ts_ms": now, "version": version, "op": op, "status": status}));
}

fn syscall_audit_status(response: &ApiEffectResponse) -> &'static str {
    if response.status >= 400 {
        return "error";
    }
    match serde_json::from_slice::<Value>(&response.body)
        .ok()
        .and_then(|value| value["status"].as_str().map(str::to_owned))
        .as_deref()
    {
        Some("blocked") => "blocked",
        _ => "ok",
    }
}

fn syscall_allowed(
    request: &ApiEffectRequest,
    shared: &SharedCapabilityState,
    op: &str,
    write: bool,
    required: &[&str],
) -> bool {
    let role = principal_role(&request.principal);
    let role_allowed = !write || role == "rw";
    let namespace_allowed = principal_strings(&request.principal, "ns")
        .into_iter()
        .any(|namespace| namespace == "workflow" || namespace == "*");
    role_allowed && namespace_allowed && capability_allowed(request, shared, op, required)
}

fn syscall_forbidden(op: &str) -> ApiEffectResponse {
    effect_error(
        StatusCode::FORBIDDEN,
        "KOGWISTAR_CAPABILITY_FORBIDDEN",
        format!("Syscall {op} requires workflow namespace, role, and capability"),
    )
}

#[derive(Clone, Debug, Deserialize)]
struct DocumentGraphProposalInput {
    doc_id: String,
    #[serde(default = "default_document_insertion_method")]
    insertion_method: String,
    nodes: Vec<serde_json::Map<String, Value>>,
    #[serde(default)]
    edges: Vec<serde_json::Map<String, Value>>,
}

fn default_document_insertion_method() -> String {
    "document_parser_v1".to_owned()
}

fn graph_validation_error(message: impl Into<String>) -> String {
    json!([{"type":"value_error","msg":message.into()}]).to_string()
}

fn validate_graph_span(value: &Value) -> Result<(), String> {
    let Some(span) = value.as_object() else {
        return Err("mention span must be an object".to_owned());
    };
    for name in [
        "collection_page_url",
        "document_page_url",
        "doc_id",
        "insertion_method",
    ] {
        if !span.get(name).is_some_and(Value::is_string) {
            return Err(format!("mention span {name} must be a string"));
        }
    }
    let start = span
        .get("start_char")
        .and_then(Value::as_i64)
        .ok_or_else(|| "mention span start_char must be an integer".to_owned())?;
    let end = span
        .get("end_char")
        .and_then(Value::as_i64)
        .ok_or_else(|| "mention span end_char must be an integer".to_owned())?;
    if start < 0 {
        return Err("mention span start_char must be >= 0".to_owned());
    }
    if end != -1 && end <= start {
        return Err("mention span end_char must be > start_char".to_owned());
    }
    Ok(())
}

fn validate_graph_mentions(value: Option<&Value>) -> Result<(), String> {
    let mentions = value
        .and_then(Value::as_array)
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "at least one mention is required".to_owned())?;
    for mention in mentions {
        if mention.get("spans").is_none()
            && (mention.get("document_page_url").is_some() || mention.get("doc_id").is_some())
        {
            validate_graph_span(mention)?;
            continue;
        }
        let spans = mention
            .get("spans")
            .and_then(Value::as_array)
            .filter(|values| !values.is_empty())
            .ok_or_else(|| "grounding spans must be a non-empty array".to_owned())?;
        for span in spans {
            validate_graph_span(span)?;
        }
    }
    Ok(())
}

fn lift_graph_pointer_mentions(
    entity: &mut serde_json::Map<String, Value>,
    doc_id: &str,
    insertion_method: &str,
) -> Result<(), String> {
    if entity.contains_key("mentions") {
        return Ok(());
    }
    let pointers = entity
        .get("metadata")
        .and_then(|metadata| metadata.get("pointers"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if pointers.is_empty() {
        return Ok(());
    }
    let mut mentions = Vec::with_capacity(pointers.len());
    for pointer in pointers {
        let source_cluster_id = pointer["source_cluster_id"]
            .as_str()
            .ok_or_else(|| "pointer source_cluster_id must be a string".to_owned())?;
        let start_char = pointer["start_char"]
            .as_i64()
            .ok_or_else(|| "pointer start_char must be an integer".to_owned())?;
        let end_char = pointer["end_char"]
            .as_i64()
            .ok_or_else(|| "pointer end_char must be an integer".to_owned())?;
        mentions.push(json!({
            "doc_id": doc_id,
            "collection_page_url": format!("doc://{doc_id}"),
            "document_page_url": format!("doc://{doc_id}#{source_cluster_id}"),
            "insertion_method": insertion_method,
            "start_char": start_char,
            "end_char": if end_char == -1 { 1_000_000_000_i64 } else { end_char },
            "excerpt": pointer["verbatim_text"].as_str().unwrap_or_default().chars().take(400).collect::<String>(),
        }));
    }
    entity.insert("mentions".to_owned(), Value::Array(mentions));
    Ok(())
}

fn validate_graph_entity(
    entity: &mut serde_json::Map<String, Value>,
    doc_id: &str,
    insertion_method: &str,
    edge: bool,
) -> Result<(), String> {
    lift_graph_pointer_mentions(entity, doc_id, insertion_method)?;
    for name in ["label", "summary"] {
        if !entity.get(name).is_some_and(Value::is_string) {
            return Err(format!("{name} must be a string"));
        }
    }
    if !entity
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|value| matches!(value, "entity" | "relationship" | "reference_pointer"))
    {
        return Err("type must be entity, relationship, or reference_pointer".to_owned());
    }
    validate_graph_mentions(entity.get("mentions"))?;
    if edge {
        if !entity.get("relation").is_some_and(Value::is_string) {
            return Err("relation must be a string".to_owned());
        }
        for name in ["source_ids", "target_ids"] {
            if !entity
                .get(name)
                .and_then(Value::as_array)
                .is_some_and(|values| values.iter().all(Value::is_string))
            {
                return Err(format!("{name} must be an array of strings"));
            }
        }
        for name in ["source_edge_ids", "target_edge_ids"] {
            let valid = entity.get(name).is_some_and(|value| {
                value.is_null()
                    || value
                        .as_array()
                        .is_some_and(|values| values.iter().all(Value::is_string))
            });
            if !valid {
                return Err(format!("{name} must be null or an array of strings"));
            }
        }
    }
    Ok(())
}

fn document_graph_validation_effect(request: &ApiEffectRequest) -> Option<ApiEffectResponse> {
    if request.method != "POST"
        || request.path_and_query.split('?').next() != Some("/api/document.validate_graph")
    {
        return None;
    }
    let mut input: DocumentGraphProposalInput = match serde_json::from_slice(&request.body) {
        Ok(value) => value,
        Err(error) => {
            return Some(effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error.to_string(),
            ));
        }
    };
    let mut node_errors = serde_json::Map::new();
    for node in &mut input.nodes {
        let key = node
            .get("id")
            .or_else(|| node.get("label"))
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        if let Err(error) =
            validate_graph_entity(node, &input.doc_id, &input.insertion_method, false)
        {
            node_errors.insert(key, json!(graph_validation_error(error)));
        }
    }
    let mut edge_errors = serde_json::Map::new();
    for edge in &mut input.edges {
        let key = edge
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("unknown-edge")
            .to_owned();
        if let Err(error) =
            validate_graph_entity(edge, &input.doc_id, &input.insertion_method, true)
        {
            edge_errors.insert(key, json!(graph_validation_error(error)));
        }
    }
    Some(json_effect(
        StatusCode::OK,
        json!({
            "ok": node_errors.is_empty() && edge_errors.is_empty(),
            "node_errors": node_errors,
            "edge_errors": edge_errors,
        }),
    ))
}

fn request_approval_syscall(
    request: &ApiEffectRequest,
    shared: &SharedCapabilityState,
    input: &SyscallInput,
) -> ApiEffectResponse {
    let action = input.args["action"]
        .as_str()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if action.is_empty() {
        return effect_error(
            StatusCode::BAD_REQUEST,
            "KOGWISTAR_INVALID_REQUEST",
            "action is required",
        );
    }
    if action == "deny" {
        return json_effect(
            StatusCode::OK,
            json!({
                "version": input.version,
                "op": input.op,
                "status": "blocked",
                "result": {},
                "error": {"reason": input.args["reason"].as_str().unwrap_or("request denied")},
            }),
        );
    }
    if !matches!(action.as_str(), "grant" | "revoke") {
        return json_effect(
            StatusCode::OK,
            json!({
                "version": input.version,
                "op": input.op,
                "status": "ok",
                "result": {"status":"requested", "reason": input.args["reason"].as_str().unwrap_or("approval pending")},
            }),
        );
    }
    let capability = input.args["capability"]
        .as_str()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if capability.is_empty() {
        return effect_error(
            StatusCode::BAD_REQUEST,
            "KOGWISTAR_INVALID_REQUEST",
            format!("capability is required for {action}"),
        );
    }
    let subject = input.args["subject"]
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| principal_subject(&request.principal));
    let mut state = shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let payload = if action == "grant" {
        let approval_action = input.args["approval_action"]
            .as_str()
            .unwrap_or("request_approval")
            .to_ascii_lowercase();
        state
            .approvals
            .entry((subject.clone(), approval_action.clone()))
            .or_default()
            .insert(capability.clone());
        json!({
            "status":"approved",
            "approval": {"subject":subject,"action":approval_action,"capabilities":[capability]},
        })
    } else {
        state
            .revoked
            .entry(subject.clone())
            .or_default()
            .insert(capability.clone());
        json!({
            "status":"revoked",
            "revocation": {"subject":subject,"capability":capability},
        })
    };
    json_effect(
        StatusCode::OK,
        json!({"version":input.version,"op":input.op,"status":"ok","result":payload}),
    )
}

fn syscall_read_effect(
    request: &ApiEffectRequest,
    shared: &SharedCapabilityState,
) -> Option<ApiEffectResponse> {
    let path = request
        .path_and_query
        .split('?')
        .next()
        .unwrap_or(&request.path_and_query);
    match (request.method.as_str(), path) {
        ("GET", "/api/syscall/v1") => {
            let mut state = shared
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|value| value.as_millis() as u64)
                .unwrap_or_default();
            state.syscall_audit.push(json!({
                "ts_ms": now,
                "version": "v1",
                "op": "list_syscalls",
                "status": "ok",
            }));
            Some(json_effect(
                StatusCode::OK,
                json!({"version": "v1", "ops": SYSCALL_OPS}),
            ))
        }
        ("GET", "/api/syscall/v1/audit") => {
            let limit =
                query_integer(&request.path_and_query, "limit", 200).clamp(0, 10_000) as usize;
            let state = shared
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let keep_from = state.syscall_audit.len().saturating_sub(limit);
            Some(json_effect(
                StatusCode::OK,
                json!({"version": "v1", "events": &state.syscall_audit[keep_from..]}),
            ))
        }
        _ => None,
    }
}

fn designer_capabilities_effect(
    request: &ApiEffectRequest,
    shared: &SharedCapabilityState,
) -> Option<ApiEffectResponse> {
    if request.method != "GET"
        || request.path_and_query.split('?').next() != Some("/designer/capabilities")
    {
        return None;
    }
    let explicit = if request.principal.get("capabilities").is_some() {
        principal_strings(&request.principal, "capabilities")
    } else {
        principal_strings(&request.principal, "caps")
    };
    let namespaces = principal_strings(&request.principal, "ns");
    if !namespaces
        .iter()
        .any(|namespace| namespace == "workflow" || namespace == "*")
        || !explicit
            .iter()
            .any(|capability| capability == "workflow.design.inspect")
        || !capability_allowed(
            request,
            shared,
            "workflow.design.inspect",
            &["workflow.design.inspect"],
        )
    {
        return Some(effect_error(
            StatusCode::FORBIDDEN,
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
            "Inspecting workflow design requires workflow.design.inspect capability",
        ));
    }
    Some(json_effect(
        StatusCode::OK,
        json!({
            "schema_version": "workflow-designer-capabilities/v1",
            "projection_schema": "workflow_design_v1",
            "design_features": {
                "undo_redo": true,
                "delta_history": true,
                "snapshot_restore": true,
                "dry_run_validation": false,
            },
            "custom_ops": {
                "allow_unregistered_ops_in_design": true,
                "allow_execution_of_unregistered_ops": false,
                "binding_statuses": ["resolved", "unresolved", "sandboxed", "plugin"],
            },
            "node_types": [{
                "type": "workflow_node",
                "display_name": "Workflow Node",
                "metadata_schema": {
                    "type": "object",
                    "properties": {
                        "wf_op": {"type": ["string", "null"]},
                        "wf_start": {"type": "boolean"},
                        "wf_terminal": {"type": "boolean"},
                        "wf_fanout": {"type": "boolean"},
                        "wf_join": {"type": "boolean"},
                    },
                },
                "flags": {
                    "supports_start": true,
                    "supports_terminal": true,
                    "supports_fanout": true,
                    "supports_join": true,
                },
            }],
            "edge_types": [{
                "type": "workflow_edge",
                "display_name": "Workflow Edge",
                "metadata_schema": {
                    "type": "object",
                    "properties": {
                        "wf_predicate": {"type": ["string", "null"]},
                        "wf_priority": {"type": "integer"},
                        "wf_is_default": {"type": "boolean"},
                        "wf_multiplicity": {"enum": ["one", "many"]},
                    },
                },
                "flags": {
                    "supports_predicate": true,
                    "supports_priority": true,
                    "supports_default": true,
                    "supports_multiplicity": true,
                },
            }],
            "runtime": {
                "resolver_found": true,
                "builtin_ops": ["llm_call", "start"],
                "nested_ops": [],
                "sandboxed_ops": [],
                "sandbox": {
                    "supports_sandboxed_ops": false,
                    "runtime_configured": false,
                },
                "state_schema": {},
            },
        }),
    ))
}

fn operational_path(path_and_query: &str) -> Option<&str> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    matches!(
        path,
        "/api/workflow/operator/dashboard"
            | "/api/workflow/lane/progress"
            | "/api/workflow/resources"
            | "/api/workflow/budget"
            | "/api/workflow/budget/history"
            | "/api/workflow/scheduler/timeline"
            | "/api/workflow/dead-letters"
            | "/api/workflow/catalog/ops"
    )
    .then_some(path)
}

fn lane_progress_value(lane: &ProjectedLaneMessage) -> Value {
    json!({
        "event_type": format!("worker.{}", lane.status),
        "message_id": lane.message_id,
        "conversation_id": lane.conversation_id,
        "inbox_id": lane.inbox_id,
        "recipient_id": lane.recipient_id,
        "sender_id": lane.sender_id,
        "status": lane.status,
        "msg_type": lane.msg_type,
        "seq": lane.seq,
        "conversation_seq": lane.conversation_seq,
        "claimed_by": lane.claimed_by,
        "retry_count": lane.retry_count,
        "run_id": lane.run_id,
        "step_id": lane.step_id,
        "correlation_id": lane.correlation_id,
        "created_at": lane.created_at,
        "available_at": lane.available_at,
    })
}

fn status_counts<'a>(values: impl Iterator<Item = &'a str>) -> serde_json::Map<String, Value> {
    let mut counts = serde_json::Map::new();
    for status in values {
        let count = counts.get(status).and_then(Value::as_u64).unwrap_or(0) + 1;
        counts.insert(status.to_owned(), json!(count));
    }
    counts
}

fn usage_history(events: &[ServerRunEvent], limit: usize) -> Vec<Value> {
    let mut values = events
        .iter()
        .filter(|event| event.event_type == "workflow.usage.v1")
        .map(|event| {
            let payload =
                serde_json::from_str::<Value>(&event.payload_json).unwrap_or_else(|_| json!({}));
            json!({
                "workspace_id": "workflow",
                "kind": "usage",
                "amount": payload["usage"]["total_tokens"].as_i64().unwrap_or_else(|| {
                    payload["usage"]["input_tokens"].as_i64().unwrap_or_default()
                        + payload["usage"]["output_tokens"].as_i64().unwrap_or_default()
                }),
                "source": payload["effect_id"],
                "run_id": event.run_id,
                "event_seq": event.seq,
                "created_at_ms": event.created_at_ms,
                "usage": payload["usage"],
            })
        })
        .collect::<Vec<_>>();
    let keep_from = values.len().saturating_sub(limit);
    values.drain(0..keep_from);
    values
}

fn resource_value(
    runs: &[ServerRun],
    lanes: &[ProjectedLaneMessage],
    usage: &[Value],
    events: &[ServerRunEvent],
) -> Value {
    let run_counts = status_counts(runs.iter().map(|run| run.status.as_str()));
    let lane_counts = status_counts(lanes.iter().map(|lane| lane.status.as_str()));
    let active = lanes.iter().filter(|lane| lane.status == "claimed").count();
    let queued = lanes.iter().filter(|lane| lane.status == "pending").count();
    let total_amount = usage
        .iter()
        .filter_map(|event| event["amount"].as_i64())
        .sum::<i64>();
    let now_seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_secs() as i64)
        .unwrap_or_default();
    let oldest_pending_seconds = lanes
        .iter()
        .filter(|lane| lane.status == "pending")
        .map(|lane| lane.available_at)
        .min();
    let parity_mismatches = events
        .iter()
        .filter(|event| {
            matches!(
                event.event_type.as_str(),
                "rust.parity_mismatch" | "workflow.parity_mismatch.v1"
            )
        })
        .count();
    json!({
        "scheduler": {
            "max_active": active.max(1),
            "max_queue": Value::Null,
            "active": active,
            "queued": queued,
            "active_by_class": {},
            "queued_by_class": {},
            "dead_letter_count": lane_counts.get("dead-letter").and_then(Value::as_u64).unwrap_or(0),
            "paused_count": runs.iter().filter(|run| run.status == "suspended").count(),
            "pause_requested_count": 0,
        },
        "runs": {
            "total_runs": runs.len(),
            "by_status": run_counts,
            "terminal_runs": runs.iter().filter(|run| matches!(run.status.as_str(), "succeeded" | "failed" | "cancelled")).count(),
        },
        "services": {"total_services": 0, "by_health": {}},
        "storage_usage_bytes": 0,
        "cost_ledger": {
            "workspace_id": "workflow",
            "event_count": usage.len(),
            "total_amount": total_amount,
            "by_kind": {"usage": usage.len()},
        },
        "budget_model": {"token_kind": "token", "time_kind": "ms", "storage_kind": "bytes"},
        "policy_infra": {"cpu_millicores": Value::Null, "memory_mb": Value::Null},
        "migration": {
            "implementation_mode": "rust",
            "contract_version": 1,
            "schema_version": 1,
            "parity_mismatch_count": parity_mismatches,
            "queue_lag": {
                "pending_count": queued,
                "oldest_pending_age_seconds": oldest_pending_seconds
                    .map(|created| now_seconds.saturating_sub(created)),
            },
            "replay_lag": {
                "events_behind": 0,
                "mode": "transactional_projection",
            },
        },
    })
}

fn operational_effect(
    request: &ApiEffectRequest,
    path: &str,
    runs: Vec<ServerRun>,
    lanes: Vec<ProjectedLaneMessage>,
    events: Vec<ServerRunEvent>,
) -> ApiEffectResponse {
    let limit = query_integer(&request.path_and_query, "limit", 200).clamp(0, 10_000) as usize;
    let usage = usage_history(&events, limit.max(1));
    let resources = resource_value(&runs, &lanes, &usage, &events);
    match path {
        "/api/workflow/catalog/ops" => json_effect(
            StatusCode::OK,
            json!([
                {
                    "op": "start",
                    "label": "Start",
                    "description": "Entry point for workflow execution.",
                    "input_schema": {},
                    "output_schema": {"type": "object"},
                    "config_schema": {"type": "object"},
                },
                {
                    "op": "llm_call",
                    "label": "LLM Call",
                    "description": "Calls an LLM and returns structured output.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"prompt": {"type": "string"}},
                        "required": ["prompt"],
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    "config_schema": {
                        "type": "object",
                        "properties": {
                            "model": {"type": "string"},
                            "temperature": {"type": "number"},
                        },
                    },
                },
            ]),
        ),
        "/api/workflow/resources" => json_effect(StatusCode::OK, resources),
        "/api/workflow/budget" => json_effect(
            StatusCode::OK,
            json!({
                "cost_ledger": resources["cost_ledger"],
                "budget_model": resources["budget_model"],
            }),
        ),
        "/api/workflow/budget/history" => json_effect(
            StatusCode::OK,
            json!({"cost_ledger": resources["cost_ledger"], "events": usage}),
        ),
        "/api/workflow/scheduler/timeline" => {
            let run_id = query_string(&request.path_and_query, "run_id");
            let mut timeline = events
                .iter()
                .filter(|event| run_id.as_ref().is_none_or(|id| id == &event.run_id))
                .cloned()
                .map(server_run_event_value)
                .collect::<Vec<_>>();
            let keep_from = timeline.len().saturating_sub(limit);
            timeline.drain(0..keep_from);
            json_effect(
                StatusCode::OK,
                json!({"run_id": run_id, "events": timeline}),
            )
        }
        "/api/workflow/dead-letters" => json_effect(
            StatusCode::OK,
            json!({
                "runs": lanes.iter()
                    .filter(|lane| lane.status == "dead-letter")
                    .take(limit)
                    .map(lane_progress_value)
                    .collect::<Vec<_>>(),
                "limit": limit,
            }),
        ),
        "/api/workflow/lane/progress" => {
            let run_id = query_string(&request.path_and_query, "run_id");
            let conversation_id = query_string(&request.path_and_query, "conversation_id");
            let mut items = events
                .iter()
                .filter(|event| run_id.as_ref().is_some_and(|id| id == &event.run_id))
                .map(|event| {
                    let mut value = server_run_event_value(event.clone());
                    if let Some(object) = value.as_object_mut() {
                        object.insert("event_type".to_owned(), json!(event.event_type));
                    }
                    value
                })
                .collect::<Vec<_>>();
            if conversation_id.is_some() {
                items.extend(
                    lanes
                        .iter()
                        .filter(|lane| {
                            run_id
                                .as_ref()
                                .is_none_or(|id| lane.run_id.as_ref() == Some(id))
                                && conversation_id
                                    .as_ref()
                                    .is_none_or(|id| &lane.conversation_id == id)
                        })
                        .map(lane_progress_value),
                );
            }
            items.truncate(limit);
            json_effect(
                StatusCode::OK,
                json!({"total": items.len(), "items": items}),
            )
        }
        "/api/workflow/operator/dashboard" => {
            let process_table = runs
                .iter()
                .take(limit)
                .cloned()
                .map(server_run_value)
                .collect::<Vec<_>>();
            let blocked_runs = runs
                .iter()
                .filter(|run| run.status == "suspended")
                .take(limit)
                .cloned()
                .map(server_run_value)
                .collect::<Vec<_>>();
            let dead_letters = lanes
                .iter()
                .filter(|lane| lane.status == "dead-letter")
                .take(limit)
                .map(lane_progress_value)
                .collect::<Vec<_>>();
            let lane_counts = status_counts(lanes.iter().map(|lane| lane.status.as_str()));
            json_effect(
                StatusCode::OK,
                json!({
                    "process_table": process_table,
                    "operator_inbox": blocked_runs.clone(),
                    "blocked_runs": blocked_runs.clone(),
                    "blocked_flow_graph": {"nodes": blocked_runs.clone(), "edges": [], "blocked_count": blocked_runs.len()},
                    "message_queue": {"total": lanes.len(), "by_status": lane_counts, "by_inbox": {}, "failed": []},
                    "dead_letters": {"runs": dead_letters, "limit": limit},
                    "capabilities": {},
                    "resources": resources,
                }),
            )
        }
        _ => unreachable!("unsupported operational path"),
    }
}

fn runtime_run_route(path_and_query: &str) -> Option<(String, Vec<String>)> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    for prefix in ["/api/runs/", "/api/workflow/runs/"] {
        if let Some(rest) = path.strip_prefix(prefix) {
            let segments = rest.split('/').collect::<Vec<_>>();
            if let Some((run_id, tail)) = segments.split_first()
                && !run_id.is_empty()
            {
                return Some((
                    (*run_id).to_owned(),
                    tail.iter().map(|value| (*value).to_owned()).collect(),
                ));
            }
        }
    }
    None
}

fn dead_letter_replay_run_id(path_and_query: &str) -> Option<String> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    let rest = path.strip_prefix("/api/workflow/dead-letters/")?;
    let run_id = rest.strip_suffix("/replay")?;
    (!run_id.is_empty() && !run_id.contains('/')).then(|| run_id.to_owned())
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ServiceRoute {
    Get,
    Events,
    Repair,
}

fn service_route(path_and_query: &str) -> Option<(String, ServiceRoute)> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    let rest = path.strip_prefix("/api/workflow/services/")?;
    let mut segments = rest.split('/');
    let service_id = segments.next()?;
    if service_id.is_empty() {
        return None;
    }
    let route = match segments.collect::<Vec<_>>().as_slice() {
        [] => ServiceRoute::Get,
        ["events"] => ServiceRoute::Events,
        ["repair"] => ServiceRoute::Repair,
        _ => return None,
    };
    Some((service_id.to_owned(), route))
}

fn service_repair_route(path_and_query: &str) -> Option<Option<String>> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    if path == "/api/workflow/services/repair" {
        return Some(None);
    }
    let (service_id, route) = service_route(path_and_query)?;
    (route == ServiceRoute::Repair).then_some(Some(service_id))
}

fn service_enabled_route(path_and_query: &str) -> Option<(String, bool)> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    let rest = path.strip_prefix("/api/workflow/services/")?;
    let (service_id, action) = rest.split_once('/')?;
    if service_id.is_empty() || action.contains('/') {
        return None;
    }
    match action {
        "enable" => Some((service_id.to_owned(), true)),
        "disable" => Some((service_id.to_owned(), false)),
        _ => None,
    }
}

fn latest_service_definition(
    events: impl IntoIterator<Item = (i64, String, String, String, String)>,
    service_id: &str,
) -> Option<Value> {
    let mut current = BTreeMap::<String, (i64, Value)>::new();
    for (seq, entity_kind, entity_id, op, payload_json) in events {
        if entity_kind != "node" {
            continue;
        }
        match op.to_ascii_uppercase().as_str() {
            "DELETE" | "TOMBSTONE" | "REMOVE" => {
                current.remove(&entity_id);
                continue;
            }
            "ADD" | "REPLACE" | "UPSERT" => {}
            _ => continue,
        }
        let Ok(payload) = serde_json::from_str::<Value>(&payload_json) else {
            continue;
        };
        if payload.is_object() {
            current.insert(entity_id, (seq, payload));
        }
    }
    current
        .into_values()
        .filter(|(_, payload)| {
            payload["metadata"]["entity_type"].as_str() == Some("service_definition")
                && payload["metadata"]["service_id"].as_str() == Some(service_id)
        })
        .max_by_key(|(seq, payload)| {
            (
                payload["metadata"]["updated_at_ms"]
                    .as_i64()
                    .or_else(|| payload["properties"]["updated_at_ms"].as_i64())
                    .unwrap_or_default(),
                *seq,
            )
        })
        .map(|(_, payload)| payload)
}

fn service_enabled_nodes(
    mut definition: Value,
    service_id: &str,
    enabled: bool,
    timestamp: u64,
) -> Result<(String, Value, String, Value), String> {
    let definition_id = format!(
        "service_def:{service_id}:{timestamp}:{}",
        uuid::Uuid::new_v4().simple()
    );
    let definition_object = definition
        .as_object_mut()
        .ok_or_else(|| "service definition must be an object".to_owned())?;
    definition_object.insert("id".to_owned(), json!(definition_id));
    let metadata = definition_object
        .get_mut("metadata")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| "service definition metadata must be an object".to_owned())?;
    metadata.insert("enabled".to_owned(), json!(enabled));
    metadata.insert("updated_at_ms".to_owned(), json!(timestamp));
    let properties = definition_object
        .get_mut("properties")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| "service definition properties must be an object".to_owned())?;
    properties.insert("enabled".to_owned(), json!(enabled));
    properties.insert("updated_at_ms".to_owned(), json!(timestamp));

    let event_type = if enabled {
        "service.enabled"
    } else {
        "service.stopped"
    };
    let event_id = format!(
        "service_evt:{service_id}:{timestamp}:{}",
        uuid::Uuid::new_v4().simple()
    );
    let event = json!({
        "id": event_id,
        "label": format!("service_event:{event_type}"),
        "type": "entity",
        "summary": format!("Service event {event_type} for {service_id}"),
        "mentions": [{"spans": [{
            "collection_page_url": format!("workflow/_wf:{service_id}"),
            "context_after": "",
            "context_before": "",
            "doc_id": format!("_wf:{service_id}"),
            "document_page_url": format!("workflow/_wf:{service_id}"),
            "end_char": 1,
            "excerpt": "",
            "page_number": 1,
            "source_cluster_id": Value::Null,
            "start_char": 0,
        }]}],
        "properties": {
            "payload_json": json!({"enabled": enabled}).to_string(),
            "enabled": enabled,
        },
        "metadata": {
            "entity_type": "service_event",
            "artifact_kind": "service_event",
            "service_id": service_id,
            "service_event_type": event_type,
            "ts_ms": timestamp,
            "in_conversation_chain": false,
            "lifecycle_status": "active",
            "redirect_to_id": Value::Null,
        },
    });
    Ok((definition_id, definition, event_id, event))
}

fn service_event_node(
    service_id: &str,
    event_type: &str,
    payload: &Value,
    timestamp: u64,
) -> (String, Value) {
    let event_id = format!(
        "service_evt:{service_id}:{timestamp}:{}",
        uuid::Uuid::new_v4().simple()
    );
    (
        event_id.clone(),
        json!({
            "id": event_id,
            "label": format!("service_event:{event_type}"),
            "type": "entity",
            "summary": format!("Service event {event_type} for {service_id}"),
            "mentions": [{"spans": [{
                "collection_page_url": format!("workflow/_wf:{service_id}"),
                "context_after": "",
                "context_before": "",
                "doc_id": format!("_wf:{service_id}"),
                "document_page_url": format!("workflow/_wf:{service_id}"),
                "end_char": 1,
                "excerpt": "",
                "page_number": 1,
                "source_cluster_id": Value::Null,
                "start_char": 0,
            }]}],
            "properties": {"payload_json": payload.to_string()},
            "metadata": {
                "entity_type": "service_event",
                "artifact_kind": "service_event",
                "service_id": service_id,
                "service_event_type": event_type,
                "ts_ms": timestamp,
                "in_conversation_chain": false,
                "lifecycle_status": "active",
                "redirect_to_id": Value::Null,
            },
        }),
    )
}

#[derive(Clone, Debug, Deserialize)]
struct ServiceDeclareInput {
    service_id: String,
    #[serde(default = "default_service_kind")]
    service_kind: String,
    #[serde(default = "default_service_target_kind")]
    target_kind: String,
    target_ref: String,
    #[serde(default)]
    target_config: serde_json::Map<String, Value>,
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default)]
    autostart: bool,
    #[serde(default)]
    restart_policy: serde_json::Map<String, Value>,
    #[serde(default = "default_heartbeat_ttl_ms")]
    heartbeat_ttl_ms: i64,
    #[serde(default)]
    trigger_specs: Vec<Value>,
}

fn default_service_kind() -> String {
    "daemon".to_owned()
}

fn default_service_target_kind() -> String {
    "workflow".to_owned()
}

fn default_heartbeat_ttl_ms() -> i64 {
    60_000
}

#[derive(Clone, Debug, Deserialize)]
struct ServiceTriggerInput {
    trigger_type: String,
    #[serde(default)]
    payload: serde_json::Map<String, Value>,
}

fn service_trigger_route(path_and_query: &str) -> Option<String> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    let rest = path.strip_prefix("/api/workflow/services/")?;
    let (service_id, action) = rest.split_once('/')?;
    (!service_id.is_empty() && action == "trigger").then(|| service_id.to_owned())
}

fn validate_service_trigger(input: &mut ServiceTriggerInput) -> Result<(), &'static str> {
    input.trigger_type = input.trigger_type.trim().to_ascii_lowercase();
    if matches!(
        input.trigger_type.as_str(),
        "schedule"
            | "message arrival"
            | "graph change"
            | "external event"
            | "autostart"
            | "restart"
    ) {
        Ok(())
    } else {
        Err("unsupported trigger_type")
    }
}

fn service_definition_runtime(definition: &Value) -> Result<(String, Value), &'static str> {
    let properties = definition
        .get("properties")
        .and_then(Value::as_object)
        .ok_or("service definition properties must be an object")?;
    let target_kind = properties
        .get("target_kind")
        .and_then(Value::as_str)
        .unwrap_or("workflow")
        .to_owned();
    let target_config = properties
        .get("target_config_json")
        .and_then(Value::as_str)
        .and_then(|value| serde_json::from_str::<Value>(value).ok())
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({}));
    Ok((target_kind, target_config))
}

fn service_trigger_is_suppressed(
    definition: &Value,
    projection: &Value,
    trigger_type: &str,
    timestamp_ms: u64,
) -> bool {
    let properties = &definition["properties"];
    if properties["enabled"].as_bool() == Some(false) {
        return true;
    }
    let trigger_specs = properties["trigger_specs_json"]
        .as_str()
        .and_then(|raw| serde_json::from_str::<Vec<Value>>(raw).ok())
        .unwrap_or_default();
    let matching = trigger_specs.into_iter().find(|spec| {
        spec["type"]
            .as_str()
            .is_some_and(|value| value.trim().eq_ignore_ascii_case(trigger_type))
    });
    let Some(spec) = matching else {
        return false;
    };
    if spec["enabled"].as_bool() == Some(false) {
        return true;
    }
    let cooldown_ms = spec["cooldown_ms"].as_u64().unwrap_or_default();
    let debounce_ms = spec["debounce_ms"].as_u64().unwrap_or_default();
    let not_before_ms = cooldown_ms.max(debounce_ms);
    let last_triggered_at_ms = projection["last_triggered_at_ms"]
        .as_u64()
        .unwrap_or_default();
    not_before_ms > 0
        && last_triggered_at_ms > 0
        && timestamp_ms.saturating_sub(last_triggered_at_ms) < not_before_ms
}

fn validate_service_declare(input: &mut ServiceDeclareInput) -> Result<(), &'static str> {
    input.service_id = input.service_id.trim().to_owned();
    input.service_kind = input.service_kind.trim().to_owned();
    input.target_kind = input.target_kind.trim().to_owned();
    input.target_ref = input.target_ref.trim().to_owned();
    if input.service_id.is_empty()
        || input.service_kind.is_empty()
        || input.target_kind.is_empty()
        || input.target_ref.is_empty()
    {
        return Err("service_id, service_kind, target_kind, and target_ref are required");
    }
    input.heartbeat_ttl_ms = input.heartbeat_ttl_ms.max(1);
    for spec in &input.trigger_specs {
        let Some(kind) = spec.get("type").and_then(Value::as_str) else {
            return Err("trigger_specs[].type is required");
        };
        if kind.trim().is_empty() {
            return Err("trigger_specs[].type is required");
        }
    }
    Ok(())
}

fn service_definition_node(input: &ServiceDeclareInput, timestamp: u64) -> (String, Value) {
    let service_id = &input.service_id;
    let entity_id = format!(
        "service_def:{service_id}:{timestamp}:{}",
        uuid::Uuid::new_v4().simple()
    );
    (
        entity_id.clone(),
        json!({
            "id": entity_id,
            "label": format!("service_definition:{service_id}"),
            "type": "entity",
            "summary": format!("Service definition {service_id}"),
            "mentions": [{"spans": [{
                "collection_page_url": format!("workflow/_wf:{service_id}"),
                "context_after": "",
                "context_before": "",
                "doc_id": format!("_wf:{service_id}"),
                "document_page_url": format!("workflow/_wf:{service_id}"),
                "end_char": 1,
                "excerpt": "",
                "page_number": 1,
                "source_cluster_id": Value::Null,
                "start_char": 0,
            }]}],
            "properties": {
                "service_id": service_id,
                "service_kind": input.service_kind,
                "target_kind": input.target_kind,
                "target_ref": input.target_ref,
                "target_config_json": Value::Object(input.target_config.clone()).to_string(),
                "enabled": input.enabled,
                "autostart": input.autostart,
                "restart_policy_json": Value::Object(input.restart_policy.clone()).to_string(),
                "heartbeat_ttl_ms": input.heartbeat_ttl_ms,
                "trigger_specs_json": Value::Array(input.trigger_specs.clone()).to_string(),
                "security_scope": "workflow",
                "storage_namespace": "workflow",
                "execution_namespace": "workflow",
                "created_at_ms": timestamp,
                "updated_at_ms": timestamp,
            },
            "metadata": {
                "entity_type": "service_definition",
                "artifact_kind": "service_definition",
                "service_id": service_id,
                "service_kind": input.service_kind,
                "target_kind": input.target_kind,
                "target_ref": input.target_ref,
                "enabled": input.enabled,
                "updated_at_ms": timestamp,
                "created_at_ms": timestamp,
                "in_conversation_chain": false,
                "lifecycle_status": "active",
                "redirect_to_id": Value::Null,
            },
        }),
    )
}

fn service_event_values(
    events: impl IntoIterator<Item = (i64, String, String, String, String)>,
    service_id: &str,
    limit: usize,
) -> Vec<Value> {
    let mut current = BTreeMap::<String, (i64, Value)>::new();
    for (seq, entity_kind, entity_id, op, payload_json) in events {
        if entity_kind != "node" {
            continue;
        }
        match op.to_ascii_uppercase().as_str() {
            "DELETE" | "TOMBSTONE" | "REMOVE" => {
                current.remove(&entity_id);
                continue;
            }
            "ADD" | "REPLACE" | "UPSERT" => {}
            _ => continue,
        }
        let Ok(payload) = serde_json::from_str::<Value>(&payload_json) else {
            continue;
        };
        if !payload.is_object() {
            continue;
        }
        current.insert(entity_id, (seq, payload));
    }

    let mut values = current
        .into_iter()
        .filter_map(|(entity_id, (seq, payload))| {
            let metadata = payload.get("metadata")?.as_object()?;
            if metadata.get("entity_type").and_then(Value::as_str) != Some("service_event")
                || metadata.get("service_id").and_then(Value::as_str) != Some(service_id)
            {
                return None;
            }
            let event_type = metadata
                .get("service_event_type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let ts_ms = metadata
                .get("ts_ms")
                .and_then(Value::as_i64)
                .unwrap_or_default();
            let properties = payload.get("properties").and_then(Value::as_object);
            let event_payload = properties
                .and_then(|value| value.get("payload_json"))
                .and_then(Value::as_str)
                .and_then(|value| serde_json::from_str::<Value>(value).ok())
                .filter(Value::is_object)
                .unwrap_or_else(|| json!({}));
            let event_id = payload
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or(&entity_id)
                .to_owned();
            Some((
                ts_ms,
                seq,
                json!({
                    "event_id": event_id,
                    "service_id": service_id,
                    "event_type": event_type,
                    "ts_ms": ts_ms,
                    "payload": event_payload,
                }),
            ))
        })
        .collect::<Vec<_>>();
    values.sort_by(|left, right| (left.0, left.1).cmp(&(right.0, right.1)));
    values.truncate(limit);
    values.into_iter().map(|(_, _, value)| value).collect()
}

fn service_projection_values(
    events: impl IntoIterator<Item = (i64, String, String, String, String)>,
) -> Result<BTreeMap<String, (i64, Value)>, String> {
    let mut current = BTreeMap::<String, (i64, Value)>::new();
    let mut latest_seq = 0_i64;
    for (seq, entity_kind, entity_id, op, payload_json) in events {
        latest_seq = latest_seq.max(seq);
        if entity_kind != "node" {
            continue;
        }
        match op.to_ascii_uppercase().as_str() {
            "DELETE" | "TOMBSTONE" | "REMOVE" => {
                current.remove(&entity_id);
                continue;
            }
            "ADD" | "REPLACE" | "UPSERT" => {}
            _ => continue,
        }
        let Ok(payload) = serde_json::from_str::<Value>(&payload_json) else {
            continue;
        };
        if payload.is_object() {
            current.insert(entity_id, (seq, payload));
        }
    }

    let mut definitions = BTreeMap::<String, (i64, i64, Value)>::new();
    let mut service_events = BTreeMap::<String, Vec<(i64, Value)>>::new();
    for (_entity_id, (seq, payload)) in current {
        let Some(metadata) = payload.get("metadata").and_then(Value::as_object) else {
            continue;
        };
        let Some(service_id) = metadata.get("service_id").and_then(Value::as_str) else {
            continue;
        };
        match metadata.get("entity_type").and_then(Value::as_str) {
            Some("service_definition") => {
                let updated_at_ms = metadata
                    .get("updated_at_ms")
                    .and_then(Value::as_i64)
                    .unwrap_or_default();
                let replace =
                    definitions
                        .get(service_id)
                        .is_none_or(|(current_updated, current_seq, _)| {
                            (updated_at_ms, seq) > (*current_updated, *current_seq)
                        });
                if replace {
                    definitions.insert(service_id.to_owned(), (updated_at_ms, seq, payload));
                }
            }
            Some("service_event") => service_events
                .entry(service_id.to_owned())
                .or_default()
                .push((seq, payload)),
            _ => {}
        }
    }

    let mut projections = BTreeMap::new();
    for (service_id, (_updated_at_ms, _definition_seq, definition)) in definitions {
        let metadata = definition["metadata"]
            .as_object()
            .ok_or_else(|| "service definition metadata must be an object".to_owned())?;
        let properties = definition["properties"]
            .as_object()
            .ok_or_else(|| "service definition properties must be an object".to_owned())?;
        let json_property = |name: &str, fallback: Value| {
            properties
                .get(name)
                .and_then(Value::as_str)
                .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                .unwrap_or(fallback)
        };
        let enabled = properties
            .get("enabled")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let created_at_ms = properties
            .get("created_at_ms")
            .and_then(Value::as_i64)
            .or_else(|| metadata.get("created_at_ms").and_then(Value::as_i64))
            .unwrap_or_default();
        let updated_at_ms = properties
            .get("updated_at_ms")
            .and_then(Value::as_i64)
            .or_else(|| metadata.get("updated_at_ms").and_then(Value::as_i64))
            .unwrap_or(created_at_ms);
        let mut projection = json!({
            "service_id": service_id,
            "service_kind": properties.get("service_kind").and_then(Value::as_str).unwrap_or("service"),
            "target_kind": properties.get("target_kind").and_then(Value::as_str).unwrap_or("workflow"),
            "target_ref": properties.get("target_ref").and_then(Value::as_str).unwrap_or_default(),
            "enabled": enabled,
            "autostart": properties.get("autostart").and_then(Value::as_bool).unwrap_or(false),
            "storage_namespace": properties.get("storage_namespace").and_then(Value::as_str).unwrap_or("workflow"),
            "execution_namespace": properties.get("execution_namespace").and_then(Value::as_str).unwrap_or("workflow"),
            "security_scope": properties.get("security_scope").and_then(Value::as_str).unwrap_or("workflow"),
            "lifecycle_status": if enabled { "degraded" } else { "stopped" },
            "health_status": if enabled { "degraded" } else { "stopped" },
            "last_heartbeat_ms": Value::Null,
            "restart_count": 0,
            "current_child_run_id": Value::Null,
            "current_child_status": Value::Null,
            "current_child_started_at_ms": Value::Null,
            "last_handled_child_run_id": Value::Null,
            "last_trigger_type": Value::Null,
            "last_triggered_at_ms": Value::Null,
            "heartbeat_ttl_ms": properties.get("heartbeat_ttl_ms").and_then(Value::as_i64).unwrap_or(60_000).max(1),
            "restart_policy": json_property("restart_policy_json", json!({})),
            "trigger_specs": json_property("trigger_specs_json", json!([])),
            "last_message_seen_ms": 0,
            "last_graph_event_seq": 0,
            "next_due_at_ms": Value::Null,
            "restart_not_before_ms": Value::Null,
            "created_at_ms": created_at_ms,
            "updated_at_ms": updated_at_ms,
        });
        let mut history = service_events.remove(&service_id).unwrap_or_default();
        history.sort_by_key(|(seq, _)| *seq);
        for (seq, event) in history {
            let metadata = &event["metadata"];
            let event_type = metadata["service_event_type"].as_str().unwrap_or_default();
            let ts_ms = metadata["ts_ms"].as_i64().unwrap_or_default();
            let payload = event["properties"]["payload_json"]
                .as_str()
                .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                .unwrap_or_else(|| json!({}));
            projection["updated_at_ms"] = json!(
                projection["updated_at_ms"]
                    .as_i64()
                    .unwrap_or_default()
                    .max(ts_ms)
            );
            match event_type {
                "service.heartbeat" => projection["last_heartbeat_ms"] = json!(ts_ms),
                "service.enabled" => {
                    projection["enabled"] = json!(payload["enabled"].as_bool().unwrap_or(true));
                }
                "service.triggered" => {
                    projection["last_trigger_type"] = payload["trigger_type"].clone();
                    projection["last_triggered_at_ms"] = json!(ts_ms);
                    projection["lifecycle_status"] = json!("starting");
                }
                "service.starting" => {
                    projection["lifecycle_status"] = json!("starting");
                    projection["health_status"] = json!("healthy");
                }
                "service.healthy" => {
                    projection["lifecycle_status"] = json!("healthy");
                    projection["health_status"] =
                        json!(payload["health_status"].as_str().unwrap_or("healthy"));
                }
                "service.degraded" => {
                    projection["lifecycle_status"] = json!("degraded");
                    projection["health_status"] =
                        json!(payload["health_status"].as_str().unwrap_or("degraded"));
                }
                "service.run_spawned" => {
                    projection["current_child_run_id"] = payload["run_id"].clone();
                    projection["current_child_started_at_ms"] = json!(ts_ms);
                    projection["current_child_status"] = json!("queued");
                    projection["lifecycle_status"] = json!("starting");
                }
                "service.restarting" => {
                    projection["lifecycle_status"] = json!("restarting");
                    projection["health_status"] = json!("degraded");
                    projection["restart_not_before_ms"] = payload["restart_not_before_ms"].clone();
                    projection["restart_count"] =
                        json!(payload["restart_count"].as_i64().unwrap_or_else(|| {
                            projection["restart_count"].as_i64().unwrap_or_default()
                        }));
                }
                "service.stopped" => {
                    let still_enabled = payload["enabled"]
                        .as_bool()
                        .unwrap_or_else(|| projection["enabled"].as_bool().unwrap_or(false));
                    projection["enabled"] = json!(still_enabled);
                    if !still_enabled {
                        projection["lifecycle_status"] = json!("stopped");
                        projection["health_status"] = json!("stopped");
                    }
                }
                "service.run_failed" => {
                    projection["current_child_status"] = payload["status"].clone();
                    projection["last_handled_child_run_id"] = payload["run_id"].clone();
                }
                _ => {}
            }
            let _ = seq;
        }
        projections.insert(service_id, (latest_seq, projection));
    }
    Ok(projections)
}

fn service_projection_write(last_seq: i64, projection: &Value) -> NamedProjectionWrite {
    NamedProjectionWrite {
        payload: projection.as_object().cloned().unwrap_or_default(),
        last_authoritative_seq: last_seq,
        last_materialized_seq: last_seq,
        projection_schema_version: 1,
        materialization_status: "ready".to_owned(),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum DesignRoute {
    Graph,
    History,
    NodeUpsert,
    NodeDelete(String),
    EdgeUpsert,
    EdgeDelete(String),
    Undo,
    Redo,
}

fn workflow_design_route(path_and_query: &str) -> Option<(String, DesignRoute)> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    let rest = path.strip_prefix("/api/workflow/design/")?;
    let mut segments = rest.split('/');
    let workflow_id = segments.next()?.to_owned();
    if workflow_id.is_empty() {
        return None;
    }
    let tail = segments.collect::<Vec<_>>();
    let route = match tail.as_slice() {
        ["graph"] => DesignRoute::Graph,
        ["history"] => DesignRoute::History,
        ["nodes"] => DesignRoute::NodeUpsert,
        ["nodes", node_id] if !node_id.is_empty() => DesignRoute::NodeDelete((*node_id).to_owned()),
        ["edges"] => DesignRoute::EdgeUpsert,
        ["edges", edge_id] if !edge_id.is_empty() => DesignRoute::EdgeDelete((*edge_id).to_owned()),
        ["undo"] => DesignRoute::Undo,
        ["redo"] => DesignRoute::Redo,
        _ => return None,
    };
    Some((workflow_id, route))
}

fn design_payload(event: &EntityEvent) -> Option<&serde_json::Map<String, Value>> {
    (event.entity_kind == "design_control")
        .then(|| event.payload.as_object())
        .flatten()
}

fn workflow_design_graph_value(workflow_id: &str, events: &[EntityEvent]) -> Result<Value, String> {
    let history = workflow_design_history_value(workflow_id, events.to_vec(), "ready")?;
    let current_version = history["current_version"].as_i64().unwrap_or(0);
    let snapshot = graph_at_version(events, current_version);
    Ok(json!({
        "workflow_id": workflow_id,
        "current_version": history["current_version"],
        "active_tip_version": history["active_tip_version"],
        "can_undo": history["can_undo"],
        "can_redo": history["can_redo"],
        "materialization_status": "ready",
        "nodes": snapshot["nodes"].as_array().cloned().unwrap_or_default(),
        "edges": snapshot["edges"].as_array().cloned().unwrap_or_default(),
    }))
}

fn designer_id(request: &ApiEffectRequest, input: &Value) -> Result<String, ApiEffectResponse> {
    let value = input["designer_id"].as_str().map(str::trim).unwrap_or("");
    if value.is_empty() {
        return Err(effect_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "KOGWISTAR_INVALID_REQUEST",
            "designer_id is required",
        ));
    }
    let subject = principal_subject(&request.principal);
    if subject != "anonymous" && subject != value.to_ascii_lowercase() {
        return Err(effect_error(
            StatusCode::FORBIDDEN,
            "KOGWISTAR_DESIGNER_SUBJECT_MISMATCH",
            "designer_id must match authenticated subject",
        ));
    }
    Ok(value.to_owned())
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_millis() as u64)
        .unwrap_or_default()
}

fn parse_design_input(request: &ApiEffectRequest) -> Result<Value, ApiEffectResponse> {
    serde_json::from_slice(&request.body).map_err(|error| {
        effect_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "KOGWISTAR_INVALID_REQUEST",
            error.to_string(),
        )
    })
}

fn graph_at_version(events: &[EntityEvent], version: i64) -> Value {
    if version <= 0 {
        return json!({"nodes": [], "edges": []});
    }
    let commits = events
        .iter()
        .filter_map(|event| {
            let payload = design_payload(event)?;
            (event.op == "MUTATION_COMMITTED").then(|| {
                (
                    payload.get("version").and_then(Value::as_i64).unwrap_or(0),
                    (
                        payload
                            .get("prev_version")
                            .and_then(Value::as_i64)
                            .unwrap_or(0),
                        payload
                            .get("target_seq")
                            .and_then(Value::as_i64)
                            .unwrap_or(0),
                        payload.get("graph").cloned(),
                    ),
                )
            })
        })
        .collect::<BTreeMap<_, _>>();
    let mut lineage = Vec::new();
    let mut cursor = version;
    let mut seen = BTreeSet::new();
    while cursor > 0 && seen.insert(cursor) {
        let Some((prev, target_seq, legacy_graph)) = commits.get(&cursor) else {
            return json!({"nodes": [], "edges": []});
        };
        lineage.push((*target_seq, legacy_graph.clone()));
        cursor = *prev;
    }
    lineage.reverse();
    let mut graph = json!({"nodes": [], "edges": []});
    for (target_seq, legacy_graph) in lineage {
        if let Some(event) = events.iter().find(|event| {
            event.seq == target_seq && matches!(event.entity_kind.as_str(), "node" | "edge")
        }) {
            apply_design_entity_event(&mut graph, event);
        } else if let Some(snapshot) = legacy_graph {
            // Transitional compatibility for pre-cutover development databases.
            graph = snapshot;
        }
    }
    graph
}

fn apply_design_entity_event(graph: &mut Value, event: &EntityEvent) {
    let object = graph.as_object_mut().expect("graph value is object");
    let key = if event.entity_kind == "node" {
        "nodes"
    } else {
        "edges"
    };
    let mut records = object
        .remove(key)
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default();
    if matches!(event.op.as_str(), "TOMBSTONE" | "DELETE") {
        records.retain(|item| item["id"].as_str() != Some(&event.entity_id));
    } else if matches!(event.op.as_str(), "ADD" | "REPLACE") {
        if let Some(index) = records
            .iter()
            .position(|item| item["id"].as_str() == Some(&event.entity_id))
        {
            records[index] = event.payload.clone();
        } else {
            records.push(event.payload.clone());
        }
    }
    object.insert(key.to_owned(), Value::Array(records));
    if event.entity_kind == "node"
        && matches!(event.op.as_str(), "TOMBSTONE" | "DELETE")
        && let Some(edges) = object.get_mut("edges").and_then(Value::as_array_mut)
    {
        edges.retain(|edge| {
            !["source_ids", "target_ids"]
                .iter()
                .filter_map(|field| edge[*field].as_array())
                .flatten()
                .any(|id| id.as_str() == Some(&event.entity_id))
        });
    }
}

fn visible_delta(before: &Value, after: &Value) -> Value {
    fn indexed(graph: &Value, key: &str) -> BTreeMap<String, Value> {
        graph[key]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|item| Some((item["id"].as_str()?.to_owned(), item.clone())))
            .collect()
    }
    let before_nodes = indexed(before, "nodes");
    let after_nodes = indexed(after, "nodes");
    let before_edges = indexed(before, "edges");
    let after_edges = indexed(after, "edges");
    json!({
        "upsert_nodes": after_nodes.iter().filter(|(id, value)| before_nodes.get(*id) != Some(*value)).map(|(_, value)| value.clone()).collect::<Vec<_>>(),
        "delete_node_ids": before_nodes.keys().filter(|id| !after_nodes.contains_key(*id)).cloned().collect::<Vec<_>>(),
        "upsert_edges": after_edges.iter().filter(|(id, value)| before_edges.get(*id) != Some(*value)).map(|(_, value)| value.clone()).collect::<Vec<_>>(),
        "delete_edge_ids": before_edges.keys().filter(|id| !after_edges.contains_key(*id)).cloned().collect::<Vec<_>>(),
    })
}

fn design_entity_event(
    action: &str,
    entity_id: &str,
    before: &Value,
    after: &Value,
) -> EntityEvent {
    let (entity_kind, op, payload) = match action {
        "node_upsert" => (
            "node",
            "ADD",
            after["nodes"]
                .as_array()
                .and_then(|items| items.iter().find(|item| item["id"] == entity_id))
                .cloned()
                .unwrap_or(Value::Null),
        ),
        "edge_upsert" => (
            "edge",
            "ADD",
            after["edges"]
                .as_array()
                .and_then(|items| items.iter().find(|item| item["id"] == entity_id))
                .cloned()
                .unwrap_or(Value::Null),
        ),
        "node_delete" => (
            "node",
            "TOMBSTONE",
            json!({"entity_id": entity_id, "reason": "workflow_design_delete"}),
        ),
        "edge_delete" => (
            "edge",
            "TOMBSTONE",
            json!({"entity_id": entity_id, "reason": "workflow_design_delete"}),
        ),
        _ => unreachable!("known design mutation action"),
    };
    let _ = before;
    EntityEvent {
        namespace: String::new(),
        seq: 0,
        event_id: uuid::Uuid::new_v4().to_string(),
        entity_kind: entity_kind.to_owned(),
        entity_id: entity_id.to_owned(),
        op: op.to_owned(),
        payload,
    }
}

fn mutate_design_graph(
    workflow_id: &str,
    route: &DesignRoute,
    input: &Value,
    graph: &mut Value,
) -> Result<(String, String, bool), ApiEffectResponse> {
    let object = graph.as_object_mut().expect("graph value is object");
    let mut nodes = object
        .remove("nodes")
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default();
    let mut edges = object
        .remove("edges")
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default();
    match route {
        DesignRoute::NodeUpsert => {
            let label = input["label"].as_str().map(str::trim).unwrap_or("");
            if label.is_empty() {
                return Err(effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    "label is required",
                ));
            }
            let node_id = input["node_id"]
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| format!("wf|{workflow_id}|n|{}", uuid::Uuid::new_v4()));
            let mut metadata = input["metadata"].as_object().cloned().unwrap_or_default();
            metadata.extend(serde_json::Map::from_iter([
                ("entity_type".to_owned(), json!("workflow_node")),
                ("workflow_id".to_owned(), json!(workflow_id)),
                (
                    "wf_op".to_owned(),
                    json!(input["op"].as_str().unwrap_or(
                        if input["terminal"].as_bool().unwrap_or(false) {
                            "end"
                        } else {
                            "noop"
                        }
                    )),
                ),
                (
                    "wf_start".to_owned(),
                    json!(input["start"].as_bool().unwrap_or(false)),
                ),
                (
                    "wf_terminal".to_owned(),
                    json!(input["terminal"].as_bool().unwrap_or(false)),
                ),
                (
                    "wf_fanout".to_owned(),
                    json!(input["fanout"].as_bool().unwrap_or(false)),
                ),
                ("designer_id".to_owned(), input["designer_id"].clone()),
            ]));
            let node = json!({
                "id": node_id, "label": label, "type": "entity",
                "summary": format!("workflow node {label}"), "doc_id": format!("workflow:{workflow_id}"),
                "mentions": [{"spans": [{"collection_page_url":"conversation/_conv:_dummy",
                    "document_page_url":"conversation/_conv:_dummy","doc_id":"_conv:_dummy","page_number":1,
                    "insertion_method":"system","start_char":0,"end_char":1,"excerpt":"","context_before":"","context_after":"",
                    "source_cluster_id":Value::Null,"verification":{"method":"system","is_verified":true,"score":1.0,"notes":""}}]}],
                "properties": {}, "metadata": metadata, "level_from_root": 0,
                "domain_id": Value::Null, "canonical_entity_id": Value::Null, "embedding": Value::Null,
            });
            if let Some(index) = nodes.iter().position(|item| item["id"] == node_id) {
                nodes[index] = node;
            } else {
                nodes.push(node);
            }
            object.insert("nodes".to_owned(), Value::Array(nodes));
            object.insert("edges".to_owned(), Value::Array(edges));
            Ok(("node_upsert".to_owned(), node_id, false))
        }
        DesignRoute::EdgeUpsert => {
            let src = input["src"].as_str().map(str::trim).unwrap_or("");
            let dst = input["dst"].as_str().map(str::trim).unwrap_or("");
            if src.is_empty() || dst.is_empty() {
                return Err(effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    "src and dst are required",
                ));
            }
            let edge_id = input["edge_id"]
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| format!("wf|{workflow_id}|e|{}", uuid::Uuid::new_v4()));
            let relation = input["relation"].as_str().unwrap_or("wf_next");
            let mut metadata = input["metadata"].as_object().cloned().unwrap_or_default();
            metadata.extend(serde_json::Map::from_iter([
                ("entity_type".to_owned(), json!("workflow_edge")),
                ("workflow_id".to_owned(), json!(workflow_id)),
                ("wf_predicate".to_owned(), input["predicate"].clone()),
                (
                    "wf_priority".to_owned(),
                    json!(input["priority"].as_i64().unwrap_or(100)),
                ),
                (
                    "wf_is_default".to_owned(),
                    json!(input["is_default"].as_bool().unwrap_or(false)),
                ),
                (
                    "wf_multiplicity".to_owned(),
                    json!(input["multiplicity"].as_str().unwrap_or("one")),
                ),
                ("designer_id".to_owned(), input["designer_id"].clone()),
            ]));
            let edge = json!({
                "id": edge_id, "source_ids": [src], "target_ids": [dst], "relation": relation,
                "label": relation, "type": "relationship", "summary": format!("workflow edge {src} -> {dst}"),
                "doc_id": format!("workflow:{workflow_id}"),
                "mentions": [{"spans": [{"collection_page_url":"conversation/_conv:_dummy",
                    "document_page_url":"conversation/_conv:_dummy","doc_id":"_conv:_dummy","page_number":1,
                    "insertion_method":"system","start_char":0,"end_char":1,"excerpt":"","context_before":"","context_after":"",
                    "source_cluster_id":Value::Null,"verification":{"method":"system","is_verified":true,"score":1.0,"notes":""}}]}],
                "properties": {}, "metadata": metadata,
                "source_edge_ids": [], "target_edge_ids": [], "domain_id": Value::Null,
                "canonical_entity_id": Value::Null, "embedding": Value::Null,
            });
            if let Some(index) = edges.iter().position(|item| item["id"] == edge_id) {
                edges[index] = edge;
            } else {
                edges.push(edge);
            }
            object.insert("nodes".to_owned(), Value::Array(nodes));
            object.insert("edges".to_owned(), Value::Array(edges));
            Ok(("edge_upsert".to_owned(), edge_id, false))
        }
        DesignRoute::NodeDelete(node_id) => {
            let before = nodes.len();
            nodes.retain(|item| item["id"] != *node_id);
            if nodes.len() == before {
                return Err(effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_NODE_NOT_FOUND",
                    format!("Unknown node_id: {node_id}"),
                ));
            }
            edges.retain(|item| {
                item["source_ids"]
                    .as_array()
                    .is_none_or(|ids| !ids.iter().any(|id| id == node_id))
                    && item["target_ids"]
                        .as_array()
                        .is_none_or(|ids| !ids.iter().any(|id| id == node_id))
            });
            object.insert("nodes".to_owned(), Value::Array(nodes));
            object.insert("edges".to_owned(), Value::Array(edges));
            Ok(("node_delete".to_owned(), node_id.clone(), true))
        }
        DesignRoute::EdgeDelete(edge_id) => {
            let before = edges.len();
            edges.retain(|item| item["id"] != *edge_id);
            if edges.len() == before {
                return Err(effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_EDGE_NOT_FOUND",
                    format!("Unknown edge_id: {edge_id}"),
                ));
            }
            object.insert("nodes".to_owned(), Value::Array(nodes));
            object.insert("edges".to_owned(), Value::Array(edges));
            Ok(("edge_delete".to_owned(), edge_id.clone(), true))
        }
        _ => unreachable!("only mutation routes reach graph mutation"),
    }
}

fn workflow_design_history_value(
    workflow_id: &str,
    events: Vec<EntityEvent>,
    materialization_status: &str,
) -> Result<Value, String> {
    let mut commits = BTreeMap::<i64, Value>::new();
    let mut dropped_ranges = Vec::new();
    let mut timeline = Vec::new();
    let mut current_version = 0_i64;
    let mut active_tip_version = 0_i64;
    let mut allocated_max_version = 0_i64;
    let latest_seq = events.iter().map(|event| event.seq).max().unwrap_or(0);
    for event in events {
        if event.entity_kind != "design_control" {
            continue;
        }
        let payload = event.payload.as_object().cloned().unwrap_or_default();
        let mut item = payload.clone();
        item.insert("seq".to_owned(), json!(event.seq));
        item.insert("op".to_owned(), json!(event.op));
        item.entry("designer_id".to_owned()).or_insert(json!(""));
        item.entry("ts_ms".to_owned()).or_insert(json!(0));
        timeline.push(Value::Object(item));
        match event.op.as_str() {
            "MUTATION_COMMITTED" => {
                let version = payload.get("version").and_then(Value::as_i64).unwrap_or(0);
                let prev_version = payload
                    .get("prev_version")
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                let target_seq = payload
                    .get("target_seq")
                    .or_else(|| payload.get("seq"))
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                commits.insert(
                    version,
                    json!({
                        "version": version,
                        "prev_version": prev_version,
                        "target_seq": target_seq,
                        "created_at_ms": payload.get("ts_ms").and_then(Value::as_i64).unwrap_or(0),
                        "entity_id": payload.get("entity_id").and_then(Value::as_str).unwrap_or(""),
                        "action": payload.get("action").and_then(Value::as_str).unwrap_or(""),
                    }),
                );
                allocated_max_version = allocated_max_version.max(version);
                current_version = version;
                active_tip_version = version;
            }
            "UNDO_APPLIED" | "REDO_APPLIED" => {
                current_version = payload
                    .get("to_version")
                    .and_then(Value::as_i64)
                    .unwrap_or(current_version);
            }
            "BRANCH_DROPPED" => {
                let start_version = payload
                    .get("drop_from_version")
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                let end_version = payload
                    .get("drop_to_version")
                    .and_then(Value::as_i64)
                    .unwrap_or(-1);
                let start_seq = payload
                    .get("drop_from_seq")
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                let end_seq = payload
                    .get("drop_to_seq")
                    .and_then(Value::as_i64)
                    .unwrap_or(-1);
                if end_version >= start_version
                    && start_version >= 0
                    && end_seq >= start_seq
                    && start_seq >= 0
                {
                    dropped_ranges.push(json!({
                        "start_version": start_version,
                        "end_version": end_version,
                        "start_seq": start_seq,
                        "end_seq": end_seq,
                    }));
                    if (start_version..=end_version).contains(&active_tip_version) {
                        active_tip_version = current_version;
                    }
                }
            }
            _ => {}
        }
    }
    fn lineage(commits: &BTreeMap<i64, Value>, version: i64) -> Result<Vec<Value>, String> {
        let mut path = Vec::new();
        let mut seen = BTreeSet::new();
        let mut current = version;
        while current > 0 {
            if !seen.insert(current) {
                return Err(format!(
                    "Workflow design lineage loop detected at version={current}"
                ));
            }
            let commit = commits.get(&current).cloned().ok_or_else(|| {
                format!("Workflow design history missing committed version={current}")
            })?;
            current = commit["prev_version"].as_i64().unwrap_or(0);
            path.push(commit);
        }
        path.reverse();
        path.insert(
            0,
            json!({"version": 0, "prev_version": 0, "target_seq": 0, "created_at_ms": 0}),
        );
        Ok(path)
    }
    let active_lineage = lineage(&commits, active_tip_version)?;
    let selected_lineage = lineage(&commits, current_version)?;
    let active_ids = active_lineage
        .iter()
        .filter_map(|item| item["version"].as_i64())
        .collect::<Vec<_>>();
    let versions = active_lineage
        .iter()
        .map(|item| {
            json!({
                "version": item["version"],
                "seq": item["target_seq"],
                "created_at_ms": item["created_at_ms"],
            })
        })
        .collect::<Vec<_>>();
    let selected_versions = selected_lineage
        .iter()
        .map(|item| {
            json!({
                "version": item["version"],
                "seq": item["target_seq"],
                "created_at_ms": item["created_at_ms"],
                "prev_version": item["prev_version"],
                "target_seq": item["target_seq"],
            })
        })
        .collect::<Vec<_>>();
    let current_seq = commits
        .get(&current_version)
        .and_then(|item| item["target_seq"].as_i64())
        .unwrap_or(0);
    let can_redo = active_ids
        .iter()
        .position(|version| *version == current_version)
        .is_some_and(|index| index + 1 < active_ids.len());
    let keep_from = timeline.len().saturating_sub(500);
    Ok(json!({
        "workflow_id": workflow_id,
        "namespace": format!("wf_design:{workflow_id}"),
        "current_version": current_version,
        "active_tip_version": active_tip_version,
        "max_version": active_tip_version,
        "allocated_max_version": allocated_max_version,
        "current_seq": current_seq,
        "can_undo": current_version > 0,
        "can_redo": can_redo,
        "versions": versions,
        "selected_versions": selected_versions,
        "dropped_ranges": dropped_ranges,
        "latest_seq": latest_seq,
        "timeline": &timeline[keep_from..],
        "commits": commits,
        "materialization_status": materialization_status,
    }))
}

fn workflow_design_projection_write(history: &Value) -> NamedProjectionWrite {
    let selected = history["selected_versions"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let mut versions = selected.clone();
    let selected_ids = selected
        .iter()
        .filter_map(|item| item["version"].as_i64())
        .collect::<BTreeSet<_>>();
    if let Some(active) = history["versions"].as_array() {
        for item in active {
            let version = item["version"].as_i64().unwrap_or(0);
            if selected_ids.contains(&version) {
                continue;
            }
            let commit = &history["commits"][version.to_string()];
            versions.push(json!({
                "version": version,
                "prev_version": commit["prev_version"].as_i64().unwrap_or(0),
                "target_seq": item["seq"].as_i64().unwrap_or(0),
                "created_at_ms": item["created_at_ms"].as_i64().unwrap_or(0),
            }));
        }
    }
    let payload = json!({
        "current_version": history["current_version"],
        "active_tip_version": history["active_tip_version"],
        "snapshot_schema_version": 1,
        "versions": versions,
        "dropped_ranges": history["dropped_ranges"],
    });
    NamedProjectionWrite {
        payload: payload.as_object().cloned().unwrap_or_default(),
        last_authoritative_seq: history["latest_seq"].as_i64().unwrap_or(0),
        last_materialized_seq: history["current_seq"].as_i64().unwrap_or(0),
        projection_schema_version: 1,
        // Rust owns event/history writes in this slice, but Python still owns the
        // legacy graph projection.  This marker makes rollback lazily rebuild it.
        materialization_status: "rust_event_only".to_owned(),
    }
}

fn capability_allowed(
    request: &ApiEffectRequest,
    shared: &SharedCapabilityState,
    action: &str,
    required: &[&str],
) -> bool {
    let mut state = shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    audit_capability(&request.principal, &mut state, action, required)
}

#[derive(Clone)]
pub struct SqliteRunApplicationService {
    store: SqliteStore,
    max_queue: usize,
    capabilities: SharedCapabilityState,
}

fn runtime_max_queue() -> usize {
    std::env::var("KOGWISTAR_RUNTIME_MAX_QUEUE")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(128)
}

impl SqliteRunApplicationService {
    fn design_events(&self, workflow_id: &str) -> Result<Vec<EntityEvent>, String> {
        let namespace = format!("wf_design:{workflow_id}");
        self.store
            .replay_raw_events(&namespace, 0, usize::MAX)
            .map(|events| {
                events
                    .into_iter()
                    .map(|event| EntityEvent {
                        namespace: event.namespace,
                        seq: event.seq,
                        event_id: event.event_id,
                        entity_kind: event.entity_kind,
                        entity_id: event.entity_id,
                        op: event.op,
                        payload: serde_json::from_str(&event.payload_json).unwrap_or(Value::Null),
                    })
                    .collect()
            })
            .map_err(|error| error.to_string())
    }

    fn workflow_design_effect(
        &self,
        request: &ApiEffectRequest,
        workflow_id: &str,
        route: DesignRoute,
    ) -> ApiEffectResponse {
        let required = if matches!(route, DesignRoute::Graph | DesignRoute::History) {
            "workflow.design.inspect"
        } else {
            "workflow.design.write"
        };
        if !capability_allowed(request, &self.capabilities, required, &[required]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                format!("Workflow design action requires {required} capability"),
            );
        }
        let events = match self.design_events(workflow_id) {
            Ok(events) => events,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        if route == DesignRoute::History {
            return self.workflow_design_history(request, workflow_id);
        }
        if route == DesignRoute::Graph {
            return match workflow_design_graph_value(workflow_id, &events) {
                Ok(value) => json_effect(StatusCode::OK, value),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_WORKFLOW_DESIGN_HISTORY_INVALID",
                    error,
                ),
            };
        }
        let input = match parse_design_input(request) {
            Ok(input) => input,
            Err(response) => return response,
        };
        let designer = match designer_id(request, &input) {
            Ok(value) => value,
            Err(response) => return response,
        };
        let history = match workflow_design_history_value(workflow_id, events.clone(), "ready") {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_WORKFLOW_DESIGN_HISTORY_INVALID",
                    error,
                );
            }
        };
        let current = history["current_version"].as_i64().unwrap_or(0);
        let active_tip = history["active_tip_version"].as_i64().unwrap_or(0);
        let namespace = format!("wf_design:{workflow_id}");
        let timestamp = now_ms();

        let outcome = self.store.transaction(|uow| {
            match route {
                DesignRoute::Undo | DesignRoute::Redo => {
                    let active = history["versions"].as_array().cloned().unwrap_or_default();
                    let ids = active.iter().filter_map(|item| item["version"].as_i64()).collect::<Vec<_>>();
                    let target = if route == DesignRoute::Undo {
                        history["selected_versions"].as_array().and_then(|items| items.iter().find(|item| item["version"].as_i64() == Some(current)))
                            .and_then(|item| item["prev_version"].as_i64()).unwrap_or(0)
                    } else {
                        ids.iter().position(|value| *value == current).and_then(|index| ids.get(index + 1)).copied().unwrap_or(current)
                    };
                    if target == current || (route == DesignRoute::Undo && current == 0) {
                        let mut value = history.clone(); value["status"] = json!("noop"); return Ok(value);
                    }
                    let op = if route == DesignRoute::Undo { "UNDO_APPLIED" } else { "REDO_APPLIED" };
                    uow.append_raw_entity_event(&namespace, SqliteNewRawEntityEvent {
                        event_id: uuid::Uuid::new_v4().to_string(), entity_kind: "design_control".to_owned(),
                        entity_id: workflow_id.to_owned(), op: op.to_owned(),
                        payload_json: json!({"designer_id": designer, "source": "rest", "ts_ms": timestamp,
                            "from_version": current, "to_version": target,
                            "target_seq": history["commits"][target.to_string()]["target_seq"].as_i64().unwrap_or(0)}).to_string(),
                    })?;
                    let folded_events = uow.replay_raw_events(&namespace, 0, usize::MAX)?.into_iter().map(|event| EntityEvent {
                        namespace: event.namespace, seq: event.seq, event_id: event.event_id, entity_kind: event.entity_kind,
                        entity_id: event.entity_id, op: event.op, payload: serde_json::from_str(&event.payload_json).unwrap_or(Value::Null),
                    }).collect::<Vec<_>>();
                    let mut folded = workflow_design_history_value(workflow_id, folded_events, "ready")
                        .map_err(kogwistar_store_sqlite::SqliteStoreError::TransactionAborted)?;
                    uow.replace_named_projection("workflow_design", workflow_id, workflow_design_projection_write(&folded))?;
                    folded["status"] = json!("ok"); Ok(folded)
                }
                DesignRoute::NodeUpsert | DesignRoute::NodeDelete(_) | DesignRoute::EdgeUpsert | DesignRoute::EdgeDelete(_) => {
                    let before = graph_at_version(&events, current);
                    let mut graph = before.clone();
                    let (action, entity_id, deleted) = mutate_design_graph(workflow_id, &route, &input, &mut graph)
                        .map_err(|response| kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(String::from_utf8_lossy(&response.body).into_owned()))?;
                    let branch = current < active_tip;
                    if branch {
                        let dropped = history["versions"].as_array().cloned().unwrap_or_default().into_iter()
                            .filter(|item| item["version"].as_i64().unwrap_or(0) > current).collect::<Vec<_>>();
                        uow.append_raw_entity_event(&namespace, SqliteNewRawEntityEvent {
                            event_id: uuid::Uuid::new_v4().to_string(), entity_kind: "design_control".to_owned(), entity_id: workflow_id.to_owned(),
                            op: "BRANCH_DROPPED".to_owned(), payload_json: json!({"designer_id": designer, "source": "rest", "ts_ms": timestamp,
                                "drop_from_version": dropped.first().and_then(|item| item["version"].as_i64()).unwrap_or(current + 1),
                                "drop_to_version": dropped.last().and_then(|item| item["version"].as_i64()).unwrap_or(active_tip),
                                "drop_from_seq": dropped.first().and_then(|item| item["seq"].as_i64()).unwrap_or(0),
                                "drop_to_seq": dropped.last().and_then(|item| item["seq"].as_i64()).unwrap_or(0)}).to_string(),
                        })?;
                    }
                    let version = history["allocated_max_version"].as_i64().unwrap_or(0) + 1;
                    let entity = design_entity_event(&action, &entity_id, &before, &graph);
                    let data_event = uow.append_raw_entity_event(&namespace, SqliteNewRawEntityEvent {
                        event_id: entity.event_id, entity_kind: entity.entity_kind, entity_id: entity.entity_id,
                        op: entity.op, payload_json: entity.payload.to_string(),
                    })?;
                    uow.append_raw_entity_event(&namespace, SqliteNewRawEntityEvent {
                        event_id: uuid::Uuid::new_v4().to_string(), entity_kind: "design_control".to_owned(), entity_id: workflow_id.to_owned(),
                        op: "MUTATION_COMMITTED".to_owned(), payload_json: json!({"designer_id": designer, "source": "rest", "ts_ms": timestamp,
                            "action": action, "entity_id": entity_id, "version": version, "prev_version": current,
                            "target_seq": data_event.event.seq}).to_string(),
                    })?;
                    uow.put_workflow_design_delta(workflow_id, WorkflowDesignDeltaWrite {
                        version, prev_version: current, target_seq: data_event.event.seq,
                        forward_json: visible_delta(&before, &graph).to_string(),
                        inverse_json: visible_delta(&graph, &before).to_string(), schema_version: 1,
                    })?;
                    if version % 50 == 0 {
                        uow.put_workflow_design_snapshot(workflow_id, WorkflowDesignSnapshotWrite {
                            version, seq: data_event.event.seq, payload_json: graph.to_string(), schema_version: 1,
                        })?;
                    }
                    let folded_events = uow.replay_raw_events(&namespace, 0, usize::MAX)?.into_iter().map(|event| EntityEvent {
                        namespace: event.namespace, seq: event.seq, event_id: event.event_id, entity_kind: event.entity_kind,
                        entity_id: event.entity_id, op: event.op, payload: serde_json::from_str(&event.payload_json).unwrap_or(Value::Null),
                    }).collect::<Vec<_>>();
                    let folded = workflow_design_history_value(workflow_id, folded_events, "ready")
                        .map_err(kogwistar_store_sqlite::SqliteStoreError::TransactionAborted)?;
                    uow.replace_named_projection("workflow_design", workflow_id, workflow_design_projection_write(&folded))?;
                    Ok(json!({"workflow_id": workflow_id, "namespace": namespace,
                        if deleted { if action == "node_delete" { "node_id" } else { "edge_id" } } else if action == "node_upsert" { "node_id" } else { "edge_id" }: entity_id,
                        "designer_id": designer, "deleted": deleted, "version": folded["current_version"],
                        "seq": data_event.event.seq, "can_undo": folded["can_undo"], "can_redo": folded["can_redo"]}))
                }
                _ => unreachable!(),
            }
        });
        match outcome {
            Ok(value) => json_effect(StatusCode::OK, value),
            Err(error)
                if error.to_string().contains("KOGWISTAR_NODE_NOT_FOUND")
                    || error.to_string().contains("KOGWISTAR_EDGE_NOT_FOUND") =>
            {
                effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_DESIGN_ENTITY_NOT_FOUND",
                    error.to_string(),
                )
            }
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_WORKFLOW_DESIGN_CONFLICT",
                error.to_string(),
            ),
        }
    }

    fn workflow_design_history(
        &self,
        request: &ApiEffectRequest,
        workflow_id: &str,
    ) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "workflow.design.inspect",
            &["workflow.design.inspect"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Inspecting workflow design requires workflow.design.inspect capability",
            );
        }
        let events = match self.design_events(workflow_id) {
            Ok(events) => events,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let history = match workflow_design_history_value(workflow_id, events, "ready") {
            Ok(history) => history,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_WORKFLOW_DESIGN_HISTORY_INVALID",
                    error,
                );
            }
        };
        match self.store.replace_named_projection(
            "workflow_design",
            workflow_id,
            workflow_design_projection_write(&history),
        ) {
            Ok(()) => json_effect(StatusCode::OK, history),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    fn service_read(&self, request: &ApiEffectRequest) -> Option<ApiEffectResponse> {
        if request.method != "GET" {
            return None;
        }
        let path = request
            .path_and_query
            .split('?')
            .next()
            .unwrap_or(&request.path_and_query);
        if path != "/api/workflow/services" && service_route(&request.path_and_query).is_none() {
            return None;
        }
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.inspect",
            &["service.inspect", "project_view"],
        ) {
            return Some(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Inspecting service requires service.inspect capability",
            ));
        }
        if path == "/api/workflow/services" {
            let limit =
                query_integer(&request.path_and_query, "limit", 200).clamp(0, 10_000) as usize;
            return Some(
                match self.store.list_named_projections("service_registry") {
                    Ok(rows) => {
                        let mut services = rows
                            .into_iter()
                            .map(|row| Value::Object(row.payload))
                            .collect::<Vec<_>>();
                        services.sort_by(|left, right| {
                            left["service_id"]
                                .as_str()
                                .cmp(&right["service_id"].as_str())
                        });
                        services.truncate(limit);
                        json_effect(StatusCode::OK, json!({"services": services}))
                    }
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                },
            );
        }
        let (service_id, route) =
            service_route(&request.path_and_query).expect("validated service route");
        if route == ServiceRoute::Events {
            let limit =
                query_integer(&request.path_and_query, "limit", 500).clamp(1, 10_000) as usize;
            return Some(
                match self.store.replay_raw_events("workflow", 0, usize::MAX) {
                    Ok(events) => json_effect(
                        StatusCode::OK,
                        json!({
                            "service_id": service_id,
                            "events": service_event_values(
                                events.into_iter().map(|event| (
                                    event.seq,
                                    event.entity_kind,
                                    event.entity_id,
                                    event.op,
                                    event.payload_json,
                                )),
                                &service_id,
                                limit,
                            ),
                        }),
                    ),
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                },
            );
        }
        Some(
            match self
                .store
                .get_named_projection("service_registry", &service_id)
            {
                Ok(Some(row)) => json_effect(StatusCode::OK, Value::Object(row.payload)),
                Ok(None) => effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_SERVICE_NOT_FOUND",
                    format!("Unknown service_id: {service_id}"),
                ),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            },
        )
    }

    fn repair_orphaned_messages(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "project_view",
            &["project_view"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Repairing claimed messages requires project_view capability",
            );
        }
        let inbox_id = query_string(&request.path_and_query, "inbox_id");
        let limit = query_integer(&request.path_and_query, "limit", 100).clamp(0, 10_000) as usize;
        match self.store.repair_orphaned_claimed_lane_messages(
            "workflow",
            inbox_id.as_deref(),
            limit,
        ) {
            Ok(repaired) => json_effect(StatusCode::OK, json!({"repaired_message_ids": repaired})),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    fn replay_dead_letter(&self, request: &ApiEffectRequest, run_id: &str) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.manage",
            &["service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Replaying dead letters requires service.manage capability",
            );
        }
        let lanes = match self.store.list_projected_lane_messages(LaneMessageFilter {
            namespace: Some("workflow".to_owned()),
            status: Some("dead-letter".to_owned()),
            limit: 10_000,
            ..LaneMessageFilter::default()
        }) {
            Ok(lanes) => lanes,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let Some(lane) = lanes
            .into_iter()
            .find(|lane| lane.run_id.as_deref() == Some(run_id))
        else {
            return json_effect(StatusCode::OK, json!({"run_id": run_id, "replayed": false}));
        };
        match self
            .store
            .update_projected_lane_message_status(&lane.message_id, "pending", None)
        {
            Ok(()) => json_effect(
                StatusCode::OK,
                json!({
                    "run_id": run_id,
                    "replayed": true,
                    "dead_letter": lane_progress_value(&lane),
                }),
            ),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    fn resume_runtime_run(&self, run_id: &str, request: &ApiEffectRequest) -> ApiEffectResponse {
        let input: RuntimeResumeRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        match self.store.resume_recorded_runtime_token(
            run_id,
            &input.workflow_id,
            &input.conversation_id,
            &input.suspended_node_id,
            &input.suspended_token_id,
            Some(input.client_result),
        ) {
            Ok(result) => json_effect(
                StatusCode::OK,
                serde_json::to_value(result).expect("recorded runtime result serializes"),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_RESUME_CONFLICT",
                error.to_string(),
            ),
        }
    }

    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        SqliteStore::open(path)
            .map(|store| Self {
                store,
                max_queue: runtime_max_queue(),
                capabilities: Arc::new(Mutex::new(CapabilityState::default())),
            })
            .map_err(|error| error.to_string())
    }

    pub fn with_max_queue(mut self, max_queue: usize) -> Self {
        self.max_queue = max_queue.max(1);
        self
    }

    fn get_run(&self, run_id: &str) -> Result<Option<ServerRun>, String> {
        self.store
            .get_server_run(run_id)
            .map_err(|error| error.to_string())
    }

    fn submit_run(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let input: RuntimeSubmitRun = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let workflow_id = input.workflow_id.trim();
        let conversation_id = input.conversation_id.trim();
        if workflow_id.is_empty() || conversation_id.is_empty() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "workflow_id and conversation_id are required",
            );
        }
        let run_id = input
            .run_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let turn_node_id = input
            .turn_node_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("wf_turn|{run_id}"));
        let message_id = format!("lane|{}", uuid::Uuid::new_v4());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|value| value.as_secs() as i64)
            .unwrap_or(0);
        let priority_class = if input.priority_class.is_empty() {
            default_runtime_priority_class()
        } else {
            input.priority_class.clone()
        };
        let runtime_kind = if input.runtime_kind.is_empty() {
            default_runtime_kind()
        } else {
            input.runtime_kind.clone()
        };
        let event_payload = json!({
            "run_id": run_id,
            "run_kind": "workflow_runtime",
            "conversation_id": conversation_id,
            "workflow_id": workflow_id,
            "status": "queued",
            "turn_node_id": turn_node_id,
        });
        let graph_plan = exact_runtime_graph_plan(
            self.store
                .get_named_projection("workflow_design", workflow_id)
                .ok()
                .flatten(),
            self.store
                .get_workflow_design_snapshot(workflow_id, i64::MAX, 1)
                .ok()
                .flatten(),
        );
        if graph_plan
            .as_ref()
            .is_some_and(|plan| submitted_runtime_graph_conflicts(plan, &input))
        {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_GRAPH_CONFLICT",
                "submitted frozen runtime graph differs from authoritative workflow design",
            );
        }
        let effective_routes = graph_plan
            .as_ref()
            .map(|plan| plan.routes.clone())
            .unwrap_or(input.runtime_routes);
        let effective_join_node_ids = graph_plan
            .as_ref()
            .map(|plan| plan.join_node_ids.clone())
            .unwrap_or(input.join_node_ids);
        let effective_start_join_mask = graph_plan
            .as_ref()
            .map(|plan| plan.start_join_mask)
            .unwrap_or(input.start_join_mask);
        let effective_start_node_id = graph_plan
            .as_ref()
            .map(|plan| plan.start_node_id.clone())
            .or_else(|| {
                input
                    .start_node_id
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "start".to_owned());
        let effective_node_ops = graph_plan
            .as_ref()
            .map(|plan| plan.node_ops.clone())
            .unwrap_or(input.node_ops);
        // Even a bare transport-level submission must hand the Python worker a
        // frozen operation.  Later continuation lanes already use this same
        // no-op fallback; leaving the admission lane as JSON null made the
        // worker correctly reject an otherwise valid durable run.
        let effective_start_op = effective_node_ops
            .get(&effective_start_node_id)
            .cloned()
            .unwrap_or_else(|| "noop".to_owned());
        let mut initial_state = input.initial_state.clone();
        if !effective_routes.is_empty() {
            initial_state.insert(
                "_rt_routes".to_owned(),
                serde_json::to_value(&effective_routes).expect("static runtime routes serialize"),
            );
        }
        if !effective_node_ops.is_empty() {
            initial_state.insert(
                "_rt_node_ops".to_owned(),
                serde_json::to_value(&effective_node_ops).expect("runtime node ops serialize"),
            );
        }
        if let Some(token_budget) = input.token_budget {
            initial_state.insert("token_budget".to_owned(), json!(token_budget));
        }
        if let Some(time_budget_ms) = input.time_budget_ms {
            initial_state.insert("time_budget_ms".to_owned(), json!(time_budget_ms));
        }
        let worker_payload = json!({
            "contract_version": 1,
            "kind": "workflow.run.execute",
            "run_id": run_id,
            "workflow_id": workflow_id,
            "conversation_id": conversation_id,
            "turn_node_id": turn_node_id,
            "user_id": input.user_id,
            "initial_state": initial_state,
            "priority_class": priority_class,
            "token_budget": input.token_budget,
            "time_budget_ms": input.time_budget_ms,
            "runtime_kind": runtime_kind,
            "join_node_ids": effective_join_node_ids,
            "start_join_mask": effective_start_join_mask,
            "start_node_id": effective_start_node_id,
            "op": effective_start_op,
            "runtime_routes": effective_routes,
        });
        if let Ok(Some(existing)) = self.store.get_server_run(&run_id) {
            let lanes = self
                .store
                .list_projected_lane_messages(LaneMessageFilter {
                    namespace: Some("workflow".to_owned()),
                    inbox_id: Some("workflow-runtime".to_owned()),
                    correlation_id: Some(run_id.clone()),
                    limit: 10,
                    ..LaneMessageFilter::default()
                })
                .unwrap_or_default();
            return match runtime_submit_retry_response(&existing, &lanes, &worker_payload) {
                Ok(Some(value)) => json_effect(StatusCode::ACCEPTED, value),
                Ok(None) | Err(_) => effect_error(
                    StatusCode::CONFLICT,
                    "KOGWISTAR_RUNTIME_ADMISSION_CONFLICT",
                    format!("run_id {run_id:?} already exists with different admission state"),
                ),
            };
        }
        let max_queue = self.max_queue;
        let outcome = self.store.immediate_transaction(|uow| {
            let active_count = ["pending", "claimed"]
                .into_iter()
                .map(|status| {
                    uow.list_projected_lane_messages(&LaneMessageFilter {
                        namespace: Some("workflow".to_owned()),
                        inbox_id: Some("workflow-runtime".to_owned()),
                        status: Some(status.to_owned()),
                        limit: max_queue,
                        ..LaneMessageFilter::default()
                    })
                    .map(|lanes| lanes.len())
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .sum::<usize>();
            if active_count >= max_queue {
                return Ok(false);
            }
            uow.create_server_run(ServerRunCreate {
                run_id: run_id.clone(),
                conversation_id: conversation_id.to_owned(),
                workflow_id: workflow_id.to_owned(),
                user_id: input.user_id.clone(),
                user_turn_node_id: turn_node_id.clone(),
                status: "queued".to_owned(),
            })?;
            let created_event = uow.append_server_run_event(
                &run_id,
                "run.created",
                serde_json::to_string(&event_payload)?,
            )?;
            let mut persisted_worker_payload = worker_payload.clone();
            persisted_worker_payload["expected_event_seq"] = json!(created_event.seq);
            uow.project_lane_message(NewProjectedLaneMessage {
                message_id: message_id.clone(),
                namespace: "workflow".to_owned(),
                purpose: "system".to_owned(),
                inbox_id: "workflow-runtime".to_owned(),
                conversation_id: conversation_id.to_owned(),
                recipient_id: "python-worker".to_owned(),
                sender_id: "rust-scheduler".to_owned(),
                msg_type: "workflow.run.execute".to_owned(),
                status: "pending".to_owned(),
                created_at: now,
                available_at: now,
                run_id: Some(run_id.clone()),
                step_id: Some(effective_start_node_id),
                correlation_id: Some(run_id.clone()),
                payload_json: Some(serde_json::to_string(&persisted_worker_payload)?),
                error_json: None,
            })?;
            Ok(true)
        });
        match outcome {
            Ok(false) => {
                return json_effect(
                    StatusCode::TOO_MANY_REQUESTS,
                    json!({
                        "admission": if matches!(priority_class.as_str(), "background" | "batch") { "rejected" } else { "deferred" },
                        "reason": "queue_full",
                        "max_queue": max_queue,
                    }),
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
            Ok(true) => {}
        }
        json_effect(
            StatusCode::ACCEPTED,
            json!({
                "run_id": run_id,
                "conversation_id": conversation_id,
                "workflow_id": workflow_id,
                "turn_node_id": turn_node_id,
                "status": "queued",
                "priority_class": priority_class,
                "token_budget": input.token_budget,
                "time_budget_ms": input.time_budget_ms,
                "admission": "accepted",
                "lane_message_id": message_id,
            }),
        )
    }

    fn claim_runtime_work(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let input: RuntimeClaimRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if input.claimed_by.trim().is_empty() || input.limit == 0 || input.lease_seconds <= 0 {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "claimed_by, positive limit, and positive lease_seconds are required",
            );
        }
        let outcome = self.store.immediate_transaction(|uow| {
            let claimed = if let Some(run_id) = input.run_id.as_deref() {
                uow.claim_projected_lane_messages_for_run(
                    "workflow",
                    "workflow-runtime",
                    run_id,
                    &input.claimed_by,
                    input.limit,
                    input.lease_seconds,
                )?
            } else {
                uow.claim_projected_lane_messages(
                    "workflow",
                    "workflow-runtime",
                    &input.claimed_by,
                    input.limit,
                    input.lease_seconds,
                )?
            };
            let mut values = Vec::new();
            for lane in claimed {
                let payload: Value =
                    serde_json::from_str(lane.payload_json.as_deref().unwrap_or("{}"))?;
                let run_id = lane.run_id.clone().unwrap_or_default();
                if uow
                    .get_server_run(&run_id)?
                    .is_some_and(|run| run.cancel_requested)
                {
                    uow.update_projected_lane_message_status(&lane.message_id, "cancelled", None)?;
                    continue;
                }
                let starts_runtime = lane.msg_type == "workflow.run.execute"
                    && payload["runtime_started"].as_bool() != Some(true);
                let expected_event_seq = if starts_runtime {
                    uow.apply_recorded_runtime_transition(
                        runtime_start_transition(&payload, &run_id),
                        false,
                    )?
                    .event_seq
                } else {
                    payload["expected_event_seq"].as_i64().ok_or_else(|| {
                        kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(format!(
                            "continuation lane message {:?} lacks expected_event_seq",
                            lane.message_id
                        ))
                    })?
                };
                let mut claimed_payload = payload.clone();
                claimed_payload["expected_event_seq"] = json!(expected_event_seq);
                if lane.msg_type == "workflow.run.execute" {
                    claimed_payload["runtime_started"] = json!(true);
                }
                uow.update_projected_lane_message_payload(
                    &lane.message_id,
                    serde_json::to_string(&claimed_payload)?,
                )?;
                values.push(runtime_claimed_work(
                    lane,
                    &input.claimed_by,
                    run_id,
                    claimed_payload,
                    expected_event_seq,
                ));
            }
            Ok(values)
        });
        match outcome {
            Ok(work) => json_effect(StatusCode::OK, json!({"work": work})),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_CLAIM_CONFLICT",
                error.to_string(),
            ),
        }
    }

    fn apply_runtime_result(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let input: RuntimeWorkerResultRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if input.transition.is_some() == input.effect.is_some() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "exactly one of transition or effect is required",
            );
        }
        let outcome = self.store.immediate_transaction(|uow| {
            let handoff = input.handoff;
            let usage = input
                .effect
                .as_ref()
                .and_then(|effect| effect.usage.clone());
            let trace_events = input
                .effect
                .as_ref()
                .map(|effect| effect.trace_events.clone())
                .unwrap_or_default();
            let (applied, transition) = if let Some(transition) = input.transition {
                let applied = uow.apply_claimed_recorded_runtime_transition(
                    handoff,
                    transition.clone(),
                    false,
                )?;
                (applied, transition)
            } else {
                let effect = input.effect.expect("effect checked above");
                let result = effect.result.clone();
                let effect_id = effect.effect_id.clone();
                let applied = uow.apply_claimed_recorded_worker_effect(handoff.clone(), effect)?;
                let state = applied.reduced.state.clone();
                (
                    applied,
                    RecordedRuntimeTransition {
                        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                        transition_id: effect_id,
                        expected_event_seq: 0,
                        kind: RecordedTransitionKind::RecordedStepSuccess,
                        run_id: handoff.run_id,
                        workflow_id: state.workflow_id.clone(),
                        conversation_id: state.conversation_id.clone(),
                        user_id: None,
                        user_turn_node_id: None,
                        step_seq: state.last_step_seq,
                        node_id: state.last_node_id.clone(),
                        token_id: state.last_token_id.clone(),
                        parent_token_id: state.last_parent_token_id.clone(),
                        initial_state: None,
                        state_update: Vec::new(),
                        update: None,
                        state_schema: serde_json::Map::new(),
                        frontier: None,
                        result,
                        wait_reason: None,
                        resume_payload: None,
                        errors: Vec::new(),
                    },
                )
            };
            let frontier = &applied.reduced.state.frontier;
            let telemetry_run_id = transition.run_id.clone();
            let telemetry_step_id = transition.node_id.clone();
            let telemetry_token_id = transition.token_id.clone();
            let telemetry_effect_id = transition.transition_id.clone();
            let unfinished = !frontier.pending.is_empty()
                || !frontier.suspended.is_empty()
                || frontier.join_outstanding.iter().any(|value| *value != 0)
                || frontier
                    .join_waiters
                    .values()
                    .any(|waiters| !waiters.is_empty());
            let applied_idempotent = applied.idempotent;
            let result = if unfinished || applied.reduced.state.status.is_terminal() {
                for (node_id, join_mask, token_id, parent_token_id) in &frontier.pending {
                    let existing_lanes = uow.list_projected_lane_messages(&LaneMessageFilter {
                        namespace: Some("workflow".to_owned()),
                        inbox_id: Some("workflow-runtime".to_owned()),
                        correlation_id: Some(transition.run_id.clone()),
                        ..LaneMessageFilter::default()
                    })?;
                    if active_runtime_lane_covers_token(&existing_lanes, token_id) {
                        continue;
                    }
                    let next_step_seq = applied.reduced.state.last_step_seq.saturating_add(1);
                    let message_id = format!(
                        "lane|{}",
                        kogwistar_contracts::stable_id(
                            "runtime.worker.request",
                            &[
                                transition.run_id.clone(),
                                token_id.clone(),
                                node_id.clone(),
                                next_step_seq.to_string(),
                            ],
                        )
                    );
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|value| value.as_secs() as i64)
                        .unwrap_or(0);
                    if uow.get_projected_lane_message(&message_id)?.is_some() {
                        continue;
                    }
                    let op = applied
                        .reduced
                        .state
                        .state
                        .get("_rt_node_ops")
                        .and_then(Value::as_object)
                        .and_then(|ops| ops.get(node_id))
                        .cloned()
                        .unwrap_or_else(|| Value::String("noop".to_owned()));
                    let payload = RuntimeStepExecutePayload::from_recorded_state(
                        &applied.reduced.state,
                        RuntimeStepExecuteRequest {
                            node_id: node_id.clone(),
                            op,
                            join_mask: *join_mask,
                            token_id: token_id.clone(),
                            parent_token_id: parent_token_id.clone(),
                            step_seq: next_step_seq,
                            expected_event_seq: applied.event_seq,
                            resume_effect: None,
                        },
                    );
                    uow.project_lane_message(NewProjectedLaneMessage {
                        message_id,
                        namespace: "workflow".to_owned(),
                        purpose: "system".to_owned(),
                        inbox_id: "workflow-runtime".to_owned(),
                        conversation_id: transition.conversation_id.clone(),
                        recipient_id: "python-worker".to_owned(),
                        sender_id: "rust-scheduler".to_owned(),
                        msg_type: "workflow.step.execute".to_owned(),
                        status: "pending".to_owned(),
                        created_at: now,
                        available_at: now,
                        run_id: Some(transition.run_id.clone()),
                        step_id: Some(node_id.clone()),
                        correlation_id: Some(transition.run_id.clone()),
                        payload_json: Some(serde_json::to_string(&payload)?),
                        error_json: None,
                    })?;
                }
                applied
            } else {
                uow.apply_recorded_runtime_transition(
                    RecordedRuntimeTransition {
                        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                        transition_id: format!("complete-{}", transition.transition_id),
                        expected_event_seq: applied.event_seq,
                        kind: RecordedTransitionKind::Complete,
                        run_id: transition.run_id,
                        workflow_id: transition.workflow_id,
                        conversation_id: transition.conversation_id,
                        user_id: None,
                        user_turn_node_id: None,
                        step_seq: transition.step_seq,
                        node_id: transition.node_id,
                        token_id: transition.token_id,
                        parent_token_id: transition.parent_token_id,
                        initial_state: None,
                        state_update: Vec::new(),
                        update: None,
                        state_schema: serde_json::Map::new(),
                        frontier: Some(RuntimeFrontier::default()),
                        result: transition.result,
                        wait_reason: None,
                        resume_payload: None,
                        errors: Vec::new(),
                    },
                    false,
                )?
            };
            if !applied_idempotent {
                if let Some(usage) = usage {
                    uow.append_server_run_event(
                        &telemetry_run_id,
                        "workflow.usage.v1",
                        serde_json::to_string(&json!({
                            "run_id": telemetry_run_id,
                            "step_id": telemetry_step_id,
                            "token_id": telemetry_token_id,
                            "effect_id": telemetry_effect_id,
                            "usage": usage,
                        }))?,
                    )?;
                }
                for (trace_index, trace) in trace_events.into_iter().enumerate() {
                    uow.append_server_run_event(
                        &telemetry_run_id,
                        "workflow.trace.v1",
                        serde_json::to_string(&json!({
                            "run_id": telemetry_run_id,
                            "step_id": telemetry_step_id,
                            "token_id": telemetry_token_id,
                            "effect_id": telemetry_effect_id,
                            "trace_index": trace_index,
                            "trace": trace,
                        }))?,
                    )?;
                }
            }
            Ok(result)
        });
        match outcome {
            Ok(result) => json_effect(
                StatusCode::OK,
                serde_json::to_value(result).expect("recorded runtime result serializes"),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_RESULT_CONFLICT",
                error.to_string(),
            ),
        }
    }
}

#[derive(Clone)]
pub struct PostgresRunApplicationService {
    store: PostgresStore,
    max_queue: usize,
    capabilities: SharedCapabilityState,
}

impl PostgresRunApplicationService {
    async fn syscall_effect(&self, request: &ApiEffectRequest, op: &str) -> ApiEffectResponse {
        let input = match syscall_input(request, op) {
            Ok(input) => input,
            Err(response) => return response,
        };
        let response = match op {
            "spawn_process" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    true,
                    &["spawn_process", "workflow.run.write"],
                ) {
                    syscall_forbidden(op)
                } else {
                    let nested = ApiEffectRequest {
                        contract_version: request.contract_version,
                        method: "POST".to_owned(),
                        path_and_query: "/api/workflow/runs".to_owned(),
                        body: serde_json::to_vec(&input.args).expect("syscall args serialize"),
                        principal: request.principal.clone(),
                    };
                    syscall_result(&input.version, op, self.submit_run(&nested).await)
                }
            }
            "terminate_process" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    true,
                    &["workflow.run.write"],
                ) {
                    syscall_forbidden(op)
                } else if let Some(run_id) = input.args["run_id"].as_str() {
                    match self.store.request_server_run_cancel(run_id).await {
                        Ok(()) => match self.store.get_server_run(run_id).await {
                            Ok(Some(run)) => syscall_result(
                                &input.version,
                                op,
                                json_effect(StatusCode::ACCEPTED, server_run_value(run)),
                            ),
                            Ok(None) => effect_error(
                                StatusCode::NOT_FOUND,
                                "KOGWISTAR_RUN_NOT_FOUND",
                                format!("Unknown run_id: {run_id}"),
                            ),
                            Err(error) => effect_error(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                "KOGWISTAR_STORE_ERROR",
                                error.to_string(),
                            ),
                        },
                        Err(error) => effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        ),
                    }
                } else {
                    effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "run_id is required",
                    )
                }
            }
            "checkpoint" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    false,
                    &["workflow.run.read"],
                ) {
                    syscall_forbidden(op)
                } else if let (Some(run_id), Some(step_seq)) = (
                    input.args["run_id"].as_str(),
                    input.args["step_seq"].as_i64(),
                ) {
                    match self.store.list_server_run_events(run_id, 0, 100_000).await {
                        Ok(events) => syscall_result(
                            &input.version,
                            op,
                            runtime_inspection_effect(
                                request,
                                run_id,
                                &["checkpoints", &step_seq.to_string()],
                                events,
                            ),
                        ),
                        Err(error) => effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        ),
                    }
                } else {
                    effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "run_id and integer step_seq are required",
                    )
                }
            }
            "resume" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    true,
                    &["workflow.run.write"],
                ) {
                    syscall_forbidden(op)
                } else if let Some(run_id) = input.args["run_id"].as_str() {
                    let nested = ApiEffectRequest {
                        contract_version: request.contract_version,
                        method: "POST".to_owned(),
                        path_and_query: format!("/api/workflow/runs/{run_id}/resume"),
                        body: serde_json::to_vec(&input.args).expect("syscall args serialize"),
                        principal: request.principal.clone(),
                    };
                    syscall_result(
                        &input.version,
                        op,
                        self.resume_runtime_run(run_id, &nested).await,
                    )
                } else {
                    effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "run_id is required",
                    )
                }
            }
            "request_approval" => {
                if !syscall_allowed(request, &self.capabilities, op, true, &["approve_action"]) {
                    syscall_forbidden(op)
                } else {
                    request_approval_syscall(request, &self.capabilities, &input)
                }
            }
            _ => unavailable_effect(request.clone()),
        };
        syscall_audit(
            &self.capabilities,
            &input.version,
            op,
            syscall_audit_status(&response),
        );
        response
    }

    async fn trigger_service(
        &self,
        request: &ApiEffectRequest,
        service_id: &str,
    ) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.manage",
            &["service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Triggering service requires service.manage capability",
            );
        }
        let mut input: ServiceTriggerInput = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if let Err(error) = validate_service_trigger(&mut input) {
            return effect_error(StatusCode::BAD_REQUEST, "KOGWISTAR_INVALID_REQUEST", error);
        }
        let timestamp = now_ms();
        let now_seconds = (timestamp / 1000) as i64;
        let max_queue = self.max_queue;
        let service_id_owned = service_id.to_owned();
        let outcome = self
            .store
            .transaction(move |uow| {
                Box::pin(async move {
                    let current = uow
                        .lock_named_projection("service_registry", &service_id_owned)
                        .await?
                        .ok_or_else(|| {
                            kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
                                "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                            )
                        })?;
                    let current_value = Value::Object(current.payload);
                    if let Some(run_id) = current_value["current_child_run_id"]
                        .as_str()
                        .filter(|value| !value.is_empty())
                        && let Some(run) = uow.get_server_run(run_id).await?
                        && !matches!(run.status.as_str(), "succeeded" | "failed" | "cancelled")
                    {
                        return Ok(Some(current_value));
                    }
                    let history = uow.replay_raw_events("workflow", 0, usize::MAX).await?;
                    let definition = latest_service_definition(
                        history.into_iter().map(|event| {
                            (
                                event.seq,
                                event.entity_kind,
                                event.entity_id,
                                event.op,
                                event.payload_json,
                            )
                        }),
                        &service_id_owned,
                    )
                    .ok_or_else(|| {
                        kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
                            "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                        )
                    })?;
                    if service_trigger_is_suppressed(
                        &definition,
                        &current_value,
                        &input.trigger_type,
                        timestamp,
                    ) {
                        return Ok(Some(current_value));
                    }
                    let (target_kind, target_config) = service_definition_runtime(&definition)
                        .map_err(|message| {
                            kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
                                message.to_owned(),
                            )
                        })?;
                    let target_ref = definition["properties"]["target_ref"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned();
                    let mut run_id = None;
                    if target_kind == "workflow" {
                        uow.lock_runtime_queue_admission().await?;
                        let pending = uow
                            .list_projected_lane_messages(&LaneMessageFilter {
                                namespace: Some("workflow".to_owned()),
                                inbox_id: Some("workflow-runtime".to_owned()),
                                status: Some("pending".to_owned()),
                                limit: max_queue,
                                ..LaneMessageFilter::default()
                            })
                            .await?;
                        let claimed = uow
                            .list_projected_lane_messages(&LaneMessageFilter {
                                namespace: Some("workflow".to_owned()),
                                inbox_id: Some("workflow-runtime".to_owned()),
                                status: Some("claimed".to_owned()),
                                limit: max_queue,
                                ..LaneMessageFilter::default()
                            })
                            .await?;
                        if pending.len() + claimed.len() >= max_queue {
                            return Ok(None);
                        }
                        let target = target_config.as_object().cloned().unwrap_or_default();
                        let conversation_id = input
                            .payload
                            .get("conversation_id")
                            .and_then(Value::as_str)
                            .or_else(|| target.get("conversation_id").and_then(Value::as_str))
                            .map(str::to_owned)
                            .unwrap_or_else(|| format!("svc:{service_id_owned}"));
                        let turn_node_id = input
                            .payload
                            .get("turn_node_id")
                            .and_then(Value::as_str)
                            .or_else(|| target.get("turn_node_id").and_then(Value::as_str))
                            .map(str::to_owned)
                            .unwrap_or_else(|| format!("wf_turn|{}", uuid::Uuid::new_v4()));
                        let user_id = input
                            .payload
                            .get("user_id")
                            .and_then(Value::as_str)
                            .or_else(|| target.get("user_id").and_then(Value::as_str))
                            .map(str::to_owned);
                        let mut initial_state = target
                            .get("initial_state")
                            .and_then(Value::as_object)
                            .cloned()
                            .unwrap_or_default();
                        if let Some(overrides) = input
                            .payload
                            .get("initial_state")
                            .and_then(Value::as_object)
                        {
                            initial_state.extend(overrides.clone());
                        }
                        let created_run_id = uuid::Uuid::new_v4().to_string();
                        let message_id = format!("lane|{}", uuid::Uuid::new_v4());
                        let graph_plan = exact_runtime_graph_plan(
                            uow.get_named_projection("workflow_design", &target_ref)
                                .await?,
                            uow.get_workflow_design_snapshot(&target_ref, i64::MAX, 1)
                                .await?,
                        );
                        let runtime_routes = graph_plan
                            .as_ref()
                            .map(|plan| plan.routes.clone())
                            .unwrap_or_default();
                        let join_node_ids = graph_plan
                            .as_ref()
                            .map(|plan| plan.join_node_ids.clone())
                            .unwrap_or_default();
                        let start_join_mask = graph_plan
                            .as_ref()
                            .map(|plan| plan.start_join_mask)
                            .unwrap_or_default();
                        let start_node_id = graph_plan
                            .as_ref()
                            .map(|plan| plan.start_node_id.clone())
                            .unwrap_or_else(|| "start".to_owned());
                        let node_ops = graph_plan
                            .as_ref()
                            .map(|plan| plan.node_ops.clone())
                            .unwrap_or_default();
                        let start_op = node_ops.get(&start_node_id).cloned();
                        if !runtime_routes.is_empty() {
                            initial_state.insert(
                                "_rt_routes".to_owned(),
                                serde_json::to_value(&runtime_routes)
                                    .expect("static runtime routes serialize"),
                            );
                        }
                        if !node_ops.is_empty() {
                            initial_state.insert(
                                "_rt_node_ops".to_owned(),
                                serde_json::to_value(&node_ops)
                                    .expect("runtime node ops serialize"),
                            );
                        }
                        uow.create_server_run(ServerRunCreate {
                            run_id: created_run_id.clone(),
                            conversation_id: conversation_id.clone(),
                            workflow_id: target_ref.clone(),
                            user_id: user_id.clone(),
                            user_turn_node_id: turn_node_id.clone(),
                            status: "queued".to_owned(),
                        })
                        .await?;
                        let created = uow
                            .append_server_run_event(
                                &created_run_id,
                                "run.created",
                                json!({
                                    "run_id": created_run_id,
                                    "run_kind": "workflow_runtime",
                                    "conversation_id": conversation_id,
                                    "workflow_id": target_ref,
                                    "status": "queued",
                                    "turn_node_id": turn_node_id,
                                })
                                .to_string(),
                            )
                            .await?;
                        let priority_class = target
                            .get("priority_class")
                            .and_then(Value::as_str)
                            .unwrap_or("background");
                        let payload = json!({
                            "contract_version": 1,
                            "kind": "workflow.run.execute",
                            "run_id": created_run_id,
                            "workflow_id": target_ref,
                            "conversation_id": conversation_id,
                            "turn_node_id": turn_node_id,
                            "user_id": user_id,
                            "initial_state": initial_state,
                            "priority_class": priority_class,
                            "token_budget": target.get("token_budget"),
                            "time_budget_ms": target.get("time_budget_ms"),
                            "runtime_kind": target.get("runtime_kind").and_then(Value::as_str).unwrap_or("sync"),
                            "join_node_ids": join_node_ids,
                            "start_join_mask": start_join_mask,
                            "start_node_id": start_node_id,
                            "op": start_op,
                            "runtime_routes": runtime_routes,
                            "expected_event_seq": created.seq,
                        });
                        uow.project_lane_message(NewProjectedLaneMessage {
                            message_id,
                            namespace: "workflow".to_owned(),
                            purpose: "system".to_owned(),
                            inbox_id: "workflow-runtime".to_owned(),
                            conversation_id,
                            recipient_id: "python-worker".to_owned(),
                            sender_id: "rust-scheduler".to_owned(),
                            msg_type: "workflow.run.execute".to_owned(),
                            status: "pending".to_owned(),
                            created_at: now_seconds,
                            available_at: now_seconds,
                            run_id: Some(created_run_id.clone()),
                            step_id: Some(start_node_id),
                            correlation_id: Some(created_run_id.clone()),
                            payload_json: Some(payload.to_string()),
                            error_json: None,
                        })
                        .await?;
                        run_id = Some(created_run_id);
                    }
                    let mut triggered_payload = input.payload.clone();
                    triggered_payload.insert(
                        "trigger_type".to_owned(),
                        json!(input.trigger_type),
                    );
                    for (event_type, payload) in [
                        (
                            "service.starting",
                            json!({"trigger_type": input.trigger_type}),
                        ),
                        ("service.triggered", Value::Object(triggered_payload)),
                    ] {
                        let (entity_id, event) = service_event_node(
                            &service_id_owned,
                            event_type,
                            &payload,
                            timestamp,
                        );
                        uow.append_raw_entity_event(
                            "workflow",
                            PostgresNewRawEntityEvent {
                                event_id: uuid::Uuid::new_v4().to_string(),
                                entity_kind: "node".to_owned(),
                                entity_id,
                                op: "ADD".to_owned(),
                                payload_json: event.to_string(),
                            },
                        )
                        .await?;
                    }
                    if let Some(run_id) = &run_id {
                        let (entity_id, event) = service_event_node(
                            &service_id_owned,
                            "service.run_spawned",
                            &json!({
                                "trigger_type": input.trigger_type,
                                "run_id": run_id,
                                "workflow_id": target_ref,
                            }),
                            timestamp,
                        );
                        uow.append_raw_entity_event(
                            "workflow",
                            PostgresNewRawEntityEvent {
                                event_id: uuid::Uuid::new_v4().to_string(),
                                entity_kind: "node".to_owned(),
                                entity_id,
                                op: "ADD".to_owned(),
                                payload_json: event.to_string(),
                            },
                        )
                        .await?;
                    }
                    let history = uow.replay_raw_events("workflow", 0, usize::MAX).await?;
                    let projections = service_projection_values(history.into_iter().map(|event| {
                        (
                            event.seq,
                            event.entity_kind,
                            event.entity_id,
                            event.op,
                            event.payload_json,
                        )
                    }))
                    .map_err(
                        kogwistar_store_postgres::PostgresStoreError::TransactionAborted,
                    )?;
                    let (last_seq, projection) = projections.get(&service_id_owned).ok_or_else(|| {
                        kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
                            "KOGWISTAR_SERVICE_TRIGGER_INVALID".to_owned(),
                        )
                    })?;
                    uow.replace_named_projection(
                        "service_registry",
                        &service_id_owned,
                        service_projection_write(*last_seq, projection),
                    )
                    .await?;
                    Ok(Some(projection.clone()))
                })
            })
            .await;
        match outcome {
            Ok(Some(projection)) => json_effect(StatusCode::OK, projection),
            Ok(None) => effect_error(
                StatusCode::TOO_MANY_REQUESTS,
                "KOGWISTAR_QUEUE_FULL",
                "workflow runtime queue is full",
            ),
            Err(error) if error.to_string().contains("KOGWISTAR_SERVICE_NOT_FOUND") => {
                effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_SERVICE_NOT_FOUND",
                    format!("Unknown service_id: {service_id}"),
                )
            }
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_SERVICE_TRIGGER_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn declare_service(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.manage",
            &["service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Declaring service requires service.manage capability",
            );
        }
        let mut input: ServiceDeclareInput = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if let Err(error) = validate_service_declare(&mut input) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error,
            );
        }
        let timestamp = now_ms();
        let service_id = input.service_id.clone();
        let outcome = self
            .store
            .transaction(move |uow| {
                Box::pin(async move {
                    let (definition_id, definition) = service_definition_node(&input, timestamp);
                    uow.append_raw_entity_event(
                        "workflow",
                        PostgresNewRawEntityEvent {
                            event_id: uuid::Uuid::new_v4().to_string(),
                            entity_kind: "node".to_owned(),
                            entity_id: definition_id,
                            op: "ADD".to_owned(),
                            payload_json: definition.to_string(),
                        },
                    )
                    .await?;
                    let lifecycle_type = if input.enabled {
                        "service.enabled"
                    } else {
                        "service.stopped"
                    };
                    let (event_id, event) = service_event_node(
                        &service_id,
                        lifecycle_type,
                        &json!({"enabled": input.enabled, "autostart": input.autostart}),
                        timestamp,
                    );
                    uow.append_raw_entity_event(
                        "workflow",
                        PostgresNewRawEntityEvent {
                            event_id: uuid::Uuid::new_v4().to_string(),
                            entity_kind: "node".to_owned(),
                            entity_id: event_id,
                            op: "ADD".to_owned(),
                            payload_json: event.to_string(),
                        },
                    )
                    .await?;
                    if input.enabled {
                        let (event_id, event) = service_event_node(
                            &service_id,
                            "service.degraded",
                            &json!({"health_status": "degraded"}),
                            timestamp,
                        );
                        uow.append_raw_entity_event(
                            "workflow",
                            PostgresNewRawEntityEvent {
                                event_id: uuid::Uuid::new_v4().to_string(),
                                entity_kind: "node".to_owned(),
                                entity_id: event_id,
                                op: "ADD".to_owned(),
                                payload_json: event.to_string(),
                            },
                        )
                        .await?;
                    }
                    let history = uow.replay_raw_events("workflow", 0, usize::MAX).await?;
                    let projections = service_projection_values(history.into_iter().map(|event| {
                        (
                            event.seq,
                            event.entity_kind,
                            event.entity_id,
                            event.op,
                            event.payload_json,
                        )
                    }))
                    .map_err(kogwistar_store_postgres::PostgresStoreError::TransactionAborted)?;
                    let (last_seq, projection) = projections.get(&service_id).ok_or_else(|| {
                        kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
                            "KOGWISTAR_SERVICE_DECLARATION_INVALID".to_owned(),
                        )
                    })?;
                    uow.replace_named_projection(
                        "service_registry",
                        &service_id,
                        service_projection_write(*last_seq, projection),
                    )
                    .await?;
                    Ok(projection.clone())
                })
            })
            .await;
        match outcome {
            Ok(projection) => json_effect(StatusCode::OK, projection),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_SERVICE_DECLARATION_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn record_service_heartbeat(
        &self,
        request: &ApiEffectRequest,
        service_id: &str,
    ) -> ApiEffectResponse {
        #[derive(Deserialize)]
        struct HeartbeatInput {
            instance_id: String,
            #[serde(default)]
            payload: serde_json::Map<String, Value>,
        }
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.heartbeat",
            &["service.heartbeat", "service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Recording service heartbeat requires service.heartbeat capability",
            );
        }
        let input: HeartbeatInput = match serde_json::from_slice::<HeartbeatInput>(&request.body) {
            Ok(value) if !value.instance_id.trim().is_empty() => value,
            Ok(_) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    "instance_id must not be empty",
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let timestamp = now_ms();
        let service_id_owned = service_id.to_owned();
        let outcome = self
            .store
            .transaction(move |uow| {
                Box::pin(async move {
                    let current = uow
                        .get_named_projection("service_registry", &service_id_owned)
                        .await?
                        .ok_or_else(|| {
                            kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
                                "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                            )
                        })?;
                    let mut projection = Value::Object(current.payload);
                    let previous_health = projection["health_status"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned();
                    let previous_error = projection["last_error"].clone();
                    let next_error = input
                        .payload
                        .get("last_error")
                        .cloned()
                        .unwrap_or(Value::Null);
                    projection["instance_id"] = json!(input.instance_id);
                    projection["last_heartbeat_ms"] = json!(timestamp);
                    projection["lifecycle_status"] = json!("healthy");
                    projection["health_status"] = json!("healthy");
                    projection["updated_at_ms"] = json!(timestamp);
                    projection["last_error"] = if next_error.is_null() {
                        Value::Null
                    } else {
                        json!(next_error.as_str().unwrap_or_default())
                    };
                    let mut last_seq = current.last_authoritative_seq;
                    if matches!(previous_health.as_str(), "degraded" | "failed" | "stopped") {
                        let (entity_id, event) = service_event_node(
                            &service_id_owned,
                            "service.healthy",
                            &json!({
                                "health_status": "healthy",
                                "instance_id": input.instance_id,
                            }),
                            timestamp,
                        );
                        last_seq = uow
                            .append_raw_entity_event(
                                "workflow",
                                PostgresNewRawEntityEvent {
                                    event_id: uuid::Uuid::new_v4().to_string(),
                                    entity_kind: "node".to_owned(),
                                    entity_id,
                                    op: "ADD".to_owned(),
                                    payload_json: event.to_string(),
                                },
                            )
                            .await?
                            .event
                            .seq;
                    }
                    if previous_error != projection["last_error"]
                        && !projection["last_error"].is_null()
                    {
                        let (entity_id, event) = service_event_node(
                            &service_id_owned,
                            "service.error_changed",
                            &json!({
                                "instance_id": input.instance_id,
                                "last_error": projection["last_error"],
                            }),
                            timestamp,
                        );
                        last_seq = uow
                            .append_raw_entity_event(
                                "workflow",
                                PostgresNewRawEntityEvent {
                                    event_id: uuid::Uuid::new_v4().to_string(),
                                    entity_kind: "node".to_owned(),
                                    entity_id,
                                    op: "ADD".to_owned(),
                                    payload_json: event.to_string(),
                                },
                            )
                            .await?
                            .event
                            .seq;
                    }
                    uow.replace_named_projection(
                        "service_registry",
                        &service_id_owned,
                        service_projection_write(last_seq, &projection),
                    )
                    .await?;
                    Ok(projection)
                })
            })
            .await;
        match outcome {
            Ok(projection) => json_effect(StatusCode::OK, projection),
            Err(error) if error.to_string().contains("KOGWISTAR_SERVICE_NOT_FOUND") => {
                effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_SERVICE_NOT_FOUND",
                    format!("Unknown service_id: {service_id}"),
                )
            }
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_SERVICE_HEARTBEAT_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn set_service_enabled(
        &self,
        request: &ApiEffectRequest,
        service_id: &str,
        enabled: bool,
    ) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.manage",
            &["service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Changing service lifecycle requires service.manage capability",
            );
        }
        let timestamp = now_ms();
        let service_id_owned = service_id.to_owned();
        let outcome = self
            .store
            .transaction(move |uow| {
                Box::pin(async move {
                    let history = uow.replay_raw_events("workflow", 0, usize::MAX).await?;
                    let definition = latest_service_definition(
                        history.into_iter().map(|event| {
                            (
                                event.seq,
                                event.entity_kind,
                                event.entity_id,
                                event.op,
                                event.payload_json,
                            )
                        }),
                        &service_id_owned,
                    )
                    .ok_or_else(|| {
                        kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
                            "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                        )
                    })?;
                    let (definition_id, definition, event_id, event) =
                        service_enabled_nodes(definition, &service_id_owned, enabled, timestamp)
                            .map_err(
                                kogwistar_store_postgres::PostgresStoreError::TransactionAborted,
                            )?;
                    uow.append_raw_entity_event(
                        "workflow",
                        PostgresNewRawEntityEvent {
                            event_id: uuid::Uuid::new_v4().to_string(),
                            entity_kind: "node".to_owned(),
                            entity_id: definition_id,
                            op: "ADD".to_owned(),
                            payload_json: definition.to_string(),
                        },
                    )
                    .await?;
                    uow.append_raw_entity_event(
                        "workflow",
                        PostgresNewRawEntityEvent {
                            event_id: uuid::Uuid::new_v4().to_string(),
                            entity_kind: "node".to_owned(),
                            entity_id: event_id,
                            op: "ADD".to_owned(),
                            payload_json: event.to_string(),
                        },
                    )
                    .await?;
                    let history = uow.replay_raw_events("workflow", 0, usize::MAX).await?;
                    let projections = service_projection_values(history.into_iter().map(|event| {
                        (
                            event.seq,
                            event.entity_kind,
                            event.entity_id,
                            event.op,
                            event.payload_json,
                        )
                    }))
                    .map_err(kogwistar_store_postgres::PostgresStoreError::TransactionAborted)?;
                    let (last_seq, projection) =
                        projections.get(&service_id_owned).ok_or_else(|| {
                            kogwistar_store_postgres::PostgresStoreError::TransactionAborted(
                                "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                            )
                        })?;
                    uow.replace_named_projection(
                        "service_registry",
                        &service_id_owned,
                        service_projection_write(*last_seq, projection),
                    )
                    .await?;
                    Ok(projection.clone())
                })
            })
            .await;
        match outcome {
            Ok(projection) => json_effect(StatusCode::OK, projection),
            Err(error) if error.to_string().contains("KOGWISTAR_SERVICE_NOT_FOUND") => {
                effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_SERVICE_NOT_FOUND",
                    format!("Unknown service_id: {service_id}"),
                )
            }
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_SERVICE_MUTATION_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn repair_service_projections(
        &self,
        request: &ApiEffectRequest,
        service_id: Option<&str>,
    ) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.manage",
            &["service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Repairing service projections requires service.manage capability",
            );
        }
        let events = match self
            .store
            .replay_raw_events("workflow", 0, usize::MAX)
            .await
        {
            Ok(events) => events,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let projections = match service_projection_values(events.into_iter().map(|event| {
            (
                event.seq,
                event.entity_kind,
                event.entity_id,
                event.op,
                event.payload_json,
            )
        })) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_SERVICE_HISTORY_INVALID",
                    error,
                );
            }
        };
        if let Some(service_id) = service_id {
            let Some((last_seq, projection)) = projections.get(service_id) else {
                return effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_SERVICE_NOT_FOUND",
                    format!("Unknown service_id: {service_id}"),
                );
            };
            return match self
                .store
                .replace_named_projection(
                    "service_registry",
                    service_id,
                    service_projection_write(*last_seq, projection),
                )
                .await
            {
                Ok(()) => json_effect(
                    StatusCode::OK,
                    json!({"service_id": service_id, "projection": projection}),
                ),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            };
        }
        let limit = query_integer(&request.path_and_query, "limit", 10_000).clamp(0, 10_000);
        let selected = projections
            .into_iter()
            .take(limit as usize)
            .collect::<Vec<_>>();
        let writes = selected.clone();
        let outcome = self
            .store
            .transaction(move |uow| {
                Box::pin(async move {
                    for (service_id, (last_seq, projection)) in &writes {
                        uow.replace_named_projection(
                            "service_registry",
                            service_id,
                            service_projection_write(*last_seq, projection),
                        )
                        .await?;
                    }
                    Ok(())
                })
            })
            .await;
        match outcome {
            Ok(()) => json_effect(
                StatusCode::OK,
                json!({
                    "repaired_service_ids": selected
                        .into_iter()
                        .map(|(service_id, _)| service_id)
                        .collect::<Vec<_>>(),
                }),
            ),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    async fn design_events(&self, workflow_id: &str) -> Result<Vec<EntityEvent>, String> {
        let namespace = format!("wf_design:{workflow_id}");
        self.store
            .replay_raw_events(&namespace, 0, usize::MAX)
            .await
            .map(|events| {
                events
                    .into_iter()
                    .map(|event| EntityEvent {
                        namespace: event.namespace,
                        seq: event.seq,
                        event_id: event.event_id,
                        entity_kind: event.entity_kind,
                        entity_id: event.entity_id,
                        op: event.op,
                        payload: serde_json::from_str(&event.payload_json).unwrap_or(Value::Null),
                    })
                    .collect()
            })
            .map_err(|error| error.to_string())
    }

    async fn workflow_design_effect(
        &self,
        request: &ApiEffectRequest,
        workflow_id: &str,
        route: DesignRoute,
    ) -> ApiEffectResponse {
        let required = if matches!(route, DesignRoute::Graph | DesignRoute::History) {
            "workflow.design.inspect"
        } else {
            "workflow.design.write"
        };
        if !capability_allowed(request, &self.capabilities, required, &[required]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                format!("Workflow design action requires {required} capability"),
            );
        }
        let events = match self.design_events(workflow_id).await {
            Ok(events) => events,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        if route == DesignRoute::History {
            return self.workflow_design_history(request, workflow_id).await;
        }
        if route == DesignRoute::Graph {
            return match workflow_design_graph_value(workflow_id, &events) {
                Ok(value) => json_effect(StatusCode::OK, value),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_WORKFLOW_DESIGN_HISTORY_INVALID",
                    error,
                ),
            };
        }
        let input = match parse_design_input(request) {
            Ok(input) => input,
            Err(response) => return response,
        };
        let designer = match designer_id(request, &input) {
            Ok(value) => value,
            Err(response) => return response,
        };
        let history = match workflow_design_history_value(workflow_id, events.clone(), "ready") {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_WORKFLOW_DESIGN_HISTORY_INVALID",
                    error,
                );
            }
        };
        let current = history["current_version"].as_i64().unwrap_or(0);
        let active_tip = history["active_tip_version"].as_i64().unwrap_or(0);
        let namespace = format!("wf_design:{workflow_id}");
        let timestamp = now_ms();
        let workflow_id_owned = workflow_id.to_owned();
        let outcome = self.store.transaction(move |uow| Box::pin(async move {
            match route {
                DesignRoute::Undo | DesignRoute::Redo => {
                    let active = history["versions"].as_array().cloned().unwrap_or_default();
                    let ids = active.iter().filter_map(|item| item["version"].as_i64()).collect::<Vec<_>>();
                    let target = if route == DesignRoute::Undo {
                        history["selected_versions"].as_array().and_then(|items| items.iter().find(|item| item["version"].as_i64() == Some(current)))
                            .and_then(|item| item["prev_version"].as_i64()).unwrap_or(0)
                    } else { ids.iter().position(|value| *value == current).and_then(|index| ids.get(index + 1)).copied().unwrap_or(current) };
                    if target == current || (route == DesignRoute::Undo && current == 0) { let mut value = history.clone(); value["status"] = json!("noop"); return Ok(value); }
                    let op = if route == DesignRoute::Undo { "UNDO_APPLIED" } else { "REDO_APPLIED" };
                    uow.append_raw_entity_event(&namespace, PostgresNewRawEntityEvent {
                        event_id: uuid::Uuid::new_v4().to_string(), entity_kind: "design_control".to_owned(), entity_id: workflow_id_owned.clone(), op: op.to_owned(),
                        payload_json: json!({"designer_id": designer, "source":"rest", "ts_ms":timestamp, "from_version":current, "to_version":target,
                            "target_seq":history["commits"][target.to_string()]["target_seq"].as_i64().unwrap_or(0)}).to_string(),
                    }).await?;
                    let folded_events = uow.replay_raw_events(&namespace, 0, usize::MAX).await?.into_iter().map(|event| EntityEvent {
                        namespace:event.namespace,seq:event.seq,event_id:event.event_id,entity_kind:event.entity_kind,entity_id:event.entity_id,op:event.op,
                        payload:serde_json::from_str(&event.payload_json).unwrap_or(Value::Null),
                    }).collect::<Vec<_>>();
                    let mut folded = workflow_design_history_value(&workflow_id_owned, folded_events, "ready")
                        .map_err(kogwistar_store_postgres::PostgresStoreError::TransactionAborted)?;
                    uow.replace_named_projection("workflow_design", &workflow_id_owned, workflow_design_projection_write(&folded)).await?;
                    folded["status"] = json!("ok"); Ok(folded)
                }
                DesignRoute::NodeUpsert | DesignRoute::NodeDelete(_) | DesignRoute::EdgeUpsert | DesignRoute::EdgeDelete(_) => {
                    let before = graph_at_version(&events, current);
                    let mut graph = before.clone();
                    let (action, entity_id, deleted) = mutate_design_graph(&workflow_id_owned, &route, &input, &mut graph)
                        .map_err(|response| kogwistar_store_postgres::PostgresStoreError::TransactionAborted(String::from_utf8_lossy(&response.body).into_owned()))?;
                    let branch = current < active_tip;
                    if branch { uow.append_raw_entity_event(&namespace, PostgresNewRawEntityEvent {
                        event_id:uuid::Uuid::new_v4().to_string(),entity_kind:"design_control".to_owned(),entity_id:workflow_id_owned.clone(),op:"BRANCH_DROPPED".to_owned(),
                        payload_json:json!({"designer_id":designer,"source":"rest","ts_ms":timestamp,"drop_from_version":current+1,"drop_to_version":active_tip,
                            "drop_from_seq":history["versions"].as_array().and_then(|items|items.iter().find(|item|item["version"].as_i64().unwrap_or(0)>current)).and_then(|item|item["seq"].as_i64()).unwrap_or(0),
                            "drop_to_seq":history["versions"].as_array().and_then(|items|items.last()).and_then(|item|item["seq"].as_i64()).unwrap_or(0)}).to_string(),
                    }).await?; }
                    let version = history["allocated_max_version"].as_i64().unwrap_or(0)+1;
                    let entity = design_entity_event(&action, &entity_id, &before, &graph);
                    let data_event = uow.append_raw_entity_event(&namespace, PostgresNewRawEntityEvent {
                        event_id:entity.event_id,entity_kind:entity.entity_kind,entity_id:entity.entity_id,op:entity.op,payload_json:entity.payload.to_string(),
                    }).await?;
                    uow.append_raw_entity_event(&namespace, PostgresNewRawEntityEvent {
                        event_id:uuid::Uuid::new_v4().to_string(),entity_kind:"design_control".to_owned(),entity_id:workflow_id_owned.clone(),op:"MUTATION_COMMITTED".to_owned(),
                        payload_json:json!({"designer_id":designer,"source":"rest","ts_ms":timestamp,"action":action,"entity_id":entity_id,"version":version,
                            "prev_version":current,"target_seq":data_event.event.seq}).to_string(),
                    }).await?;
                    uow.put_workflow_design_delta(&workflow_id_owned, WorkflowDesignDeltaWrite {
                        version,prev_version:current,target_seq:data_event.event.seq,
                        forward_json:visible_delta(&before,&graph).to_string(),inverse_json:visible_delta(&graph,&before).to_string(),schema_version:1,
                    }).await?;
                    if version % 50 == 0 { uow.put_workflow_design_snapshot(&workflow_id_owned, WorkflowDesignSnapshotWrite {
                        version,seq:data_event.event.seq,payload_json:graph.to_string(),schema_version:1,
                    }).await?; }
                    let folded_events = uow.replay_raw_events(&namespace,0,usize::MAX).await?.into_iter().map(|event|EntityEvent{
                        namespace:event.namespace,seq:event.seq,event_id:event.event_id,entity_kind:event.entity_kind,entity_id:event.entity_id,op:event.op,
                        payload:serde_json::from_str(&event.payload_json).unwrap_or(Value::Null),}).collect::<Vec<_>>();
                    let folded=workflow_design_history_value(&workflow_id_owned,folded_events,"ready").map_err(kogwistar_store_postgres::PostgresStoreError::TransactionAborted)?;
                    uow.replace_named_projection("workflow_design",&workflow_id_owned,workflow_design_projection_write(&folded)).await?;
                    Ok(json!({"workflow_id":workflow_id_owned,"namespace":namespace,
                        if deleted {if action=="node_delete"{"node_id"}else{"edge_id"}}else if action=="node_upsert"{"node_id"}else{"edge_id"}:entity_id,
                        "designer_id":designer,"deleted":deleted,"version":folded["current_version"],"seq":data_event.event.seq,"can_undo":folded["can_undo"],"can_redo":folded["can_redo"]}))
                }
                _ => unreachable!(),
            }
        })).await;
        match outcome {
            Ok(value) => json_effect(StatusCode::OK, value),
            Err(error)
                if error.to_string().contains("KOGWISTAR_NODE_NOT_FOUND")
                    || error.to_string().contains("KOGWISTAR_EDGE_NOT_FOUND") =>
            {
                effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_DESIGN_ENTITY_NOT_FOUND",
                    error.to_string(),
                )
            }
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_WORKFLOW_DESIGN_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn workflow_design_history(
        &self,
        request: &ApiEffectRequest,
        workflow_id: &str,
    ) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "workflow.design.inspect",
            &["workflow.design.inspect"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Inspecting workflow design requires workflow.design.inspect capability",
            );
        }
        let namespace = format!("wf_design:{workflow_id}");
        let events = match self
            .store
            .replay_raw_events(&namespace, 0, usize::MAX)
            .await
        {
            Ok(events) => events
                .into_iter()
                .map(|event| EntityEvent {
                    namespace: event.namespace,
                    seq: event.seq,
                    event_id: event.event_id,
                    entity_kind: event.entity_kind,
                    entity_id: event.entity_id,
                    op: event.op,
                    payload: serde_json::from_str(&event.payload_json).unwrap_or(Value::Null),
                })
                .collect(),
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let history = match workflow_design_history_value(workflow_id, events, "ready") {
            Ok(history) => history,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_WORKFLOW_DESIGN_HISTORY_INVALID",
                    error,
                );
            }
        };
        match self
            .store
            .replace_named_projection(
                "workflow_design",
                workflow_id,
                workflow_design_projection_write(&history),
            )
            .await
        {
            Ok(()) => json_effect(StatusCode::OK, history),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    async fn service_read(&self, request: &ApiEffectRequest) -> Option<ApiEffectResponse> {
        if request.method != "GET" {
            return None;
        }
        let path = request
            .path_and_query
            .split('?')
            .next()
            .unwrap_or(&request.path_and_query);
        if path != "/api/workflow/services" && service_route(&request.path_and_query).is_none() {
            return None;
        }
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.inspect",
            &["service.inspect", "project_view"],
        ) {
            return Some(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Inspecting service requires service.inspect capability",
            ));
        }
        if path == "/api/workflow/services" {
            let limit =
                query_integer(&request.path_and_query, "limit", 200).clamp(0, 10_000) as usize;
            return Some(
                match self.store.list_named_projections("service_registry").await {
                    Ok(rows) => {
                        let mut services = rows
                            .into_iter()
                            .map(|row| Value::Object(row.payload))
                            .collect::<Vec<_>>();
                        services.sort_by(|left, right| {
                            left["service_id"]
                                .as_str()
                                .cmp(&right["service_id"].as_str())
                        });
                        services.truncate(limit);
                        json_effect(StatusCode::OK, json!({"services": services}))
                    }
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                },
            );
        }
        let (service_id, route) =
            service_route(&request.path_and_query).expect("validated service route");
        if route == ServiceRoute::Events {
            let limit =
                query_integer(&request.path_and_query, "limit", 500).clamp(1, 10_000) as usize;
            return Some(
                match self
                    .store
                    .replay_raw_events("workflow", 0, usize::MAX)
                    .await
                {
                    Ok(events) => json_effect(
                        StatusCode::OK,
                        json!({
                            "service_id": service_id,
                            "events": service_event_values(
                                events.into_iter().map(|event| (
                                    event.seq,
                                    event.entity_kind,
                                    event.entity_id,
                                    event.op,
                                    event.payload_json,
                                )),
                                &service_id,
                                limit,
                            ),
                        }),
                    ),
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                },
            );
        }
        Some(
            match self
                .store
                .get_named_projection("service_registry", &service_id)
                .await
            {
                Ok(Some(row)) => json_effect(StatusCode::OK, Value::Object(row.payload)),
                Ok(None) => effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_SERVICE_NOT_FOUND",
                    format!("Unknown service_id: {service_id}"),
                ),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            },
        )
    }

    async fn repair_orphaned_messages(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "project_view",
            &["project_view"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Repairing claimed messages requires project_view capability",
            );
        }
        let inbox_id = query_string(&request.path_and_query, "inbox_id");
        let limit = query_integer(&request.path_and_query, "limit", 100).clamp(0, 10_000) as usize;
        match self
            .store
            .repair_orphaned_claimed_lane_messages("workflow", inbox_id.as_deref(), limit)
            .await
        {
            Ok(repaired) => json_effect(StatusCode::OK, json!({"repaired_message_ids": repaired})),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    async fn replay_dead_letter(
        &self,
        request: &ApiEffectRequest,
        run_id: &str,
    ) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.manage",
            &["service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Replaying dead letters requires service.manage capability",
            );
        }
        let lanes = match self
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                namespace: Some("workflow".to_owned()),
                status: Some("dead-letter".to_owned()),
                limit: 10_000,
                ..LaneMessageFilter::default()
            })
            .await
        {
            Ok(lanes) => lanes,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let Some(lane) = lanes
            .into_iter()
            .find(|lane| lane.run_id.as_deref() == Some(run_id))
        else {
            return json_effect(StatusCode::OK, json!({"run_id": run_id, "replayed": false}));
        };
        match self
            .store
            .update_projected_lane_message_status(&lane.message_id, "pending", None)
            .await
        {
            Ok(()) => json_effect(
                StatusCode::OK,
                json!({
                    "run_id": run_id,
                    "replayed": true,
                    "dead_letter": lane_progress_value(&lane),
                }),
            ),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    pub fn from_dsn(dsn: &str, schema: &str) -> Result<Self, String> {
        let dsn = dsn
            .strip_prefix("postgresql+psycopg://")
            .map(|rest| format!("postgresql://{rest}"))
            .or_else(|| {
                dsn.strip_prefix("postgresql+psycopg2://")
                    .map(|rest| format!("postgresql://{rest}"))
            })
            .unwrap_or_else(|| dsn.to_owned());
        PostgresStore::from_dsn(&dsn, schema)
            .map(|store| Self {
                store,
                max_queue: runtime_max_queue(),
                capabilities: Arc::new(Mutex::new(CapabilityState::default())),
            })
            .map_err(|error| error.to_string())
    }

    pub async fn ensure_schema(&self) -> Result<(), String> {
        self.store
            .ensure_schema()
            .await
            .map_err(|error| error.to_string())
    }

    pub fn with_max_queue(mut self, max_queue: usize) -> Self {
        self.max_queue = max_queue.max(1);
        self
    }

    async fn submit_run(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let input: RuntimeSubmitRun = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let workflow_id = input.workflow_id.trim().to_owned();
        let conversation_id = input.conversation_id.trim().to_owned();
        if workflow_id.is_empty() || conversation_id.is_empty() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "workflow_id and conversation_id are required",
            );
        }
        let run_id = input
            .run_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let turn_node_id = input
            .turn_node_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("wf_turn|{run_id}"));
        let message_id = format!("lane|{}", uuid::Uuid::new_v4());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|value| value.as_secs() as i64)
            .unwrap_or(0);
        let priority_class = if input.priority_class.is_empty() {
            "foreground".to_owned()
        } else {
            input.priority_class.clone()
        };
        let runtime_kind = if input.runtime_kind.is_empty() {
            "sync".to_owned()
        } else {
            input.runtime_kind.clone()
        };
        let graph_plan = match (
            self.store
                .get_named_projection("workflow_design", &workflow_id)
                .await,
            self.store
                .get_workflow_design_snapshot(&workflow_id, i64::MAX, 1)
                .await,
        ) {
            (Ok(projection), Ok(snapshot)) => exact_runtime_graph_plan(projection, snapshot),
            _ => None,
        };
        if graph_plan
            .as_ref()
            .is_some_and(|plan| submitted_runtime_graph_conflicts(plan, &input))
        {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_GRAPH_CONFLICT",
                "submitted frozen runtime graph differs from authoritative workflow design",
            );
        }
        let effective_routes = graph_plan
            .as_ref()
            .map(|plan| plan.routes.clone())
            .unwrap_or(input.runtime_routes);
        let effective_join_node_ids = graph_plan
            .as_ref()
            .map(|plan| plan.join_node_ids.clone())
            .unwrap_or(input.join_node_ids);
        let effective_start_join_mask = graph_plan
            .as_ref()
            .map(|plan| plan.start_join_mask)
            .unwrap_or(input.start_join_mask);
        let effective_start_node_id = graph_plan
            .as_ref()
            .map(|plan| plan.start_node_id.clone())
            .or_else(|| {
                input
                    .start_node_id
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "start".to_owned());
        let effective_node_ops = graph_plan
            .as_ref()
            .map(|plan| plan.node_ops.clone())
            .unwrap_or(input.node_ops);
        // Keep PostgreSQL admission wire-identical to SQLite: the worker must
        // receive a frozen operation, never a null value requiring a node-id
        // resolver guess.
        let effective_start_op = effective_node_ops
            .get(&effective_start_node_id)
            .cloned()
            .unwrap_or_else(|| "noop".to_owned());
        let mut initial_state = input.initial_state;
        if !effective_routes.is_empty() {
            initial_state.insert(
                "_rt_routes".to_owned(),
                serde_json::to_value(&effective_routes).expect("static runtime routes serialize"),
            );
        }
        if !effective_node_ops.is_empty() {
            initial_state.insert(
                "_rt_node_ops".to_owned(),
                serde_json::to_value(&effective_node_ops).expect("runtime node ops serialize"),
            );
        }
        if let Some(value) = input.token_budget {
            initial_state.insert("token_budget".to_owned(), json!(value));
        }
        if let Some(value) = input.time_budget_ms {
            initial_state.insert("time_budget_ms".to_owned(), json!(value));
        }
        let response = json!({
            "run_id": run_id,
            "conversation_id": conversation_id,
            "workflow_id": workflow_id,
            "turn_node_id": turn_node_id,
            "status": "queued",
            "priority_class": priority_class,
            "token_budget": input.token_budget,
            "time_budget_ms": input.time_budget_ms,
            "admission": "accepted",
            "lane_message_id": message_id,
        });
        if let Ok(Some(existing)) = self.store.get_server_run(&run_id).await {
            let lanes = self
                .store
                .list_projected_lane_messages(LaneMessageFilter {
                    namespace: Some("workflow".to_owned()),
                    inbox_id: Some("workflow-runtime".to_owned()),
                    correlation_id: Some(run_id.clone()),
                    limit: 10,
                    ..LaneMessageFilter::default()
                })
                .await
                .unwrap_or_default();
            let expected_worker_payload = json!({
                "contract_version": 1,
                "kind": "workflow.run.execute",
                "run_id": run_id,
                "workflow_id": workflow_id,
                "conversation_id": conversation_id,
                "turn_node_id": turn_node_id,
                "user_id": input.user_id,
                "initial_state": initial_state,
                "priority_class": priority_class,
                "token_budget": input.token_budget,
                "time_budget_ms": input.time_budget_ms,
                "runtime_kind": runtime_kind,
                "join_node_ids": effective_join_node_ids,
                "start_join_mask": effective_start_join_mask,
                "start_node_id": effective_start_node_id,
                "op": effective_start_op,
                "runtime_routes": effective_routes,
            });
            return match runtime_submit_retry_response(&existing, &lanes, &expected_worker_payload)
            {
                Ok(Some(value)) => json_effect(StatusCode::ACCEPTED, value),
                Ok(None) | Err(_) => effect_error(
                    StatusCode::CONFLICT,
                    "KOGWISTAR_RUNTIME_ADMISSION_CONFLICT",
                    format!("run_id {run_id:?} already exists with different admission state"),
                ),
            };
        }
        let max_queue = self.max_queue;
        let queued_priority_class = priority_class.clone();
        let queue_full_admission = if matches!(priority_class.as_str(), "background" | "batch") {
            "rejected"
        } else {
            "deferred"
        };
        let outcome = self
            .store
            .transaction(|uow| {
                Box::pin(async move {
                    uow.lock_runtime_queue_admission().await?;
                    let pending = uow
                        .list_projected_lane_messages(&LaneMessageFilter {
                            namespace: Some("workflow".to_owned()),
                            inbox_id: Some("workflow-runtime".to_owned()),
                            status: Some("pending".to_owned()),
                            limit: max_queue,
                            ..LaneMessageFilter::default()
                        })
                        .await?;
                    let claimed = uow
                        .list_projected_lane_messages(&LaneMessageFilter {
                            namespace: Some("workflow".to_owned()),
                            inbox_id: Some("workflow-runtime".to_owned()),
                            status: Some("claimed".to_owned()),
                            limit: max_queue,
                            ..LaneMessageFilter::default()
                        })
                        .await?;
                    if pending.len() + claimed.len() >= max_queue {
                        return Ok(false);
                    }
                    uow.create_server_run(ServerRunCreate {
                        run_id: run_id.clone(),
                        conversation_id: conversation_id.clone(),
                        workflow_id: workflow_id.clone(),
                        user_id: input.user_id.clone(),
                        user_turn_node_id: turn_node_id.clone(),
                        status: "queued".to_owned(),
                    })
                    .await?;
                    let created = uow
                        .append_server_run_event(
                            &run_id,
                            "run.created",
                            serde_json::to_string(&json!({
                                "run_id": run_id,
                                "run_kind": "workflow_runtime",
                                "conversation_id": conversation_id,
                                "workflow_id": workflow_id,
                                "status": "queued",
                                "turn_node_id": turn_node_id,
                            }))?,
                        )
                        .await?;
                    let payload = json!({
                        "contract_version": 1,
                        "kind": "workflow.run.execute",
                        "run_id": run_id,
                        "workflow_id": workflow_id,
                        "conversation_id": conversation_id,
                        "turn_node_id": turn_node_id,
                        "user_id": input.user_id,
                        "initial_state": initial_state,
                        "priority_class": queued_priority_class,
                        "token_budget": input.token_budget,
                        "time_budget_ms": input.time_budget_ms,
                        "runtime_kind": runtime_kind,
                        "join_node_ids": effective_join_node_ids,
                        "start_join_mask": effective_start_join_mask,
                        "start_node_id": effective_start_node_id,
                        "op": effective_start_op,
                        "runtime_routes": effective_routes,
                        "expected_event_seq": created.seq,
                    });
                    uow.project_lane_message(NewProjectedLaneMessage {
                        message_id,
                        namespace: "workflow".to_owned(),
                        purpose: "system".to_owned(),
                        inbox_id: "workflow-runtime".to_owned(),
                        conversation_id,
                        recipient_id: "python-worker".to_owned(),
                        sender_id: "rust-scheduler".to_owned(),
                        msg_type: "workflow.run.execute".to_owned(),
                        status: "pending".to_owned(),
                        created_at: now,
                        available_at: now,
                        run_id: Some(run_id.clone()),
                        step_id: Some(effective_start_node_id),
                        correlation_id: Some(run_id),
                        payload_json: Some(serde_json::to_string(&payload)?),
                        error_json: None,
                    })
                    .await?;
                    Ok(true)
                })
            })
            .await;
        match outcome {
            Ok(true) => json_effect(StatusCode::ACCEPTED, response),
            Ok(false) => json_effect(
                StatusCode::TOO_MANY_REQUESTS,
                json!({
                    "admission": queue_full_admission,
                    "reason": "queue_full",
                    "max_queue": max_queue,
                }),
            ),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    async fn claim_runtime_work(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let input: RuntimeClaimRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if input.claimed_by.trim().is_empty() || input.limit == 0 || input.lease_seconds <= 0 {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "claimed_by, positive limit, and positive lease_seconds are required",
            );
        }
        let outcome = self
            .store
            .transaction(|uow| {
                Box::pin(async move {
                    let claimed = if let Some(run_id) = input.run_id.as_deref() {
                        uow.claim_projected_lane_messages_for_run(
                            "workflow",
                            "workflow-runtime",
                            run_id,
                            &input.claimed_by,
                            input.limit,
                            input.lease_seconds,
                        )
                        .await?
                    } else {
                        uow.claim_projected_lane_messages(
                            "workflow",
                            "workflow-runtime",
                            &input.claimed_by,
                            input.limit,
                            input.lease_seconds,
                        )
                        .await?
                    };
                    let mut values = Vec::new();
                    for lane in claimed {
                        let mut payload: Value =
                            serde_json::from_str(lane.payload_json.as_deref().unwrap_or("{}"))?;
                        let run_id = lane.run_id.clone().unwrap_or_default();
                        if uow
                            .get_server_run(&run_id)
                            .await?
                            .is_some_and(|run| run.cancel_requested)
                        {
                            uow.update_projected_lane_message_status(
                                &lane.message_id,
                                "cancelled",
                                None,
                            )
                            .await?;
                            continue;
                        }
                        let starts_runtime = lane.msg_type == "workflow.run.execute"
                            && payload["runtime_started"].as_bool() != Some(true);
                        let expected_event_seq = if starts_runtime {
                            uow.apply_recorded_runtime_transition(
                                runtime_start_transition(&payload, &run_id),
                                false,
                            )
                            .await?
                            .event_seq
                        } else {
                            payload["expected_event_seq"].as_i64().ok_or_else(|| {
                                kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                    format!(
                                        "continuation lane message {:?} lacks expected_event_seq",
                                        lane.message_id
                                    ),
                                )
                            })?
                        };
                        payload["expected_event_seq"] = json!(expected_event_seq);
                        if lane.msg_type == "workflow.run.execute" {
                            payload["runtime_started"] = json!(true);
                        }
                        uow.update_projected_lane_message_payload(
                            &lane.message_id,
                            serde_json::to_string(&payload)?,
                        )
                        .await?;
                        values.push(runtime_claimed_work(
                            lane,
                            &input.claimed_by,
                            run_id,
                            payload,
                            expected_event_seq,
                        ));
                    }
                    Ok(values)
                })
            })
            .await;
        match outcome {
            Ok(work) => json_effect(StatusCode::OK, json!({"work": work})),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_CLAIM_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn apply_runtime_result(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let input: RuntimeWorkerResultRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if input.transition.is_some() == input.effect.is_some() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "exactly one of transition or effect is required",
            );
        }
        let outcome = self
            .store
            .transaction(|uow| {
                Box::pin(async move {
                    let handoff = input.handoff;
                    let (applied, effect_id, terminal_result, usage, trace_events) =
                        if let Some(transition) = input.transition {
                            let effect_id = transition.transition_id.clone();
                            let terminal_result = transition.result.clone();
                            let applied = uow
                                .apply_claimed_recorded_runtime_transition(
                                    handoff, transition, false,
                                )
                                .await?;
                            (applied, effect_id, terminal_result, None, Vec::new())
                        } else {
                            let effect = input.effect.expect("effect checked above");
                            let effect_id = effect.effect_id.clone();
                            let terminal_result = effect.result.clone();
                            let usage = effect.usage.clone();
                            let trace_events = effect.trace_events.clone();
                            let applied = uow
                                .apply_claimed_recorded_worker_effect(handoff, effect)
                                .await?;
                            (applied, effect_id, terminal_result, usage, trace_events)
                        };
                    let state = applied.reduced.state.clone();
                    let frontier = &state.frontier;
                    let unfinished = !frontier.pending.is_empty()
                        || !frontier.suspended.is_empty()
                        || frontier.join_outstanding.iter().any(|value| *value != 0)
                        || frontier
                            .join_waiters
                            .values()
                            .any(|waiters| !waiters.is_empty());
                    let applied_idempotent = applied.idempotent;
                    let result = if unfinished || state.status.is_terminal() {
                        for (node_id, join_mask, token_id, parent_token_id) in &frontier.pending {
                            let existing_lanes = uow
                                .list_projected_lane_messages(&LaneMessageFilter {
                                    namespace: Some("workflow".to_owned()),
                                    inbox_id: Some("workflow-runtime".to_owned()),
                                    correlation_id: Some(state.run_id.clone()),
                                    ..LaneMessageFilter::default()
                                })
                                .await?;
                            if active_runtime_lane_covers_token(&existing_lanes, token_id) {
                                continue;
                            }
                            let next_step_seq = state.last_step_seq.saturating_add(1);
                            let message_id = format!(
                                "lane|{}",
                                kogwistar_contracts::stable_id(
                                    "runtime.worker.request",
                                    &[
                                        state.run_id.clone(),
                                        token_id.clone(),
                                        node_id.clone(),
                                        next_step_seq.to_string(),
                                    ],
                                )
                            );
                            let now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|value| value.as_secs() as i64)
                                .unwrap_or(0);
                            if uow.get_projected_lane_message(&message_id).await?.is_some() {
                                continue;
                            }
                            let op = state
                                .state
                                .get("_rt_node_ops")
                                .and_then(Value::as_object)
                                .and_then(|ops| ops.get(node_id))
                                .cloned()
                                .unwrap_or_else(|| Value::String("noop".to_owned()));
                            let payload = RuntimeStepExecutePayload::from_recorded_state(
                                &state,
                                RuntimeStepExecuteRequest {
                                    node_id: node_id.clone(),
                                    op,
                                    join_mask: *join_mask,
                                    token_id: token_id.clone(),
                                    parent_token_id: parent_token_id.clone(),
                                    step_seq: next_step_seq,
                                    expected_event_seq: applied.event_seq,
                                    resume_effect: None,
                                },
                            );
                            uow.project_lane_message(NewProjectedLaneMessage {
                                message_id,
                                namespace: "workflow".to_owned(),
                                purpose: "system".to_owned(),
                                inbox_id: "workflow-runtime".to_owned(),
                                conversation_id: state.conversation_id.clone(),
                                recipient_id: "python-worker".to_owned(),
                                sender_id: "rust-scheduler".to_owned(),
                                msg_type: "workflow.step.execute".to_owned(),
                                status: "pending".to_owned(),
                                created_at: now,
                                available_at: now,
                                run_id: Some(state.run_id.clone()),
                                step_id: Some(node_id.clone()),
                                correlation_id: Some(state.run_id.clone()),
                                payload_json: Some(serde_json::to_string(&payload)?),
                                error_json: None,
                            })
                            .await?;
                        }
                        applied
                    } else {
                        uow.apply_recorded_runtime_transition(
                            RecordedRuntimeTransition {
                                contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                                transition_id: format!("complete-{effect_id}"),
                                expected_event_seq: applied.event_seq,
                                kind: RecordedTransitionKind::Complete,
                                run_id: state.run_id,
                                workflow_id: state.workflow_id,
                                conversation_id: state.conversation_id,
                                user_id: None,
                                user_turn_node_id: None,
                                step_seq: state.last_step_seq,
                                node_id: state.last_node_id,
                                token_id: state.last_token_id,
                                parent_token_id: state.last_parent_token_id,
                                initial_state: None,
                                state_update: Vec::new(),
                                update: None,
                                state_schema: serde_json::Map::new(),
                                frontier: Some(RuntimeFrontier::default()),
                                result: terminal_result,
                                wait_reason: None,
                                resume_payload: None,
                                errors: Vec::new(),
                            },
                            false,
                        )
                        .await?
                    };
                    if !applied_idempotent {
                        if let Some(usage) = usage {
                            uow.append_server_run_event(
                                &result.reduced.state.run_id,
                                "workflow.usage.v1",
                                serde_json::to_string(&json!({
                                    "run_id": result.reduced.state.run_id,
                                    "step_id": result.reduced.state.last_node_id,
                                    "token_id": result.reduced.state.last_token_id,
                                    "effect_id": effect_id,
                                    "usage": usage,
                                }))?,
                            )
                            .await?;
                        }
                        for (trace_index, trace) in trace_events.into_iter().enumerate() {
                            uow.append_server_run_event(
                                &result.reduced.state.run_id,
                                "workflow.trace.v1",
                                serde_json::to_string(&json!({
                                    "run_id": result.reduced.state.run_id,
                                    "step_id": result.reduced.state.last_node_id,
                                    "token_id": result.reduced.state.last_token_id,
                                    "effect_id": effect_id,
                                    "trace_index": trace_index,
                                    "trace": trace,
                                }))?,
                            )
                            .await?;
                        }
                    }
                    Ok(result)
                })
            })
            .await;
        match outcome {
            Ok(result) => json_effect(
                StatusCode::OK,
                serde_json::to_value(result).expect("recorded runtime result serializes"),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_RESULT_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn resume_runtime_run(
        &self,
        run_id: &str,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let input: RuntimeResumeRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        match self
            .store
            .resume_recorded_runtime_token(
                run_id,
                &input.workflow_id,
                &input.conversation_id,
                &input.suspended_node_id,
                &input.suspended_token_id,
                Some(input.client_result),
            )
            .await
        {
            Ok(result) => json_effect(
                StatusCode::OK,
                serde_json::to_value(result).expect("recorded runtime result serializes"),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_RESUME_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn execute_async(&self, request: ApiEffectRequest) -> ApiEffectResponse {
        if let Some(response) = visualization_shell_effect(&request) {
            return response;
        }
        if let Some(response) = document_graph_validation_effect(&request) {
            return response;
        }
        if let Some(response) = capability_effect(&request, &self.capabilities) {
            return response;
        }
        if let Some(response) = syscall_read_effect(&request, &self.capabilities) {
            return response;
        }
        if request.method == "POST"
            && let Some(op) = syscall_operation(&request.path_and_query)
        {
            return self.syscall_effect(&request, &op).await;
        }
        if let Some(response) = designer_capabilities_effect(&request, &self.capabilities) {
            return response;
        }
        if let Some((workflow_id, route)) = workflow_design_route(&request.path_and_query) {
            return self
                .workflow_design_effect(&request, &workflow_id, route)
                .await;
        }
        if let Some(response) = self.service_read(&request).await {
            return response;
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next() == Some("/api/workflow/services")
        {
            return self.declare_service(&request).await;
        }
        if request.method == "POST"
            && let Some(service_id) = service_trigger_route(&request.path_and_query)
        {
            return self.trigger_service(&request, &service_id).await;
        }
        if request.method == "POST"
            && let Some(service_id) = service_repair_route(&request.path_and_query)
        {
            return self
                .repair_service_projections(&request, service_id.as_deref())
                .await;
        }
        if request.method == "POST"
            && let Some((service_id, enabled)) = service_enabled_route(&request.path_and_query)
        {
            return self
                .set_service_enabled(&request, &service_id, enabled)
                .await;
        }
        if request.method == "POST"
            && let Some((service_id, action)) = request
                .path_and_query
                .split('?')
                .next()
                .and_then(|path| path.strip_prefix("/api/workflow/services/"))
                .and_then(|rest| rest.split_once('/'))
            && action == "heartbeat"
        {
            return self.record_service_heartbeat(&request, service_id).await;
        }
        if request.method == "POST"
            && let Some(run_id) = dead_letter_replay_run_id(&request.path_and_query)
        {
            return self.replay_dead_letter(&request, &run_id).await;
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next()
                == Some("/api/workflow/messages/repair-orphans")
        {
            return self.repair_orphaned_messages(&request).await;
        }
        if request.method == "POST" && request.path_and_query == "/internal/runtime/claim" {
            return self.claim_runtime_work(&request).await;
        }
        if request.method == "POST" && request.path_and_query == "/internal/runtime/results" {
            return self.apply_runtime_result(&request).await;
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next() == Some("/api/workflow/runs")
        {
            return self.submit_run(&request).await;
        }
        if request.method == "GET"
            && let Some(path) = operational_path(&request.path_and_query)
        {
            let limit =
                query_integer(&request.path_and_query, "limit", 200).clamp(1, 10_000) as usize;
            let runs = match self.store.list_server_runs(None, None, None, 10_000).await {
                Ok(runs) => runs,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    );
                }
            };
            let lanes = match self
                .store
                .list_projected_lane_messages(LaneMessageFilter {
                    namespace: Some("workflow".to_owned()),
                    limit: 10_000,
                    ..LaneMessageFilter::default()
                })
                .await
            {
                Ok(lanes) => lanes,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    );
                }
            };
            let mut events = Vec::new();
            for run in &runs {
                match self
                    .store
                    .list_server_run_events(&run.run_id, 0, limit)
                    .await
                {
                    Ok(mut run_events) => events.append(&mut run_events),
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        );
                    }
                }
            }
            return operational_effect(&request, path, runs, lanes, events);
        }
        let Some((run_id, tail)) = runtime_run_route(&request.path_and_query) else {
            return unavailable_effect(request);
        };
        let run = match self.store.get_server_run(&run_id).await {
            Ok(Some(run)) => run,
            Ok(None) => {
                return effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_RUN_NOT_FOUND",
                    format!("Unknown run_id: {run_id}"),
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let tail = tail.iter().map(String::as_str).collect::<Vec<_>>();
        if request.method == "GET" && is_runtime_inspection_tail(&tail) {
            return match self.store.list_server_run_events(&run_id, 0, 200_000).await {
                Ok(events) => runtime_inspection_effect(&request, &run_id, &tail, events),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            };
        }
        match (request.method.as_str(), tail.as_slice()) {
            ("GET", []) => json_effect(StatusCode::OK, server_run_value(run)),
            ("GET", ["events", "poll"]) => {
                let after_seq = query_integer(&request.path_and_query, "after_seq", 0);
                let limit = query_integer(&request.path_and_query, "limit", 500).clamp(1, 10_000);
                match self
                    .store
                    .list_server_run_events(&run_id, after_seq, limit as usize)
                    .await
                {
                    Ok(events) => json_effect(
                        StatusCode::OK,
                        json!({
                            "run_id": run_id,
                            "events": events.into_iter().map(server_run_event_value).collect::<Vec<_>>(),
                        }),
                    ),
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                }
            }
            ("GET", ["events"]) => {
                let after_seq = query_integer(&request.path_and_query, "after_seq", 0);
                match self
                    .store
                    .list_server_run_events(&run_id, after_seq, 10_000)
                    .await
                {
                    Ok(events) => {
                        let mut body = String::new();
                        for event in events {
                            let payload = serde_json::from_str::<Value>(&event.payload_json)
                                .unwrap_or_else(|_| json!({}));
                            let mut event_payload = json!({
                                "run_id": event.run_id,
                                "event_type": event.event_type,
                                "created_at_ms": event.created_at_ms,
                            });
                            if let (Some(target), Some(source)) =
                                (event_payload.as_object_mut(), payload.as_object())
                            {
                                target.extend(source.clone());
                            }
                            let event_type =
                                event_payload["event_type"].as_str().unwrap_or("run.event");
                            body.push_str(&sse_frame(
                                event_type,
                                &event_payload,
                                Some(&event.seq.to_string()),
                            ));
                        }
                        ApiEffectResponse {
                            status: StatusCode::OK.as_u16(),
                            content_type: "text/event-stream; charset=utf-8".to_owned(),
                            body: body.into_bytes(),
                        }
                    }
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                }
            }
            ("GET", ["resume-contract"]) => match self
                .store
                .read_recorded_runtime_state(&run_id, &run.workflow_id, &run.conversation_id)
                .await
            {
                Ok(Some(state)) if !state.frontier.suspended.is_empty() => {
                    json_effect(StatusCode::OK, resume_contract_value(state))
                }
                Ok(Some(_)) => effect_error(
                    StatusCode::CONFLICT,
                    "KOGWISTAR_RUNTIME_NOT_SUSPENDED",
                    "run has no suspended token",
                ),
                Ok(None) => effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_RUNTIME_STATE_NOT_FOUND",
                    format!("No recorded runtime state for {run_id}"),
                ),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            },
            ("POST", ["resume"]) => self.resume_runtime_run(&run_id, &request).await,
            ("POST", ["cancel"]) => match self.store.request_server_run_cancel(&run_id).await {
                Ok(()) => match self.store.get_server_run(&run_id).await {
                    Ok(Some(updated)) => {
                        json_effect(StatusCode::ACCEPTED, server_run_value(updated))
                    }
                    Ok(None) => effect_error(
                        StatusCode::NOT_FOUND,
                        "KOGWISTAR_RUN_NOT_FOUND",
                        format!("Unknown run_id: {run_id}"),
                    ),
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                },
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            },
            _ => unavailable_effect(request),
        }
    }
}

impl ApplicationService for PostgresRunApplicationService {
    fn execute(
        &self,
        request: ApiEffectRequest,
    ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
        Box::pin(async move { self.execute_async(request).await })
    }
}

impl ApplicationService for SqliteRunApplicationService {
    fn execute(
        &self,
        request: ApiEffectRequest,
    ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
        Box::pin(async move { self.execute_sync(request) })
    }
}

impl SqliteRunApplicationService {
    fn syscall_effect(&self, request: &ApiEffectRequest, op: &str) -> ApiEffectResponse {
        let input = match syscall_input(request, op) {
            Ok(input) => input,
            Err(response) => return response,
        };
        let response = match op {
            "spawn_process" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    true,
                    &["spawn_process", "workflow.run.write"],
                ) {
                    syscall_forbidden(op)
                } else {
                    let nested = ApiEffectRequest {
                        contract_version: request.contract_version,
                        method: "POST".to_owned(),
                        path_and_query: "/api/workflow/runs".to_owned(),
                        body: serde_json::to_vec(&input.args).expect("syscall args serialize"),
                        principal: request.principal.clone(),
                    };
                    syscall_result(&input.version, op, self.submit_run(&nested))
                }
            }
            "terminate_process" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    true,
                    &["workflow.run.write"],
                ) {
                    syscall_forbidden(op)
                } else if let Some(run_id) = input.args["run_id"].as_str() {
                    match self.store.request_server_run_cancel(run_id) {
                        Ok(()) => match self.store.get_server_run(run_id) {
                            Ok(Some(run)) => syscall_result(
                                &input.version,
                                op,
                                json_effect(StatusCode::ACCEPTED, server_run_value(run)),
                            ),
                            Ok(None) => effect_error(
                                StatusCode::NOT_FOUND,
                                "KOGWISTAR_RUN_NOT_FOUND",
                                format!("Unknown run_id: {run_id}"),
                            ),
                            Err(error) => effect_error(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                "KOGWISTAR_STORE_ERROR",
                                error.to_string(),
                            ),
                        },
                        Err(error) => effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        ),
                    }
                } else {
                    effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "run_id is required",
                    )
                }
            }
            "checkpoint" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    false,
                    &["workflow.run.read"],
                ) {
                    syscall_forbidden(op)
                } else if let (Some(run_id), Some(step_seq)) = (
                    input.args["run_id"].as_str(),
                    input.args["step_seq"].as_i64(),
                ) {
                    match self.store.list_server_run_events(run_id, 0, 100_000) {
                        Ok(events) => syscall_result(
                            &input.version,
                            op,
                            runtime_inspection_effect(
                                request,
                                run_id,
                                &["checkpoints", &step_seq.to_string()],
                                events,
                            ),
                        ),
                        Err(error) => effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        ),
                    }
                } else {
                    effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "run_id and integer step_seq are required",
                    )
                }
            }
            "resume" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    true,
                    &["workflow.run.write"],
                ) {
                    syscall_forbidden(op)
                } else if let Some(run_id) = input.args["run_id"].as_str() {
                    let nested = ApiEffectRequest {
                        contract_version: request.contract_version,
                        method: "POST".to_owned(),
                        path_and_query: format!("/api/workflow/runs/{run_id}/resume"),
                        body: serde_json::to_vec(&input.args).expect("syscall args serialize"),
                        principal: request.principal.clone(),
                    };
                    syscall_result(&input.version, op, self.resume_runtime_run(run_id, &nested))
                } else {
                    effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "run_id is required",
                    )
                }
            }
            "request_approval" => {
                if !syscall_allowed(request, &self.capabilities, op, true, &["approve_action"]) {
                    syscall_forbidden(op)
                } else {
                    request_approval_syscall(request, &self.capabilities, &input)
                }
            }
            _ => unavailable_effect(request.clone()),
        };
        syscall_audit(
            &self.capabilities,
            &input.version,
            op,
            syscall_audit_status(&response),
        );
        response
    }

    fn trigger_service(&self, request: &ApiEffectRequest, service_id: &str) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.manage",
            &["service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Triggering service requires service.manage capability",
            );
        }
        let mut input: ServiceTriggerInput = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if let Err(error) = validate_service_trigger(&mut input) {
            return effect_error(StatusCode::BAD_REQUEST, "KOGWISTAR_INVALID_REQUEST", error);
        }
        let timestamp = now_ms();
        let now_seconds = (timestamp / 1000) as i64;
        let max_queue = self.max_queue;
        let outcome = self.store.transaction(|uow| {
            let current = uow
                .get_named_projection("service_registry", service_id)?
                .ok_or_else(|| {
                    kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                        "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                    )
                })?;
            let current_value = Value::Object(current.payload);
            if let Some(run_id) = current_value["current_child_run_id"]
                .as_str()
                .filter(|value| !value.is_empty())
                && let Some(run) = uow.get_server_run(run_id)?
                && !matches!(run.status.as_str(), "succeeded" | "failed" | "cancelled")
            {
                return Ok(Some(current_value));
            }
            let history = uow.replay_raw_events("workflow", 0, usize::MAX)?;
            let definition = latest_service_definition(
                history.into_iter().map(|event| {
                    (
                        event.seq,
                        event.entity_kind,
                        event.entity_id,
                        event.op,
                        event.payload_json,
                    )
                }),
                service_id,
            )
            .ok_or_else(|| {
                kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                    "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                )
            })?;
            if service_trigger_is_suppressed(
                &definition,
                &current_value,
                &input.trigger_type,
                timestamp,
            ) {
                return Ok(Some(current_value));
            }
            let (target_kind, target_config) = service_definition_runtime(&definition).map_err(
                |message| {
                    kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                        message.to_owned(),
                    )
                },
            )?;
            let target_ref = definition["properties"]["target_ref"]
                .as_str()
                .unwrap_or_default()
                .to_owned();
            let mut run_id = None;
            if target_kind == "workflow" {
                let active_count = ["pending", "claimed"]
                    .into_iter()
                    .map(|status| {
                        uow.list_projected_lane_messages(&LaneMessageFilter {
                            namespace: Some("workflow".to_owned()),
                            inbox_id: Some("workflow-runtime".to_owned()),
                            status: Some(status.to_owned()),
                            limit: max_queue,
                            ..LaneMessageFilter::default()
                        })
                        .map(|lanes| lanes.len())
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .sum::<usize>();
                if active_count >= max_queue {
                    return Ok(None);
                }
                let target = target_config.as_object().cloned().unwrap_or_default();
                let conversation_id = input
                    .payload
                    .get("conversation_id")
                    .and_then(Value::as_str)
                    .or_else(|| target.get("conversation_id").and_then(Value::as_str))
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("svc:{service_id}"));
                let turn_node_id = input
                    .payload
                    .get("turn_node_id")
                    .and_then(Value::as_str)
                    .or_else(|| target.get("turn_node_id").and_then(Value::as_str))
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("wf_turn|{}", uuid::Uuid::new_v4()));
                let user_id = input
                    .payload
                    .get("user_id")
                    .and_then(Value::as_str)
                    .or_else(|| target.get("user_id").and_then(Value::as_str))
                    .map(str::to_owned);
                let mut initial_state = target
                    .get("initial_state")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                if let Some(overrides) = input
                    .payload
                    .get("initial_state")
                    .and_then(Value::as_object)
                {
                    initial_state.extend(overrides.clone());
                }
                let created_run_id = uuid::Uuid::new_v4().to_string();
                let message_id = format!("lane|{}", uuid::Uuid::new_v4());
                let graph_plan = exact_runtime_graph_plan(
                    uow.get_named_projection("workflow_design", &target_ref)?,
                    uow.get_workflow_design_snapshot(&target_ref, i64::MAX, 1)?,
                );
                let runtime_routes = graph_plan
                    .as_ref()
                    .map(|plan| plan.routes.clone())
                    .unwrap_or_default();
                let join_node_ids = graph_plan
                    .as_ref()
                    .map(|plan| plan.join_node_ids.clone())
                    .unwrap_or_default();
                let start_join_mask = graph_plan
                    .as_ref()
                    .map(|plan| plan.start_join_mask)
                    .unwrap_or_default();
                let start_node_id = graph_plan
                    .as_ref()
                    .map(|plan| plan.start_node_id.clone())
                    .unwrap_or_else(|| "start".to_owned());
                let node_ops = graph_plan
                    .as_ref()
                    .map(|plan| plan.node_ops.clone())
                    .unwrap_or_default();
                let start_op = node_ops.get(&start_node_id).cloned();
                if !runtime_routes.is_empty() {
                    initial_state.insert(
                        "_rt_routes".to_owned(),
                        serde_json::to_value(&runtime_routes)
                            .expect("static runtime routes serialize"),
                    );
                }
                if !node_ops.is_empty() {
                    initial_state.insert(
                        "_rt_node_ops".to_owned(),
                        serde_json::to_value(&node_ops).expect("runtime node ops serialize"),
                    );
                }
                uow.create_server_run(ServerRunCreate {
                    run_id: created_run_id.clone(),
                    conversation_id: conversation_id.clone(),
                    workflow_id: target_ref.clone(),
                    user_id: user_id.clone(),
                    user_turn_node_id: turn_node_id.clone(),
                    status: "queued".to_owned(),
                })?;
                let created = uow.append_server_run_event(
                    &created_run_id,
                    "run.created",
                    json!({
                        "run_id": created_run_id,
                        "run_kind": "workflow_runtime",
                        "conversation_id": conversation_id,
                        "workflow_id": target_ref,
                        "status": "queued",
                        "turn_node_id": turn_node_id,
                    })
                    .to_string(),
                )?;
                let priority_class = target
                    .get("priority_class")
                    .and_then(Value::as_str)
                    .unwrap_or("background");
                let payload = json!({
                    "contract_version": 1,
                    "kind": "workflow.run.execute",
                    "run_id": created_run_id,
                    "workflow_id": target_ref,
                    "conversation_id": conversation_id,
                    "turn_node_id": turn_node_id,
                    "user_id": user_id,
                    "initial_state": initial_state,
                    "priority_class": priority_class,
                    "token_budget": target.get("token_budget"),
                    "time_budget_ms": target.get("time_budget_ms"),
                    "runtime_kind": target.get("runtime_kind").and_then(Value::as_str).unwrap_or("sync"),
                    "join_node_ids": join_node_ids,
                    "start_join_mask": start_join_mask,
                    "start_node_id": start_node_id,
                    "op": start_op,
                    "runtime_routes": runtime_routes,
                    "expected_event_seq": created.seq,
                });
                uow.project_lane_message(NewProjectedLaneMessage {
                    message_id,
                    namespace: "workflow".to_owned(),
                    purpose: "system".to_owned(),
                    inbox_id: "workflow-runtime".to_owned(),
                    conversation_id,
                    recipient_id: "python-worker".to_owned(),
                    sender_id: "rust-scheduler".to_owned(),
                    msg_type: "workflow.run.execute".to_owned(),
                    status: "pending".to_owned(),
                    created_at: now_seconds,
                    available_at: now_seconds,
                    run_id: Some(created_run_id.clone()),
                    step_id: Some(start_node_id),
                    correlation_id: Some(created_run_id.clone()),
                    payload_json: Some(payload.to_string()),
                    error_json: None,
                })?;
                run_id = Some(created_run_id);
            }
            let mut triggered_payload = input.payload.clone();
            triggered_payload.insert(
                "trigger_type".to_owned(),
                json!(input.trigger_type),
            );
            for (event_type, payload) in [
                (
                    "service.starting",
                    json!({"trigger_type": input.trigger_type}),
                ),
                ("service.triggered", Value::Object(triggered_payload)),
            ] {
                let (entity_id, event) =
                    service_event_node(service_id, event_type, &payload, timestamp);
                uow.append_raw_entity_event(
                    "workflow",
                    SqliteNewRawEntityEvent {
                        event_id: uuid::Uuid::new_v4().to_string(),
                        entity_kind: "node".to_owned(),
                        entity_id,
                        op: "ADD".to_owned(),
                        payload_json: event.to_string(),
                    },
                )?;
            }
            if let Some(run_id) = &run_id {
                let (entity_id, event) = service_event_node(
                    service_id,
                    "service.run_spawned",
                    &json!({
                        "trigger_type": input.trigger_type,
                        "run_id": run_id,
                        "workflow_id": target_ref,
                    }),
                    timestamp,
                );
                uow.append_raw_entity_event(
                    "workflow",
                    SqliteNewRawEntityEvent {
                        event_id: uuid::Uuid::new_v4().to_string(),
                        entity_kind: "node".to_owned(),
                        entity_id,
                        op: "ADD".to_owned(),
                        payload_json: event.to_string(),
                    },
                )?;
            }
            let history = uow.replay_raw_events("workflow", 0, usize::MAX)?;
            let projections = service_projection_values(history.into_iter().map(|event| {
                (
                    event.seq,
                    event.entity_kind,
                    event.entity_id,
                    event.op,
                    event.payload_json,
                )
            }))
            .map_err(kogwistar_store_sqlite::SqliteStoreError::TransactionAborted)?;
            let (last_seq, projection) = projections.get(service_id).ok_or_else(|| {
                kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                    "KOGWISTAR_SERVICE_TRIGGER_INVALID".to_owned(),
                )
            })?;
            uow.replace_named_projection(
                "service_registry",
                service_id,
                service_projection_write(*last_seq, projection),
            )?;
            Ok(Some(projection.clone()))
        });
        match outcome {
            Ok(Some(projection)) => json_effect(StatusCode::OK, projection),
            Ok(None) => effect_error(
                StatusCode::TOO_MANY_REQUESTS,
                "KOGWISTAR_QUEUE_FULL",
                "workflow runtime queue is full",
            ),
            Err(error) if error.to_string().contains("KOGWISTAR_SERVICE_NOT_FOUND") => {
                effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_SERVICE_NOT_FOUND",
                    format!("Unknown service_id: {service_id}"),
                )
            }
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_SERVICE_TRIGGER_CONFLICT",
                error.to_string(),
            ),
        }
    }

    fn declare_service(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.manage",
            &["service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Declaring service requires service.manage capability",
            );
        }
        let mut input: ServiceDeclareInput = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if let Err(error) = validate_service_declare(&mut input) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error,
            );
        }
        let timestamp = now_ms();
        let service_id = input.service_id.clone();
        let outcome = self.store.transaction(|uow| {
            let (definition_id, definition) = service_definition_node(&input, timestamp);
            uow.append_raw_entity_event(
                "workflow",
                SqliteNewRawEntityEvent {
                    event_id: uuid::Uuid::new_v4().to_string(),
                    entity_kind: "node".to_owned(),
                    entity_id: definition_id,
                    op: "ADD".to_owned(),
                    payload_json: definition.to_string(),
                },
            )?;
            let lifecycle_type = if input.enabled {
                "service.enabled"
            } else {
                "service.stopped"
            };
            let (event_id, event) = service_event_node(
                &service_id,
                lifecycle_type,
                &json!({"enabled": input.enabled, "autostart": input.autostart}),
                timestamp,
            );
            uow.append_raw_entity_event(
                "workflow",
                SqliteNewRawEntityEvent {
                    event_id: uuid::Uuid::new_v4().to_string(),
                    entity_kind: "node".to_owned(),
                    entity_id: event_id,
                    op: "ADD".to_owned(),
                    payload_json: event.to_string(),
                },
            )?;
            if input.enabled {
                let (event_id, event) = service_event_node(
                    &service_id,
                    "service.degraded",
                    &json!({"health_status": "degraded"}),
                    timestamp,
                );
                uow.append_raw_entity_event(
                    "workflow",
                    SqliteNewRawEntityEvent {
                        event_id: uuid::Uuid::new_v4().to_string(),
                        entity_kind: "node".to_owned(),
                        entity_id: event_id,
                        op: "ADD".to_owned(),
                        payload_json: event.to_string(),
                    },
                )?;
            }
            let history = uow.replay_raw_events("workflow", 0, usize::MAX)?;
            let projections = service_projection_values(history.into_iter().map(|event| {
                (
                    event.seq,
                    event.entity_kind,
                    event.entity_id,
                    event.op,
                    event.payload_json,
                )
            }))
            .map_err(kogwistar_store_sqlite::SqliteStoreError::TransactionAborted)?;
            let (last_seq, projection) = projections.get(&service_id).ok_or_else(|| {
                kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                    "KOGWISTAR_SERVICE_DECLARATION_INVALID".to_owned(),
                )
            })?;
            uow.replace_named_projection(
                "service_registry",
                &service_id,
                service_projection_write(*last_seq, projection),
            )?;
            Ok(projection.clone())
        });
        match outcome {
            Ok(projection) => json_effect(StatusCode::OK, projection),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_SERVICE_DECLARATION_CONFLICT",
                error.to_string(),
            ),
        }
    }

    fn record_service_heartbeat(
        &self,
        request: &ApiEffectRequest,
        service_id: &str,
    ) -> ApiEffectResponse {
        #[derive(Deserialize)]
        struct HeartbeatInput {
            instance_id: String,
            #[serde(default)]
            payload: serde_json::Map<String, Value>,
        }
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.heartbeat",
            &["service.heartbeat", "service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Recording service heartbeat requires service.heartbeat capability",
            );
        }
        let input: HeartbeatInput = match serde_json::from_slice::<HeartbeatInput>(&request.body) {
            Ok(value) if !value.instance_id.trim().is_empty() => value,
            Ok(_) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    "instance_id must not be empty",
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let timestamp = now_ms();
        let outcome = self.store.transaction(|uow| {
            let current = uow
                .get_named_projection("service_registry", service_id)?
                .ok_or_else(|| {
                    kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                        "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                    )
                })?;
            let mut projection = Value::Object(current.payload);
            let previous_health = projection["health_status"]
                .as_str()
                .unwrap_or_default()
                .to_owned();
            let previous_error = projection["last_error"].clone();
            let next_error = input
                .payload
                .get("last_error")
                .cloned()
                .unwrap_or(Value::Null);
            projection["instance_id"] = json!(input.instance_id);
            projection["last_heartbeat_ms"] = json!(timestamp);
            projection["lifecycle_status"] = json!("healthy");
            projection["health_status"] = json!("healthy");
            projection["updated_at_ms"] = json!(timestamp);
            projection["last_error"] = if next_error.is_null() {
                Value::Null
            } else {
                json!(next_error.as_str().unwrap_or_default())
            };
            let mut last_seq = current.last_authoritative_seq;
            if matches!(previous_health.as_str(), "degraded" | "failed" | "stopped") {
                let (entity_id, event) = service_event_node(
                    service_id,
                    "service.healthy",
                    &json!({
                        "health_status": "healthy",
                        "instance_id": input.instance_id,
                    }),
                    timestamp,
                );
                last_seq = uow
                    .append_raw_entity_event(
                        "workflow",
                        SqliteNewRawEntityEvent {
                            event_id: uuid::Uuid::new_v4().to_string(),
                            entity_kind: "node".to_owned(),
                            entity_id,
                            op: "ADD".to_owned(),
                            payload_json: event.to_string(),
                        },
                    )?
                    .event
                    .seq;
            }
            if previous_error != projection["last_error"] && !projection["last_error"].is_null() {
                let (entity_id, event) = service_event_node(
                    service_id,
                    "service.error_changed",
                    &json!({
                        "instance_id": input.instance_id,
                        "last_error": projection["last_error"],
                    }),
                    timestamp,
                );
                last_seq = uow
                    .append_raw_entity_event(
                        "workflow",
                        SqliteNewRawEntityEvent {
                            event_id: uuid::Uuid::new_v4().to_string(),
                            entity_kind: "node".to_owned(),
                            entity_id,
                            op: "ADD".to_owned(),
                            payload_json: event.to_string(),
                        },
                    )?
                    .event
                    .seq;
            }
            uow.replace_named_projection(
                "service_registry",
                service_id,
                service_projection_write(last_seq, &projection),
            )?;
            Ok(projection)
        });
        match outcome {
            Ok(projection) => json_effect(StatusCode::OK, projection),
            Err(error) if error.to_string().contains("KOGWISTAR_SERVICE_NOT_FOUND") => {
                effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_SERVICE_NOT_FOUND",
                    format!("Unknown service_id: {service_id}"),
                )
            }
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_SERVICE_HEARTBEAT_CONFLICT",
                error.to_string(),
            ),
        }
    }

    fn set_service_enabled(
        &self,
        request: &ApiEffectRequest,
        service_id: &str,
        enabled: bool,
    ) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.manage",
            &["service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Changing service lifecycle requires service.manage capability",
            );
        }
        let timestamp = now_ms();
        let outcome = self.store.transaction(|uow| {
            let history = uow.replay_raw_events("workflow", 0, usize::MAX)?;
            let definition = latest_service_definition(
                history.into_iter().map(|event| {
                    (
                        event.seq,
                        event.entity_kind,
                        event.entity_id,
                        event.op,
                        event.payload_json,
                    )
                }),
                service_id,
            )
            .ok_or_else(|| {
                kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                    "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                )
            })?;
            let (definition_id, definition, event_id, event) =
                service_enabled_nodes(definition, service_id, enabled, timestamp)
                    .map_err(kogwistar_store_sqlite::SqliteStoreError::TransactionAborted)?;
            uow.append_raw_entity_event(
                "workflow",
                SqliteNewRawEntityEvent {
                    event_id: uuid::Uuid::new_v4().to_string(),
                    entity_kind: "node".to_owned(),
                    entity_id: definition_id,
                    op: "ADD".to_owned(),
                    payload_json: definition.to_string(),
                },
            )?;
            uow.append_raw_entity_event(
                "workflow",
                SqliteNewRawEntityEvent {
                    event_id: uuid::Uuid::new_v4().to_string(),
                    entity_kind: "node".to_owned(),
                    entity_id: event_id,
                    op: "ADD".to_owned(),
                    payload_json: event.to_string(),
                },
            )?;
            let history = uow.replay_raw_events("workflow", 0, usize::MAX)?;
            let projections = service_projection_values(history.into_iter().map(|event| {
                (
                    event.seq,
                    event.entity_kind,
                    event.entity_id,
                    event.op,
                    event.payload_json,
                )
            }))
            .map_err(kogwistar_store_sqlite::SqliteStoreError::TransactionAborted)?;
            let (last_seq, projection) = projections.get(service_id).ok_or_else(|| {
                kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                    "KOGWISTAR_SERVICE_NOT_FOUND".to_owned(),
                )
            })?;
            uow.replace_named_projection(
                "service_registry",
                service_id,
                service_projection_write(*last_seq, projection),
            )?;
            Ok(projection.clone())
        });
        match outcome {
            Ok(projection) => json_effect(StatusCode::OK, projection),
            Err(error) if error.to_string().contains("KOGWISTAR_SERVICE_NOT_FOUND") => {
                effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_SERVICE_NOT_FOUND",
                    format!("Unknown service_id: {service_id}"),
                )
            }
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_SERVICE_MUTATION_CONFLICT",
                error.to_string(),
            ),
        }
    }

    fn repair_service_projections(
        &self,
        request: &ApiEffectRequest,
        service_id: Option<&str>,
    ) -> ApiEffectResponse {
        if !capability_allowed(
            request,
            &self.capabilities,
            "service.manage",
            &["service.manage"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Repairing service projections requires service.manage capability",
            );
        }
        let events = match self.store.replay_raw_events("workflow", 0, usize::MAX) {
            Ok(events) => events,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let projections = match service_projection_values(events.into_iter().map(|event| {
            (
                event.seq,
                event.entity_kind,
                event.entity_id,
                event.op,
                event.payload_json,
            )
        })) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_SERVICE_HISTORY_INVALID",
                    error,
                );
            }
        };
        if let Some(service_id) = service_id {
            let Some((last_seq, projection)) = projections.get(service_id) else {
                return effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_SERVICE_NOT_FOUND",
                    format!("Unknown service_id: {service_id}"),
                );
            };
            return match self.store.replace_named_projection(
                "service_registry",
                service_id,
                service_projection_write(*last_seq, projection),
            ) {
                Ok(()) => json_effect(
                    StatusCode::OK,
                    json!({"service_id": service_id, "projection": projection}),
                ),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            };
        }
        let limit = query_integer(&request.path_and_query, "limit", 10_000).clamp(0, 10_000);
        let selected = projections
            .into_iter()
            .take(limit as usize)
            .collect::<Vec<_>>();
        let outcome = self.store.transaction(|uow| {
            for (service_id, (last_seq, projection)) in &selected {
                uow.replace_named_projection(
                    "service_registry",
                    service_id,
                    service_projection_write(*last_seq, projection),
                )?;
            }
            Ok(())
        });
        match outcome {
            Ok(()) => json_effect(
                StatusCode::OK,
                json!({
                    "repaired_service_ids": selected
                        .into_iter()
                        .map(|(service_id, _)| service_id)
                        .collect::<Vec<_>>(),
                }),
            ),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    fn execute_sync(&self, request: ApiEffectRequest) -> ApiEffectResponse {
        if let Some(response) = visualization_shell_effect(&request) {
            return response;
        }
        if let Some(response) = document_graph_validation_effect(&request) {
            return response;
        }
        if let Some(response) = capability_effect(&request, &self.capabilities) {
            return response;
        }
        if let Some(response) = syscall_read_effect(&request, &self.capabilities) {
            return response;
        }
        if request.method == "POST"
            && let Some(op) = syscall_operation(&request.path_and_query)
        {
            return self.syscall_effect(&request, &op);
        }
        if let Some(response) = designer_capabilities_effect(&request, &self.capabilities) {
            return response;
        }
        if let Some((workflow_id, route)) = workflow_design_route(&request.path_and_query) {
            return self.workflow_design_effect(&request, &workflow_id, route);
        }
        if let Some(response) = self.service_read(&request) {
            return response;
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next() == Some("/api/workflow/services")
        {
            return self.declare_service(&request);
        }
        if request.method == "POST"
            && let Some(service_id) = service_trigger_route(&request.path_and_query)
        {
            return self.trigger_service(&request, &service_id);
        }
        if request.method == "POST"
            && let Some(service_id) = service_repair_route(&request.path_and_query)
        {
            return self.repair_service_projections(&request, service_id.as_deref());
        }
        if request.method == "POST"
            && let Some((service_id, enabled)) = service_enabled_route(&request.path_and_query)
        {
            return self.set_service_enabled(&request, &service_id, enabled);
        }
        if request.method == "POST"
            && let Some((service_id, action)) = request
                .path_and_query
                .split('?')
                .next()
                .and_then(|path| path.strip_prefix("/api/workflow/services/"))
                .and_then(|rest| rest.split_once('/'))
            && action == "heartbeat"
        {
            return self.record_service_heartbeat(&request, service_id);
        }
        if request.method == "POST"
            && let Some(run_id) = dead_letter_replay_run_id(&request.path_and_query)
        {
            return self.replay_dead_letter(&request, &run_id);
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next()
                == Some("/api/workflow/messages/repair-orphans")
        {
            return self.repair_orphaned_messages(&request);
        }
        if request.method == "POST" && request.path_and_query == "/internal/runtime/claim" {
            return self.claim_runtime_work(&request);
        }
        if request.method == "POST" && request.path_and_query == "/internal/runtime/results" {
            return self.apply_runtime_result(&request);
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next() == Some("/api/workflow/runs")
        {
            return self.submit_run(&request);
        }
        if request.method == "GET"
            && let Some(path) = operational_path(&request.path_and_query)
        {
            let limit =
                query_integer(&request.path_and_query, "limit", 200).clamp(1, 10_000) as usize;
            let runs = match self.store.list_server_runs(None, None, None, 10_000) {
                Ok(runs) => runs,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    );
                }
            };
            let lanes = match self.store.list_projected_lane_messages(LaneMessageFilter {
                namespace: Some("workflow".to_owned()),
                limit: 10_000,
                ..LaneMessageFilter::default()
            }) {
                Ok(lanes) => lanes,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    );
                }
            };
            let mut events = Vec::new();
            for run in &runs {
                match self.store.list_server_run_events(&run.run_id, 0, limit) {
                    Ok(mut run_events) => events.append(&mut run_events),
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        );
                    }
                }
            }
            return operational_effect(&request, path, runs, lanes, events);
        }
        let Some((run_id, tail)) = runtime_run_route(&request.path_and_query) else {
            return unavailable_effect(request);
        };
        let run = match self.get_run(&run_id) {
            Ok(Some(run)) => run,
            Ok(None) => {
                return effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_RUN_NOT_FOUND",
                    format!("Unknown run_id: {run_id}"),
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let tail = tail.iter().map(String::as_str).collect::<Vec<_>>();
        if request.method == "GET" && is_runtime_inspection_tail(&tail) {
            return match self.store.list_server_run_events(&run_id, 0, 200_000) {
                Ok(events) => runtime_inspection_effect(&request, &run_id, &tail, events),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            };
        }
        match (request.method.as_str(), tail.as_slice()) {
            ("GET", []) => json_effect(StatusCode::OK, server_run_value(run)),
            ("GET", ["events", "poll"]) => {
                let after_seq = query_integer(&request.path_and_query, "after_seq", 0);
                let limit = query_integer(&request.path_and_query, "limit", 500).clamp(1, 10_000);
                match self
                    .store
                    .list_server_run_events(&run_id, after_seq, limit as usize)
                {
                    Ok(events) => json_effect(
                        StatusCode::OK,
                        json!({
                            "run_id": run_id,
                            "events": events.into_iter().map(server_run_event_value).collect::<Vec<_>>(),
                        }),
                    ),
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                }
            }
            ("GET", ["events"]) => {
                let after_seq = query_integer(&request.path_and_query, "after_seq", 0);
                match self
                    .store
                    .list_server_run_events(&run_id, after_seq, 10_000)
                {
                    Ok(events) => {
                        let mut body = String::new();
                        for event in events {
                            let payload = serde_json::from_str::<Value>(&event.payload_json)
                                .unwrap_or_else(|_| json!({}));
                            let mut event_payload = json!({
                                "run_id": event.run_id,
                                "event_type": event.event_type,
                                "created_at_ms": event.created_at_ms,
                            });
                            if let (Some(target), Some(source)) =
                                (event_payload.as_object_mut(), payload.as_object())
                            {
                                target.extend(source.clone());
                            }
                            let event_type =
                                event_payload["event_type"].as_str().unwrap_or("run.event");
                            body.push_str(&sse_frame(
                                event_type,
                                &event_payload,
                                Some(&event.seq.to_string()),
                            ));
                        }
                        ApiEffectResponse {
                            status: StatusCode::OK.as_u16(),
                            content_type: "text/event-stream; charset=utf-8".to_owned(),
                            body: body.into_bytes(),
                        }
                    }
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                }
            }
            ("GET", ["resume-contract"]) => match self.store.read_recorded_runtime_state(
                &run_id,
                &run.workflow_id,
                &run.conversation_id,
            ) {
                Ok(Some(state)) if !state.frontier.suspended.is_empty() => {
                    json_effect(StatusCode::OK, resume_contract_value(state))
                }
                Ok(Some(_)) => effect_error(
                    StatusCode::CONFLICT,
                    "KOGWISTAR_RUNTIME_NOT_SUSPENDED",
                    "run has no suspended token",
                ),
                Ok(None) => effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_RUNTIME_STATE_NOT_FOUND",
                    format!("No recorded runtime state for {run_id}"),
                ),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            },
            ("POST", ["resume"]) => self.resume_runtime_run(&run_id, &request),
            ("POST", ["cancel"]) => match self.store.request_server_run_cancel(&run_id) {
                Ok(()) => match self.get_run(&run_id) {
                    Ok(Some(updated)) => {
                        json_effect(StatusCode::ACCEPTED, server_run_value(updated))
                    }
                    Ok(None) => effect_error(
                        StatusCode::NOT_FOUND,
                        "KOGWISTAR_RUN_NOT_FOUND",
                        format!("Unknown run_id: {run_id}"),
                    ),
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error,
                    ),
                },
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            },
            _ => unavailable_effect(request),
        }
    }
}

#[derive(Clone)]
pub struct ServerState {
    pub api: ApiState,
    pub application: Arc<dyn ApplicationService>,
}

#[derive(Debug, Serialize)]
struct ApiErrorBody {
    code: &'static str,
    message: &'static str,
}

fn roles_from_headers(headers: &HeaderMap) -> Vec<String> {
    headers
        .get("x-kogwistar-roles")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect()
}

fn jwt_algorithm(value: &str) -> Option<Algorithm> {
    match value {
        "HS256" => Some(Algorithm::HS256),
        "RS256" => Some(Algorithm::RS256),
        _ => None,
    }
}

fn jwt_algorithm_name(algorithm: Algorithm) -> &'static str {
    match algorithm {
        Algorithm::HS256 => "HS256",
        Algorithm::RS256 => "RS256",
        _ => "unsupported",
    }
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get("authorization")?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("bearer")
        .then_some(token.trim())
}

fn jwt_roles(headers: &HeaderMap, config: &AuthConfig) -> Result<Vec<String>, &'static str> {
    let claims = jwt_claims(headers, config)?;
    let mut roles = Vec::new();
    if let Some(role) = claims.get("role").and_then(Value::as_str) {
        roles.push(role.to_ascii_lowercase());
    }
    if let Some(values) = claims.get("roles").and_then(Value::as_array) {
        roles.extend(
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_ascii_lowercase),
        );
    }
    Ok(roles)
}

fn jwt_claims(headers: &HeaderMap, config: &AuthConfig) -> Result<Value, &'static str> {
    let token = bearer_token(headers).ok_or("Missing bearer token")?;
    let configured_algorithm = jwt_algorithm(config.algorithm.as_deref().unwrap_or("HS256"))
        .ok_or("Unsupported JWT algorithm")?;
    let header = decode_header(token).map_err(|_| "Invalid bearer token")?;
    if header.alg != configured_algorithm {
        return Err("JWT algorithm mismatch");
    }
    let key = if let Some(jwks_json) = config.jwks_json.as_deref() {
        let key_id = header.kid.as_deref().ok_or("JWT kid is required")?;
        let set = serde_json::from_str::<jsonwebtoken::jwk::JwkSet>(jwks_json)
            .map_err(|_| "Invalid JWKS JSON")?;
        let jwk = set.find(key_id).ok_or("JWT kid is not trusted")?;
        if jwk.common.key_algorithm.is_some_and(|key_algorithm| {
            key_algorithm.to_string() != jwt_algorithm_name(configured_algorithm)
        }) {
            return Err("JWT JWK algorithm mismatch");
        }
        DecodingKey::from_jwk(jwk).map_err(|_| "Invalid JWK")?
    } else {
        let key_text = config
            .key
            .as_deref()
            .ok_or("JWT verification key is not configured")?;
        match configured_algorithm {
            Algorithm::HS256 => DecodingKey::from_secret(key_text.as_bytes()),
            Algorithm::RS256 => DecodingKey::from_rsa_pem(key_text.as_bytes())
                .map_err(|_| "Invalid RSA public key")?,
            _ => return Err("Unsupported JWT algorithm"),
        }
    };
    let mut validation = Validation::new(configured_algorithm);
    if let Some(issuer) = &config.issuer {
        validation.set_issuer(&[issuer]);
    }
    if let Some(audience) = &config.audience {
        validation.set_audience(&[audience]);
    } else {
        validation.validate_aud = false;
    }
    Ok(decode::<Value>(token, &key, &validation)
        .map_err(|_| "Invalid bearer token")?
        .claims)
}

fn request_roles(headers: &HeaderMap, config: &AuthConfig) -> Result<Vec<String>, &'static str> {
    if config.configured() {
        jwt_roles(headers, config)
    } else {
        Ok(roles_from_headers(headers))
    }
}

fn request_principal(headers: &HeaderMap, config: &AuthConfig) -> Value {
    if config.configured() {
        return jwt_claims(headers, config).unwrap_or_else(|_| json!({}));
    }
    let roles = roles_from_headers(headers);
    let role = if roles.iter().any(|role| role == "rw") {
        "rw"
    } else {
        "ro"
    };
    json!({"sub": "anonymous", "role": role, "roles": roles, "ns": "workflow"})
}

fn has_required_role(headers: &HeaderMap, required_roles: &[String], config: &AuthConfig) -> bool {
    request_roles(headers, config).is_ok_and(|roles| {
        authorize(&ApiAuthRequest {
            roles,
            required_roles: required_roles.to_vec(),
        })
    })
}

fn forbidden_response() -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(ApiErrorBody {
            code: "KOGWISTAR_AUTH_FORBIDDEN",
            message: "required role missing",
        }),
    )
        .into_response()
}

fn auth_error(status: StatusCode, detail: impl Into<String>) -> Response {
    (status, Json(json!({"detail": detail.into()}))).into_response()
}

fn normalized_string_or_list(value: Option<&Value>, default: Value) -> Value {
    match value {
        None | Some(Value::Null) => default,
        Some(Value::String(raw)) if raw.contains(',') => json!(
            raw.split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .collect::<Vec<_>>()
        ),
        Some(Value::String(raw)) if raw.is_empty() => default,
        Some(value) => value.clone(),
    }
}

#[derive(Clone, Debug, Deserialize)]
struct OidcProviderConfig {
    #[serde(default, rename = "name")]
    _name: String,
    discovery_url: String,
    redirect_uri: String,
    client_id: String,
    #[serde(default)]
    client_secret: String,
    #[serde(default)]
    issuer: Option<String>,
    #[serde(default)]
    scopes: Vec<String>,
    #[serde(default = "default_true")]
    allowed: bool,
    #[serde(default = "default_true")]
    required_email: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct OidcProvidersConfig {
    #[serde(default)]
    default_provider: Option<String>,
    #[serde(default)]
    providers: BTreeMap<String, OidcProviderConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct LoginQuery {
    #[serde(default)]
    redirect_uri: Option<String>,
    #[serde(default)]
    provider: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct CallbackQuery {
    #[serde(default)]
    state: Option<String>,
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    error_description: Option<String>,
}

fn request_cookie(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(';'))
        .filter_map(|part| part.trim().split_once('='))
        .find_map(|(cookie_name, value)| (cookie_name == name).then(|| value.to_owned()))
}

fn delete_login_cookies(response: &mut Response) {
    for name in [
        "auth_state",
        "auth_pkce_verifier",
        "auth_nonce",
        "auth_provider",
    ] {
        let cookie = format!("{name}=; HttpOnly; Max-Age=0; Path=/; SameSite=Lax");
        if let Ok(value) = HeaderValue::from_str(&cookie) {
            response.headers_mut().append(header::SET_COOKIE, value);
        }
    }
}

fn random_token() -> String {
    format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    )
}

fn oidc_providers(config: &AuthConfig) -> Result<OidcProvidersConfig, &'static str> {
    let raw = config
        .oidc_providers_json
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or("OIDC provider is not configured")?;
    serde_json::from_str(raw).map_err(|_| "OIDC provider configuration is invalid")
}

enum AuthStore {
    Sqlite(SqliteAuthStore),
    Postgres(PostgresAuthStore),
}

impl AuthStore {
    async fn from_config(config: &AuthConfig) -> Result<Self, String> {
        let url = config
            .auth_db_url
            .as_deref()
            .unwrap_or("sqlite:///auth.sqlite");
        if let Some(path) = url.strip_prefix("sqlite:///") {
            return SqliteAuthStore::open(path)
                .map(Self::Sqlite)
                .map_err(|error| error.to_string());
        }
        if url.starts_with("postgresql://") || url.starts_with("postgres://") {
            let store = PostgresAuthStore::from_dsn(url).map_err(|error| error.to_string())?;
            store
                .ensure_schema()
                .await
                .map_err(|error| error.to_string())?;
            return Ok(Self::Postgres(store));
        }
        Err("AUTH_DB_URL backend is not supported".to_owned())
    }

    async fn resolve_external_identity(
        &self,
        request: ResolveExternalIdentity,
    ) -> Result<AuthUser, String> {
        match self {
            Self::Sqlite(store) => store.resolve_external_identity(request).await,
            Self::Postgres(store) => store.resolve_external_identity(request).await,
        }
        .map_err(|error| error.to_string())
    }
}

fn mint_app_token(config: &AuthConfig, user: &AuthUser) -> Result<String, &'static str> {
    let secret = config
        .key
        .as_deref()
        .filter(|value| !value.is_empty())
        .ok_or("JWT secret is not configured")?;
    if config.algorithm.as_deref().unwrap_or("HS256") != "HS256" {
        return Err("App token minting requires JWT_ALG=HS256");
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_secs())
        .unwrap_or_default();
    let mut claims = json!({
        "sub": user.email,
        "user_id": user.user_id,
        "email": user.email,
        "name": user.display_name,
        "role": user.global_role.as_deref().unwrap_or("ro"),
        "ns": normalized_string_or_list(
            user.global_ns.as_ref().map(|value| json!(value)).as_ref(),
            json!("docs"),
        ),
        "iat": now,
        "exp": now + 4 * 60 * 60,
        "iss": config.issuer.as_deref().unwrap_or("local"),
    });
    if let Some(audience) = &config.audience {
        claims["aud"] = json!(audience);
    }
    encode(
        &Header::new(Algorithm::HS256),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .map_err(|_| "JWT encoding failed")
}

fn redirect_response(location: String) -> Response {
    let mut response = StatusCode::TEMPORARY_REDIRECT.into_response();
    match HeaderValue::from_str(&location) {
        Ok(value) => {
            response.headers_mut().insert(header::LOCATION, value);
            response
        }
        Err(_) => auth_error(StatusCode::INTERNAL_SERVER_ERROR, "Invalid redirect URL"),
    }
}

fn append_login_cookie(response: &mut Response, name: &str, value: &str) {
    let cookie = format!("{name}={value}; HttpOnly; Max-Age=600; Path=/; SameSite=Lax");
    if let Ok(value) = HeaderValue::from_str(&cookie) {
        response.headers_mut().append(header::SET_COOKIE, value);
    }
}

async fn auth_login_handler(
    State(state): State<Arc<ServerState>>,
    Query(query): Query<LoginQuery>,
) -> Response {
    if state.api.auth.mode.eq_ignore_ascii_case("dev") {
        let ui_url = query
            .redirect_uri
            .or_else(|| std::env::var("UI_URL").ok())
            .unwrap_or_else(|| "/".to_owned());
        let Some(secret) = state
            .api
            .auth
            .key
            .as_deref()
            .filter(|value| !value.is_empty())
        else {
            return auth_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "JWT secret is not configured",
            );
        };
        if state.api.auth.algorithm.as_deref().unwrap_or("HS256") != "HS256" {
            return auth_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Dev token minting requires JWT_ALG=HS256",
            );
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|value| value.as_secs())
            .unwrap_or_default();
        let email =
            std::env::var("DEV_AUTH_EMAIL").unwrap_or_else(|_| "dev@example.com".to_owned());
        let mut claims = json!({
            "sub": email,
            "user_id": std::env::var("DEV_AUTH_SUBJECT").unwrap_or_else(|_| "dev".to_owned()),
            "email": email,
            "name": std::env::var("DEV_AUTH_NAME").unwrap_or_else(|_| "Dev User".to_owned()),
            "role": std::env::var("DEV_AUTH_ROLE").unwrap_or_else(|_| "ro".to_owned()),
            "ns": normalized_string_or_list(
                std::env::var("DEV_AUTH_NS").ok().as_ref().map(|value| json!(value)).as_ref(),
                json!(["docs", "conversation", "workflow", "wisdom"]),
            ),
            "iat": now,
            "exp": now + 4 * 60 * 60,
            "iss": state.api.auth.issuer.as_deref().unwrap_or("local"),
        });
        if let Some(audience) = &state.api.auth.audience {
            claims["aud"] = json!(audience);
        }
        return match encode(
            &Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(secret.as_bytes()),
        ) {
            Ok(token) => redirect_response(format!("{ui_url}?token={token}")),
            Err(_) => auth_error(StatusCode::INTERNAL_SERVER_ERROR, "JWT encoding failed"),
        };
    }

    if query.redirect_uri.is_some() {
        return auth_error(
            StatusCode::BAD_REQUEST,
            "redirect_uri override is only allowed when AUTH_MODE=dev",
        );
    }
    let providers = match oidc_providers(&state.api.auth) {
        Ok(providers) => providers,
        Err(message) => return auth_error(StatusCode::BAD_REQUEST, message),
    };
    let provider_name = query.provider.or(providers.default_provider).or_else(|| {
        (providers.providers.len() == 1)
            .then(|| providers.providers.keys().next().cloned())
            .flatten()
    });
    let Some(provider_name) = provider_name else {
        return auth_error(StatusCode::BAD_REQUEST, "OIDC provider is not configured");
    };
    let Some(provider) = providers
        .providers
        .get(&provider_name)
        .filter(|provider| provider.allowed)
    else {
        return auth_error(
            StatusCode::BAD_REQUEST,
            format!("OIDC provider {provider_name:?} is not configured"),
        );
    };
    let discovery = match reqwest::get(&provider.discovery_url).await {
        Ok(response) => match response.error_for_status() {
            Ok(response) => match response.json::<Value>().await {
                Ok(value) => value,
                Err(_) => {
                    return auth_error(
                        StatusCode::BAD_GATEWAY,
                        "OIDC discovery response is invalid",
                    );
                }
            },
            Err(_) => {
                return auth_error(StatusCode::BAD_GATEWAY, "OIDC discovery request failed");
            }
        },
        Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC discovery request failed"),
    };
    let Some(endpoint) = discovery["authorization_endpoint"].as_str() else {
        return auth_error(
            StatusCode::BAD_GATEWAY,
            "OIDC discovery response is invalid",
        );
    };
    let state_token = random_token();
    let verifier = random_token();
    let nonce = random_token();
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(Sha256::digest(verifier.as_bytes()));
    let mut authorization_url = match url::Url::parse(endpoint) {
        Ok(url) => url,
        Err(_) => {
            return auth_error(
                StatusCode::BAD_GATEWAY,
                "OIDC authorization endpoint is invalid",
            );
        }
    };
    let scopes = if provider.scopes.is_empty() {
        "openid email profile".to_owned()
    } else {
        provider.scopes.join(" ")
    };
    authorization_url.query_pairs_mut().extend_pairs([
        ("client_id", provider.client_id.as_str()),
        ("response_type", "code"),
        ("scope", scopes.as_str()),
        ("redirect_uri", provider.redirect_uri.as_str()),
        ("state", state_token.as_str()),
        ("code_challenge", challenge.as_str()),
        ("code_challenge_method", "S256"),
        ("nonce", nonce.as_str()),
    ]);
    let mut response = redirect_response(authorization_url.into());
    append_login_cookie(&mut response, "auth_state", &state_token);
    append_login_cookie(&mut response, "auth_pkce_verifier", &verifier);
    append_login_cookie(&mut response, "auth_nonce", &nonce);
    append_login_cookie(&mut response, "auth_provider", &provider_name);
    response
}

async fn auth_callback_handler(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Query(query): Query<CallbackQuery>,
) -> Response {
    if state.api.auth.mode.eq_ignore_ascii_case("dev") {
        return auth_error(
            StatusCode::BAD_REQUEST,
            "OIDC callback disabled (AUTH_MODE=dev)",
        );
    }
    if let Some(error) = query.error {
        return auth_error(
            StatusCode::BAD_REQUEST,
            format!(
                "OIDC Error: {error} - {}",
                query.error_description.unwrap_or_default()
            ),
        );
    }
    let Some(code) = query.code.filter(|value| !value.is_empty()) else {
        return auth_error(StatusCode::BAD_REQUEST, "Missing authorization code");
    };
    let Some(state_value) = query.state else {
        return auth_error(StatusCode::BAD_REQUEST, "Invalid state");
    };
    if request_cookie(&headers, "auth_state").as_deref() != Some(state_value.as_str()) {
        return auth_error(StatusCode::BAD_REQUEST, "Invalid state");
    }
    let Some(verifier) = request_cookie(&headers, "auth_pkce_verifier") else {
        return auth_error(StatusCode::BAD_REQUEST, "Missing PKCE verifier");
    };
    let Some(nonce) = request_cookie(&headers, "auth_nonce") else {
        return auth_error(StatusCode::BAD_REQUEST, "Missing nonce");
    };
    let Some(provider_name) = request_cookie(&headers, "auth_provider") else {
        return auth_error(StatusCode::BAD_REQUEST, "Missing provider");
    };
    let providers = match oidc_providers(&state.api.auth) {
        Ok(providers) => providers,
        Err(message) => return auth_error(StatusCode::BAD_REQUEST, message),
    };
    let Some(provider) = providers
        .providers
        .get(&provider_name)
        .filter(|provider| provider.allowed)
    else {
        return auth_error(
            StatusCode::BAD_REQUEST,
            format!("OIDC provider {provider_name:?} is not configured"),
        );
    };
    let client = reqwest::Client::new();
    let discovery = match client.get(&provider.discovery_url).send().await {
        Ok(response) => match response.error_for_status() {
            Ok(response) => match response.json::<Value>().await {
                Ok(value) => value,
                Err(_) => {
                    return auth_error(
                        StatusCode::BAD_GATEWAY,
                        "OIDC discovery response is invalid",
                    );
                }
            },
            Err(_) => {
                return auth_error(StatusCode::BAD_GATEWAY, "OIDC discovery request failed");
            }
        },
        Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC discovery request failed"),
    };
    let Some(token_endpoint) = discovery["token_endpoint"].as_str() else {
        return auth_error(
            StatusCode::BAD_GATEWAY,
            "OIDC discovery response is invalid",
        );
    };
    let mut token_form = vec![
        ("client_id", provider.client_id.as_str()),
        ("grant_type", "authorization_code"),
        ("code", code.as_str()),
        ("redirect_uri", provider.redirect_uri.as_str()),
        ("code_verifier", verifier.as_str()),
    ];
    if !provider.client_secret.is_empty() {
        token_form.push(("client_secret", provider.client_secret.as_str()));
    }
    let tokens = match client.post(token_endpoint).form(&token_form).send().await {
        Ok(response) => match response.error_for_status() {
            Ok(response) => match response.json::<Value>().await {
                Ok(value) => value,
                Err(_) => {
                    return auth_error(StatusCode::BAD_GATEWAY, "OIDC token response is invalid");
                }
            },
            Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC token exchange failed"),
        },
        Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC token exchange failed"),
    };
    let Some(id_token) = tokens["id_token"].as_str() else {
        return auth_error(StatusCode::BAD_REQUEST, "Missing id_token");
    };
    let Some(access_token) = tokens["access_token"].as_str() else {
        return auth_error(StatusCode::BAD_REQUEST, "Missing access_token");
    };
    let Some(jwks_uri) = discovery["jwks_uri"].as_str() else {
        return auth_error(
            StatusCode::BAD_GATEWAY,
            "OIDC discovery response is invalid",
        );
    };
    let jwks = match client.get(jwks_uri).send().await {
        Ok(response) => match response.error_for_status() {
            Ok(response) => match response.json::<jsonwebtoken::jwk::JwkSet>().await {
                Ok(value) => value,
                Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC JWKS is invalid"),
            },
            Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC JWKS request failed"),
        },
        Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC JWKS request failed"),
    };
    let token_header = match decode_header(id_token) {
        Ok(header) => header,
        Err(_) => return auth_error(StatusCode::UNAUTHORIZED, "Invalid id_token"),
    };
    if token_header.alg != Algorithm::RS256 {
        return auth_error(StatusCode::UNAUTHORIZED, "Invalid id_token algorithm");
    }
    let Some(kid) = token_header.kid.as_deref() else {
        return auth_error(StatusCode::UNAUTHORIZED, "id_token missing kid");
    };
    let Some(jwk) = jwks.find(kid) else {
        return auth_error(StatusCode::UNAUTHORIZED, "id_token kid is not trusted");
    };
    let key = match DecodingKey::from_jwk(jwk) {
        Ok(key) => key,
        Err(_) => return auth_error(StatusCode::UNAUTHORIZED, "Invalid id_token key"),
    };
    let issuer = provider
        .issuer
        .as_deref()
        .or_else(|| discovery["issuer"].as_str());
    let Some(issuer) = issuer else {
        return auth_error(
            StatusCode::BAD_GATEWAY,
            "OIDC discovery response is invalid",
        );
    };
    let mut validation = Validation::new(Algorithm::RS256);
    validation.set_audience(&[provider.client_id.as_str()]);
    validation.set_issuer(&[issuer]);
    validation.set_required_spec_claims(&["exp", "iat", "iss", "aud", "sub"]);
    validation.leeway = 60;
    let claims = match decode::<Value>(id_token, &key, &validation) {
        Ok(token) => token.claims,
        Err(_) => return auth_error(StatusCode::UNAUTHORIZED, "Invalid id_token"),
    };
    if claims["nonce"].as_str() != Some(nonce.as_str()) {
        return auth_error(StatusCode::UNAUTHORIZED, "Invalid nonce");
    }
    if claims["aud"]
        .as_array()
        .is_some_and(|audiences| audiences.len() > 1)
        && claims["azp"].as_str() != Some(provider.client_id.as_str())
    {
        return auth_error(
            StatusCode::UNAUTHORIZED,
            "id_token azp does not match client_id",
        );
    }
    let Some(userinfo_endpoint) = discovery["userinfo_endpoint"].as_str() else {
        return auth_error(
            StatusCode::BAD_GATEWAY,
            "OIDC discovery response is invalid",
        );
    };
    let userinfo = match client
        .get(userinfo_endpoint)
        .bearer_auth(access_token)
        .send()
        .await
    {
        Ok(response) => match response.error_for_status() {
            Ok(response) => match response.json::<Value>().await {
                Ok(value) => value,
                Err(_) => {
                    return auth_error(StatusCode::BAD_GATEWAY, "OIDC userinfo is invalid");
                }
            },
            Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC userinfo request failed"),
        },
        Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC userinfo request failed"),
    };
    let Some(subject) = claims["sub"].as_str() else {
        return auth_error(StatusCode::UNAUTHORIZED, "Invalid id_token");
    };
    if userinfo["sub"].as_str() != Some(subject) {
        return auth_error(
            StatusCode::UNAUTHORIZED,
            "userinfo subject does not match validated id_token subject",
        );
    }
    let id_email = claims["email"].as_str();
    let userinfo_email = userinfo["email"].as_str();
    if id_email.is_some() && userinfo_email.is_some() && id_email != userinfo_email {
        return auth_error(
            StatusCode::UNAUTHORIZED,
            "userinfo email does not match validated id_token email",
        );
    }
    let email = id_email.or(userinfo_email);
    if provider.required_email && email.is_none() {
        return auth_error(StatusCode::BAD_REQUEST, "OIDC identity missing email");
    }
    let Some(email) = email else {
        return auth_error(StatusCode::BAD_REQUEST, "OIDC identity missing email");
    };
    let auth_store = match AuthStore::from_config(&state.api.auth).await {
        Ok(store) => store,
        Err(error) => {
            return auth_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Auth store failed: {error}"),
            );
        }
    };
    let user = match auth_store
        .resolve_external_identity(ResolveExternalIdentity {
            issuer: issuer.to_owned(),
            subject: subject.to_owned(),
            email: email.to_owned(),
            display_name: claims["name"]
                .as_str()
                .or_else(|| userinfo["name"].as_str())
                .map(str::to_owned),
            new_user_id: uuid::Uuid::new_v4().to_string(),
            default_role: "ro".to_owned(),
            default_ns: "docs".to_owned(),
        })
        .await
    {
        Ok(user) => user,
        Err(error) => {
            return auth_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Auth store failed: {error}"),
            );
        }
    };
    let token = match mint_app_token(&state.api.auth, &user) {
        Ok(token) => token,
        Err(message) => return auth_error(StatusCode::INTERNAL_SERVER_ERROR, message),
    };
    let ui_url = std::env::var("UI_URL").unwrap_or_else(|_| "/".to_owned());
    let mut response = redirect_response(format!("{ui_url}?token={token}"));
    delete_login_cookies(&mut response);
    response
}

async fn dev_token_handler(
    State(state): State<Arc<ServerState>>,
    Json(input): Json<Value>,
) -> Response {
    if !state.api.auth.mode.eq_ignore_ascii_case("dev") {
        return auth_error(StatusCode::NOT_FOUND, "Dev token endpoint is disabled");
    }
    let Some(secret) = state
        .api
        .auth
        .key
        .as_deref()
        .filter(|value| !value.is_empty())
    else {
        return auth_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "JWT secret is not configured",
        );
    };
    let algorithm = state.api.auth.algorithm.as_deref().unwrap_or("HS256");
    if algorithm != "HS256" {
        return auth_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Dev token minting requires JWT_ALG=HS256",
        );
    }
    let role = input["role"].as_str().unwrap_or("ro");
    if !matches!(role, "ro" | "rw") {
        return auth_error(StatusCode::BAD_REQUEST, "role must be one of ['ro', 'rw']");
    }
    let username = input["username"].as_str().unwrap_or("dev");
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_secs())
        .unwrap_or_default();
    let mut claims = json!({
        "sub": username,
        "ns": normalized_string_or_list(input.get("ns"), json!("docs")),
        "role": role,
        "iat": now,
        "exp": now + 4 * 60 * 60,
        "iss": state.api.auth.issuer.as_deref().unwrap_or("local"),
    });
    if let Some(capabilities) = input.get("capabilities")
        && !capabilities.is_null()
    {
        claims["capabilities"] = normalized_string_or_list(Some(capabilities), Value::Null);
    }
    if let Some(audience) = &state.api.auth.audience {
        claims["aud"] = json!(audience);
    }
    match encode(
        &Header::new(Algorithm::HS256),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    ) {
        Ok(token) => Json(json!({"token": token})).into_response(),
        Err(_) => auth_error(StatusCode::INTERNAL_SERVER_ERROR, "JWT encoding failed"),
    }
}

async fn auth_me_handler(State(state): State<Arc<ServerState>>, headers: HeaderMap) -> Response {
    if !state.api.auth.configured() {
        return auth_error(StatusCode::UNAUTHORIZED, "Not authenticated");
    }
    let claims = match jwt_claims(&headers, &state.api.auth) {
        Ok(claims) => claims,
        Err(_) => return auth_error(StatusCode::UNAUTHORIZED, "Not authenticated"),
    };
    let user_id = claims["user_id"]
        .as_str()
        .or_else(|| claims["sub"].as_str())
        .unwrap_or_default();
    if user_id.is_empty() {
        return auth_error(StatusCode::UNAUTHORIZED, "Not authenticated");
    }
    Json(json!({
        "user_id": user_id,
        "email": claims["email"].as_str().or_else(|| claims["sub"].as_str()),
        "display_name": claims["name"],
        "is_active": true,
        "role": claims["role"],
        "ns": claims["ns"],
    }))
    .into_response()
}

async fn auth_logout_handler() -> Json<Value> {
    Json(json!({"ok": true}))
}

async fn health_handler(State(state): State<Arc<ServerState>>) -> Json<Value> {
    let mut value = health_response(&state.api.health);
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "implementation".to_owned(),
            serde_json::to_value(&state.api.implementation)
                .expect("implementation snapshot serializes"),
        );
    }
    Json(value)
}

async fn openapi_handler() -> Response {
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        FROZEN_OPENAPI_JSON,
    )
        .into_response()
}

#[derive(Debug, Default, Deserialize)]
struct SseQuery {
    #[serde(default)]
    event: Option<String>,
    #[serde(default)]
    id: Option<String>,
}

async fn sse_handler(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Query(query): Query<SseQuery>,
) -> Response {
    if !has_required_role(&headers, &state.api.required_roles, &state.api.auth) {
        return forbidden_response();
    }
    let body = sse_frame(
        query.event.as_deref().unwrap_or("ready"),
        &json!({"ok": true}),
        query.id.as_deref(),
    );
    (
        StatusCode::OK,
        [("content-type", "text/event-stream; charset=utf-8")],
        body,
    )
        .into_response()
}

async fn mcp_handler(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    OriginalUri(uri): OriginalUri,
    Json(payload): Json<Value>,
) -> Response {
    if !has_required_role(&headers, &state.api.required_roles, &state.api.auth) {
        return forbidden_response();
    }
    let id = payload.get("id").cloned().unwrap_or(Value::Null);
    let method = payload
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if method == "tools/list" {
        let contract: Value = serde_json::from_str(FROZEN_MCP_TOOLS_JSON)
            .expect("committed MCP tool contract is valid JSON");
        let surface = match uri.path() {
            "/mcp/conversation" => "conversation",
            "/mcp/workflow" => "workflow",
            _ => "root",
        };
        return Json(mcp_result(
            id,
            json!({"tools": contract["surfaces"][surface].clone()}),
        ))
        .into_response();
    }
    if method == "tools/call" {
        let principal = request_principal(&headers, &state.api.auth);
        let params = payload.get("params").cloned().unwrap_or_else(|| json!({}));
        let tool_name = params
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let arguments = params
            .get("arguments")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let run_id = arguments
            .get("run_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let request = match tool_name {
            "workflow.run_status" | "conversation.run_status" if !run_id.is_empty() => {
                Some(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: format!("/api/workflow/runs/{run_id}"),
                    body: Vec::new(),
                    principal: principal.clone(),
                })
            }
            "workflow.run_cancel" | "conversation.cancel_run" if !run_id.is_empty() => {
                Some(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: format!("/api/workflow/runs/{run_id}/cancel"),
                    body: Vec::new(),
                    principal: principal.clone(),
                })
            }
            "workflow.run_events" if !run_id.is_empty() => {
                let after_seq = arguments
                    .get("after_seq")
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                let limit = arguments
                    .get("limit")
                    .and_then(Value::as_i64)
                    .unwrap_or(500);
                Some(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: format!(
                        "/api/workflow/runs/{run_id}/events/poll?after_seq={after_seq}&limit={limit}"
                    ),
                    body: Vec::new(),
                    principal: principal.clone(),
                })
            }
            "workflow.run_checkpoint_get" if !run_id.is_empty() => arguments
                .get("step_seq")
                .and_then(Value::as_i64)
                .map(|step_seq| ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: format!("/api/workflow/runs/{run_id}/checkpoints/{step_seq}"),
                    body: Vec::new(),
                    principal: principal.clone(),
                }),
            "workflow.run_replay" if !run_id.is_empty() => arguments
                .get("target_step_seq")
                .and_then(Value::as_i64)
                .map(|target_step_seq| ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: format!(
                        "/api/workflow/runs/{run_id}/replay?target_step_seq={target_step_seq}"
                    ),
                    body: Vec::new(),
                    principal: principal.clone(),
                }),
            "workflow.run_resume_contract" if !run_id.is_empty() => Some(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: format!("/api/workflow/runs/{run_id}/resume-contract"),
                body: Vec::new(),
                principal: principal.clone(),
            }),
            "workflow.run_submit" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/runs".to_owned(),
                body: serde_json::to_vec(&arguments).expect("MCP arguments serialize"),
                principal: principal.clone(),
            }),
            "workflow.run_resume" if !run_id.is_empty() => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: format!("/api/workflow/runs/{run_id}/resume"),
                body: serde_json::to_vec(&json!({
                    "suspended_node_id": arguments.get("suspended_node_id"),
                    "suspended_token_id": arguments.get("suspended_token_id"),
                    "client_result": arguments.get("client_result"),
                    "workflow_id": arguments.get("workflow_id"),
                    "conversation_id": arguments.get("conversation_id"),
                }))
                .expect("MCP arguments serialize"),
                principal: principal.clone(),
            }),
            "workflow.budget_snapshot" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/budget".to_owned(),
                body: Vec::new(),
                principal: principal.clone(),
            }),
            "workflow.budget_history" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: format!(
                    "/api/workflow/budget/history?limit={}",
                    arguments
                        .get("limit")
                        .and_then(Value::as_i64)
                        .unwrap_or(200)
                ),
                body: Vec::new(),
                principal: principal.clone(),
            }),
            "workflow.operator_dashboard" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: format!(
                    "/api/workflow/operator/dashboard?limit={}",
                    arguments
                        .get("limit")
                        .and_then(Value::as_i64)
                        .unwrap_or(100)
                ),
                body: Vec::new(),
                principal: principal.clone(),
            }),
            "workflow.scheduler_timeline" => {
                let limit = arguments
                    .get("limit")
                    .and_then(Value::as_i64)
                    .unwrap_or(200);
                let run_query = arguments
                    .get("run_id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .map(|run_id| format!("&run_id={run_id}"))
                    .unwrap_or_default();
                Some(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: format!(
                        "/api/workflow/scheduler/timeline?limit={limit}{run_query}"
                    ),
                    body: Vec::new(),
                    principal: principal.clone(),
                })
            }
            "workflow.dead_letters" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: format!(
                    "/api/workflow/dead-letters?limit={}",
                    arguments
                        .get("limit")
                        .and_then(Value::as_i64)
                        .unwrap_or(100)
                ),
                body: Vec::new(),
                principal: principal.clone(),
            }),
            "workflow.dead_letter_replay" if !run_id.is_empty() => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: format!("/api/workflow/dead-letters/{run_id}/replay"),
                body: Vec::new(),
                principal: principal.clone(),
            }),
            "workflow.message_orphans_repair" => {
                let limit = arguments
                    .get("limit")
                    .and_then(Value::as_i64)
                    .unwrap_or(100);
                let inbox = arguments
                    .get("inbox_id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .map(|value| format!("&inbox_id={value}"))
                    .unwrap_or_default();
                Some(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: format!(
                        "/api/workflow/messages/repair-orphans?limit={limit}{inbox}"
                    ),
                    body: Vec::new(),
                    principal: principal.clone(),
                })
            }
            "workflow.service_list" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: format!(
                    "/api/workflow/services?limit={}",
                    arguments
                        .get("limit")
                        .and_then(Value::as_i64)
                        .unwrap_or(200)
                ),
                body: Vec::new(),
                principal: principal.clone(),
            }),
            "workflow.service_get" => arguments
                .get("service_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(|service_id| ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: format!("/api/workflow/services/{service_id}"),
                    body: Vec::new(),
                    principal: principal.clone(),
                }),
            "workflow.service_events" => arguments
                .get("service_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(|service_id| ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: format!(
                        "/api/workflow/services/{service_id}/events?limit={}",
                        arguments
                            .get("limit")
                            .and_then(Value::as_i64)
                            .unwrap_or(500)
                    ),
                    body: Vec::new(),
                    principal: principal.clone(),
                }),
            "workflow.service_repair" => arguments
                .get("service_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(|service_id| ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: format!("/api/workflow/services/{service_id}/repair"),
                    body: Vec::new(),
                    principal: principal.clone(),
                }),
            name @ ("workflow.service_enable" | "workflow.service_disable") => arguments
                .get("service_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(|service_id| ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: format!(
                        "/api/workflow/services/{service_id}/{}",
                        if name == "workflow.service_enable" {
                            "enable"
                        } else {
                            "disable"
                        }
                    ),
                    body: Vec::new(),
                    principal: principal.clone(),
                }),
            "workflow.service_heartbeat" => arguments
                .get("service_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(|service_id| ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: format!("/api/workflow/services/{service_id}/heartbeat"),
                    body: serde_json::to_vec(&arguments).expect("MCP arguments serialize"),
                    principal: principal.clone(),
                }),
            "workflow.service_trigger" => arguments
                .get("service_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(|service_id| ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: format!("/api/workflow/services/{service_id}/trigger"),
                    body: serde_json::to_vec(&json!({
                        "trigger_type": arguments.get("trigger_type").cloned(),
                        "payload": arguments
                            .get("payload")
                            .filter(|value| value.is_object())
                            .cloned()
                            .unwrap_or_else(|| json!({})),
                    }))
                    .expect("MCP arguments serialize"),
                    principal: principal.clone(),
                }),
            "workflow.service_declare" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: serde_json::to_vec(&arguments).expect("MCP arguments serialize"),
                principal: principal.clone(),
            }),
            "workflow.services_repair" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: format!(
                    "/api/workflow/services/repair?limit={}",
                    arguments
                        .get("limit")
                        .and_then(Value::as_i64)
                        .unwrap_or(10_000)
                ),
                body: Vec::new(),
                principal: principal.clone(),
            }),
            "workflow.design_history" => arguments
                .get("workflow_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(|workflow_id| ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: format!("/api/workflow/design/{workflow_id}/history"),
                    body: Vec::new(),
                    principal: principal.clone(),
                }),
            "workflow.design_node_upsert" => arguments
                .get("workflow_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(|workflow_id| ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: format!("/api/workflow/design/{workflow_id}/nodes"),
                    body: serde_json::to_vec(&arguments).expect("MCP arguments serialize"),
                    principal: principal.clone(),
                }),
            "workflow.design_edge_upsert" => arguments
                .get("workflow_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(|workflow_id| ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: format!("/api/workflow/design/{workflow_id}/edges"),
                    body: serde_json::to_vec(&arguments).expect("MCP arguments serialize"),
                    principal: principal.clone(),
                }),
            "workflow.design_node_delete" => arguments
                .get("workflow_id")
                .and_then(Value::as_str)
                .zip(arguments.get("node_id").and_then(Value::as_str))
                .map(|(workflow_id, node_id)| ApiEffectRequest {
                    contract_version: 1,
                    method: "DELETE".to_owned(),
                    path_and_query: format!("/api/workflow/design/{workflow_id}/nodes/{node_id}"),
                    body: serde_json::to_vec(&arguments).expect("MCP arguments serialize"),
                    principal: principal.clone(),
                }),
            "workflow.design_edge_delete" => arguments
                .get("workflow_id")
                .and_then(Value::as_str)
                .zip(arguments.get("edge_id").and_then(Value::as_str))
                .map(|(workflow_id, edge_id)| ApiEffectRequest {
                    contract_version: 1,
                    method: "DELETE".to_owned(),
                    path_and_query: format!("/api/workflow/design/{workflow_id}/edges/{edge_id}"),
                    body: serde_json::to_vec(&arguments).expect("MCP arguments serialize"),
                    principal: principal.clone(),
                }),
            name @ ("workflow.design_undo" | "workflow.design_redo") => arguments
                .get("workflow_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(|workflow_id| ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: format!(
                        "/api/workflow/design/{workflow_id}/{}",
                        if name == "workflow.design_undo" {
                            "undo"
                        } else {
                            "redo"
                        }
                    ),
                    body: serde_json::to_vec(&arguments).expect("MCP arguments serialize"),
                    principal: principal.clone(),
                }),
            "workflow.visibility_snapshot" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/visibility".to_owned(),
                body: Vec::new(),
                principal: principal.clone(),
            }),
            "workflow.capabilities_snapshot" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/capabilities".to_owned(),
                body: Vec::new(),
                principal: principal.clone(),
            }),
            "workflow.capability_approve" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/capabilities/approve".to_owned(),
                body: serde_json::to_vec(&arguments).expect("MCP arguments serialize"),
                principal: principal.clone(),
            }),
            "workflow.capability_revoke" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/capabilities/revoke".to_owned(),
                body: serde_json::to_vec(&arguments).expect("MCP arguments serialize"),
                principal: principal.clone(),
            }),
            _ => None,
        };
        let Some(request) = request else {
            return Json(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32602, "message": "Tool not cut over or invalid arguments"},
            }))
            .into_response();
        };
        let response = state.application.execute(request).await;
        if !(200..300).contains(&response.status) {
            return Json(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32000,
                    "message": "Rust application service error",
                    "data": serde_json::from_slice::<Value>(&response.body).unwrap_or(Value::Null),
                },
            }))
            .into_response();
        }
        let result = serde_json::from_slice::<Value>(&response.body).unwrap_or(Value::Null);
        return Json(mcp_result(id, result)).into_response();
    }
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": -32601, "message": "Method not found"},
        })),
    )
        .into_response()
}

async fn application_effect_handler(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    method: Method,
    OriginalUri(uri): OriginalUri,
    body: Bytes,
) -> Response {
    if !has_required_role(&headers, &state.api.required_roles, &state.api.auth) {
        return forbidden_response();
    }
    let mut path_and_query = uri.to_string();
    if path_and_query.ends_with("/events")
        && !path_and_query.contains("after_seq=")
        && let Some(last_event_id) = headers
            .get("last-event-id")
            .and_then(|value| value.to_str().ok())
            .filter(|value| value.parse::<i64>().is_ok())
    {
        path_and_query.push_str("?after_seq=");
        path_and_query.push_str(last_event_id);
    }
    let response = state
        .application
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: method.to_string(),
            path_and_query,
            body: body.to_vec(),
            principal: request_principal(&headers, &state.api.auth),
        })
        .await;
    let status = StatusCode::from_u16(response.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (
        status,
        [("content-type", response.content_type)],
        Body::from(response.body),
    )
        .into_response()
}

fn method_filter(method: &str) -> MethodFilter {
    match method {
        "GET" => MethodFilter::GET,
        "POST" => MethodFilter::POST,
        "PUT" => MethodFilter::PUT,
        "PATCH" => MethodFilter::PATCH,
        "DELETE" => MethodFilter::DELETE,
        "HEAD" => MethodFilter::HEAD,
        "OPTIONS" => MethodFilter::OPTIONS,
        other => panic!("unsupported frozen OpenAPI method {other}"),
    }
}

/// Real Rust HTTP transport. Business capabilities are injected into this
/// router as they cut over; handlers remain transport-only.
pub fn router(state: ApiState) -> Router {
    router_with_application(state, Arc::new(UnavailableApplicationService))
}

pub fn router_with_application(
    state: ApiState,
    application: Arc<dyn ApplicationService>,
) -> Router {
    let mut router = Router::new()
        .route("/api/events", get(sse_handler))
        .route("/mcp", post(mcp_handler))
        .route("/mcp/conversation", post(mcp_handler))
        .route("/mcp/workflow", post(mcp_handler));
    router = router
        .route("/internal/runtime/claim", post(application_effect_handler))
        .route(
            "/internal/runtime/results",
            post(application_effect_handler),
        );
    for &(method, path) in FROZEN_OPENAPI_ROUTES {
        if (method == "GET"
            && matches!(
                path,
                "/health" | "/api/auth/login" | "/api/auth/callback" | "/api/auth/me"
            ))
            || (method == "POST" && matches!(path, "/api/auth/logout" | "/auth/dev-token"))
        {
            continue;
        }
        router = router.route(path, on(method_filter(method), application_effect_handler));
    }
    router
        .route("/health", get(health_handler))
        .route("/api/auth/login", get(auth_login_handler))
        .route("/api/auth/callback", get(auth_callback_handler))
        .route("/api/auth/me", get(auth_me_handler))
        .route("/api/auth/logout", post(auth_logout_handler))
        .route("/auth/dev-token", post(dev_token_handler))
        .route("/openapi.json", get(openapi_handler))
        .with_state(Arc::new(ServerState {
            api: state,
            application,
        }))
}

pub async fn serve(
    listener: tokio::net::TcpListener,
    state: ApiState,
) -> Result<(), std::io::Error> {
    axum::serve(listener, router(state)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::Request;
    use tower::ServiceExt;

    fn remove_test_sqlite(path: &std::path::Path) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        loop {
            match std::fs::remove_file(path) {
                Ok(()) => return,
                Err(error) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    let _ = error;
                }
                Err(error) => panic!("failed to remove test SQLite file {path:?}: {error}"),
            }
        }
    }

    fn snapshot() -> HealthSnapshot {
        HealthSnapshot {
            backend: "pg".to_owned(),
            persist_directory: "root".to_owned(),
            conversation_persist_directory: "conversation".to_owned(),
            workflow_persist_directory: "workflow".to_owned(),
            wisdom_persist_directory: "wisdom".to_owned(),
            pg_schema_base: Some("kogwistar".to_owned()),
        }
    }

    #[test]
    fn health_preserves_python_shape() {
        let value = health_response(&snapshot());
        assert_eq!(value["ok"], true);
        assert_eq!(value["backend"], "pg");
        assert_eq!(value["pg_schema_base"], "kogwistar");
    }

    #[test]
    fn auth_sse_mcp_and_cli_contracts_are_versioned_shapes() {
        assert!(authorize(&ApiAuthRequest {
            roles: vec!["admin".to_owned()],
            required_roles: vec!["admin".to_owned()],
        }));
        assert_eq!(
            sse_frame("run", &json!({"ok": true}), Some("7")),
            "id: 7\nevent: run\ndata: {\"ok\":true}\n\n"
        );
        assert_eq!(mcp_result(json!(1), json!({"ok": true}))["jsonrpc"], "2.0");
        assert_eq!(cli_health(&snapshot()), "ok backend=pg");
    }

    #[test]
    fn jwt_auth_verifies_signature_claims_expiry_and_role_lattice() {
        use axum::http::HeaderValue;
        use jsonwebtoken::{EncodingKey, Header, encode};
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let config = AuthConfig {
            mode: "oidc".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some("test-secret".to_owned()),
            jwks_json: None,
            issuer: Some("issuer".to_owned()),
            audience: Some("audience".to_owned()),
            oidc_providers_json: None,
            auth_db_url: None,
        };
        let token = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": "user-1",
                "role": "rw",
                "iss": "issuer",
                "aud": "audience",
                "exp": now + 300,
            }),
            &EncodingKey::from_secret(b"test-secret"),
        )
        .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        assert_eq!(jwt_roles(&headers, &config).unwrap(), vec!["rw"]);
        assert!(has_required_role(&headers, &["ro".to_owned()], &config));

        let expired = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "role": "rw",
                "iss": "issuer",
                "aud": "audience",
                "exp": now - 300,
            }),
            &EncodingKey::from_secret(b"test-secret"),
        )
        .unwrap();
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {expired}")).unwrap(),
        );
        assert!(jwt_roles(&headers, &config).is_err());

        let mut forged_headers = HeaderMap::new();
        forged_headers.insert("x-kogwistar-roles", HeaderValue::from_static("rw"));
        assert!(!has_required_role(
            &forged_headers,
            &["ro".to_owned()],
            &config
        ));
    }

    #[test]
    fn jwks_kid_selection_supports_rotation_and_rejects_unknown_keys() {
        use axum::http::HeaderValue;
        use jsonwebtoken::{EncodingKey, Header, encode};
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let claims = json!({"sub": "user", "role": "ro", "exp": now + 300});
        let token = |kid: &str, secret: &[u8]| {
            let mut header = Header::new(Algorithm::HS256);
            header.kid = Some(kid.to_owned());
            encode(&header, &claims, &EncodingKey::from_secret(secret)).unwrap()
        };
        let config = |jwks_json: Value| AuthConfig {
            mode: "oidc".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: None,
            jwks_json: Some(serde_json::to_string(&jwks_json).unwrap()),
            issuer: None,
            audience: None,
            oidc_providers_json: None,
            auth_db_url: None,
        };
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {}", token("new", b"new-secret"))).unwrap(),
        );
        let rotating = config(json!({
            "keys": [
                {"kty": "oct", "kid": "old", "alg": "HS256", "k": "b2xkLXNlY3JldA"},
                {"kty": "oct", "kid": "new", "alg": "HS256", "k": "bmV3LXNlY3JldA"}
            ]
        }));
        assert_eq!(jwt_roles(&headers, &rotating).unwrap(), vec!["ro"]);

        let old_only = config(json!({
            "keys": [
                {"kty": "oct", "kid": "old", "alg": "HS256", "k": "b2xkLXNlY3JldA"}
            ]
        }));
        assert_eq!(
            jwt_roles(&headers, &old_only),
            Err("JWT kid is not trusted")
        );

        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {}", token("new", b"forged-secret"))).unwrap(),
        );
        assert_eq!(jwt_roles(&headers, &rotating), Err("Invalid bearer token"));
    }

    #[test]
    fn frozen_route_registry_covers_committed_openapi() {
        assert_eq!(FROZEN_OPENAPI_ROUTES.len(), 82);
        assert!(FROZEN_OPENAPI_ROUTES.contains(&("GET", "/health")));
        assert!(FROZEN_OPENAPI_ROUTES.contains(&("POST", "/api/workflow/runs")));
        assert!(FROZEN_OPENAPI_ROUTES.contains(&("GET", "/api/runs/{run_id}/events")));
        let embedded: Value = serde_json::from_str(FROZEN_OPENAPI_JSON).unwrap();
        assert_eq!(embedded["paths"].as_object().unwrap().len(), 80);
    }

    #[test]
    fn sqlite_application_route_inventory_has_no_unclassified_cutover_drift() {
        let path = std::env::temp_dir().join(format!(
            "kogwistar-api-route-inventory-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let transport_routes = [
            ("GET", "/health"),
            ("GET", "/api/auth/login"),
            ("GET", "/api/auth/callback"),
            ("GET", "/api/auth/me"),
            ("POST", "/api/auth/logout"),
            ("POST", "/auth/dev-token"),
        ];
        let mut unavailable = Vec::new();
        for &(method, template) in FROZEN_OPENAPI_ROUTES {
            if transport_routes.contains(&(method, template)) {
                continue;
            }
            let path_and_query = template
                .replace("{run_id}", "inventory-run")
                .replace("{conversation_id}", "inventory-conversation")
                .replace("{workflow_id}", "inventory-workflow")
                .replace("{service_id}", "inventory-service")
                .replace("{step_seq}", "0")
                .replace("{node_id}", "inventory-node")
                .replace("{edge_id}", "inventory-edge")
                .replace("{doc_id}", "inventory-document")
                .replace("{op}", "checkpoint");
            let response = service.execute_sync(ApiEffectRequest {
                contract_version: 1,
                method: method.to_owned(),
                path_and_query,
                body: b"{}".to_vec(),
                principal: json!({
                    "sub": "inventory-admin",
                    "role": "admin",
                    "capabilities": ["*"]
                }),
            });
            if response.status == StatusCode::NOT_IMPLEMENTED.as_u16() {
                unavailable.push(format!("{method} {template}"));
            }
        }
        drop(service);
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            unavailable,
            PENDING_SERVER_CUTOVER_ROUTES
                .iter()
                .map(|(method, path)| format!("{method} {path}"))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn frozen_mcp_tools_are_unique_and_versioned() {
        let contract: Value = serde_json::from_str(FROZEN_MCP_TOOLS_JSON).unwrap();
        assert_eq!(contract["contract_version"], "1.0.0");
        for (surface, expected) in [("root", 12), ("conversation", 5), ("workflow", 41)] {
            let tools = contract["surfaces"][surface].as_array().unwrap();
            assert_eq!(tools.len(), expected);
            let mut names = tools
                .iter()
                .map(|tool| tool["name"].as_str().unwrap())
                .collect::<Vec<_>>();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), tools.len());
        }
    }

    #[tokio::test]
    async fn router_serves_health_auth_sse_and_mcp() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use tower::ServiceExt;

        let app = router(ApiState {
            health: snapshot(),
            required_roles: vec!["reader".to_owned()],
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig::default(),
        });
        let health = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
        let health_body = to_bytes(health.into_body(), 4096).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&health_body).unwrap()["backend"],
            "pg"
        );

        let openapi = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(openapi.status(), StatusCode::OK);
        let openapi_body = to_bytes(openapi.into_body(), 2_000_000).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&openapi_body).unwrap(),
            serde_json::from_str::<Value>(FROZEN_OPENAPI_JSON).unwrap()
        );

        let forbidden = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/events")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(forbidden.status(), StatusCode::FORBIDDEN);

        let events = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/events?event=run&id=7")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(events.status(), StatusCode::OK);
        assert_eq!(
            events.headers()["content-type"],
            "text/event-stream; charset=utf-8"
        );

        let mcp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp.status(), StatusCode::OK);

        let pending_cutover = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(pending_cutover.status(), StatusCode::NOT_IMPLEMENTED);
    }

    #[tokio::test]
    async fn dev_token_me_and_logout_share_verified_jwt_contract() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use tower::ServiceExt;

        let app = router(ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "dev".to_owned(),
                algorithm: Some("HS256".to_owned()),
                key: Some("dev-secret".to_owned()),
                jwks_json: None,
                issuer: Some("local".to_owned()),
                audience: None,
                oidc_providers_json: None,
                auth_db_url: None,
            },
        });
        let token_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/auth/dev-token")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"username":"dev@example.com","role":"rw","ns":"docs,workflow","capabilities":"project_view,workflow.run.read"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(token_response.status(), StatusCode::OK);
        let token_body = to_bytes(token_response.into_body(), 8192).await.unwrap();
        let token = serde_json::from_slice::<Value>(&token_body).unwrap()["token"]
            .as_str()
            .unwrap()
            .to_owned();
        let me = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/auth/me")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(me.status(), StatusCode::OK);
        let me_body = to_bytes(me.into_body(), 8192).await.unwrap();
        let me_value = serde_json::from_slice::<Value>(&me_body).unwrap();
        assert_eq!(me_value["email"], "dev@example.com");
        assert_eq!(me_value["role"], "rw");
        assert_eq!(me_value["ns"], json!(["docs", "workflow"]));

        let forged = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/auth/me")
                    .header("authorization", "Bearer forged")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(forged.status(), StatusCode::UNAUTHORIZED);
        let logout = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/logout")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(logout.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn oidc_login_preserves_provider_pkce_and_cookie_contract() {
        use axum::body::Body;
        use axum::http::Request;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tower::ServiceExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let discovery_task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 2048];
            let size = stream.read(&mut request).await.unwrap();
            assert!(String::from_utf8_lossy(&request[..size]).starts_with("GET /discovery "));
            let body = format!(r#"{{"authorization_endpoint":"http://{address}/authorize"}}"#);
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        let app = router(ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "oidc".to_owned(),
                algorithm: None,
                key: None,
                jwks_json: None,
                issuer: None,
                audience: None,
                oidc_providers_json: Some(
                    json!({
                        "default_provider": "test",
                        "providers": {
                            "test": {
                                "name": "test",
                                "discovery_url": format!("http://{address}/discovery"),
                                "redirect_uri": "http://localhost/api/auth/callback",
                                "client_id": "client-1",
                                "client_secret": "secret",
                                "scopes": ["openid", "email", "profile"],
                                "allowed": true
                            }
                        }
                    })
                    .to_string(),
                ),
                auth_db_url: None,
            },
        });
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/auth/login")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        discovery_task.await.unwrap();
        assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
        let location = response.headers()[header::LOCATION].to_str().unwrap();
        let authorization_url = url::Url::parse(location).unwrap();
        assert_eq!(authorization_url.path(), "/authorize");
        let query = authorization_url
            .query_pairs()
            .into_owned()
            .collect::<BTreeMap<_, _>>();
        assert_eq!(query["client_id"], "client-1");
        assert_eq!(query["response_type"], "code");
        assert_eq!(query["scope"], "openid email profile");
        assert_eq!(query["redirect_uri"], "http://localhost/api/auth/callback");
        assert_eq!(query["code_challenge_method"], "S256");
        assert_eq!(query["state"].len(), 64);
        assert_eq!(query["nonce"].len(), 64);
        assert!(!query["code_challenge"].contains('='));
        let cookies = response
            .headers()
            .get_all(header::SET_COOKIE)
            .iter()
            .map(|value| value.to_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(cookies.len(), 4);
        for cookie_name in [
            "auth_state=",
            "auth_pkce_verifier=",
            "auth_nonce=",
            "auth_provider=test",
        ] {
            assert!(cookies.iter().any(|cookie| {
                cookie.starts_with(cookie_name)
                    && cookie.contains("HttpOnly")
                    && cookie.contains("Max-Age=600")
            }));
        }
    }

    #[tokio::test]
    async fn oidc_login_rejects_production_redirect_override_before_network_io() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use tower::ServiceExt;

        let app = router(ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "oidc".to_owned(),
                oidc_providers_json: Some(r#"{"providers":{}}"#.to_owned()),
                auth_db_url: None,
                ..AuthConfig::default()
            },
        });
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/auth/login?redirect_uri=https://attacker.example")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["detail"],
            "redirect_uri override is only allowed when AUTH_MODE=dev"
        );
    }

    #[tokio::test]
    async fn oidc_callback_validates_rs256_and_persists_identity_before_minting() {
        use axum::body::Body;
        use axum::http::Request;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tower::ServiceExt;

        const PRIVATE_KEY: &str = r#"-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDZEMrHkpSKPPCq
DATB6Q7At9YZVbLq4acttd+yM1ffjpEXfaMfpurxcpHOIoZOO4ddHUDqtDj+cPbH
T1CyNwrSl1jHLe1o+qZRkhPuuKS5ferI9vB0w2vsTynB7rqR45SPO3+ViCJDPmsq
x2qsX/pTXC4a9RRkOTXdEYg7t47jDxim822/AIRu2mSjA7GjGLoDpMTFm1/wTotm
DVb6/hy9iaSf5SO8qd8ROHH1vbRAtY1lcc4Sx/P2+Ngm94QEdKV/tCqQAV5yCPTR
2OcM5aDjyY7gyTY0kd9Gkw2He0o49INcdsGQSeIVE75ITC/vqommk4OgI2gCtd3g
MWDWm4dDAgMBAAECggEAPxjpASNkR1zYjm2o8l8XWUD3HO0y8aD/kkOEj43qNMOB
/KSaRuij8eSeap/Rj6sxOYl35eHWkWvv6Fbve6aRYE77UQbSNMprj1mZrrKAu6TV
G27gzehClnIajtOg6yiO9iXS+/oTD5300/4czZemshWhF1f3gfy5YhYnFkjQ4cJo
E1+5FnB8dgQLSk9clD47zVkFySWoSN1H7OOfwojtxfpgr/wN+/wHuJY425gaFc+7
VmQUg5RGtdciVYmwTV0tQIsqcqeM82/f2e554EzAOck+vX1PKhq3Ll8Z9S0rln57
o1HADCHBzHQ2APrr518BT/jYwsl+kGQPKI+IcmaE2QKBgQDwIkxejcjHvbg140s4
PyJ6yq3bthNF8XaXXMoa14lPQtTy316mn+uoNBwL9YFhEm9ByHBJ1aLugv2ZulMj
sC837+VZye6nl8fCiebGP8Guy4vHFgm4/Lk3HOkbr4Zd2Iwj5A9tlF/iMyjbHkdN
vAUCmkyhyJNpYk26EKijH0CG2QKBgQDnaE7Rx0/mXqhuNjAL9PypuVpqsA/3j3ZL
sJP+vrkEL8DoQGeOEFHGHGJlxmK/lgSmHcov34Zt4nhxa04BgpsIcd6C3OCaSJpX
Cw+KKit1CKPm5Bka2L9XtxaDKHSFH67K9pngc+QPlrEQeXH3xcCKxv0T50SdyZQ+
HSfvEWyFewKBgHx3OqBT2zr0sjN0QXvA9a0xupXENQ8uzeo8lSD+kNQ9bsUIVDYH
dA02HUdxlALtnC87pkAO9KmtyabRteAspPzYYkd87C9/83F5Kt2dFFX2eNfTK2zv
yUywtn68JugjotfDkN+aZWyIWefhNNIs32fu9ENzBD0+T81ebxpFy5tZAoGAG5rb
3DaUl3yvRwZ70NFW2sBbwuJh5Txd9kWIQhlqZM91ib81G0NjHekA6/cwjH5O66oe
Fnvpw24CxDTyx0dXSziaPK4wtPb4Qm31WpwRNxLiyoZnYEZ+/O3AZ8EJtV/EMD4e
uSHaEOn/EWILcG1MvMFkK12pV9FWN9quitxfP8UCgYEAvffs3ep6KgW8zGtQxlWt
emV55w55nDC3OYHQpvv41roViMihNU5EhxF0YNjus744wLxTSyWc29YgkSqZGRwK
NGj8qC7iDj7eHst5dr2KCfToUOTBidV7ynv8RZ5LyChcKzOiEDh/EqlEfGt5xYEI
6Qv6B43kPAqwE3JgGUyb4Wo=
-----END PRIVATE KEY-----"#;
        const JWK: &str = r#"{"kty":"RSA","kid":"test-key","alg":"RS256","use":"sig","n":"2RDKx5KUijzwqgwEwekOwLfWGVWy6uGnLbXfsjNX346RF32jH6bq8XKRziKGTjuHXR1A6rQ4_nD2x09QsjcK0pdYxy3taPqmUZIT7rikuX3qyPbwdMNr7E8pwe66keOUjzt_lYgiQz5rKsdqrF_6U1wuGvUUZDk13RGIO7eO4w8YpvNtvwCEbtpkowOxoxi6A6TExZtf8E6LZg1W-v4cvYmkn-UjvKnfEThx9b20QLWNZXHOEsfz9vjYJveEBHSlf7QqkAFecgj00djnDOWg48mO4Mk2NJHfRpMNh3tKOPSDXHbBkEniFRO-SEwv76qJppODoCNoArXd4DFg1puHQw","e":"AQAB"}"#;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let issuer = format!("http://{address}");
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut id_header = Header::new(Algorithm::RS256);
        id_header.kid = Some("test-key".to_owned());
        let id_token = encode(
            &id_header,
            &json!({
                "iss": issuer,
                "aud": "client-1",
                "sub": "subject-1",
                "email": "alice@example.com",
                "name": "Alice",
                "nonce": "nonce-1",
                "iat": now,
                "exp": now + 300,
            }),
            &EncodingKey::from_rsa_pem(PRIVATE_KEY.as_bytes()).unwrap(),
        )
        .unwrap();
        let mock_issuer = issuer.clone();
        let mock = tokio::spawn(async move {
            for _ in 0..4 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = vec![0_u8; 8192];
                let size = stream.read(&mut request).await.unwrap();
                let request = String::from_utf8_lossy(&request[..size]);
                let (status, content_type, body) = if request.starts_with("GET /discovery ") {
                    (
                        "200 OK",
                        "application/json",
                        json!({
                            "issuer": mock_issuer,
                            "authorization_endpoint": format!("http://{address}/authorize"),
                            "token_endpoint": format!("http://{address}/token"),
                            "jwks_uri": format!("http://{address}/jwks"),
                            "userinfo_endpoint": format!("http://{address}/userinfo"),
                        })
                        .to_string(),
                    )
                } else if request.starts_with("POST /token ") {
                    assert!(request.contains("code=auth-code"));
                    assert!(request.contains("code_verifier=verifier-1"));
                    (
                        "200 OK",
                        "application/json",
                        json!({"access_token": "access-1", "id_token": id_token}).to_string(),
                    )
                } else if request.starts_with("GET /jwks ") {
                    (
                        "200 OK",
                        "application/json",
                        format!(r#"{{"keys":[{JWK}]}}"#),
                    )
                } else if request.starts_with("GET /userinfo ") {
                    assert!(request.contains("authorization: Bearer access-1"));
                    (
                        "200 OK",
                        "application/json",
                        json!({"sub": "subject-1", "email": "alice@example.com", "name": "Alice"})
                            .to_string(),
                    )
                } else {
                    ("404 Not Found", "text/plain", "not found".to_owned())
                };
                stream
                    .write_all(
                        format!(
                            "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
            }
        });
        let auth_path = std::env::temp_dir().join(format!(
            "kogwistar-oidc-callback-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let config = AuthConfig {
            mode: "oidc".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some("app-secret".to_owned()),
            jwks_json: None,
            issuer: Some("local".to_owned()),
            audience: None,
            oidc_providers_json: Some(
                json!({
                    "default_provider": "test",
                    "providers": {
                        "test": {
                            "name": "test",
                            "discovery_url": format!("http://{address}/discovery"),
                            "redirect_uri": "http://localhost/api/auth/callback",
                            "issuer": issuer,
                            "client_id": "client-1",
                            "client_secret": "secret",
                            "required_email": true,
                            "allowed": true,
                        }
                    }
                })
                .to_string(),
            ),
            auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        };
        let app = router(ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: config.clone(),
        });
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/auth/callback?code=auth-code&state=state-1")
                    .header(
                        header::COOKIE,
                        "auth_state=state-1; auth_pkce_verifier=verifier-1; auth_nonce=nonce-1; auth_provider=test",
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        mock.await.unwrap();
        assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
        let location = response.headers()[header::LOCATION].to_str().unwrap();
        let token = location.split_once("?token=").unwrap().1;
        let mut token_headers = HeaderMap::new();
        token_headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        let app_claims = jwt_claims(&token_headers, &config).unwrap();
        assert_eq!(app_claims["email"], "alice@example.com");
        assert_eq!(app_claims["role"], "ro");
        assert_eq!(app_claims["ns"], "docs");
        assert_eq!(
            response
                .headers()
                .get_all(header::SET_COOKIE)
                .iter()
                .count(),
            4
        );
        let auth_store = SqliteAuthStore::open(&auth_path).unwrap();
        let identity = auth_store
            .external_identity(&issuer, "subject-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(identity.email.as_deref(), Some("alice@example.com"));
        assert_eq!(
            auth_store
                .auth_user(&identity.user_id)
                .await
                .unwrap()
                .unwrap()
                .display_name
                .as_deref(),
            Some("Alice")
        );
        drop(auth_store);
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_visibility_and_capabilities_enforce_claims_across_rest_and_mcp() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{EncodingKey, Header, encode};
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-capabilities-{nonce}.sqlite3"));
        let auth = AuthConfig {
            mode: "dev".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some("capability-secret".to_owned()),
            jwks_json: None,
            issuer: Some("local".to_owned()),
            audience: None,
            oidc_providers_json: None,
            auth_db_url: None,
        };
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth,
            },
            Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
        );
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let token = |claims: Value| {
            encode(
                &Header::new(Algorithm::HS256),
                &claims,
                &EncodingKey::from_secret(b"capability-secret"),
            )
            .unwrap()
        };
        let read_token = token(json!({
            "sub": "reader-1",
            "role": "ro",
            "ns": ["workflow"],
            "capabilities": ["project_view", "read_security_scope"],
            "tenant": "tenant-a",
            "workspace": "workspace-b",
            "project": "project-c",
            "iss": "local",
            "exp": now + 300,
        }));
        let visibility = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/visibility")
                    .header("authorization", format!("Bearer {read_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(visibility.status(), StatusCode::OK);
        let visibility = serde_json::from_slice::<Value>(
            &to_bytes(visibility.into_body(), 32768).await.unwrap(),
        )
        .unwrap();
        assert_eq!(visibility["current_subject"], "reader-1");
        assert_eq!(visibility["namespaces"]["execution_namespace"], "workflow");
        assert_eq!(visibility["security_scope"], "tenant-a");
        assert_eq!(
            visibility["storage_security_mapping"]["security_scope_path"],
            "tenant-a/workspace-b/project-c"
        );

        let denied_token = token(json!({
            "sub": "denied",
            "role": "ro",
            "ns": ["workflow"],
            "capabilities": ["workflow.run.read"],
            "iss": "local",
            "exp": now + 300,
        }));
        let denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/capabilities")
                    .header("authorization", format!("Bearer {denied_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);

        let admin_token = token(json!({
            "sub": "admin-1",
            "role": "rw",
            "ns": ["workflow"],
            "capabilities": ["project_view", "approve_action"],
            "iss": "local",
            "exp": now + 300,
        }));
        let approve = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/capabilities/approve")
                    .header("authorization", format!("Bearer {admin_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"action":"spawn_process","capabilities":["spawn_process","workflow.run.write"]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(approve.status(), StatusCode::ACCEPTED);
        let approve =
            serde_json::from_slice::<Value>(&to_bytes(approve.into_body(), 32768).await.unwrap())
                .unwrap();
        assert!(
            approve["effective_capabilities"]
                .as_array()
                .unwrap()
                .contains(&json!("spawn_process"))
        );
        assert_eq!(
            approve["audit_log"].as_array().unwrap().last().unwrap()["outcome"],
            "allow"
        );

        let revoke = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/capabilities/revoke")
                    .header("authorization", format!("Bearer {admin_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"capability":"spawn_process"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(revoke.status(), StatusCode::ACCEPTED);
        let revoke =
            serde_json::from_slice::<Value>(&to_bytes(revoke.into_body(), 32768).await.unwrap())
                .unwrap();
        assert!(
            !revoke["effective_capabilities"]
                .as_array()
                .unwrap()
                .contains(&json!("spawn_process"))
        );

        let mcp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("authorization", format!("Bearer {read_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"workflow.visibility_snapshot","arguments":{}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp.status(), StatusCode::OK);
        let mcp = serde_json::from_slice::<Value>(&to_bytes(mcp.into_body(), 32768).await.unwrap())
            .unwrap();
        assert_eq!(mcp["result"]["current_subject"], "reader-1");

        let syscall_list = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/syscall/v1")
                    .header("authorization", format!("Bearer {read_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(syscall_list.status(), StatusCode::OK);
        let syscall_list = serde_json::from_slice::<Value>(
            &to_bytes(syscall_list.into_body(), 16_384).await.unwrap(),
        )
        .unwrap();
        assert_eq!(syscall_list["version"], "v1");
        assert!(
            syscall_list["ops"]
                .as_array()
                .unwrap()
                .contains(&json!("resume"))
        );
        let blocked_approval = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/syscall/v1/request_approval")
                    .header("authorization", format!("Bearer {admin_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"version":"v1","op":"request_approval","args":{"action":"deny","reason":"manual deny"}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(blocked_approval.status(), StatusCode::OK);
        let blocked_approval = serde_json::from_slice::<Value>(
            &to_bytes(blocked_approval.into_body(), 16_384)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(blocked_approval["status"], "blocked");
        assert_eq!(blocked_approval["error"]["reason"], "manual deny");
        let unsupported = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/syscall/v1/send_message")
                    .header("authorization", format!("Bearer {admin_token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"version":"v1","op":"send_message","args":{}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unsupported.status(), StatusCode::NOT_IMPLEMENTED);
        let syscall_audit = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/syscall/v1/audit?limit=1")
                    .header("authorization", format!("Bearer {read_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(syscall_audit.status(), StatusCode::OK);
        let syscall_audit = serde_json::from_slice::<Value>(
            &to_bytes(syscall_audit.into_body(), 16_384).await.unwrap(),
        )
        .unwrap();
        assert_eq!(syscall_audit["events"][0]["op"], "send_message");
        assert_eq!(syscall_audit["events"][0]["status"], "error");

        let designer = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/designer/capabilities")
                    .header("authorization", format!("Bearer {read_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(designer.status(), StatusCode::FORBIDDEN);
        let designer_token = token(json!({
            "sub": "designer-1",
            "role": "ro",
            "ns": ["workflow"],
            "capabilities": ["workflow.design.inspect"],
            "iss": "local",
            "exp": now + 300,
        }));
        let designer = app
            .oneshot(
                Request::builder()
                    .uri("/designer/capabilities")
                    .header("authorization", format!("Bearer {designer_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(designer.status(), StatusCode::OK);
        let designer =
            serde_json::from_slice::<Value>(&to_bytes(designer.into_body(), 65_536).await.unwrap())
                .unwrap();
        assert_eq!(
            designer["schema_version"],
            "workflow-designer-capabilities/v1"
        );
        assert_eq!(designer["node_types"][0]["type"], "workflow_node");
        assert!(
            designer["node_types"][0]["metadata_schema"]["properties"]
                .get("wf_join")
                .is_some()
        );
        assert!(
            designer["edge_types"][0]["metadata_schema"]["properties"]
                .get("wf_predicate")
                .is_some()
        );
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_design_history_folds_branch_and_repairs_projection() {
        use axum::body::{Body, to_bytes};
        use kogwistar_store_sqlite::NewRawEntityEvent;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-design-history-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        let namespace = "wf_design:wf-history";
        let append = |event_id: &str, op: &str, payload: Value| {
            store
                .append_raw_entity_event(
                    namespace,
                    NewRawEntityEvent {
                        event_id: event_id.to_owned(),
                        entity_kind: "design_control".to_owned(),
                        entity_id: "wf-history".to_owned(),
                        op: op.to_owned(),
                        payload_json: payload.to_string(),
                    },
                )
                .unwrap();
        };
        append(
            "commit-1",
            "MUTATION_COMMITTED",
            json!({"version": 1, "prev_version": 0, "target_seq": 1, "ts_ms": 10, "designer_id": "alice", "action": "node_upsert", "entity_id": "n1"}),
        );
        append(
            "commit-2",
            "MUTATION_COMMITTED",
            json!({"version": 2, "prev_version": 1, "target_seq": 2, "ts_ms": 20, "designer_id": "alice", "action": "node_upsert", "entity_id": "n2"}),
        );
        append(
            "undo-1",
            "UNDO_APPLIED",
            json!({"from_version": 2, "to_version": 1, "ts_ms": 30, "designer_id": "alice"}),
        );
        append(
            "drop-2",
            "BRANCH_DROPPED",
            json!({"drop_from_version": 2, "drop_to_version": 2, "drop_from_seq": 2, "drop_to_seq": 2, "ts_ms": 40, "designer_id": "alice"}),
        );
        append(
            "commit-3",
            "MUTATION_COMMITTED",
            json!({"version": 3, "prev_version": 1, "target_seq": 5, "ts_ms": 50, "designer_id": "alice", "action": "edge_upsert", "entity_id": "e3"}),
        );
        store
            .replace_named_projection(
                "workflow_design",
                "wf-history",
                NamedProjectionWrite {
                    payload: json!({"current_version": 999}).as_object().unwrap().clone(),
                    last_authoritative_seq: 5,
                    last_materialized_seq: 999,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )
            .unwrap();
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/design/wf-history/history".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "workflow",
                    "capabilities": ["workflow.design.inspect"],
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let history = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(history["current_version"], 3);
        assert_eq!(history["active_tip_version"], 3);
        assert_eq!(history["allocated_max_version"], 3);
        assert_eq!(history["current_seq"], 5);
        assert_eq!(history["can_undo"], true);
        assert_eq!(history["can_redo"], false);
        assert_eq!(
            history["versions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item["version"].as_i64().unwrap())
                .collect::<Vec<_>>(),
            [0, 1, 3]
        );
        assert_eq!(history["selected_versions"].as_array().unwrap().len(), 3);
        assert_eq!(history["dropped_ranges"][0]["start_version"], 2);
        assert_eq!(history["timeline"].as_array().unwrap().len(), 5);
        let repaired = store
            .get_named_projection("workflow_design", "wf-history")
            .unwrap()
            .unwrap();
        assert_eq!(repaired.payload["current_version"], 3);
        assert_eq!(repaired.payload["active_tip_version"], 3);
        assert_eq!(repaired.last_authoritative_seq, 5);
        assert_eq!(repaired.last_materialized_seq, 5);
        assert_eq!(repaired.materialization_status, "rust_event_only");
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(service.clone()),
        );
        let mcp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"history","method":"tools/call","params":{"name":"workflow.design_history","arguments":{"workflow_id":"wf-history"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp.status(), StatusCode::OK);
        let mcp_body = to_bytes(mcp.into_body(), 65_536).await.unwrap();
        let mcp_value = serde_json::from_slice::<Value>(&mcp_body).unwrap();
        assert_eq!(mcp_value["result"]["current_version"], 3);
        drop(service);
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_design_capability_mutates_graph_and_undoes_atomically() {
        let path = std::env::temp_dir().join(format!(
            "kogwistar-design-write-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let store = SqliteStore::open(&path).unwrap();
        let principal = json!({"sub":"alice","role":"rw","ns":"workflow","capabilities":["workflow.design.write","workflow.design.inspect"]});
        let call = |method: &str, path: &str, body: Value| {
            service.execute_sync(ApiEffectRequest {
                contract_version: 1,
                method: method.to_owned(),
                path_and_query: path.to_owned(),
                body: serde_json::to_vec(&body).unwrap(),
                principal: principal.clone(),
            })
        };
        for body in [
            json!({"designer_id":"alice","node_id":"start","label":"Start","op":"start","start":true}),
            json!({"designer_id":"alice","node_id":"end","label":"End","op":"end","terminal":true}),
        ] {
            assert_eq!(
                call("POST", "/api/workflow/design/wf-write/nodes", body).status,
                200
            );
        }
        assert_eq!(call("POST", "/api/workflow/design/wf-write/edges", json!({
            "designer_id":"alice","edge_id":"edge","src":"start","dst":"end","is_default":true
        })).status, 200);
        let graph = call("GET", "/api/workflow/design/wf-write/graph", json!({}));
        let graph: Value = serde_json::from_slice(&graph.body).unwrap();
        assert_eq!(graph["nodes"].as_array().unwrap().len(), 2);
        assert_eq!(graph["edges"].as_array().unwrap().len(), 1);
        let events = store
            .replay_raw_events("wf_design:wf-write", 0, usize::MAX)
            .unwrap();
        assert_eq!(
            events
                .iter()
                .map(|event| (event.entity_kind.as_str(), event.op.as_str()))
                .collect::<Vec<_>>(),
            [
                ("node", "ADD"),
                ("design_control", "MUTATION_COMMITTED"),
                ("node", "ADD"),
                ("design_control", "MUTATION_COMMITTED"),
                ("edge", "ADD"),
                ("design_control", "MUTATION_COMMITTED"),
            ]
        );
        let delta = store
            .get_workflow_design_delta("wf-write", 3, 1)
            .unwrap()
            .unwrap();
        assert_eq!(delta.target_seq, 5);
        assert_eq!(
            serde_json::from_str::<Value>(&delta.forward_json).unwrap()["upsert_edges"][0]["id"],
            "edge"
        );
        assert_eq!(
            store
                .get_named_projection("workflow_design", "wf-write")
                .unwrap()
                .unwrap()
                .materialization_status,
            "rust_event_only"
        );
        assert_eq!(
            call(
                "POST",
                "/api/workflow/design/wf-write/undo",
                json!({"designer_id":"alice"})
            )
            .status,
            200
        );
        let graph = call("GET", "/api/workflow/design/wf-write/graph", json!({}));
        let graph: Value = serde_json::from_slice(&graph.body).unwrap();
        assert!(graph["edges"].as_array().unwrap().is_empty());
        assert_eq!(
            call(
                "POST",
                "/api/workflow/design/wf-write/redo",
                json!({"designer_id":"alice"})
            )
            .status,
            200
        );
        assert_eq!(
            call(
                "DELETE",
                "/api/workflow/design/wf-write/edges/edge",
                json!({"designer_id":"alice"})
            )
            .status,
            200
        );
        let key = "design-secret";
        let auth = AuthConfig {
            mode: "jwt".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some(key.to_owned()),
            ..AuthConfig::default()
        };
        let claims = json!({"sub":"alice","role":"rw","roles":["reader","rw"],"ns":"workflow","capabilities":["workflow.design.write","workflow.design.inspect"],"exp":4102444800_u64});
        let token = encode(
            &Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(key.as_bytes()),
        )
        .unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth,
            },
            Arc::new(service.clone()),
        );
        let mcp = app.oneshot(Request::builder().method("POST").uri("/mcp/workflow")
            .header("content-type", "application/json").header("authorization", format!("Bearer {token}"))
            .body(Body::from(r#"{"jsonrpc":"2.0","id":"node","method":"tools/call","params":{"name":"workflow.design_node_upsert","arguments":{"workflow_id":"wf-mcp-design","designer_id":"alice","node_id":"mcp-node","label":"MCP"}}}"#)).unwrap()).await.unwrap();
        assert_eq!(mcp.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&to_bytes(mcp.into_body(), 65_536).await.unwrap()).unwrap();
        assert_eq!(body["result"]["node_id"], "mcp-node");
        drop(service);
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn postgres_design_history_folds_and_repairs_projection_when_dsn_available() {
        use kogwistar_store_postgres::NewRawEntityEvent;

        let Some(dsn) = std::env::var("KOGWISTAR_TEST_PG_DSN").ok() else {
            return;
        };
        let schema = format!("rust_api_design_history_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        for (event_id, op, payload) in [
            (
                "commit-1",
                "MUTATION_COMMITTED",
                json!({"version": 1, "prev_version": 0, "target_seq": 1, "ts_ms": 10, "designer_id": "alice", "action": "node_upsert", "entity_id": "n1"}),
            ),
            (
                "commit-2",
                "MUTATION_COMMITTED",
                json!({"version": 2, "prev_version": 1, "target_seq": 2, "ts_ms": 20, "designer_id": "alice", "action": "edge_upsert", "entity_id": "e2"}),
            ),
            (
                "undo-1",
                "UNDO_APPLIED",
                json!({"from_version": 2, "to_version": 1, "ts_ms": 30, "designer_id": "alice"}),
            ),
        ] {
            service
                .store
                .append_raw_entity_event(
                    "wf_design:wf-pg-history",
                    NewRawEntityEvent {
                        event_id: event_id.to_owned(),
                        entity_kind: "design_control".to_owned(),
                        entity_id: "wf-pg-history".to_owned(),
                        op: op.to_owned(),
                        payload_json: payload.to_string(),
                    },
                )
                .await
                .unwrap();
        }
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/design/wf-pg-history/history".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "workflow",
                    "capabilities": ["workflow.design.inspect"],
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let history = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(history["current_version"], 1);
        assert_eq!(history["active_tip_version"], 2);
        assert_eq!(history["can_redo"], true);
        assert_eq!(history["latest_seq"], 3);
        let projection = service
            .store
            .get_named_projection("workflow_design", "wf-pg-history")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(projection.payload["current_version"], 1);
        assert_eq!(projection.payload["active_tip_version"], 2);
        assert_eq!(projection.last_authoritative_seq, 3);
        assert_eq!(projection.last_materialized_seq, 1);
        drop(service);

        let cleanup_dsn = dsn
            .strip_prefix("postgresql+psycopg://")
            .map(|rest| format!("postgresql://{rest}"))
            .or_else(|| {
                dsn.strip_prefix("postgresql+psycopg2://")
                    .map(|rest| format!("postgresql://{rest}"))
            })
            .unwrap_or(dsn);
        let (client, connection) = tokio_postgres::connect(&cleanup_dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        let connection_task = tokio::spawn(connection);
        client
            .batch_execute(&format!("DROP SCHEMA {schema} CASCADE"))
            .await
            .unwrap();
        drop(client);
        connection_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn postgres_service_events_fold_authoritative_history_when_dsn_available() {
        use kogwistar_store_postgres::NewRawEntityEvent;

        let Some(dsn) = std::env::var("KOGWISTAR_TEST_PG_DSN").ok() else {
            return;
        };
        let schema = format!("rust_api_service_events_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        for (event_id, entity_id, op, payload) in [
            (
                "service-event-1",
                "service_evt:svc-pg:1",
                "ADD",
                json!({
                    "id": "service_evt:svc-pg:1",
                    "metadata": {
                        "entity_type": "service_event",
                        "service_id": "svc-pg",
                        "service_event_type": "service.starting",
                        "ts_ms": 20,
                    },
                    "properties": {"payload_json": r#"{"phase":"old"}"#},
                }),
            ),
            (
                "service-event-2",
                "service_evt:svc-pg:1",
                "REPLACE",
                json!({
                    "id": "service_evt:svc-pg:1",
                    "metadata": {
                        "entity_type": "service_event",
                        "service_id": "svc-pg",
                        "service_event_type": "service.started",
                        "ts_ms": 20,
                    },
                    "properties": {"payload_json": r#"{"phase":"new"}"#},
                }),
            ),
            (
                "service-event-other",
                "service_evt:other:1",
                "ADD",
                json!({
                    "id": "service_evt:other:1",
                    "metadata": {
                        "entity_type": "service_event",
                        "service_id": "other",
                        "service_event_type": "service.healthy",
                        "ts_ms": 1,
                    },
                    "properties": {"payload_json": "{}"},
                }),
            ),
        ] {
            service
                .store
                .append_raw_entity_event(
                    "workflow",
                    NewRawEntityEvent {
                        event_id: event_id.to_owned(),
                        entity_kind: "node".to_owned(),
                        entity_id: entity_id.to_owned(),
                        op: op.to_owned(),
                        payload_json: payload.to_string(),
                    },
                )
                .await
                .unwrap();
        }
        let response = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/services/svc-pg/events?limit=10".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "workflow",
                    "capabilities": ["service.inspect", "project_view"],
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let body = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(body["service_id"], "svc-pg");
        assert_eq!(body["events"].as_array().unwrap().len(), 1);
        assert_eq!(body["events"][0]["event_type"], "service.started");
        assert_eq!(body["events"][0]["payload"]["phase"], "new");
        let principal = json!({
            "sub": "operator",
            "role": "rw",
            "ns": "workflow",
            "capabilities": ["service.manage", "service.inspect", "project_view"],
        });
        let declared = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: serde_json::to_vec(&json!({
                    "service_id": "svc-pg-trigger",
                    "target_kind": "workflow",
                    "target_ref": "wf-pg-trigger",
                    "target_config": {"conversation_id":"conv-pg-trigger"},
                }))
                .unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(declared.status, StatusCode::OK.as_u16());
        let trigger = || ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/services/svc-pg-trigger/trigger".to_owned(),
            body: serde_json::to_vec(&json!({
                "trigger_type": "external event",
                "payload": {"source":"postgres-live"},
            }))
            .unwrap(),
            principal: principal.clone(),
        };
        let service_clone = service.clone();
        let (first, second) = tokio::join!(
            service.execute_async(trigger()),
            service_clone.execute_async(trigger()),
        );
        assert_eq!(first.status, StatusCode::OK.as_u16());
        assert_eq!(second.status, StatusCode::OK.as_u16());
        let first = serde_json::from_slice::<Value>(&first.body).unwrap();
        let second = serde_json::from_slice::<Value>(&second.body).unwrap();
        let run_id = first["current_child_run_id"].as_str().unwrap();
        assert_eq!(second["current_child_run_id"], run_id);
        assert!(
            service
                .store
                .get_server_run(run_id)
                .await
                .unwrap()
                .is_some()
        );
        let lanes = service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                correlation_id: Some(run_id.to_owned()),
                limit: 10,
                ..LaneMessageFilter::default()
            })
            .await
            .unwrap();
        assert_eq!(lanes.len(), 1);
        drop(service);

        let cleanup_dsn = dsn
            .strip_prefix("postgresql+psycopg://")
            .map(|rest| format!("postgresql://{rest}"))
            .or_else(|| {
                dsn.strip_prefix("postgresql+psycopg2://")
                    .map(|rest| format!("postgresql://{rest}"))
            })
            .unwrap_or(dsn);
        let (client, connection) = tokio_postgres::connect(&cleanup_dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        let connection_task = tokio::spawn(connection);
        client
            .batch_execute(&format!("DROP SCHEMA {schema} CASCADE"))
            .await
            .unwrap();
        drop(client);
        connection_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn postgres_design_capability_mutates_and_undoes_when_dsn_available() {
        let Some(dsn) = std::env::var("KOGWISTAR_TEST_PG_DSN").ok() else {
            return;
        };
        let schema = format!("rust_api_design_write_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        let principal = json!({"sub":"alice","role":"rw","ns":"workflow","capabilities":["workflow.design.write","workflow.design.inspect"]});
        for body in [
            json!({"designer_id":"alice","node_id":"start","label":"Start","op":"start","start":true}),
            json!({"designer_id":"alice","node_id":"end","label":"End","op":"end","terminal":true}),
        ] {
            let response = service
                .execute_async(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: "/api/workflow/design/wf-pg-write/nodes".to_owned(),
                    body: serde_json::to_vec(&body).unwrap(),
                    principal: principal.clone(),
                })
                .await;
            assert_eq!(response.status, 200);
        }
        let edge = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/design/wf-pg-write/edges".to_owned(),
                body: serde_json::to_vec(
                    &json!({"designer_id":"alice","edge_id":"edge","src":"start","dst":"end"}),
                )
                .unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(edge.status, 200);
        let undo = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/design/wf-pg-write/undo".to_owned(),
                body: serde_json::to_vec(&json!({"designer_id":"alice"})).unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(undo.status, 200);
        let graph = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/design/wf-pg-write/graph".to_owned(),
                body: Vec::new(),
                principal,
            })
            .await;
        let graph: Value = serde_json::from_slice(&graph.body).unwrap();
        assert!(graph["edges"].as_array().unwrap().is_empty());
        drop(service);
        let cleanup_dsn = dsn
            .strip_prefix("postgresql+psycopg://")
            .map(|rest| format!("postgresql://{rest}"))
            .or_else(|| {
                dsn.strip_prefix("postgresql+psycopg2://")
                    .map(|rest| format!("postgresql://{rest}"))
            })
            .unwrap_or(dsn);
        let (client, connection) = tokio_postgres::connect(&cleanup_dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        let connection_task = tokio::spawn(connection);
        client
            .batch_execute(&format!("DROP SCHEMA {schema} CASCADE"))
            .await
            .unwrap();
        drop(client);
        connection_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn sqlite_service_events_fold_authoritative_history_for_rest_and_mcp() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store_sqlite::NewRawEntityEvent;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        fn node(
            id: &str,
            service_id: &str,
            event_type: &str,
            ts_ms: i64,
            event_payload_json: &str,
        ) -> Value {
            json!({
                "id": id,
                "metadata": {
                    "entity_type": "service_event",
                    "service_id": service_id,
                    "service_event_type": event_type,
                    "ts_ms": ts_ms,
                },
                "properties": {"payload_json": event_payload_json},
            })
        }

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-service-events-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        let append = |event_id: &str, entity_id: &str, op: &str, payload: Value| {
            store
                .append_raw_entity_event(
                    "workflow",
                    NewRawEntityEvent {
                        event_id: event_id.to_owned(),
                        entity_kind: "node".to_owned(),
                        entity_id: entity_id.to_owned(),
                        op: op.to_owned(),
                        payload_json: payload.to_string(),
                    },
                )
                .unwrap();
        };
        append(
            "e1-add",
            "service_evt:svc-a:1",
            "ADD",
            node(
                "service_evt:svc-a:1",
                "svc-a",
                "service.starting",
                20,
                r#"{"phase":"old"}"#,
            ),
        );
        append(
            "e2-replace",
            "service_evt:svc-a:1",
            "REPLACE",
            node(
                "service_evt:svc-a:1",
                "svc-a",
                "service.started",
                20,
                r#"{"phase":"new"}"#,
            ),
        );
        append(
            "e3-other",
            "service_evt:svc-b:1",
            "ADD",
            node("service_evt:svc-b:1", "svc-b", "service.healthy", 1, "{}"),
        );
        append(
            "e4-earlier",
            "service_evt:svc-a:2",
            "ADD",
            node(
                "service_evt:svc-a:2",
                "svc-a",
                "service.healthy",
                10,
                "not-json",
            ),
        );
        append("e5-delete", "service_evt:svc-a:2", "DELETE", json!({}));
        append(
            "e6-unrelated",
            "node-a",
            "ADD",
            json!({"id":"node-a","metadata":{"entity_type":"node"}}),
        );

        let service = SqliteRunApplicationService::open(&path).unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(service.clone()),
        );
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/services/svc-a/events?limit=1")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body =
            serde_json::from_slice::<Value>(&to_bytes(response.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(body["service_id"], "svc-a");
        assert_eq!(body["events"].as_array().unwrap().len(), 1);
        assert_eq!(body["events"][0]["event_type"], "service.started");
        assert_eq!(body["events"][0]["payload"]["phase"], "new");

        let missing = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/services/missing/events")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::OK);
        let missing =
            serde_json::from_slice::<Value>(&to_bytes(missing.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(missing["events"], json!([]));

        let forbidden = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/services/svc-a/events".to_owned(),
                body: Vec::new(),
                principal: json!({"sub":"alice","role":"ro","capabilities":[]}),
            })
            .await;
        assert_eq!(forbidden.status, StatusCode::FORBIDDEN.as_u16());

        let mcp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"service-events","method":"tools/call","params":{"name":"workflow.service_events","arguments":{"service_id":"svc-a","limit":10}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp.status(), StatusCode::OK);
        let mcp =
            serde_json::from_slice::<Value>(&to_bytes(mcp.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(mcp["result"]["events"][0]["event_type"], "service.started");

        drop(service);
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_service_repair_rebuilds_disposable_projection_for_rest_and_mcp() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store_sqlite::NewRawEntityEvent;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-service-repair-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        let definition = json!({
            "id": "service_def:svc-repair:1",
            "metadata": {
                "entity_type": "service_definition",
                "service_id": "svc-repair",
                "updated_at_ms": 10,
                "created_at_ms": 10,
            },
            "properties": {
                "service_id": "svc-repair",
                "service_kind": "daemon",
                "target_kind": "workflow",
                "target_ref": "wf-repair",
                "target_config_json": "{}",
                "enabled": true,
                "autostart": false,
                "restart_policy_json": "{}",
                "heartbeat_ttl_ms": 60000,
                "trigger_specs_json": "[]",
                "security_scope": "workflow",
                "storage_namespace": "workflow",
                "execution_namespace": "workflow",
                "created_at_ms": 10,
                "updated_at_ms": 10,
            },
        });
        store
            .append_raw_entity_event(
                "workflow",
                NewRawEntityEvent {
                    event_id: "definition".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: "service_def:svc-repair:1".to_owned(),
                    op: "ADD".to_owned(),
                    payload_json: definition.to_string(),
                },
            )
            .unwrap();
        store
            .append_raw_entity_event(
                "workflow",
                NewRawEntityEvent {
                    event_id: "disabled".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: "service_evt:svc-repair:2".to_owned(),
                    op: "ADD".to_owned(),
                    payload_json: json!({
                        "id": "service_evt:svc-repair:2",
                        "metadata": {
                            "entity_type": "service_event",
                            "service_id": "svc-repair",
                            "service_event_type": "service.stopped",
                            "ts_ms": 20,
                        },
                        "properties": {"payload_json": r#"{"enabled":false}"#},
                    })
                    .to_string(),
                },
            )
            .unwrap();
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let principal = json!({
            "sub": "operator",
            "role": "rw",
            "ns": "workflow",
            "capabilities": ["service.manage", "service.inspect", "project_view"],
        });
        let declared = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: serde_json::to_vec(&json!({
                    "service_id": "svc-declared",
                    "service_kind": "daemon",
                    "target_kind": "workflow",
                    "target_ref": "wf-declared",
                    "target_config": {"conversation_id": "conv-declared"},
                    "enabled": true,
                    "autostart": true,
                    "heartbeat_ttl_ms": 1234,
                    "trigger_specs": [{"type":"external event","enabled":true}],
                }))
                .unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(declared.status, StatusCode::OK.as_u16());
        let declared = serde_json::from_slice::<Value>(&declared.body).unwrap();
        assert_eq!(declared["service_id"], "svc-declared");
        assert_eq!(declared["target_ref"], "wf-declared");
        assert_eq!(declared["autostart"], true);
        assert_eq!(declared["health_status"], "degraded");
        assert_eq!(declared["heartbeat_ttl_ms"], 1234);
        let triggered = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services/svc-declared/trigger".to_owned(),
                body: serde_json::to_vec(&json!({
                    "trigger_type": "external event",
                    "payload": {"source": "test"},
                }))
                .unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(triggered.status, StatusCode::OK.as_u16());
        let triggered = serde_json::from_slice::<Value>(&triggered.body).unwrap();
        let run_id = triggered["current_child_run_id"]
            .as_str()
            .unwrap()
            .to_owned();
        assert_eq!(triggered["last_trigger_type"], "external event");
        assert_eq!(triggered["current_child_status"], "queued");
        assert!(store.get_server_run(&run_id).unwrap().is_some());
        let lanes = store
            .list_projected_lane_messages(kogwistar_store::LaneMessageFilter {
                correlation_id: Some(run_id.clone()),
                limit: 10,
                ..kogwistar_store::LaneMessageFilter::default()
            })
            .unwrap();
        assert_eq!(lanes.len(), 1);
        let retriggered = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services/svc-declared/trigger".to_owned(),
                body: serde_json::to_vec(&json!({
                    "trigger_type": "external event",
                    "payload": {},
                }))
                .unwrap(),
                principal: principal.clone(),
            })
            .await;
        let retriggered = serde_json::from_slice::<Value>(&retriggered.body).unwrap();
        assert_eq!(retriggered["current_child_run_id"], run_id);
        assert_eq!(
            store
                .list_projected_lane_messages(kogwistar_store::LaneMessageFilter {
                    correlation_id: Some(run_id.clone()),
                    limit: 10,
                    ..kogwistar_store::LaneMessageFilter::default()
                })
                .unwrap()
                .len(),
            1
        );
        let repaired = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services/svc-repair/repair".to_owned(),
                body: Vec::new(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(repaired.status, StatusCode::OK.as_u16());
        let body = serde_json::from_slice::<Value>(&repaired.body).unwrap();
        assert_eq!(body["projection"]["service_id"], "svc-repair");
        assert_eq!(body["projection"]["enabled"], false);
        assert_eq!(body["projection"]["lifecycle_status"], "stopped");
        store
            .clear_named_projection("service_registry", "svc-repair")
            .unwrap();

        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(service.clone()),
        );
        let mcp_trigger = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"trigger","method":"tools/call","params":{"name":"workflow.service_trigger","arguments":{"service_id":"svc-declared","trigger_type":"external event","payload":{"source":"mcp"}}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_trigger.status(), StatusCode::OK);
        let mcp_trigger = serde_json::from_slice::<Value>(
            &to_bytes(mcp_trigger.into_body(), 16_384).await.unwrap(),
        )
        .unwrap();
        assert_eq!(mcp_trigger["result"]["current_child_run_id"], run_id);
        let mcp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"repair","method":"tools/call","params":{"name":"workflow.service_repair","arguments":{"service_id":"svc-repair"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp.status(), StatusCode::OK);
        let mcp =
            serde_json::from_slice::<Value>(&to_bytes(mcp.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(mcp["result"]["projection"]["enabled"], false);
        let forbidden = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services/repair".to_owned(),
                body: Vec::new(),
                principal: json!({"sub":"reader","role":"ro","capabilities":["service.inspect"]}),
            })
            .await;
        assert_eq!(forbidden.status, StatusCode::FORBIDDEN.as_u16());
        let disabled = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services/svc-repair/disable".to_owned(),
                body: Vec::new(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(disabled.status, StatusCode::OK.as_u16());
        let disabled = serde_json::from_slice::<Value>(&disabled.body).unwrap();
        assert_eq!(disabled["enabled"], false);
        assert_eq!(disabled["lifecycle_status"], "stopped");
        let enabled = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services/svc-repair/enable".to_owned(),
                body: Vec::new(),
                principal,
            })
            .await;
        assert_eq!(enabled.status, StatusCode::OK.as_u16());
        let enabled = serde_json::from_slice::<Value>(&enabled.body).unwrap();
        assert_eq!(enabled["enabled"], true);
        // Python rebuild keeps the prior stopped lifecycle until the next
        // supervisor health tick emits a new lifecycle event.
        assert_eq!(enabled["lifecycle_status"], "stopped");
        let heartbeat = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services/svc-repair/heartbeat".to_owned(),
                body: serde_json::to_vec(&json!({
                    "instance_id": "worker-1",
                    "payload": {"last_error": "recovering"},
                }))
                .unwrap(),
                principal: json!({
                    "sub": "worker",
                    "role": "rw",
                    "capabilities": ["service.heartbeat", "service.manage"],
                }),
            })
            .await;
        assert_eq!(heartbeat.status, StatusCode::OK.as_u16());
        let heartbeat = serde_json::from_slice::<Value>(&heartbeat.body).unwrap();
        assert_eq!(heartbeat["instance_id"], "worker-1");
        assert_eq!(heartbeat["health_status"], "healthy");
        assert_eq!(heartbeat["last_error"], "recovering");
        let history = store.replay_raw_events("workflow", 0, usize::MAX).unwrap();
        assert!(history.iter().any(|event| {
            event.payload_json.contains("service.enabled")
                && event.entity_id.starts_with("service_evt:svc-repair:")
        }));
        assert!(history.iter().any(|event| {
            event.payload_json.contains("service.error_changed")
                && event.entity_id.starts_with("service_evt:svc-repair:")
        }));
        drop(service);
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn service_route_and_event_fold_reject_unrelated_inputs() {
        assert_eq!(
            service_route("/api/workflow/services/svc/events?limit=1"),
            Some(("svc".to_owned(), ServiceRoute::Events))
        );
        assert_eq!(service_route("/api/workflow/services/svc/unknown"), None);
        assert_eq!(
            service_repair_route("/api/workflow/services/repair?limit=2"),
            Some(None)
        );
        assert_eq!(
            service_repair_route("/api/workflow/services/svc/repair"),
            Some(Some("svc".to_owned()))
        );
        assert_eq!(
            service_enabled_route("/api/workflow/services/svc/disable"),
            Some(("svc".to_owned(), false))
        );
        assert!(
            service_event_values(
                [(
                    1,
                    "node".to_owned(),
                    "bad".to_owned(),
                    "ADD".to_owned(),
                    "not-json".to_owned(),
                )],
                "svc",
                10,
            )
            .is_empty()
        );
        let projection = json!({"last_triggered_at_ms": 900});
        assert!(service_trigger_is_suppressed(
            &json!({"properties":{"enabled":false,"trigger_specs_json":"[]"}}),
            &projection,
            "external event",
            1_000,
        ));
        assert!(service_trigger_is_suppressed(
            &json!({"properties":{"enabled":true,"trigger_specs_json":r#"[{"type":"external event","enabled":true,"cooldown_ms":200}]"#}}),
            &projection,
            "external event",
            1_000,
        ));
        assert!(!service_trigger_is_suppressed(
            &json!({"properties":{"enabled":true,"trigger_specs_json":r#"[{"type":"graph change","enabled":false}]"#}}),
            &projection,
            "external event",
            1_000,
        ));
        assert_eq!(PENDING_SYSCALL_CUTOVER_OPS.len(), 5);
        assert!(PENDING_SYSCALL_CUTOVER_OPS.contains(&"invoke_tool"));
        assert!(!PENDING_SYSCALL_CUTOVER_OPS.contains(&"spawn_process"));
    }

    #[test]
    fn document_graph_validation_matches_provenance_and_pointer_contract() {
        let request = |body: Value| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/document.validate_graph".to_owned(),
            body: serde_json::to_vec(&body).unwrap(),
            principal: json!({}),
        };
        let valid = document_graph_validation_effect(&request(json!({
            "doc_id":"doc-validate",
            "insertion_method":"pytest",
            "nodes":[{
                "id":"node-1",
                "label":"Node",
                "type":"entity",
                "summary":"Valid node",
                "metadata":{"pointers":[{
                    "source_cluster_id":"cluster-1",
                    "start_char":0,
                    "end_char":4,
                    "verbatim_text":"Node",
                }]},
            }],
            "edges":[{
                "id":"edge-1",
                "label":"Edge",
                "type":"relationship",
                "summary":"Valid edge",
                "relation":"links",
                "source_ids":["node-1"],
                "target_ids":["node-1"],
                "source_edge_ids":[],
                "target_edge_ids":[],
                "mentions":[{"spans":[{
                    "collection_page_url":"doc://doc-validate",
                    "document_page_url":"doc://doc-validate#cluster-1",
                    "doc_id":"doc-validate",
                    "insertion_method":"pytest",
                    "start_char":0,
                    "end_char":4,
                }]}],
            }],
        })))
        .unwrap();
        let valid = serde_json::from_slice::<Value>(&valid.body).unwrap();
        assert_eq!(valid, json!({"ok":true,"node_errors":{},"edge_errors":{}}));

        let invalid = document_graph_validation_effect(&request(json!({
            "doc_id":"doc-invalid",
            "nodes":[{"id":"node-bad","label":"Bad","type":"entity","summary":"","mentions":[]}],
            "edges":[{"id":"edge-bad","label":"Bad","type":"relationship","summary":"","mentions":[],"relation":"links","source_ids":[],"target_ids":[],"source_edge_ids":[],"target_edge_ids":[]}],
        })))
        .unwrap();
        let invalid = serde_json::from_slice::<Value>(&invalid.body).unwrap();
        assert_eq!(invalid["ok"], false);
        assert!(invalid["node_errors"]["node-bad"].is_string());
        assert!(invalid["edge_errors"]["edge-bad"].is_string());
    }

    #[test]
    fn visualization_shells_embed_packaged_templates_without_jinja_drift() {
        for (path, marker) in [
            ("/viz/cytoscape?doc_id=doc-1&mode=graph", "Cytoscape"),
            ("/viz/d3?doc_id=doc-1&mode=graph", "D3 graph viewer"),
            ("/viz/go?doc_id=doc-1&mode=graph", "GoJS"),
        ] {
            let response = visualization_shell_effect(&ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: path.to_owned(),
                body: Vec::new(),
                principal: json!({}),
            })
            .unwrap();
            assert_eq!(response.status, StatusCode::OK.as_u16());
            assert_eq!(response.content_type, "text/html; charset=utf-8");
            let body = String::from_utf8(response.body).unwrap();
            assert!(body.contains(marker));
            assert!(!body.contains("{{"));
            assert!(!body.contains("{%"));
        }
    }

    #[test]
    fn runtime_start_tracks_only_reachable_join_obligations() {
        assert_eq!(runtime_join_outstanding(4, 0b0101), vec![1, 0, 1, 0]);
    }

    #[test]
    fn python_runtime_wire_vectors_round_trip_through_rust_dtos() {
        let vectors: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/adr015-runtime-wire-v1.json"
        )))
        .unwrap();
        let submit: RuntimeSubmitRun = serde_json::from_value(vectors["submit"].clone()).unwrap();
        let claim: RuntimeClaimRequest = serde_json::from_value(vectors["claim"].clone()).unwrap();
        let result: RuntimeWorkerResultRequest =
            serde_json::from_value(vectors["result_effect"].clone()).unwrap();
        let transition_result: RuntimeWorkerResultRequest =
            serde_json::from_value(vectors["result_transition"].clone()).unwrap();
        let resume: RuntimeResumeRequest =
            serde_json::from_value(vectors["resume"].clone()).unwrap();
        let sqlite_work: RuntimeClaimedWork =
            serde_json::from_value(vectors["claimed_work_sqlite"].clone()).unwrap();
        let postgres_work: RuntimeClaimedWork =
            serde_json::from_value(vectors["claimed_work_postgres"].clone()).unwrap();
        let step_execute: kogwistar_runtime::RuntimeStepExecutePayload =
            serde_json::from_value(vectors["step_execute"].clone()).unwrap();
        assert_eq!(serde_json::to_value(submit).unwrap(), vectors["submit"]);
        assert_eq!(serde_json::to_value(claim).unwrap(), vectors["claim"]);
        assert_eq!(
            serde_json::to_value(result).unwrap(),
            vectors["result_effect"]
        );
        assert_eq!(
            serde_json::to_value(transition_result).unwrap(),
            vectors["result_transition"]
        );
        assert_eq!(serde_json::to_value(resume).unwrap(), vectors["resume"]);
        assert_eq!(
            serde_json::to_value(sqlite_work).unwrap(),
            vectors["claimed_work_sqlite"]
        );
        assert_eq!(
            serde_json::to_value(postgres_work).unwrap(),
            vectors["claimed_work_postgres"]
        );
        assert_eq!(
            serde_json::to_value(step_execute).unwrap(),
            vectors["step_execute"]
        );
    }

    #[test]
    fn runtime_start_builder_preserves_all_python_wire_identity() {
        let vectors: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/adr015-runtime-wire-v1.json"
        )))
        .unwrap();
        let mut payload = vectors["submit"].clone();
        payload["expected_event_seq"] = json!(7);
        let transition = runtime_start_transition(&payload, "run-1");
        assert_eq!(transition.run_id, "run-1");
        assert_eq!(transition.workflow_id, "wf-1");
        assert_eq!(transition.conversation_id, "conv-1");
        assert_eq!(transition.user_id.as_deref(), Some("user-1"));
        assert_eq!(transition.user_turn_node_id, None);
        assert_eq!(transition.node_id.as_deref(), Some("entry"));
        assert_eq!(transition.token_id.as_deref(), Some("run-1"));
        assert_eq!(transition.expected_event_seq, 7);
        let frontier = transition.frontier.unwrap();
        assert_eq!(frontier.pending[0].0, "entry");
        assert_eq!(frontier.pending[0].1, 1);
        assert_eq!(frontier.join_node_ids, vec!["join"]);
        assert_eq!(frontier.join_outstanding, vec![1]);
    }

    #[tokio::test]
    async fn sqlite_run_service_handles_get_events_and_cancel() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store::NamedProjectionWrite;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-api-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        store
            .create_server_run(kogwistar_store::ServerRunCreate {
                run_id: "run-1".to_owned(),
                conversation_id: "conversation-1".to_owned(),
                workflow_id: "workflow-1".to_owned(),
                user_id: Some("user-1".to_owned()),
                user_turn_node_id: "turn-1".to_owned(),
                status: "running".to_owned(),
            })
            .unwrap();
        store
            .append_server_run_event("run-1", "run.started", r#"{"source":"test"}"#.to_owned())
            .unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
        );

        let submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"run_id":"caller-run-id","workflow_id":"workflow-submit","conversation_id":"conversation-submit","initial_state":{"seed":1},"priority_class":"foreground","runtime_kind":"sync","start_node_id":"entry","node_ops":{"entry":"begin"}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(submit.status(), StatusCode::ACCEPTED);
        let submit_body = to_bytes(submit.into_body(), 8192).await.unwrap();
        let submitted = serde_json::from_slice::<Value>(&submit_body).unwrap();
        let submitted_run_id = submitted["run_id"].as_str().unwrap();
        assert_eq!(submitted_run_id, "caller-run-id");
        let retried_submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"run_id":"caller-run-id","workflow_id":"workflow-submit","conversation_id":"conversation-submit","initial_state":{"seed":1},"priority_class":"foreground","runtime_kind":"sync","start_node_id":"entry","node_ops":{"entry":"begin"}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(retried_submit.status(), StatusCode::ACCEPTED);
        let retried_submit = serde_json::from_slice::<Value>(
            &to_bytes(retried_submit.into_body(), 8192).await.unwrap(),
        )
        .unwrap();
        assert_eq!(retried_submit["idempotent"], true);
        assert_eq!(retried_submit["run_id"], "caller-run-id");

        let conflicting_submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"run_id":"caller-run-id","workflow_id":"workflow-submit","conversation_id":"conversation-submit","initial_state":{"seed":2},"priority_class":"foreground","runtime_kind":"sync","start_node_id":"entry","node_ops":{"entry":"begin"}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(conflicting_submit.status(), StatusCode::CONFLICT);
        assert_eq!(
            store
                .get_server_run(submitted_run_id)
                .unwrap()
                .unwrap()
                .status,
            "queued"
        );
        assert_eq!(
            store
                .list_server_run_events(submitted_run_id, 0, 10)
                .unwrap()[0]
                .event_type,
            "run.created"
        );
        let claimed = store
            .claim_projected_lane_messages("workflow", "workflow-runtime", "python-worker-1", 1, 60)
            .unwrap();
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].run_id.as_deref(), Some(submitted_run_id));
        assert_eq!(claimed[0].msg_type, "workflow.run.execute");
        assert_eq!(claimed[0].claimed_by.as_deref(), Some("python-worker-1"));

        let submitted_message_id = claimed[0].message_id.clone();
        // Requeue the direct-store claim; HTTP claim owns the tested handoff.
        store
            .requeue_projected_lane_message(&submitted_message_id, "python-worker-1", None, 0)
            .unwrap();
        let claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"python-worker-2","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let claim_status = claim.status();
        let claim_body = to_bytes(claim.into_body(), 16384).await.unwrap();
        assert_eq!(
            claim_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&claim_body)
        );
        let claim_value = serde_json::from_slice::<Value>(&claim_body).unwrap();
        let work = &claim_value["work"][0];
        assert_eq!(work["run_id"], submitted_run_id);
        assert_eq!(work["step_id"], "entry");
        assert_eq!(work["payload"]["start_node_id"], "entry");
        assert_eq!(work["payload"]["op"], "begin");
        let result_payload = json!({
            "handoff": {
                "message_id": work["message_id"],
                "claimed_by": work["claimed_by"],
                "run_id": work["run_id"],
                "step_id": work["step_id"],
                "correlation_id": work["correlation_id"],
            },
            "transition": {
                "contract_version": 1,
                "transition_id": format!("result-{submitted_run_id}"),
                "expected_event_seq": work["expected_event_seq"],
                "kind": "recorded_step_success",
                "run_id": submitted_run_id,
                "workflow_id": "workflow-submit",
                "conversation_id": "conversation-submit",
                "step_seq": 0,
                "node_id": "entry",
                "token_id": submitted_run_id,
                "parent_token_id": null,
                "state_update": [["u", {"answer": "worker"}]],
                "frontier": {
                    "pending": [],
                    "suspended": [],
                    "join_node_ids": [],
                    "join_outstanding": [],
                    "join_waiters": {},
                },
                "result": {"answer": "worker"},
            }
        });
        let result_bytes = serde_json::to_vec(&result_payload).unwrap();
        let result = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(result_bytes.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let result_status = result.status();
        let result_body = to_bytes(result.into_body(), 16384).await.unwrap();
        assert_eq!(
            result_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&result_body)
        );
        let duplicate_result = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(result_bytes))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(duplicate_result.status(), StatusCode::OK);
        assert_eq!(
            store
                .get_server_run(submitted_run_id)
                .unwrap()
                .unwrap()
                .status,
            "succeeded"
        );
        assert_eq!(
            store
                .get_projected_lane_message(&submitted_message_id)
                .unwrap()
                .unwrap()
                .status,
            "completed"
        );
        assert_eq!(
            store
                .read_recorded_runtime_state(
                    submitted_run_id,
                    "workflow-submit",
                    "conversation-submit",
                )
                .unwrap()
                .unwrap()
                .state["answer"],
            "worker"
        );
        for (path, required_keys) in [
            (
                "/api/workflow/resources",
                &[
                    "scheduler",
                    "runs",
                    "cost_ledger",
                    "budget_model",
                    "migration",
                ][..],
            ),
            ("/api/workflow/budget", &["cost_ledger", "budget_model"][..]),
            (
                "/api/workflow/budget/history?limit=10",
                &["cost_ledger", "events"][..],
            ),
            (
                "/api/workflow/lane/progress?conversation_id=conversation-submit&limit=10",
                &["items", "total"][..],
            ),
            (
                "/api/workflow/operator/dashboard?limit=10",
                &["process_table", "message_queue", "resources"][..],
            ),
            (
                "/api/workflow/scheduler/timeline?limit=10",
                &["run_id", "events"][..],
            ),
            (
                "/api/workflow/dead-letters?limit=10",
                &["runs", "limit"][..],
            ),
            ("/api/workflow/catalog/ops", &[][..]),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(path)
                        .header("x-kogwistar-roles", "reader")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{path}");
            let body = to_bytes(response.into_body(), 65_536).await.unwrap();
            let value = serde_json::from_slice::<Value>(&body).unwrap();
            if path == "/api/workflow/catalog/ops" {
                assert_eq!(value[0]["op"], "start");
                assert_eq!(value[1]["op"], "llm_call");
            }
            for key in required_keys {
                assert!(value.get(key).is_some(), "{path} lacks {key}: {value}");
            }
        }
        let dead_lane = |message_id: &str, run_id: &str| NewProjectedLaneMessage {
            message_id: message_id.to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            conversation_id: "dead-conversation".to_owned(),
            recipient_id: "python-worker".to_owned(),
            sender_id: "rust-scheduler".to_owned(),
            msg_type: "workflow.run.execute".to_owned(),
            status: "dead-letter".to_owned(),
            created_at: 1,
            available_at: 1,
            run_id: Some(run_id.to_owned()),
            step_id: Some("start".to_owned()),
            correlation_id: Some(format!("corr-{run_id}")),
            payload_json: Some("{}".to_owned()),
            error_json: Some(r#"{"error":"boom"}"#.to_owned()),
        };
        store
            .project_lane_message(dead_lane("dead-rest-message", "dead-rest-run"))
            .unwrap();
        let replay = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/dead-letters/dead-rest-run/replay")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(replay.status(), StatusCode::OK);
        let replay =
            serde_json::from_slice::<Value>(&to_bytes(replay.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(replay["replayed"], true);
        assert_eq!(replay["dead_letter"]["status"], "dead-letter");
        assert_eq!(
            store
                .get_projected_lane_message("dead-rest-message")
                .unwrap()
                .unwrap()
                .status,
            "pending"
        );
        store
            .update_projected_lane_message_status("dead-rest-message", "completed", None)
            .unwrap();
        let missing_replay = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/dead-letters/missing/replay")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let missing_replay = serde_json::from_slice::<Value>(
            &to_bytes(missing_replay.into_body(), 16_384).await.unwrap(),
        )
        .unwrap();
        assert_eq!(
            missing_replay,
            json!({"run_id": "missing", "replayed": false})
        );
        for prefix in ["/api/runs", "/api/workflow/runs"] {
            let steps = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(format!("{prefix}/{submitted_run_id}/steps"))
                        .header("x-kogwistar-roles", "reader")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(steps.status(), StatusCode::OK);
            let steps_body = to_bytes(steps.into_body(), 16_384).await.unwrap();
            let steps_value = serde_json::from_slice::<Value>(&steps_body).unwrap();
            assert_eq!(steps_value["steps"].as_array().unwrap().len(), 1);
            assert_eq!(steps_value["steps"][0]["workflow_node_id"], "entry");
            assert_eq!(steps_value["steps"][0]["result"]["answer"], "worker");

            let checkpoints = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(format!("{prefix}/{submitted_run_id}/checkpoints"))
                        .header("x-kogwistar-roles", "reader")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(checkpoints.status(), StatusCode::OK);
            let checkpoints_body = to_bytes(checkpoints.into_body(), 16_384).await.unwrap();
            let checkpoints_value = serde_json::from_slice::<Value>(&checkpoints_body).unwrap();
            assert_eq!(
                checkpoints_value["checkpoints"].as_array().unwrap().len(),
                1
            );
            assert_eq!(
                checkpoints_value["checkpoints"][0]["state"]["answer"],
                "worker"
            );

            for suffix in ["checkpoints/0", "replay?target_step_seq=0"] {
                let response = app
                    .clone()
                    .oneshot(
                        Request::builder()
                            .uri(format!("{prefix}/{submitted_run_id}/{suffix}"))
                            .header("x-kogwistar-roles", "reader")
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let body = to_bytes(response.into_body(), 16_384).await.unwrap();
                assert_eq!(
                    serde_json::from_slice::<Value>(&body).unwrap()["state"]["answer"],
                    "worker"
                );
            }
        }
        for (tool_name, arguments, expected_key) in [
            (
                "workflow.run_checkpoint_get",
                json!({"run_id": submitted_run_id, "step_seq": 0}),
                "state",
            ),
            (
                "workflow.run_replay",
                json!({"run_id": submitted_run_id, "target_step_seq": 0}),
                "state",
            ),
            ("workflow.budget_snapshot", json!({}), "cost_ledger"),
            (
                "workflow.operator_dashboard",
                json!({"limit": 10}),
                "resources",
            ),
            (
                "workflow.scheduler_timeline",
                json!({"run_id": submitted_run_id, "limit": 10}),
                "events",
            ),
            ("workflow.dead_letters", json!({"limit": 10}), "runs"),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/mcp/workflow")
                        .header("content-type", "application/json")
                        .header("x-kogwistar-roles", "reader")
                        .body(Body::from(
                            serde_json::to_vec(&json!({
                                "jsonrpc": "2.0",
                                "id": tool_name,
                                "method": "tools/call",
                                "params": {"name": tool_name, "arguments": arguments},
                            }))
                            .unwrap(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = to_bytes(response.into_body(), 65_536).await.unwrap();
            let value = serde_json::from_slice::<Value>(&body).unwrap();
            assert!(
                value["result"].get(expected_key).is_some(),
                "{tool_name} failed: {value}"
            );
        }
        store
            .project_lane_message(dead_lane("dead-mcp-message", "dead-mcp-run"))
            .unwrap();
        let replay = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"dead-replay","method":"tools/call","params":{"name":"workflow.dead_letter_replay","arguments":{"run_id":"dead-mcp-run"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let replay =
            serde_json::from_slice::<Value>(&to_bytes(replay.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(replay["result"]["replayed"], true);
        assert_eq!(
            store
                .get_projected_lane_message("dead-mcp-message")
                .unwrap()
                .unwrap()
                .status,
            "pending"
        );
        store
            .update_projected_lane_message_status("dead-mcp-message", "completed", None)
            .unwrap();

        let orphan_lane = |message_id: &str, inbox_id: &str| NewProjectedLaneMessage {
            message_id: message_id.to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: inbox_id.to_owned(),
            conversation_id: "orphan-conversation".to_owned(),
            recipient_id: "python-worker".to_owned(),
            sender_id: "rust-scheduler".to_owned(),
            msg_type: "workflow.run.execute".to_owned(),
            status: "pending".to_owned(),
            created_at: 1,
            available_at: 0,
            run_id: Some(format!("run-{message_id}")),
            step_id: Some("start".to_owned()),
            correlation_id: Some(format!("corr-{message_id}")),
            payload_json: Some("{}".to_owned()),
            error_json: None,
        };
        store
            .project_lane_message(orphan_lane("expired-orphan", "repair-rest"))
            .unwrap();
        store
            .project_lane_message(orphan_lane("active-claim", "repair-active"))
            .unwrap();
        store
            .project_lane_message(orphan_lane("mcp-orphan", "repair-mcp"))
            .unwrap();
        store
            .claim_projected_lane_messages("workflow", "repair-rest", "expired-owner", 1, -1)
            .unwrap();
        store
            .claim_projected_lane_messages("workflow", "repair-active", "active-owner", 1, 60)
            .unwrap();
        store
            .claim_projected_lane_messages("workflow", "repair-mcp", "mcp-owner", 1, -1)
            .unwrap();
        let repaired = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/messages/repair-orphans?inbox_id=repair-rest&limit=10")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(repaired.status(), StatusCode::OK);
        let repaired =
            serde_json::from_slice::<Value>(&to_bytes(repaired.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(repaired["repaired_message_ids"], json!(["expired-orphan"]));
        assert_eq!(
            store
                .get_projected_lane_message("active-claim")
                .unwrap()
                .unwrap()
                .status,
            "claimed"
        );
        let repaired = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"orphan-repair","method":"tools/call","params":{"name":"workflow.message_orphans_repair","arguments":{"inbox_id":"repair-mcp","limit":10}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let repaired =
            serde_json::from_slice::<Value>(&to_bytes(repaired.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(
            repaired["result"]["repaired_message_ids"],
            json!(["mcp-orphan"])
        );
        for message_id in ["expired-orphan", "active-claim", "mcp-orphan"] {
            store
                .update_projected_lane_message_status(message_id, "completed", None)
                .unwrap();
        }

        store
            .replace_named_projection(
                "service_registry",
                "svc-b",
                NamedProjectionWrite {
                    payload: serde_json::Map::from_iter([
                        ("service_id".to_owned(), json!("svc-b")),
                        ("enabled".to_owned(), json!(true)),
                        ("health_status".to_owned(), json!("healthy")),
                    ]),
                    last_authoritative_seq: 1,
                    last_materialized_seq: 1,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )
            .unwrap();
        store
            .replace_named_projection(
                "service_registry",
                "svc-a",
                NamedProjectionWrite {
                    payload: serde_json::Map::from_iter([
                        ("service_id".to_owned(), json!("svc-a")),
                        ("enabled".to_owned(), json!(false)),
                        ("health_status".to_owned(), json!("stopped")),
                    ]),
                    last_authoritative_seq: 2,
                    last_materialized_seq: 2,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )
            .unwrap();
        let services = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/services?limit=1")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(services.status(), StatusCode::OK);
        let services =
            serde_json::from_slice::<Value>(&to_bytes(services.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(services["services"][0]["service_id"], "svc-a");
        let service = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"service-get","method":"tools/call","params":{"name":"workflow.service_get","arguments":{"service_id":"svc-b"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let service =
            serde_json::from_slice::<Value>(&to_bytes(service.into_body(), 16_384).await.unwrap())
                .unwrap();
        assert_eq!(service["result"]["health_status"], "healthy");
        let missing_service = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/services/missing")
                    .header("x-kogwistar-roles", "reader,rw")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_service.status(), StatusCode::NOT_FOUND);

        // A non-terminal result must durably schedule exactly one successor.
        let two_step_submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"workflow_id":"workflow-two-step","conversation_id":"conversation-two-step","initial_state":{"seed":2}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let two_step_body = to_bytes(two_step_submit.into_body(), 8192).await.unwrap();
        let two_step_run_id = serde_json::from_slice::<Value>(&two_step_body).unwrap()["run_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let first_claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"python-worker-3","limit":10,"lease_seconds":1}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let first_claim_body = to_bytes(first_claim.into_body(), 16384).await.unwrap();
        let first_claim_value = serde_json::from_slice::<Value>(&first_claim_body).unwrap();
        let first_work = first_claim_value["work"]
            .as_array()
            .unwrap()
            .iter()
            .find(|work| work["run_id"] == two_step_run_id)
            .unwrap();
        let successor_token_id = format!("{two_step_run_id}:finish");
        let mut first_result_payload = json!({
            "handoff": {
                "message_id": first_work["message_id"],
                "claimed_by": first_work["claimed_by"],
                "run_id": first_work["run_id"],
                "step_id": first_work["step_id"],
                "correlation_id": first_work["correlation_id"],
            },
            "transition": {
                "contract_version": 1,
                "transition_id": format!("first-{two_step_run_id}"),
                "expected_event_seq": first_work["expected_event_seq"],
                "kind": "recorded_step_success",
                "run_id": two_step_run_id,
                "workflow_id": "workflow-two-step",
                "conversation_id": "conversation-two-step",
                "step_seq": 0,
                "node_id": "start",
                "token_id": two_step_run_id,
                "parent_token_id": null,
                "state_update": [["u", {"first": true}]],
                "frontier": {
                    "pending": [["finish", 0, successor_token_id, two_step_run_id]],
                    "suspended": [],
                    "join_node_ids": [],
                    "join_outstanding": [],
                    "join_waiters": {},
                },
            }
        });
        let mut forged_owner_payload = first_result_payload.clone();
        forged_owner_payload["handoff"]["claimed_by"] = json!("forged-worker");
        let forged_owner = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&forged_owner_payload).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(forged_owner.status(), StatusCode::CONFLICT);
        let first_message_id = first_work["message_id"].as_str().unwrap();
        let lane_after_forgery = store
            .get_projected_lane_message(first_message_id)
            .unwrap()
            .unwrap();
        assert_eq!(lane_after_forgery.status, "claimed");
        assert_eq!(
            lane_after_forgery.claimed_by.as_deref(),
            Some("python-worker-3")
        );

        let mut invalid_transition_payload = first_result_payload.clone();
        invalid_transition_payload["transition"]["expected_event_seq"] = json!(999_999);
        let invalid_transition = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&invalid_transition_payload).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid_transition.status(), StatusCode::CONFLICT);
        let lane_after_invalid = store
            .get_projected_lane_message(first_message_id)
            .unwrap()
            .unwrap();
        assert_eq!(lane_after_invalid.status, "claimed");
        assert_eq!(
            lane_after_invalid.claimed_by.as_deref(),
            Some("python-worker-3")
        );
        assert!(
            store
                .read_recorded_runtime_state(
                    &two_step_run_id,
                    "workflow-two-step",
                    "conversation-two-step",
                )
                .unwrap()
                .unwrap()
                .state
                .get("first")
                .is_none(),
            "rejected result must not partially update state"
        );
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let reclaimed = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"python-worker-reclaimed","limit":10,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let reclaimed_status = reclaimed.status();
        let reclaimed_body = to_bytes(reclaimed.into_body(), 16384).await.unwrap();
        assert_eq!(
            reclaimed_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&reclaimed_body)
        );
        let reclaimed_value = serde_json::from_slice::<Value>(&reclaimed_body).unwrap();
        let reclaimed_work = &reclaimed_value["work"][0];
        assert_eq!(reclaimed_work["message_id"], first_work["message_id"]);
        assert_eq!(reclaimed_work["claimed_by"], "python-worker-reclaimed");
        let stale_owner = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&first_result_payload).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stale_owner.status(), StatusCode::CONFLICT);
        first_result_payload["handoff"]["claimed_by"] = reclaimed_work["claimed_by"].clone();
        first_result_payload["transition"]["expected_event_seq"] =
            reclaimed_work["expected_event_seq"].clone();
        let first_result_bytes = serde_json::to_vec(&first_result_payload).unwrap();
        for body in [first_result_bytes.clone(), first_result_bytes] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/internal/runtime/results")
                        .header("content-type", "application/json")
                        .header("x-kogwistar-roles", "reader")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        let second_claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"python-worker-4","limit":10,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let second_claim_body = to_bytes(second_claim.into_body(), 16384).await.unwrap();
        let second_claim_value = serde_json::from_slice::<Value>(&second_claim_body).unwrap();
        let second_work = second_claim_value["work"].as_array().unwrap();
        assert_eq!(
            second_work.len(),
            1,
            "exact retry duplicated successor lane"
        );
        assert_eq!(second_work[0]["run_id"], two_step_run_id);
        assert_eq!(second_work[0]["step_id"], "finish");
        assert_eq!(second_work[0]["payload"]["token_id"], successor_token_id);
        let second_result_payload = json!({
            "handoff": {
                "message_id": second_work[0]["message_id"],
                "claimed_by": second_work[0]["claimed_by"],
                "run_id": second_work[0]["run_id"],
                "step_id": second_work[0]["step_id"],
                "correlation_id": second_work[0]["correlation_id"],
            },
            "transition": {
                "contract_version": 1,
                "transition_id": format!("second-{two_step_run_id}"),
                "expected_event_seq": second_work[0]["expected_event_seq"],
                "kind": "recorded_step_success",
                "run_id": two_step_run_id,
                "workflow_id": "workflow-two-step",
                "conversation_id": "conversation-two-step",
                "step_seq": 1,
                "node_id": "finish",
                "token_id": successor_token_id,
                "parent_token_id": two_step_run_id,
                "state_update": [["u", {"second": true}]],
                "frontier": {
                    "pending": [],
                    "suspended": [],
                    "join_node_ids": [],
                    "join_outstanding": [],
                    "join_waiters": {},
                },
                "result": {"done": true},
            }
        });
        let second_result = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&second_result_payload).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let second_result_status = second_result.status();
        let second_result_body = to_bytes(second_result.into_body(), 16384).await.unwrap();
        assert_eq!(
            second_result_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&second_result_body)
        );
        assert_eq!(
            store
                .get_server_run(&two_step_run_id)
                .unwrap()
                .unwrap()
                .status,
            "succeeded"
        );
        let reopened = SqliteStore::open(&path).unwrap();
        let reopened_state = reopened
            .read_recorded_runtime_state(
                &two_step_run_id,
                "workflow-two-step",
                "conversation-two-step",
            )
            .unwrap()
            .unwrap();
        assert_eq!(reopened_state.state["first"], true);
        assert_eq!(reopened_state.state["second"], true);
        drop(reopened);

        // Restricted worker effects cannot replace the scheduler frontier.
        let effect_submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"workflow_id":"workflow-effect","conversation_id":"conversation-effect","join_node_ids":["join"],"start_join_mask":1,"runtime_routes":[{"source_node_id":"start","target_node_id":"left","join_mask":1},{"source_node_id":"start","target_node_id":"right","join_mask":1},{"source_node_id":"left","target_node_id":"join","join_mask":1},{"source_node_id":"right","target_node_id":"join","join_mask":1}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let effect_submit_body = to_bytes(effect_submit.into_body(), 8192).await.unwrap();
        let effect_run_id = serde_json::from_slice::<Value>(&effect_submit_body).unwrap()["run_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let effect_claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"effect-worker","limit":10,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let effect_claim_body = to_bytes(effect_claim.into_body(), 16384).await.unwrap();
        let effect_claim_value = serde_json::from_slice::<Value>(&effect_claim_body).unwrap();
        let effect_work = effect_claim_value["work"]
            .as_array()
            .unwrap()
            .iter()
            .find(|work| work["run_id"] == effect_run_id)
            .unwrap();
        let effect_id = format!("effect-{effect_run_id}");
        let effect_payload = json!({
            "handoff": {
                "message_id": effect_work["message_id"],
                "claimed_by": effect_work["claimed_by"],
                "run_id": effect_work["run_id"],
                "step_id": effect_work["step_id"],
                "correlation_id": effect_work["correlation_id"],
            },
            "effect": {
                "contract_version": 1,
                "effect_id": effect_id,
                "state_update": [["u", {"effect_first": true}]],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_cost": 0.01},
                "trace_events": [{"type": "step_completed", "span_id": "effect-start"}],
                "successors": [{"node_id": "forged", "join_mask": 0}],
            }
        });
        store
            .append_server_run_event(&effect_run_id, "test.competing", "{}".to_owned())
            .unwrap();
        let effect_bytes = serde_json::to_vec(&effect_payload).unwrap();
        for body in [effect_bytes.clone(), effect_bytes] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/internal/runtime/results")
                        .header("content-type", "application/json")
                        .header("x-kogwistar-roles", "reader")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        let effect_event_types = store
            .list_server_run_events(&effect_run_id, 0, usize::MAX)
            .unwrap()
            .into_iter()
            .map(|event| event.event_type)
            .collect::<Vec<_>>();
        assert_eq!(
            effect_event_types
                .iter()
                .filter(|event_type| event_type.as_str() == "workflow.usage.v1")
                .count(),
            1
        );
        assert_eq!(
            effect_event_types
                .iter()
                .filter(|event_type| event_type.as_str() == "workflow.trace.v1")
                .count(),
            1
        );
        let effect_state = store
            .read_recorded_runtime_state(&effect_run_id, "workflow-effect", "conversation-effect")
            .unwrap()
            .unwrap();
        assert_eq!(effect_state.frontier.pending.len(), 2);
        let left = effect_state
            .frontier
            .pending
            .iter()
            .find(|token| token.0 == "left")
            .unwrap();
        let right = effect_state
            .frontier
            .pending
            .iter()
            .find(|token| token.0 == "right")
            .unwrap();
        assert_eq!(left.2, effect_run_id);
        assert_eq!(right.3.as_deref(), Some(effect_run_id.as_str()));
        assert_ne!(right.2, effect_run_id);
        let pending_effect_lanes = store
            .list_projected_lane_messages(kogwistar_store::LaneMessageFilter {
                namespace: Some("workflow".to_owned()),
                inbox_id: Some("workflow-runtime".to_owned()),
                status: Some("pending".to_owned()),
                correlation_id: Some(effect_run_id.clone()),
                ..kogwistar_store::LaneMessageFilter::default()
            })
            .unwrap();
        assert_eq!(
            pending_effect_lanes.len(),
            2,
            "effect retry duplicated dispatch"
        );
        assert_eq!(pending_effect_lanes[0].step_id.as_deref(), Some("left"));
        assert_eq!(pending_effect_lanes[1].step_id.as_deref(), Some("right"));

        let parallel_claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"effect-parallel-worker","limit":10,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let parallel_claim_body = to_bytes(parallel_claim.into_body(), 16384).await.unwrap();
        let parallel_claim_value = serde_json::from_slice::<Value>(&parallel_claim_body).unwrap();
        let parallel_work = parallel_claim_value["work"].as_array().unwrap();
        assert_eq!(parallel_work.len(), 2);
        assert_eq!(parallel_work[0]["step_id"], "left");
        assert_eq!(parallel_work[1]["step_id"], "right");

        let submit_effect = |app: Router, work: Value, successors: Value, final_result: Value| {
            let effect_run_id = effect_run_id.clone();
            async move {
                let node = work["step_id"].as_str().unwrap().to_owned();
                let mut effect = json!({
                    "contract_version": 1,
                    "effect_id": format!("effect-{node}-{effect_run_id}"),
                    "state_update": [["u", {(format!("effect_{node}")): true}]],
                    "successors": successors,
                });
                if !final_result.is_null() {
                    effect["result"] = final_result;
                }
                app.clone()
                    .oneshot(
                        Request::builder()
                            .method("POST")
                            .uri("/internal/runtime/results")
                            .header("content-type", "application/json")
                            .header("x-kogwistar-roles", "reader")
                            .body(Body::from(
                                serde_json::to_vec(&json!({
                                    "handoff": {
                                        "message_id": work["message_id"],
                                        "claimed_by": work["claimed_by"],
                                        "run_id": work["run_id"],
                                        "step_id": work["step_id"],
                                        "correlation_id": work["correlation_id"],
                                    },
                                    "effect": effect,
                                }))
                                .unwrap(),
                            ))
                            .unwrap(),
                    )
                    .await
                    .unwrap()
            }
        };
        let to_join = json!([{"node_id": "forged", "join_mask": 0}]);
        let early_right = submit_effect(
            app.clone(),
            parallel_work[1].clone(),
            to_join.clone(),
            Value::Null,
        )
        .await;
        assert_eq!(early_right.status(), StatusCode::CONFLICT);
        let left_result = submit_effect(
            app.clone(),
            parallel_work[0].clone(),
            to_join.clone(),
            Value::Null,
        )
        .await;
        let result_status = left_result.status();
        let result_body = to_bytes(left_result.into_body(), 16384).await.unwrap();
        assert_eq!(
            result_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&result_body)
        );
        let restarted_app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
        );
        let right_result = submit_effect(
            restarted_app.clone(),
            parallel_work[1].clone(),
            to_join,
            Value::Null,
        )
        .await;
        let result_status = right_result.status();
        let result_body = to_bytes(right_result.into_body(), 16384).await.unwrap();
        assert_eq!(
            result_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&result_body)
        );
        let join_claim = restarted_app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"effect-join-worker","limit":10,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let join_claim_body = to_bytes(join_claim.into_body(), 16384).await.unwrap();
        let join_claim_value = serde_json::from_slice::<Value>(&join_claim_body).unwrap();
        let join_work = join_claim_value["work"].as_array().unwrap();
        assert_eq!(join_work.len(), 1);
        assert_eq!(join_work[0]["step_id"], "join");
        let join_result = submit_effect(
            restarted_app,
            join_work[0].clone(),
            json!([]),
            json!({"effect_done": true}),
        )
        .await;
        let join_status = join_result.status();
        let join_body = to_bytes(join_result.into_body(), 16384).await.unwrap();
        assert_eq!(
            join_status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&join_body)
        );
        let final_effect_state = store
            .read_recorded_runtime_state(&effect_run_id, "workflow-effect", "conversation-effect")
            .unwrap()
            .unwrap();
        assert!(final_effect_state.frontier.pending.is_empty());
        assert!(final_effect_state.status.is_terminal());
        assert_eq!(final_effect_state.state["effect_first"], true);
        assert_eq!(final_effect_state.state["effect_left"], true);
        assert_eq!(final_effect_state.state["effect_right"], true);
        assert_eq!(final_effect_state.state["effect_join"], true);
        assert!(final_effect_state.frontier.join_outstanding.is_empty());
        assert!(final_effect_state.frontier.join_waiters.is_empty());
        assert_eq!(
            store
                .get_server_run(&effect_run_id)
                .unwrap()
                .unwrap()
                .status,
            "succeeded"
        );

        let cancelled_submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"workflow_id":"workflow-cancel","conversation_id":"conversation-cancel"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let cancelled_submit_body = to_bytes(cancelled_submit.into_body(), 8192).await.unwrap();
        let cancelled_run_id =
            serde_json::from_slice::<Value>(&cancelled_submit_body).unwrap()["run_id"]
                .as_str()
                .unwrap()
                .to_owned();
        let cancel_before_claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/workflow/runs/{cancelled_run_id}/cancel"))
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cancel_before_claim.status(), StatusCode::ACCEPTED);
        let cancel_body = to_bytes(cancel_before_claim.into_body(), 8192)
            .await
            .unwrap();
        let cancelled_run = serde_json::from_slice::<Value>(&cancel_body).unwrap();
        assert_eq!(cancelled_run["status"], "cancelled");
        assert_eq!(cancelled_run["terminal"], true);
        let cancelled_claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"must-not-run","limit":10,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let cancelled_claim_body = to_bytes(cancelled_claim.into_body(), 8192).await.unwrap();
        assert!(
            serde_json::from_slice::<Value>(&cancelled_claim_body).unwrap()["work"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        let cancelled_lanes = store
            .list_projected_lane_messages(kogwistar_store::LaneMessageFilter {
                correlation_id: Some(cancelled_run_id.clone()),
                ..kogwistar_store::LaneMessageFilter::default()
            })
            .unwrap();
        assert_eq!(cancelled_lanes.len(), 1);
        assert_eq!(cancelled_lanes[0].status, "cancelled");
        let cancelled_state = store
            .read_recorded_runtime_state(
                &cancelled_run_id,
                "workflow-cancel",
                "conversation-cancel",
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            cancelled_state.status,
            kogwistar_runtime::RecordedRunStatus::Cancelled
        );
        assert!(cancelled_state.frontier.pending.is_empty());

        let claimed_cancel_submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"workflow_id":"workflow-claimed-cancel","conversation_id":"conversation-claimed-cancel"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let claimed_cancel_submit_body = to_bytes(claimed_cancel_submit.into_body(), 8192)
            .await
            .unwrap();
        let claimed_cancel_run_id = serde_json::from_slice::<Value>(&claimed_cancel_submit_body)
            .unwrap()["run_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let claimed_cancel_claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"cancelled-worker","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let claimed_cancel_claim_body = to_bytes(claimed_cancel_claim.into_body(), 8192)
            .await
            .unwrap();
        let claimed_cancel_work =
            serde_json::from_slice::<Value>(&claimed_cancel_claim_body).unwrap()["work"][0].clone();
        let claimed_cancel = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/workflow/runs/{claimed_cancel_run_id}/cancel"))
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(claimed_cancel.status(), StatusCode::ACCEPTED);
        let stale_result = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "handoff": {
                                "message_id": claimed_cancel_work["message_id"],
                                "claimed_by": claimed_cancel_work["claimed_by"],
                                "run_id": claimed_cancel_work["run_id"],
                                "step_id": claimed_cancel_work["step_id"],
                                "correlation_id": claimed_cancel_work["correlation_id"],
                            },
                            "effect": {
                                "contract_version": 1,
                                "effect_id": format!("stale-{claimed_cancel_run_id}"),
                                "state_update": [["u", {"must_not_apply": true}]],
                                "successors": [],
                                "result": {"must_not_apply": true},
                            }
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stale_result.status(), StatusCode::CONFLICT);
        let claimed_cancel_state = store
            .read_recorded_runtime_state(
                &claimed_cancel_run_id,
                "workflow-claimed-cancel",
                "conversation-claimed-cancel",
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            claimed_cancel_state.status,
            kogwistar_runtime::RecordedRunStatus::Cancelled
        );
        assert!(claimed_cancel_state.state.get("must_not_apply").is_none());

        drop(app);
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
        );
        let get = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/runs/run-1")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(get.status(), StatusCode::OK);
        let get_body = to_bytes(get.into_body(), 8192).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&get_body).unwrap()["status"],
            "running"
        );

        let events = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/runs/run-1/events/poll?after_seq=0&limit=10")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(events.status(), StatusCode::OK);
        let event_body = to_bytes(events.into_body(), 8192).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&event_body).unwrap()["events"][0]["event_type"],
            "run.started"
        );

        let cancel = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs/run-1/cancel")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cancel.status(), StatusCode::ACCEPTED);
        let cancel_body = to_bytes(cancel.into_body(), 8192).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&cancel_body).unwrap()["status"],
            "cancelling"
        );

        let sse = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/workflow/runs/run-1/events")
                    .header("x-kogwistar-roles", "reader")
                    .header("last-event-id", "0")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(sse.status(), StatusCode::OK);
        assert_eq!(
            sse.headers()["content-type"],
            "text/event-stream; charset=utf-8"
        );
        let sse_body =
            String::from_utf8(to_bytes(sse.into_body(), 8192).await.unwrap().to_vec()).unwrap();
        assert!(sse_body.contains("event: run.started"));
        assert!(sse_body.contains("id: 1"));

        let mcp_status = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"workflow.run_status","arguments":{"run_id":"run-1"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_status.status(), StatusCode::OK);
        let mcp_body = to_bytes(mcp_status.into_body(), 8192).await.unwrap();
        let mcp_value = serde_json::from_slice::<Value>(&mcp_body).unwrap();
        assert_eq!(mcp_value["id"], 7);
        assert_eq!(mcp_value["result"]["run_id"], "run-1");

        let mcp_events = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"workflow.run_events","arguments":{"run_id":"run-1","after_seq":0,"limit":10}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let mcp_event_body = to_bytes(mcp_events.into_body(), 8192).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&mcp_event_body).unwrap()["result"]["events"][0]["event_type"],
            "run.started"
        );
        drop(app);
        drop(store);
        remove_test_sqlite(&path);
    }

    #[tokio::test]
    async fn sqlite_submit_enforces_atomic_queue_backpressure() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-backpressure-{nonce}.sqlite3"));
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(
                SqliteRunApplicationService::open(&path)
                    .unwrap()
                    .with_max_queue(1),
            ),
        );
        let submit = |priority_class: &str| {
            Request::builder()
                .method("POST")
                .uri("/api/workflow/runs")
                .header("content-type", "application/json")
                .header("x-kogwistar-roles", "reader")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "workflow_id": "backpressure-wf",
                        "conversation_id": "backpressure-conv",
                        "priority_class": priority_class,
                    }))
                    .unwrap(),
                ))
                .unwrap()
        };
        let first = app.clone().oneshot(submit("foreground")).await.unwrap();
        assert_eq!(first.status(), StatusCode::ACCEPTED);
        let claimed = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"backpressure-worker","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(claimed.status(), StatusCode::OK);
        let claimed =
            serde_json::from_slice::<Value>(&to_bytes(claimed.into_body(), 8192).await.unwrap())
                .unwrap();
        assert_eq!(claimed["work"][0]["payload"]["op"], "noop");
        for (priority_class, admission) in [("foreground", "deferred"), ("batch", "rejected")] {
            let response = app.clone().oneshot(submit(priority_class)).await.unwrap();
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
            let body = to_bytes(response.into_body(), 8192).await.unwrap();
            let value = serde_json::from_slice::<Value>(&body).unwrap();
            assert_eq!(value["admission"], admission);
            assert_eq!(value["reason"], "queue_full");
            assert_eq!(value["max_queue"], 1);
        }
        let store = SqliteStore::open(&path).unwrap();
        assert_eq!(
            store.list_server_runs(None, None, None, 10).unwrap().len(),
            1
        );
        assert_eq!(
            store
                .list_projected_lane_messages(LaneMessageFilter {
                    namespace: Some("workflow".to_owned()),
                    inbox_id: Some("workflow-runtime".to_owned()),
                    ..LaneMessageFilter::default()
                })
                .unwrap()
                .len(),
            1
        );
        drop(store);
        drop(app);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_syscall_spawn_and_terminate_use_authoritative_run_service() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-syscall-{nonce}.sqlite3"));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let principal = json!({
            "sub":"operator",
            "role":"rw",
            "ns":"workflow",
            "capabilities":["spawn_process","workflow.run.write","workflow.run.read"],
        });
        let spawned = service.execute_sync(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/spawn_process".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"spawn_process",
                "args":{
                    "workflow_id":"wf-syscall",
                    "conversation_id":"conv-syscall",
                    "initial_state":{"source":"syscall"},
                },
            }))
            .unwrap(),
            principal: principal.clone(),
        });
        assert_eq!(spawned.status, StatusCode::OK.as_u16());
        let spawned = serde_json::from_slice::<Value>(&spawned.body).unwrap();
        assert_eq!(spawned["version"], "v1");
        assert_eq!(spawned["op"], "spawn_process");
        assert_eq!(spawned["status"], "ok");
        let run_id = spawned["result"]["run_id"].as_str().unwrap();
        assert!(service.store.get_server_run(run_id).unwrap().is_some());
        let terminated = service.execute_sync(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/terminate_process".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"terminate_process",
                "args":{"run_id":run_id},
            }))
            .unwrap(),
            principal,
        });
        assert_eq!(terminated.status, StatusCode::OK.as_u16());
        let terminated = serde_json::from_slice::<Value>(&terminated.body).unwrap();
        assert_eq!(terminated["result"]["cancel_requested"], true);
        assert!(
            service
                .store
                .get_server_run(run_id)
                .unwrap()
                .unwrap()
                .cancel_requested
        );
        drop(service);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_worker_suspend_resume_is_durable_and_scheduler_owned() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-resume-{nonce}.sqlite3"));
        let reopen = || {
            router_with_application(
                ApiState {
                    health: snapshot(),
                    required_roles: vec!["reader".to_owned()],
                    implementation: ImplementationSnapshot::default(),
                    auth: AuthConfig::default(),
                },
                Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
            )
        };
        let mut app = reopen();
        let call = |app: Router, method: &'static str, uri: String, body: Value| async move {
            let response = app
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(uri)
                        .header("content-type", "application/json")
                        .header("x-kogwistar-roles", "reader")
                        .body(Body::from(if body.is_null() {
                            Vec::new()
                        } else {
                            serde_json::to_vec(&body).unwrap()
                        }))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            let bytes = to_bytes(response.into_body(), 32768).await.unwrap();
            (status, serde_json::from_slice::<Value>(&bytes).unwrap())
        };
        let (status, submitted) = call(
            app.clone(),
            "POST",
            "/api/workflow/runs".to_owned(),
            json!({"workflow_id":"resume-wf","conversation_id":"resume-conv","turn_node_id":"turn-original"}),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED);
        let run_id = submitted["run_id"].as_str().unwrap().to_owned();
        drop(app);
        app = reopen();
        let (_, claimed) = call(
            app.clone(),
            "POST",
            "/internal/runtime/claim".to_owned(),
            json!({"claimed_by":"suspender","limit":1,"lease_seconds":60}),
        )
        .await;
        let work = &claimed["work"][0];
        drop(app);
        app = reopen();
        let suspend = json!({
            "handoff": {
                "message_id": work["message_id"],
                "claimed_by": work["claimed_by"],
                "run_id": work["run_id"],
                "step_id": work["step_id"],
                "correlation_id": work["correlation_id"],
            },
            "effect": {
                "contract_version": 1,
                "effect_id": format!("suspend-{run_id}"),
                "status": "suspended",
                "state_update": [["u", {"before_suspend": true}]],
                "successors": [],
                "wait_reason": "approval",
                "resume_payload": {"question": "continue?"},
            }
        });
        let (status, _) = call(
            app.clone(),
            "POST",
            "/internal/runtime/results".to_owned(),
            suspend,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        drop(app);
        app = reopen();
        let (status, contract) = call(
            app.clone(),
            "GET",
            format!("/api/workflow/runs/{run_id}/resume-contract"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(contract["wait_reason"], "approval");
        assert_eq!(contract["suspended"][0][0], "start");
        assert_eq!(contract["runtime_routes"], json!([]));
        assert_eq!(contract["node_ops"], json!({}));
        let token_id = contract["suspended"][0][2].as_str().unwrap().to_owned();
        let resume_body = json!({
            "suspended_node_id": "start",
            "suspended_token_id": token_id,
            "client_result": {
                "status": "success",
                "state_update": [["u", {"approved": true}]],
                "successors": [],
                "route_next": [],
                "result": {"workflow_status": "succeeded"}
            },
            "workflow_id": "resume-wf",
            "conversation_id": "resume-conv",
            "turn_node_id": "unused-compatible-field",
        });
        let (status, resumed) = call(
            app.clone(),
            "POST",
            format!("/api/workflow/runs/{run_id}/resume"),
            resume_body,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(resumed["state"]["status"], "running");
        assert_eq!(resumed["state"]["wait_reason"], Value::Null);
        drop(app);
        app = reopen();
        let (_, claimed) = call(
            app.clone(),
            "POST",
            "/internal/runtime/claim".to_owned(),
            json!({"claimed_by":"finisher","limit":1,"lease_seconds":60}),
        )
        .await;
        let work = &claimed["work"][0];
        assert_eq!(work["step_id"], "start");
        assert_eq!(work["payload"]["turn_node_id"], "turn-original");
        assert_eq!(work["payload"]["resume_effect"]["status"], "success");
        assert_eq!(
            work["payload"]["resume_effect"]["state_update"][0][1]["approved"],
            true
        );
        let (status, _) = call(
            app.clone(),
            "POST",
            "/internal/runtime/results".to_owned(),
            json!({
                "handoff": {
                    "message_id": work["message_id"],
                    "claimed_by": work["claimed_by"],
                    "run_id": work["run_id"],
                    "step_id": work["step_id"],
                    "correlation_id": work["correlation_id"],
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": format!("finish-{run_id}"),
                    "state_update": [["u", {"after_resume": true}]],
                    "successors": [],
                    "result": {"done": true},
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        drop(app);
        app = reopen();
        let (status, run) = call(
            app,
            "GET",
            format!("/api/workflow/runs/{run_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(run["status"], "succeeded");
        assert_eq!(run["result"]["done"], true);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_submit_freezes_exact_ready_graph_snapshot_as_authoritative_routes() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-graph-route-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        store
            .replace_named_projection(
                "workflow_design",
                "graph-wf",
                NamedProjectionWrite {
                    payload: serde_json::Map::from_iter([
                        ("current_version".to_owned(), json!(1)),
                        ("snapshot_schema_version".to_owned(), json!(1)),
                    ]),
                    last_authoritative_seq: 1,
                    last_materialized_seq: 1,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )
            .unwrap();
        store
            .put_workflow_design_snapshot(
                "graph-wf",
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({
                        "nodes": [
                            {"id":"graph-start","metadata":{"wf_start":true}},
                            {"id":"graph-end","metadata":{"wf_terminal":true}}
                        ],
                        "edges": [{
                            "id":"edge-1",
                            "source_ids":["graph-start"],
                            "target_ids":["graph-end"],
                            "metadata":{}
                        }]
                    }))
                    .unwrap(),
                },
            )
            .unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
        );
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"workflow_id":"graph-wf","conversation_id":"graph-conv","runtime_routes":[{"source_node_id":"start","target_node_id":"forged","join_mask":0}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"graph-worker","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let claim_body = to_bytes(claim.into_body(), 16384).await.unwrap();
        let work = serde_json::from_slice::<Value>(&claim_body).unwrap()["work"][0].clone();
        assert_eq!(work["step_id"], "graph-start");
        let result = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "handoff": {
                                "message_id": work["message_id"],
                                "claimed_by": work["claimed_by"],
                                "run_id": work["run_id"],
                                "step_id": work["step_id"],
                                "correlation_id": work["correlation_id"],
                            },
                            "effect": {
                                "contract_version": 1,
                                "effect_id": format!("graph-start-{}", work["run_id"].as_str().unwrap()),
                                "successors": [{"node_id":"worker-forged","join_mask":0}],
                            }
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(result.status(), StatusCode::OK);
        let next = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"graph-worker-2","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let next_body = to_bytes(next.into_body(), 16384).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&next_body).unwrap()["work"][0]["step_id"],
            "graph-end"
        );
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_predicate_snapshot_accepts_only_graph_valid_worker_selection() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-predicate-route-{nonce}.sqlite3"));
        let store = SqliteStore::open(&path).unwrap();
        store
            .replace_named_projection(
                "workflow_design",
                "predicate-wf",
                NamedProjectionWrite {
                    payload: serde_json::Map::from_iter([
                        ("current_version".to_owned(), json!(1)),
                        ("snapshot_schema_version".to_owned(), json!(1)),
                    ]),
                    last_authoritative_seq: 1,
                    last_materialized_seq: 1,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )
            .unwrap();
        store
            .put_workflow_design_snapshot(
                "predicate-wf",
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({
                        "nodes": [
                            {"id":"gate","metadata":{"wf_start":true}},
                            {"id":"left","metadata":{"wf_terminal":true}},
                            {"id":"right","metadata":{"wf_terminal":true}}
                        ],
                        "edges": [
                            {"id":"to-left","source_ids":["gate"],"target_ids":["left"],"metadata":{"wf_predicate":"if_true"}},
                            {"id":"to-right","source_ids":["gate"],"target_ids":["right"],"metadata":{"wf_predicate":"if_false","wf_is_default":true}}
                        ]
                    }))
                    .unwrap(),
                },
            )
            .unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
        );
        let submit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/workflow/runs")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"workflow_id":"predicate-wf","conversation_id":"predicate-conv"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(submit.status(), StatusCode::ACCEPTED);
        let claim = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"predicate-worker","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let claim_body = to_bytes(claim.into_body(), 16_384).await.unwrap();
        let work = serde_json::from_slice::<Value>(&claim_body).unwrap()["work"][0].clone();
        let envelope = |successor: Value| {
            json!({
                "handoff": {
                    "message_id": work["message_id"],
                    "claimed_by": work["claimed_by"],
                    "run_id": work["run_id"],
                    "step_id": work["step_id"],
                    "correlation_id": work["correlation_id"],
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": format!("predicate-{}", work["run_id"].as_str().unwrap()),
                    "successors": [successor],
                }
            })
        };
        let forged = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&envelope(json!({"node_id":"forged","join_mask":0})))
                            .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(forged.status(), StatusCode::CONFLICT);
        let accepted = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/results")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        serde_json::to_vec(&envelope(json!({"node_id":"right","join_mask":0})))
                            .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::OK);
        let next = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/runtime/claim")
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(
                        r#"{"claimed_by":"predicate-finisher","limit":1,"lease_seconds":60}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let next_body = to_bytes(next.into_body(), 16_384).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&next_body).unwrap()["work"][0]["step_id"],
            "right"
        );
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn tcp_transport_serves_health_over_real_socket() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(serve(
            listener,
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
        ));
        let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
        stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        let response = String::from_utf8(response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(response.contains("\"backend\":\"pg\""));
        server.abort();
    }
}
