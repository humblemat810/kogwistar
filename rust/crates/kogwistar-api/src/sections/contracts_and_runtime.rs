pub const FROZEN_OPENAPI_JSON: &str = include_str!(concat!(env!("OUT_DIR"), "/openapi.json"));
pub const FROZEN_MCP_TOOLS_JSON: &str = include_str!(concat!(env!("OUT_DIR"), "/mcp-tools.json"));
const CYTOSCAPE_TEMPLATE: &str = include_str!(concat!(env!("OUT_DIR"), "/cytoscape.html"));
const D3_TEMPLATE: &str = include_str!(concat!(env!("OUT_DIR"), "/d3.html"));
const GO_TEMPLATE: &str = include_str!(concat!(env!("OUT_DIR"), "/go.html"));

const INTERNAL_RUNTIME_CLAIM_ROUTE: &str = "/internal/runtime/claim";
const INTERNAL_RUNTIME_RESULTS_ROUTE: &str = "/internal/runtime/results";
const INTERNAL_RUNTIME_NATIVE_EXECUTE_ROUTE: &str = "/internal/runtime/native/execute";
const INTERNAL_GRAPH_MUTATIONS_ROUTE: &str = "/internal/graph/mutations";
const INTERNAL_CONVERSATION_TRANSCRIPT_ROUTE: &str = "/internal/conversation/transcript";
const INTERNAL_CONVERSATION_CONTEXT_SNAPSHOT_ROUTE: &str =
    "/internal/conversation/context-snapshot";
const INTERNAL_GRAPH_NEIGHBORS_ROUTE: &str = "/internal/graph/neighbors";
const INTERNAL_GRAPH_VECTOR_QUERY_ROUTE: &str = "/internal/graph/vector-query";
const INTERNAL_GRAPH_FIND_EDGES_ROUTE: &str = "/internal/graph/find-edges";
const INTERNAL_GRAPH_SHORTEST_PATH_ROUTE: &str = "/internal/graph/shortest-path";
const INTERNAL_GRAPH_K_HOP_ROUTE: &str = "/internal/graph/k-hop";

const INTERNAL_APPLICATION_EFFECT_ROUTES: &[&str] = &[
    INTERNAL_RUNTIME_CLAIM_ROUTE,
    INTERNAL_RUNTIME_RESULTS_ROUTE,
    INTERNAL_RUNTIME_NATIVE_EXECUTE_ROUTE,
    INTERNAL_DOCUMENT_PROJECTION_ROUTE,
    INTERNAL_GRAPH_MUTATIONS_ROUTE,
    INTERNAL_CONVERSATION_TRANSCRIPT_ROUTE,
    INTERNAL_CONVERSATION_CONTEXT_SNAPSHOT_ROUTE,
    INTERNAL_GRAPH_NEIGHBORS_ROUTE,
    INTERNAL_GRAPH_VECTOR_QUERY_ROUTE,
    INTERNAL_GRAPH_FIND_EDGES_ROUTE,
    INTERNAL_GRAPH_SHORTEST_PATH_ROUTE,
    INTERNAL_GRAPH_K_HOP_ROUTE,
];

fn rust_native_attested(mut value: Value) -> Value {
    attach_rust_native_seam_attestation(&mut value);
    value
}

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

/// Return whether the Rust application currently owns a frozen MCP operation.
///
/// The shared catalog describes the complete product surface, including
/// Python-owned operations. A Rust server must not advertise those operations
/// until it has an equivalent handler and authority path.
fn rust_mcp_tool_supported(surface: &str, name: &str) -> bool {
    match surface {
        "workflow" => matches!(
            name,
            "workflow.budget_history"
                | "workflow.budget_snapshot"
                | "workflow.capabilities_snapshot"
                | "workflow.capability_approve"
                | "workflow.capability_revoke"
                | "workflow.dead_letter_replay"
                | "workflow.dead_letters"
                | "workflow.design_edge_delete"
                | "workflow.design_edge_upsert"
                | "workflow.design_history"
                | "workflow.design_node_delete"
                | "workflow.design_node_upsert"
                | "workflow.design_redo"
                | "workflow.design_undo"
                | "workflow.message_orphans_repair"
                | "workflow.operator_dashboard"
                | "workflow.run_cancel"
                | "workflow.run_checkpoint_get"
                | "workflow.run_events"
                | "workflow.run_replay"
                | "workflow.run_resume"
                | "workflow.run_resume_contract"
                | "workflow.run_status"
                | "workflow.run_submit"
                | "workflow.scheduler_timeline"
                | "workflow.service_declare"
                | "workflow.service_disable"
                | "workflow.service_enable"
                | "workflow.service_events"
                | "workflow.service_get"
                | "workflow.service_heartbeat"
                | "workflow.service_list"
                | "workflow.service_repair"
                | "workflow.service_trigger"
                | "workflow.services_repair"
                | "workflow.tool_audit"
                | "workflow.visibility_snapshot"
        ),
        "conversation" => matches!(name, "conversation.run_status" | "conversation.cancel_run"),
        "root" => matches!(
            name,
            "kg_find_edges" | "kg_k_hop" | "kg_neighbors" | "kg_shortest_path"
        ),
        _ => false,
    }
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
    fn graph_authority_capability(&self) -> GraphAuthorityCapability {
        GraphAuthorityCapability::Unsupported
    }

    fn document_authority_capability(&self) -> DocumentAuthorityCapability {
        DocumentAuthorityCapability::Unsupported
    }

    /// Advertise only operations this concrete authority can execute.
    ///
    /// Keeping discovery on the application service prevents an unavailable or
    /// partially wired backend from claiming the static catalog's operations.
    fn supports_mcp_tool(&self, _surface: &str, _name: &str) -> bool {
        false
    }

    /// Report which frozen HTTP operations this concrete service can execute.
    ///
    /// Transport registration is deliberately not used as evidence: the
    /// default service is unavailable, and partially wired services must not
    /// claim the capabilities of the complete route catalog.
    fn supports_http_operation(&self, _method: &str, _path: &str) -> bool {
        false
    }

    fn implemented_http_operations(&self) -> usize {
        FROZEN_OPENAPI_ROUTES
            .iter()
            .filter(|(method, path)| self.supports_http_operation(method, path))
            .count()
    }

    fn execute(
        &self,
        request: ApiEffectRequest,
    ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>>;
}

#[derive(Default)]
pub struct UnavailableApplicationService;

impl ApplicationService for UnavailableApplicationService {
    fn supports_mcp_tool(&self, _surface: &str, _name: &str) -> bool {
        false
    }

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
    let node_versions = state
        .state
        .get("_rt_node_versions")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let node_environment_ids = state
        .state
        .get("_rt_node_environment_ids")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let node_executor_targets = state
        .state
        .get("_rt_node_executor_targets")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let mut contract = json!({
        "run_id": state.run_id,
        "status": state.status,
        "wait_reason": state.wait_reason,
        "resume_payload": state.resume_payload,
        "suspended": state.frontier.suspended,
        "last_step_seq": state.last_step_seq,
        "runtime_routes": state.static_routes,
        "node_ops": node_ops,
        "node_versions": node_versions,
        "node_environment_ids": node_environment_ids,
    });
    if node_executor_targets
        .as_object()
        .is_some_and(|targets| !targets.is_empty())
    {
        contract["node_executor_targets"] = node_executor_targets;
    }
    contract
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
    node_versions: std::collections::BTreeMap<String, String>,
    node_environment_ids: std::collections::BTreeMap<String, String>,
    node_executor_targets: std::collections::BTreeMap<String, String>,
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
    let submitted_versions = if input.node_versions.is_empty() {
        input
            .node_ops
            .keys()
            .map(|node_id| (node_id.clone(), "v1".to_owned()))
            .collect()
    } else {
        input.node_versions.clone()
    };
    let submitted_environment_ids = input.node_environment_ids.clone();
    graph_plan.start_node_id != start_node_id
        || graph_plan.join_node_ids != input.join_node_ids
        || graph_plan.start_join_mask != input.start_join_mask
        || canonical_routes(&graph_plan.routes) != canonical_routes(&input.runtime_routes)
        || graph_plan.node_ops != input.node_ops
        || graph_plan.node_versions != submitted_versions
        || graph_plan.node_environment_ids != submitted_environment_ids
        || graph_plan.node_executor_targets != input.node_executor_targets
}

fn runtime_join_outstanding(join_count: usize, start_join_mask: i64) -> Vec<i64> {
    (0..join_count)
        .map(|index| i64::from(start_join_mask & (1_i64 << index) != 0))
        .collect()
}

fn default_runtime_priority_class() -> String {
    "foreground".to_owned()
}

fn native_operation_name(operation: &str) -> Option<&str> {
    kogwistar_runtime::native_operation_name(operation)
}

fn validate_native_operation_registry(
    operations: &std::collections::BTreeMap<String, String>,
    versions: &std::collections::BTreeMap<String, String>,
    executor_targets: &std::collections::BTreeMap<String, String>,
    registry: &kogwistar_runtime::NativeOperationRegistry,
) -> Result<(), (&'static str, String)> {
    for (node_id, target) in executor_targets {
        let Some(operation) = operations.get(node_id) else {
            return Err((
                "KOGWISTAR_INVALID_EXECUTOR_TARGET",
                format!("executor target references unknown workflow node {node_id:?}"),
            ));
        };
        if !matches!(
            target.as_str(),
            kogwistar_runtime::PYTHON_EXECUTOR_TARGET
                | kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET
        ) {
            return Err((
                "KOGWISTAR_INVALID_EXECUTOR_TARGET",
                format!("workflow node {node_id:?} has unsupported executor target {target:?}"),
            ));
        }
        if target == kogwistar_runtime::PYTHON_EXECUTOR_TARGET
            && native_operation_name(operation).is_some()
        {
            return Err((
                "KOGWISTAR_INVALID_EXECUTOR_TARGET",
                format!("Rust-native operation for workflow node {node_id:?} cannot target Python"),
            ));
        }
    }
    for (node_id, operation) in operations {
        let target = executor_targets.get(node_id).map(String::as_str);
        let rust_targeted = target == Some(kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET);
        let legacy_native_name = target.is_none() && operation.starts_with("rust-native/");
        let Some(name) = kogwistar_runtime::native_operation_for_target(operation, target) else {
            if (rust_targeted || legacy_native_name)
                && (operation.trim().is_empty()
                    || (operation.starts_with("rust-native/")
                        && native_operation_name(operation).is_none()))
            {
                return Err((
                    "KOGWISTAR_INVALID_NATIVE_OPERATION",
                    format!("node {node_id:?} has an empty Rust-native operation name"),
                ));
            }
            continue;
        };
        let version = versions.get(node_id).map(String::as_str).unwrap_or("v1");
        if !registry.supports(name, version) {
            return Err((
                "KOGWISTAR_NATIVE_OPERATION_UNREGISTERED",
                format!(
                    "Rust-native operation {name:?} version {version:?} for node {node_id:?} is not registered"
                ),
            ));
        }
    }
    Ok(())
}

fn add_executor_target(payload: &mut Value, operation: &str, executor_target: Option<&str>) {
    if let Some(target) = executor_target.or_else(|| {
        native_operation_name(operation).map(|_| kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET)
    }) {
        payload["executor_target"] = json!(target);
    }
}

fn runtime_worker_recipient_for_operation(
    operation: &str,
    environment_id: Option<&str>,
    executor_target: Option<&str>,
) -> String {
    let target = executor_target.or_else(|| {
        native_operation_name(operation).map(|_| kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET)
    });
    if target == Some(kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET) {
        kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned() + "-worker"
    } else {
        runtime_worker_recipient(environment_id)
    }
}

fn runtime_worker_recipient_for_state(state: &RecordedRuntimeState, node_id: &str) -> String {
    let operation = state
        .state
        .get("_rt_node_ops")
        .and_then(Value::as_object)
        .and_then(|ops| ops.get(node_id))
        .and_then(Value::as_str)
        .unwrap_or("noop");
    let environment_id = state
        .state
        .get("_rt_node_environment_ids")
        .and_then(Value::as_object)
        .and_then(|ids| ids.get(node_id))
        .and_then(Value::as_str);
    let executor_target = state
        .state
        .get("_rt_node_executor_targets")
        .and_then(Value::as_object)
        .and_then(|targets| targets.get(node_id))
        .and_then(Value::as_str);
    runtime_worker_recipient_for_operation(operation, environment_id, executor_target)
}

fn install_authoritative_node_executor_targets(
    state: &mut serde_json::Map<String, Value>,
    targets: &std::collections::BTreeMap<String, String>,
) {
    state.remove("_rt_node_executor_targets");
    if !targets.is_empty() {
        state.insert(
            "_rt_node_executor_targets".to_owned(),
            serde_json::to_value(targets).expect("runtime node executor targets serialize"),
        );
    }
}

fn effective_runtime_claim_recipients(values: &[String]) -> Result<Vec<String>, String> {
    if values.is_empty() {
        return Ok(vec!["python-worker".to_owned()]);
    }
    let mut unique = BTreeSet::new();
    let mut ordered = Vec::with_capacity(values.len());
    for value in values {
        if value.trim().is_empty() || value.trim() != value || value.len() > 256 {
            return Err(
                "recipient_ids must contain canonical non-empty values up to 256 bytes".to_owned(),
            );
        }
        if !unique.insert(value.clone()) {
            return Err("recipient_ids must not contain duplicates".to_owned());
        }
        ordered.push(value.clone());
    }
    Ok(ordered)
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
    #[serde(default)]
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    node_versions: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    node_environment_ids: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    node_executor_targets: BTreeMap<String, String>,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    recipient_ids: Vec<String>,
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
struct RuntimeNativeExecuteRequest {
    handoff: RecordedWorkerHandoff,
    payload: Value,
}

fn native_step_request_from_payload(
    handoff: &RecordedWorkerHandoff,
    payload: Value,
) -> Result<kogwistar_runtime::RuntimeNativeStepRequest, String> {
    if handoff.claimed_by != "rust-native-worker" {
        return Err("native execution requires the rust-native worker claim".to_owned());
    }
    if payload["kind"] == "workflow.run.execute" {
        if payload["executor_target"]
            .as_str()
            .is_some_and(|target| target != kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET)
        {
            return Err("native run payload has a conflicting executor target".to_owned());
        }
        let run_id = payload["run_id"]
            .as_str()
            .ok_or_else(|| "native run payload lacks run_id".to_owned())?;
        let workflow_id = payload["workflow_id"]
            .as_str()
            .ok_or_else(|| "native run payload lacks workflow_id".to_owned())?;
        let conversation_id = payload["conversation_id"]
            .as_str()
            .ok_or_else(|| "native run payload lacks conversation_id".to_owned())?;
        let node_id = payload["start_node_id"]
            .as_str()
            .ok_or_else(|| "native run payload lacks start_node_id".to_owned())?;
        let executor_target = payload["executor_target"].as_str();
        let operation = payload["op"]
            .as_str()
            .and_then(|operation| {
                kogwistar_runtime::native_operation_for_target(operation, executor_target)
            })
            .ok_or_else(|| {
                "native run payload must resolve to a registered Rust operation".to_owned()
            })?;
        let runtime_routes = serde_json::from_value(payload["runtime_routes"].clone())
            .map_err(|error| format!("invalid native runtime routes: {error}"))?;
        if handoff.run_id != run_id || handoff.step_id != node_id {
            return Err("native run payload does not match the claimed handoff".to_owned());
        }
        let op_version = match payload.get("op_version") {
            None => "v1".to_owned(),
            Some(Value::String(version)) => version.clone(),
            Some(_) => return Err("native run payload has an invalid op_version".to_owned()),
        };
        return Ok(kogwistar_runtime::RuntimeNativeStepRequest {
            executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: operation.to_owned(),
            op_version,
            run_id: run_id.to_owned(),
            workflow_id: workflow_id.to_owned(),
            node_id: node_id.to_owned(),
            conversation_id: conversation_id.to_owned(),
            turn_node_id: payload["turn_node_id"].as_str().map(str::to_owned),
            token_id: run_id.to_owned(),
            parent_token_id: None,
            step_seq: 0,
            state: serde_json::from_value(payload["initial_state"].clone())
                .map_err(|error| format!("invalid native initial state: {error}"))?,
            runtime_routes,
        });
    }

    let payload: kogwistar_runtime::RuntimeStepExecutePayload = serde_json::from_value(payload)
        .map_err(|error| format!("invalid native step payload: {error}"))?;
    let operation = payload
        .op
        .as_str()
        .and_then(|operation| {
            kogwistar_runtime::native_operation_for_target(
                operation,
                payload.executor_target.as_deref(),
            )
        })
        .ok_or_else(|| {
            "native step payload must resolve to a registered Rust operation".to_owned()
        })?;
    if payload
        .executor_target
        .as_deref()
        .is_some_and(|target| target != kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET)
    {
        return Err("native step payload has a conflicting executor target".to_owned());
    }
    if handoff.run_id != payload.run_id || handoff.step_id != payload.node_id {
        return Err("native step payload does not match the claimed handoff".to_owned());
    }
    Ok(kogwistar_runtime::RuntimeNativeStepRequest {
        executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
        operation: operation.to_owned(),
        op_version: payload.op_version,
        run_id: payload.run_id,
        workflow_id: payload.workflow_id,
        node_id: payload.node_id,
        conversation_id: payload.conversation_id,
        turn_node_id: payload.turn_node_id,
        token_id: payload.token_id,
        parent_token_id: payload.parent_token_id,
        step_seq: payload.step_seq,
        state: payload.state,
        runtime_routes: payload.runtime_routes,
    })
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
    let node_versions = nodes
        .iter()
        .filter_map(|node| {
            let node_id = node["id"].as_str()?;
            let version = node["metadata"]["wf_version"]
                .as_str()
                .or_else(|| node["metadata"]["wf.version"].as_str())
                .unwrap_or("v1");
            Some((node_id.to_owned(), version.to_owned()))
        })
        .collect();
    let node_environment_ids = nodes
        .iter()
        .filter_map(|node| {
            let node_id = node["id"].as_str()?;
            let environment_id = node["metadata"]["wf_environment_id"]
                .as_str()
                .or_else(|| node["metadata"]["wf.environment_id"].as_str())?
                .trim();
            if environment_id.is_empty() {
                return None;
            }
            Some((node_id.to_owned(), environment_id.to_owned()))
        })
        .collect();
    let mut node_executor_targets = std::collections::BTreeMap::new();
    for node in nodes {
        let Some(raw_target) = node["metadata"].get("wf_executor_target") else {
            continue;
        };
        let node_id = node["id"].as_str()?;
        let target = raw_target.as_str()?;
        if !matches!(
            target,
            kogwistar_runtime::PYTHON_EXECUTOR_TARGET
                | kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET
        ) {
            return None;
        }
        node_executor_targets.insert(node_id.to_owned(), target.to_owned());
    }
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
        node_versions,
        node_environment_ids,
        node_executor_targets,
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
