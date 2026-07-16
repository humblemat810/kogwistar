//! Transport-neutral Phase-5 API contracts.
//!
//! Network transports adapt these typed decisions. They do not own domain
//! policy, authentication role checks, SSE framing, or JSON-RPC envelopes.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use kogwistar_runtime::{
    RECORDED_RUNTIME_CONTRACT_VERSION, RecordedRuntimeState, RecordedRuntimeTransition,
    RecordedTransitionKind, RecordedWorkerHandoff, RecordedWorkerSuccessEffect, RuntimeFrontier,
};
use kogwistar_store::{
    LaneMessageFilter, NamedProjection, NewProjectedLaneMessage, ProjectedLaneMessage, ServerRun,
    ServerRunCreate, ServerRunEvent, WorkflowDesignSnapshot,
};
use kogwistar_store_postgres::PostgresStore;
use kogwistar_store_sqlite::SqliteStore;

use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{OriginalUri, Query, State},
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{MethodFilter, get, on, post},
};

include!(concat!(env!("OUT_DIR"), "/frozen_routes.rs"));
pub const FROZEN_OPENAPI_JSON: &str = include_str!(concat!(env!("OUT_DIR"), "/openapi.json"));
pub const FROZEN_MCP_TOOLS_JSON: &str = include_str!(concat!(env!("OUT_DIR"), "/mcp-tools.json"));

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
    pub algorithm: Option<String>,
    pub key: Option<String>,
    pub issuer: Option<String>,
    pub audience: Option<String>,
}

impl AuthConfig {
    pub fn from_environment() -> Self {
        Self {
            algorithm: std::env::var("JWT_ALG")
                .ok()
                .or_else(|| Some("HS256".to_owned())),
            key: std::env::var("JWT_SECRET").ok(),
            issuer: std::env::var("JWT_ISS").ok(),
            audience: std::env::var("JWT_AUD").ok(),
        }
    }

    fn configured(&self) -> bool {
        self.key.as_deref().is_some_and(|key| !key.is_empty())
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

impl Default for ImplementationSnapshot {
    fn default() -> Self {
        Self {
            mode: "rust".to_owned(),
            contract_version: 1,
            schema_version: 1,
            frozen_route_operations: FROZEN_OPENAPI_ROUTES.len(),
            // health + submit + two aliases each for run GET/events/cancel + poll on
            // conversation API. Static transport-only /api/events and /mcp
            // are not part of the frozen OpenAPI operation count.
            implemented_route_operations: 9,
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

fn resume_contract_value(state: RecordedRuntimeState) -> Value {
    json!({
        "run_id": state.run_id,
        "status": state.status,
        "wait_reason": state.wait_reason,
        "resume_payload": state.resume_payload,
        "suspended": state.frontier.suspended,
        "last_step_seq": state.last_step_seq,
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

#[derive(Clone, Debug)]
struct RuntimeGraphPlan {
    start_node_id: String,
    join_node_ids: Vec<String>,
    start_join_mask: i64,
    routes: Vec<kogwistar_runtime::RuntimeStaticRoute>,
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
    let mut join_node_ids = nodes
        .iter()
        .filter(|node| node["metadata"]["wf_join"].as_bool() == Some(true))
        .filter_map(|node| node["id"].as_str().map(str::to_owned))
        .collect::<Vec<_>>();
    join_node_ids.sort();
    if join_node_ids.len() >= i64::BITS as usize {
        return None;
    }
    let mut topology_edges = Vec::new();
    for edge in edges {
        if edge["metadata"]["wf_predicate"]
            .as_str()
            .is_some_and(|predicate| !predicate.is_empty())
        {
            return None;
        }
        for source in string_list(&edge["source_ids"]) {
            for target in string_list(&edge["target_ids"]) {
                topology_edges.push([source.clone(), target]);
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
    let routes = topology_edges
        .into_iter()
        .map(
            |[source_node_id, target_node_id]| kogwistar_runtime::RuntimeStaticRoute {
                source_node_id,
                join_mask: mask(&target_node_id),
                target_node_id,
            },
        )
        .collect();
    Some(RuntimeGraphPlan {
        start_join_mask: mask(&start_node_id),
        start_node_id,
        join_node_ids,
        routes,
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

#[derive(Clone)]
pub struct SqliteRunApplicationService {
    store: SqliteStore,
}

impl SqliteRunApplicationService {
    fn resume_runtime_run(&self, run_id: &str, request: &ApiEffectRequest) -> ApiEffectResponse {
        #[derive(Deserialize)]
        struct ResumeRun {
            suspended_node_id: String,
            suspended_token_id: String,
            #[serde(default)]
            client_result: Value,
            workflow_id: String,
            conversation_id: String,
        }
        let input: ResumeRun = match serde_json::from_slice(&request.body) {
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
            .map(|store| Self { store })
            .map_err(|error| error.to_string())
    }

    fn get_run(&self, run_id: &str) -> Result<Option<ServerRun>, String> {
        self.store
            .get_server_run(run_id)
            .map_err(|error| error.to_string())
    }

    fn submit_run(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        #[derive(Deserialize)]
        struct SubmitRun {
            workflow_id: String,
            conversation_id: String,
            #[serde(default)]
            turn_node_id: Option<String>,
            #[serde(default)]
            user_id: Option<String>,
            #[serde(default)]
            initial_state: serde_json::Map<String, Value>,
            #[serde(default = "default_priority_class")]
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
        }

        fn default_priority_class() -> String {
            "foreground".to_owned()
        }
        fn default_runtime_kind() -> String {
            "sync".to_owned()
        }

        let input: SubmitRun = match serde_json::from_slice(&request.body) {
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
        let run_id = uuid::Uuid::new_v4().to_string();
        let turn_node_id = input
            .turn_node_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("wf_turn|{}", uuid::Uuid::new_v4()));
        let message_id = format!("lane|{}", uuid::Uuid::new_v4());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|value| value.as_secs() as i64)
            .unwrap_or(0);
        let priority_class = if input.priority_class.is_empty() {
            default_priority_class()
        } else {
            input.priority_class
        };
        let runtime_kind = if input.runtime_kind.is_empty() {
            default_runtime_kind()
        } else {
            input.runtime_kind
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
            .unwrap_or_else(|| "start".to_owned());
        let mut initial_state = input.initial_state.clone();
        if !effective_routes.is_empty() {
            initial_state.insert(
                "_rt_routes".to_owned(),
                serde_json::to_value(&effective_routes).expect("static runtime routes serialize"),
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
        });
        let outcome = self.store.immediate_transaction(|uow| {
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
            Ok(())
        });
        if let Err(error) = outcome {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            );
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
        #[derive(Deserialize)]
        struct ClaimRequest {
            claimed_by: String,
            #[serde(default = "default_limit")]
            limit: usize,
            #[serde(default = "default_lease")]
            lease_seconds: i64,
        }
        fn default_limit() -> usize {
            1
        }
        fn default_lease() -> i64 {
            60
        }
        let input: ClaimRequest = match serde_json::from_slice(&request.body) {
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
            let claimed = uow.claim_projected_lane_messages(
                "workflow",
                "workflow-runtime",
                &input.claimed_by,
                input.limit,
                input.lease_seconds,
            )?;
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
                let workflow_id = payload["workflow_id"].as_str().unwrap_or_default();
                let conversation_id = payload["conversation_id"].as_str().unwrap_or_default();
                let starts_runtime = lane.msg_type == "workflow.run.execute"
                    && payload["runtime_started"].as_bool() != Some(true);
                let expected_event_seq = if starts_runtime {
                    let start_node_id = payload["start_node_id"].as_str().unwrap_or("start");
                    let turn_node_id = payload["turn_node_id"].as_str().unwrap_or_default();
                    let initial_state = payload["initial_state"]
                        .as_object()
                        .cloned()
                        .unwrap_or_default();
                    uow.apply_recorded_runtime_transition(
                        RecordedRuntimeTransition {
                            contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                            transition_id: format!("start-{run_id}"),
                            expected_event_seq: payload["expected_event_seq"]
                                .as_i64()
                                .unwrap_or_default(),
                            kind: RecordedTransitionKind::Start,
                            run_id: run_id.clone(),
                            workflow_id: workflow_id.to_owned(),
                            conversation_id: conversation_id.to_owned(),
                            user_id: payload["user_id"].as_str().map(str::to_owned),
                            user_turn_node_id: Some(turn_node_id.to_owned()),
                            step_seq: 0,
                            node_id: Some(start_node_id.to_owned()),
                            token_id: Some(run_id.clone()),
                            parent_token_id: None,
                            initial_state: Some(initial_state),
                            state_update: Vec::new(),
                            update: None,
                            state_schema: serde_json::Map::new(),
                            frontier: Some(RuntimeFrontier {
                                pending: vec![(
                                    start_node_id.to_owned(),
                                    payload["start_join_mask"].as_i64().unwrap_or_default(),
                                    run_id.clone(),
                                    None,
                                )],
                                join_outstanding: vec![
                                    1;
                                    payload["join_node_ids"]
                                        .as_array()
                                        .map(Vec::len)
                                        .unwrap_or_default()
                                ],
                                join_waiters: payload["join_node_ids"]
                                    .as_array()
                                    .into_iter()
                                    .flatten()
                                    .filter_map(Value::as_str)
                                    .map(|node_id| (node_id.to_owned(), Vec::new()))
                                    .collect(),
                                join_node_ids: payload["join_node_ids"]
                                    .as_array()
                                    .into_iter()
                                    .flatten()
                                    .filter_map(Value::as_str)
                                    .map(str::to_owned)
                                    .collect(),
                                ..RuntimeFrontier::default()
                            }),
                            result: None,
                            wait_reason: None,
                            resume_payload: None,
                            errors: Vec::new(),
                        },
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
                values.push(json!({
                    "message_id": lane.message_id,
                    "claimed_by": input.claimed_by,
                    "run_id": run_id,
                    "step_id": lane.step_id,
                    "correlation_id": lane.correlation_id,
                    "payload": claimed_payload,
                    "expected_event_seq": expected_event_seq,
                    "lease_until": lane.lease_until,
                }));
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
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct WorkerResultRequest {
            handoff: RecordedWorkerHandoff,
            #[serde(default)]
            transition: Option<RecordedRuntimeTransition>,
            #[serde(default)]
            effect: Option<RecordedWorkerSuccessEffect>,
        }
        let input: WorkerResultRequest = match serde_json::from_slice(&request.body) {
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
                        payload_json: Some(serde_json::to_string(&json!({
                            "contract_version": 1,
                            "kind": "workflow.step.execute",
                            "run_id": transition.run_id,
                            "workflow_id": transition.workflow_id,
                            "conversation_id": transition.conversation_id,
                            "node_id": node_id,
                            "join_mask": join_mask,
                            "token_id": token_id,
                            "parent_token_id": parent_token_id,
                            "step_seq": next_step_seq,
                            "expected_event_seq": applied.event_seq,
                            "state": applied.reduced.state.state,
                        }))?),
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
}

impl PostgresRunApplicationService {
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
            .map(|store| Self { store })
            .map_err(|error| error.to_string())
    }

    pub async fn ensure_schema(&self) -> Result<(), String> {
        self.store
            .ensure_schema()
            .await
            .map_err(|error| error.to_string())
    }

    async fn submit_run(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        #[derive(Deserialize)]
        struct SubmitRun {
            workflow_id: String,
            conversation_id: String,
            #[serde(default)]
            turn_node_id: Option<String>,
            #[serde(default)]
            user_id: Option<String>,
            #[serde(default)]
            initial_state: serde_json::Map<String, Value>,
            #[serde(default)]
            priority_class: String,
            #[serde(default)]
            token_budget: Option<i64>,
            #[serde(default)]
            time_budget_ms: Option<i64>,
            #[serde(default)]
            runtime_kind: String,
            #[serde(default)]
            join_node_ids: Vec<String>,
            #[serde(default)]
            start_join_mask: i64,
            #[serde(default)]
            runtime_routes: Vec<kogwistar_runtime::RuntimeStaticRoute>,
        }
        let input: SubmitRun = match serde_json::from_slice(&request.body) {
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
        let run_id = uuid::Uuid::new_v4().to_string();
        let turn_node_id = input
            .turn_node_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("wf_turn|{}", uuid::Uuid::new_v4()));
        let message_id = format!("lane|{}", uuid::Uuid::new_v4());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|value| value.as_secs() as i64)
            .unwrap_or(0);
        let priority_class = if input.priority_class.is_empty() {
            "foreground".to_owned()
        } else {
            input.priority_class
        };
        let runtime_kind = if input.runtime_kind.is_empty() {
            "sync".to_owned()
        } else {
            input.runtime_kind
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
            .unwrap_or_else(|| "start".to_owned());
        let mut initial_state = input.initial_state;
        if !effective_routes.is_empty() {
            initial_state.insert(
                "_rt_routes".to_owned(),
                serde_json::to_value(&effective_routes).expect("static runtime routes serialize"),
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
        let outcome = self
            .store
            .transaction(|uow| {
                Box::pin(async move {
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
                        "priority_class": priority_class,
                        "token_budget": input.token_budget,
                        "time_budget_ms": input.time_budget_ms,
                        "runtime_kind": runtime_kind,
                        "join_node_ids": effective_join_node_ids,
                        "start_join_mask": effective_start_join_mask,
                        "start_node_id": effective_start_node_id,
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
                    .await
                })
            })
            .await;
        match outcome {
            Ok(()) => json_effect(StatusCode::ACCEPTED, response),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    async fn claim_runtime_work(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        #[derive(Deserialize)]
        struct ClaimRequest {
            claimed_by: String,
            #[serde(default = "default_limit")]
            limit: usize,
            #[serde(default = "default_lease")]
            lease_seconds: i64,
        }
        fn default_limit() -> usize {
            1
        }
        fn default_lease() -> i64 {
            60
        }
        let input: ClaimRequest = match serde_json::from_slice(&request.body) {
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
                    let claimed = uow
                        .claim_projected_lane_messages(
                            "workflow",
                            "workflow-runtime",
                            &input.claimed_by,
                            input.limit,
                            input.lease_seconds,
                        )
                        .await?;
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
                            let start_node_id =
                                payload["start_node_id"].as_str().unwrap_or("start");
                            let initial_state = payload["initial_state"]
                                .as_object()
                                .cloned()
                                .unwrap_or_default();
                            uow.apply_recorded_runtime_transition(
                                RecordedRuntimeTransition {
                                    contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                                    transition_id: format!("start-{run_id}"),
                                    expected_event_seq: payload["expected_event_seq"]
                                        .as_i64()
                                        .unwrap_or_default(),
                                    kind: RecordedTransitionKind::Start,
                                    run_id: run_id.clone(),
                                    workflow_id: payload["workflow_id"]
                                        .as_str()
                                        .unwrap_or_default()
                                        .to_owned(),
                                    conversation_id: payload["conversation_id"]
                                        .as_str()
                                        .unwrap_or_default()
                                        .to_owned(),
                                    user_id: payload["user_id"].as_str().map(str::to_owned),
                                    user_turn_node_id: payload["turn_node_id"]
                                        .as_str()
                                        .map(str::to_owned),
                                    step_seq: 0,
                                    node_id: Some(start_node_id.to_owned()),
                                    token_id: Some(run_id.clone()),
                                    parent_token_id: None,
                                    initial_state: Some(initial_state),
                                    state_update: Vec::new(),
                                    update: None,
                                    state_schema: serde_json::Map::new(),
                                    frontier: Some(RuntimeFrontier {
                                        pending: vec![(
                                            start_node_id.to_owned(),
                                            payload["start_join_mask"].as_i64().unwrap_or_default(),
                                            run_id.clone(),
                                            None,
                                        )],
                                        join_outstanding: vec![
                                            1;
                                            payload["join_node_ids"]
                                                .as_array()
                                                .map(Vec::len)
                                                .unwrap_or_default()
                                        ],
                                        join_waiters: payload["join_node_ids"]
                                            .as_array()
                                            .into_iter()
                                            .flatten()
                                            .filter_map(Value::as_str)
                                            .map(|node_id| (node_id.to_owned(), Vec::new()))
                                            .collect(),
                                        join_node_ids: payload["join_node_ids"]
                                            .as_array()
                                            .into_iter()
                                            .flatten()
                                            .filter_map(Value::as_str)
                                            .map(str::to_owned)
                                            .collect(),
                                        ..RuntimeFrontier::default()
                                    }),
                                    result: None,
                                    wait_reason: None,
                                    resume_payload: None,
                                    errors: Vec::new(),
                                },
                                false,
                            )
                            .await?
                            .event_seq
                        } else {
                            payload["expected_event_seq"].as_i64().unwrap_or_default()
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
                        values.push(json!({
                            "message_id": lane.message_id,
                            "claimed_by": input.claimed_by,
                            "run_id": run_id,
                            "step_id": lane.step_id,
                            "correlation_id": lane.correlation_id,
                            "payload": payload,
                            "expected_event_seq": expected_event_seq,
                            "lease_until": lane.lease_until,
                        }));
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
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct WorkerResultRequest {
            handoff: RecordedWorkerHandoff,
            effect: RecordedWorkerSuccessEffect,
        }
        let input: WorkerResultRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let outcome = self
            .store
            .transaction(|uow| {
                Box::pin(async move {
                    let handoff = input.handoff;
                    let effect_id = input.effect.effect_id.clone();
                    let terminal_result = input.effect.result.clone();
                    let usage = input.effect.usage.clone();
                    let trace_events = input.effect.trace_events.clone();
                    let applied = uow
                        .apply_claimed_recorded_worker_effect(handoff, input.effect)
                        .await?;
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
                                payload_json: Some(serde_json::to_string(&json!({
                                    "contract_version": 1,
                                    "kind": "workflow.step.execute",
                                    "run_id": state.run_id,
                                    "workflow_id": state.workflow_id,
                                    "conversation_id": state.conversation_id,
                                    "node_id": node_id,
                                    "join_mask": join_mask,
                                    "token_id": token_id,
                                    "parent_token_id": parent_token_id,
                                    "step_seq": next_step_seq,
                                    "expected_event_seq": applied.event_seq,
                                    "state": state.state,
                                }))?),
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
        #[derive(Deserialize)]
        struct ResumeRun {
            suspended_node_id: String,
            suspended_token_id: String,
            #[serde(default)]
            client_result: Value,
            workflow_id: String,
            conversation_id: String,
        }
        let input: ResumeRun = match serde_json::from_slice(&request.body) {
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
    fn execute_sync(&self, request: ApiEffectRequest) -> ApiEffectResponse {
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

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get("authorization")?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("bearer")
        .then_some(token.trim())
}

fn jwt_roles(headers: &HeaderMap, config: &AuthConfig) -> Result<Vec<String>, &'static str> {
    let token = bearer_token(headers).ok_or("Missing bearer token")?;
    let configured_algorithm = jwt_algorithm(config.algorithm.as_deref().unwrap_or("HS256"))
        .ok_or("Unsupported JWT algorithm")?;
    let header = decode_header(token).map_err(|_| "Invalid bearer token")?;
    if header.alg != configured_algorithm {
        return Err("JWT algorithm mismatch");
    }
    let key_text = config
        .key
        .as_deref()
        .ok_or("JWT secret is not configured")?;
    let key = match configured_algorithm {
        Algorithm::HS256 => DecodingKey::from_secret(key_text.as_bytes()),
        Algorithm::RS256 => {
            DecodingKey::from_rsa_pem(key_text.as_bytes()).map_err(|_| "Invalid RSA public key")?
        }
        _ => return Err("Unsupported JWT algorithm"),
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
    let claims = decode::<Value>(token, &key, &validation)
        .map_err(|_| "Invalid bearer token")?
        .claims;
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

fn request_roles(headers: &HeaderMap, config: &AuthConfig) -> Result<Vec<String>, &'static str> {
    if config.configured() {
        jwt_roles(headers, config)
    } else {
        Ok(roles_from_headers(headers))
    }
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
                })
            }
            "workflow.run_cancel" | "conversation.cancel_run" if !run_id.is_empty() => {
                Some(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: format!("/api/workflow/runs/{run_id}/cancel"),
                    body: Vec::new(),
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
                })
            }
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
        if method == "GET" && path == "/health" {
            continue;
        }
        router = router.route(path, on(method_filter(method), application_effect_handler));
    }
    router
        .route("/health", get(health_handler))
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
            algorithm: Some("HS256".to_owned()),
            key: Some("test-secret".to_owned()),
            issuer: Some("issuer".to_owned()),
            audience: Some("audience".to_owned()),
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
    fn frozen_route_registry_covers_committed_openapi() {
        assert_eq!(FROZEN_OPENAPI_ROUTES.len(), 82);
        assert!(FROZEN_OPENAPI_ROUTES.contains(&("GET", "/health")));
        assert!(FROZEN_OPENAPI_ROUTES.contains(&("POST", "/api/workflow/runs")));
        assert!(FROZEN_OPENAPI_ROUTES.contains(&("GET", "/api/runs/{run_id}/events")));
        let embedded: Value = serde_json::from_str(FROZEN_OPENAPI_JSON).unwrap();
        assert_eq!(embedded["paths"].as_object().unwrap().len(), 80);
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
    async fn sqlite_run_service_handles_get_events_and_cancel() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
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
                        r#"{"workflow_id":"workflow-submit","conversation_id":"conversation-submit","initial_state":{"seed":1},"priority_class":"foreground","runtime_kind":"sync"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(submit.status(), StatusCode::ACCEPTED);
        let submit_body = to_bytes(submit.into_body(), 8192).await.unwrap();
        let submitted = serde_json::from_slice::<Value>(&submit_body).unwrap();
        let submitted_run_id = submitted["run_id"].as_str().unwrap();
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
                "node_id": "start",
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
            json!({"workflow_id":"resume-wf","conversation_id":"resume-conv"}),
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
        let token_id = contract["suspended"][0][2].as_str().unwrap().to_owned();
        let resume_body = json!({
            "suspended_node_id": "start",
            "suspended_token_id": token_id,
            "client_result": {"approved": true},
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
        assert_eq!(work["payload"]["resume_payload"]["approved"], true);
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
