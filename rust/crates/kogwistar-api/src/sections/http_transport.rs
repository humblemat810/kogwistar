async fn health_handler(State(state): State<Arc<ServerState>>) -> Json<Value> {
    let mut value = health_response(&state.api.health);
    if let Some(object) = value.as_object_mut() {
        let mut implementation = state.api.implementation.clone();
        implementation.implemented_route_operations =
            state.application.implemented_http_operations();
        let effective_server_cutover_ready = implementation.effective_server_cutover_ready();
        object.insert(
            "implementation".to_owned(),
            serde_json::to_value(&implementation).expect("implementation snapshot serializes"),
        );
        if let Some(implementation) = object
            .get_mut("implementation")
            .and_then(Value::as_object_mut)
        {
            implementation.insert(
                "effective_server_cutover_ready".to_owned(),
                json!(effective_server_cutover_ready),
            );
        }
        let capability = state.application.graph_authority_capability();
        object.insert(
            "graph_authority".to_owned(),
            json!({
                "level": capability.as_str(),
                "single_mutation": capability.supports_single_mutation(),
                "atomic_index_jobs": capability.supports_atomic_index_jobs(),
            }),
        );
        let document_capability = state.application.document_authority_capability();
        object.insert(
            "document_authority".to_owned(),
            json!({
                "level": document_capability.as_str(),
                "projection_upsert": document_capability.supports_projection_upsert(),
            }),
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
    let surface = match uri.path() {
        "/mcp/conversation" => "conversation",
        "/mcp/workflow" => "workflow",
        _ => "root",
    };
    if method == "tools/list" {
        let contract: Value = serde_json::from_str(FROZEN_MCP_TOOLS_JSON)
            .expect("committed MCP tool contract is valid JSON");
        let tools = contract["surfaces"][surface]
            .as_array()
            .map(|tools| {
                tools
                    .iter()
                    .filter(|tool| {
                        tool.get("name")
                            .and_then(Value::as_str)
                            .is_some_and(|name| state.application.supports_mcp_tool(surface, name))
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        return Json(mcp_result(id, json!({"tools": tools}))).into_response();
    }
    if method == "tools/call" {
        let principal = request_principal(&headers, &state.api.auth);
        let params = payload.get("params").cloned().unwrap_or_else(|| json!({}));
        let tool_name = params
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !state.application.supports_mcp_tool(surface, tool_name) {
            return Json(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32602, "message": "Tool not advertised by this application service"},
            }))
            .into_response();
        }
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
            "kg_find_edges" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/find-edges".to_owned(),
                body: serde_json::to_vec(&arguments).expect("MCP graph arguments serialize"),
                principal: principal.clone(),
            }),
            "kg_shortest_path" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/shortest-path".to_owned(),
                body: serde_json::to_vec(&arguments).expect("MCP graph arguments serialize"),
                principal: principal.clone(),
            }),
            "kg_k_hop" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/k-hop".to_owned(),
                body: serde_json::to_vec(&arguments).expect("MCP graph arguments serialize"),
                principal: principal.clone(),
            }),
            "kg_neighbors" => arguments
                .get("rid")
                .and_then(Value::as_str)
                .filter(|rid| !rid.trim().is_empty())
                .map(|rid| {
                    let mut body = json!({
                        "rid": rid,
                        "truth_graph": "knowledge",
                    });
                    if let Some(doc_id) = arguments.get("doc_id").and_then(Value::as_str) {
                        body["doc_id"] = json!(doc_id);
                    }
                    ApiEffectRequest {
                        contract_version: 1,
                        method: "POST".to_owned(),
                        path_and_query: "/internal/graph/neighbors".to_owned(),
                        body: serde_json::to_vec(&body).expect("MCP graph arguments serialize"),
                        principal: principal.clone(),
                    }
                }),
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
            "workflow.tool_audit" => {
                let conversation_id =
                    match arguments.get("conversation_id") {
                        None | Some(Value::Null) => None,
                        Some(Value::String(value)) if !value.is_empty() => Some(value.as_str()),
                        _ => return Json(json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {"code": -32602, "message": "Invalid tool-audit arguments"},
                        }))
                        .into_response(),
                    };
                let limit = match arguments.get("limit") {
                    None => 200,
                    Some(Value::Number(value)) => match value.as_i64() {
                        Some(value) => value,
                        None => {
                            return Json(json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "error": {"code": -32602, "message": "Invalid tool-audit limit"},
                            }))
                            .into_response();
                        }
                    },
                    _ => {
                        return Json(json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {"code": -32602, "message": "Invalid tool-audit limit"},
                        }))
                        .into_response();
                    }
                };
                let mut query = url::form_urlencoded::Serializer::new(String::new());
                query.append_pair("limit", &limit.to_string());
                if let Some(conversation_id) = conversation_id {
                    query.append_pair("conversation_id", conversation_id);
                }
                Some(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: format!("/api/workflow/tools/audit?{}", query.finish()),
                    body: Vec::new(),
                    principal: principal.clone(),
                })
            }
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
    for path in INTERNAL_APPLICATION_EFFECT_ROUTES {
        router = router.route(path, post(application_effect_handler));
    }
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
