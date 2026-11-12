async fn health_handler(State(state): State<Arc<ServerState>>) -> Json<Value> {
    let mut value = health_response(&state.api.health);
    if let Some(object) = value.as_object_mut() {
        let inline_native_worker_enabled = std::env::var("KOGWISTAR_RUNTIME_NATIVE_INLINE")
            .ok()
            .is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            });
        object.insert(
            "native_inline_worker_enabled".to_owned(),
            json!(inline_native_worker_enabled),
        );
        let inline_embedding_worker_enabled = std::env::var("KOGWISTAR_EMBEDDING_WORKER_INLINE")
            .ok()
            .is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            });
        object.insert(
            "embedding_inline_worker_enabled".to_owned(),
            json!(inline_embedding_worker_enabled),
        );
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

fn map_graph_short_ids(ids: &[Value], map_key: &str) -> Result<Vec<String>, ()> {
    let mut mapped = ids.iter()
        .map(|value| {
            let canonical_id = value.as_str().filter(|id| !id.trim().is_empty()).ok_or(())?;
            crate::short_id_store::short_id_for(".shortids", map_key, canonical_id).map_err(|_| ())
        })
        .collect::<Result<Vec<_>, _>>()?;
    mapped.sort();
    Ok(mapped)
}

fn mcp_tool_success_result(result: Value) -> Value {
    let text = serde_json::to_string(&result).unwrap_or_else(|_| "null".to_owned());
    let mut response = json!({"content": [{"type": "text", "text": text}]});
    if result.is_object() {
        response["structuredContent"] = result;
    }
    response
}

async fn mcp_tool_call_error_result(response: Response, id: Value) -> Response {
    if !response.status().is_success() {
        return response;
    }
    let (parts, body) = response.into_parts();
    let Ok(body) = axum::body::to_bytes(body, 1024 * 1024).await else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };
    let Ok(payload) = serde_json::from_slice::<Value>(&body) else {
        return Response::from_parts(parts, Body::from(body));
    };
    let Some(error) = payload.get("error") else {
        return Response::from_parts(parts, Body::from(body));
    };
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("Tool call failed");
    Json(mcp_result(
        id,
        json!({
            "content": [{"type": "text", "text": message}],
            "isError": true,
        }),
    ))
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
    let response_mode = headers
        .get(header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .map(|accept| {
            let accepts_json = accept.split(',').any(|item| {
                item.trim()
                    .split(';')
                    .next()
                    .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("application/json"))
            });
            let accepts_sse = accept.split(',').any(|item| {
                item.trim()
                    .split(';')
                    .next()
                    .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("text/event-stream"))
            });
            (accepts_json, accepts_sse)
        });
    let response_mode = response_mode.filter(|_| {
        !headers
            .get(header::ACCEPT)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|accept| {
                accept.split(',').any(|item| {
                    item.trim()
                        .split(';')
                        .next()
                        .is_some_and(|mime| mime.trim() == "*/*")
                })
            })
    });
    if response_mode.is_some_and(|(json, sse)| !json || !sse) {
        return StatusCode::NOT_ACCEPTABLE.into_response();
    }
    let is_initialize = payload.get("method").and_then(Value::as_str) == Some("initialize");
    let session_owner = request_mcp_session_owner(&headers, &state.api.auth);
    let incoming_protocol_version = headers
        .get("mcp-protocol-version")
        .and_then(|value| value.to_str().ok());
    if !is_initialize
        && incoming_protocol_version
            .is_some_and(|version| {
                !matches!(
                    version,
                    "2024-11-05" | "2025-03-26" | "2025-06-18" | "2025-11-25"
                )
            })
    {
        return StatusCode::BAD_REQUEST.into_response();
    }

    let requested_initialize_version = is_initialize.then(|| {
        payload
            .pointer("/params/protocolVersion")
            .and_then(Value::as_str)
            .unwrap_or("2025-11-25")
    });
    let negotiated_initialize_version = requested_initialize_version.map(|version| match version {
        "2024-11-05" | "2025-03-26" | "2025-06-18" | "2025-11-25" => version,
        _ => "2025-11-25",
    });
    let incoming_session_id = headers
        .get("mcp-session-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let mut existing_session_version = None;
    if let Some(session_id) = incoming_session_id.as_deref() {
        let requested_session_version = incoming_protocol_version.or(negotiated_initialize_version);
        match state.mcp_sessions.touch(
            session_id,
            &session_owner,
            requested_session_version,
        ) {
            Ok(version) => existing_session_version = Some(version),
            Err(McpSessionError::ProtocolVersionMismatch) => {
                return StatusCode::BAD_REQUEST.into_response();
            }
            Err(McpSessionError::NotFound) => {
                return StatusCode::NOT_FOUND.into_response();
            }
            Err(_) => return StatusCode::BAD_REQUEST.into_response(),
        }
    } else if !is_initialize && incoming_protocol_version.is_some() {
        return StatusCode::BAD_REQUEST.into_response();
    }

    let mut session_id_to_return = incoming_session_id;
    let mut created_new_session = false;
    if is_initialize && session_id_to_return.is_none() {
        let protocol_version = negotiated_initialize_version.unwrap_or("2025-11-25");
        let Some(session_id) = state
            .mcp_sessions
            .open(session_owner.clone(), protocol_version.to_owned())
        else {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {"code": -32603, "message": "MCP session capacity reached"}
                })),
            )
                .into_response();
        };
        session_id_to_return = Some(session_id);
        created_new_session = true;
    }
    let request_protocol_version = existing_session_version
        .or_else(|| negotiated_initialize_version.map(str::to_owned))
        .or_else(|| incoming_protocol_version.map(str::to_owned));
    let standard_tool_call = response_mode.is_some()
        && payload.get("method").and_then(Value::as_str) == Some("tools/call");
    let tool_call_id = payload.get("id").cloned().unwrap_or(Value::Null);
    let mut response = mcp_dispatch(
        state.clone(),
        headers.clone(),
        uri,
        payload,
        response_mode.is_some(),
    )
    .await;
    if standard_tool_call {
        response = mcp_tool_call_error_result(response, tool_call_id).await;
    }
    if response.status().is_success() {
        if let Some(session_id) = session_id_to_return.as_deref() {
            if let Ok(value) = HeaderValue::from_str(session_id) {
                response.headers_mut().insert("mcp-session-id", value);
            }
        }
    } else if created_new_session && let Some(session_id) = session_id_to_return.as_deref() {
        let _ = state.mcp_sessions.close(session_id, &session_owner, None);
    }
    if let Some(protocol_version) = request_protocol_version {
        response.headers_mut().insert(
            "mcp-protocol-version",
            HeaderValue::from_str(&protocol_version)
                .unwrap_or_else(|_| HeaderValue::from_static("2025-11-25")),
        );
    }
    if response_mode.is_none() || response.status() != StatusCode::OK {
        return response;
    }
    let protocol_version = response
        .headers()
        .get("mcp-protocol-version")
        .cloned();
    let session_id = response.headers().get("mcp-session-id").cloned();
    let Ok(body) = axum::body::to_bytes(response.into_body(), usize::MAX).await else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };
    let mut sse = Vec::with_capacity(body.len() + 32);
    sse.extend_from_slice(b"event: message\ndata: ");
    sse.extend_from_slice(&body);
    sse.extend_from_slice(b"\n\n");
    let mut response = (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "text/event-stream; charset=utf-8"),
            (header::CACHE_CONTROL, "no-cache, no-transform"),
            (header::CONNECTION, "keep-alive"),
        ],
        sse,
    )
        .into_response();
    if let Some(protocol_version) = protocol_version {
        response
            .headers_mut()
            .insert("mcp-protocol-version", protocol_version);
    }
    if let Some(session_id) = session_id {
        response.headers_mut().insert("mcp-session-id", session_id);
    }
    response
}

async fn mcp_delete_handler(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> Response {
    if !has_required_role(&headers, &state.api.required_roles, &state.api.auth) {
        return forbidden_response();
    }
    let Some(session_id) = headers
        .get("mcp-session-id")
        .and_then(|value| value.to_str().ok())
    else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let protocol_version = headers
        .get("mcp-protocol-version")
        .and_then(|value| value.to_str().ok());
    let session_owner = request_mcp_session_owner(&headers, &state.api.auth);
    match state
        .mcp_sessions
        .close(session_id, &session_owner, protocol_version)
    {
        Ok(()) => StatusCode::OK.into_response(),
        Err(McpSessionError::ProtocolVersionMismatch) => StatusCode::BAD_REQUEST.into_response(),
        Err(McpSessionError::NotFound) => StatusCode::NOT_FOUND.into_response(),
        Err(_) => StatusCode::BAD_REQUEST.into_response(),
    }
}

async fn mcp_get_handler(State(state): State<Arc<ServerState>>, headers: HeaderMap) -> Response {
    if !has_required_role(&headers, &state.api.required_roles, &state.api.auth) {
        return forbidden_response();
    }
    let accepts_sse = headers
        .get(header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|accept| {
            accept.split(',').any(|item| {
                item.trim()
                    .split(';')
                    .next()
                    .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("text/event-stream"))
            })
        });
    if !accepts_sse {
        return StatusCode::NOT_ACCEPTABLE.into_response();
    }
    let Some(session_id) = headers
        .get("mcp-session-id")
        .and_then(|value| value.to_str().ok())
    else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let protocol_version = headers
        .get("mcp-protocol-version")
        .and_then(|value| value.to_str().ok());
    let session_owner = request_mcp_session_owner(&headers, &state.api.auth);
    let last_event_id = headers
        .get("last-event-id")
        .and_then(|value| value.to_str().ok());
    let subscription = match state.mcp_sessions.subscribe_get(
        session_id,
        &session_owner,
        protocol_version,
        last_event_id,
    ) {
        Ok(subscription) => subscription,
        Err(McpSessionError::ProtocolVersionMismatch | McpSessionError::InvalidLastEventId) => {
            return StatusCode::BAD_REQUEST.into_response();
        }
        Err(McpSessionError::NotFound) => return StatusCode::NOT_FOUND.into_response(),
        Err(McpSessionError::GetStreamAlreadyActive) => {
            return StatusCode::CONFLICT.into_response();
        }
        Err(McpSessionError::EventHistoryUnavailable) => {
            return StatusCode::CONFLICT.into_response();
        }
        Err(McpSessionError::InvalidOutboundMessage | McpSessionError::OutboundMessageTooLarge) => {
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };
    let replay = subscription.replay;
    let mut receiver = subscription.receiver;
    let registry = state.mcp_sessions.clone();
    let session_id = session_id.to_owned();
    let stream_session_id = session_id.clone();
    // Python parity: MCPToolRegistry.http_app delegates this session-bound GET SSE
    // stream to the official StreamableHTTPSessionManager transport.
    let stream_guard = McpGetStreamGuard::new(registry, stream_session_id);
    let stream = async_stream::stream! {
        let _stream_guard = stream_guard;
        for event in replay {
            let event_id = event.event_id();
            let frame = sse_frame("message", event.message(), Some(&event_id));
            yield Ok::<Bytes, std::convert::Infallible>(Bytes::from(frame));
        }
        let mut heartbeat = tokio::time::interval_at(
            tokio::time::Instant::now() + Duration::from_secs(15),
            Duration::from_secs(15),
        );
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                event = receiver.recv() => match event {
                    Ok(event) => {
                        let event_id = event.event_id();
                        let frame = sse_frame("message", event.message(), Some(&event_id));
                        yield Ok(Bytes::from(frame));
                    }
                    // A reconnect can replay from Last-Event-ID; do not silently skip messages.
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)
                        | tokio::sync::broadcast::error::RecvError::Closed) => break,
                },
                _ = heartbeat.tick() => yield Ok(Bytes::from_static(b": keep-alive\n\n")),
            }
        }
    };
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-cache, no-transform"),
    );
    response.headers_mut().insert(
        header::CONNECTION,
        HeaderValue::from_static("keep-alive"),
    );
    response.headers_mut().insert(
        "mcp-session-id",
        HeaderValue::from_str(session_id.as_str()).unwrap_or_else(|_| HeaderValue::from_static("")),
    );
    if let Some(protocol_version) = protocol_version {
        response.headers_mut().insert(
            "mcp-protocol-version",
            HeaderValue::from_str(protocol_version)
                .unwrap_or_else(|_| HeaderValue::from_static("2025-11-25")),
        );
    }
    response
}

async fn mcp_dispatch(
    state: Arc<ServerState>,
    headers: HeaderMap,
    uri: axum::http::Uri,
    payload: Value,
    standard_mcp: bool,
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
    if method == "initialize" {
        let valid_params = payload
            .get("params")
            .and_then(Value::as_object)
            .is_some_and(|params| {
                params.get("protocolVersion").is_some_and(Value::is_string)
                    && params.get("capabilities").is_some_and(Value::is_object)
                    && params.get("clientInfo").and_then(Value::as_object).is_some_and(
                        |client| {
                            client.get("name").is_some_and(Value::is_string)
                                && client.get("version").is_some_and(Value::is_string)
                        },
                    )
            });
        if !valid_params {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32602, "message": "Invalid initialize parameters"}
                })),
            )
                .into_response();
        }
        let requested_version = payload
            .pointer("/params/protocolVersion")
            .and_then(Value::as_str)
            .unwrap_or("2025-11-25");
        let protocol_version = match requested_version {
            "2024-11-05" | "2025-03-26" | "2025-06-18" | "2025-11-25" => requested_version,
            _ => "2025-11-25",
        };
        let server_name = match surface {
            "conversation" => "Conversation MCP",
            "workflow" => "Workflow Diagnostics MCP",
            _ => "KnowledgeEngine + MCP + Admin",
        };
        return Json(mcp_result(
            id,
            json!({
                "protocolVersion": protocol_version,
                "capabilities": {"tools": {"listChanged": false}},
                "serverInfo": {"name": server_name, "version": env!("CARGO_PKG_VERSION")},
            }),
        ))
        .into_response();
    }
    if method == "notifications/initialized" {
        return StatusCode::ACCEPTED.into_response();
    }
    if method == "ping" {
        return Json(mcp_result(id, json!({}))).into_response();
    }
    if method == "tools/list" {
        let principal = request_principal(&headers, &state.api.auth);
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
                            .is_some_and(|name| {
                                state
                                    .application
                                    .supports_mcp_tool_for_principal(surface, name, &principal)
                            })
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
        if !state
            .application
            .supports_mcp_tool_for_principal(surface, tool_name, &principal)
        {
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
            "document_id_from_file_name" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: INTERNAL_DOCUMENT_LOOKUP_ROUTE.to_owned(),
                body: serde_json::to_vec(&arguments)
                    .expect("document lookup arguments serialize"),
                principal: principal.clone(),
            }),
            "kg_load_persisted" => arguments
                .get("inp")
                .and_then(Value::as_object)
                .filter(|_| arguments.len() == 1)
                .map(|input| ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: INTERNAL_GRAPH_LOAD_PERSISTED_ROUTE.to_owned(),
                    body: serde_json::to_vec(input)
                        .expect("load persisted arguments serialize"),
                    principal: principal.clone(),
                }),
            "kg_propose_bruteforce" => arguments
                .get("inp")
                .and_then(Value::as_object)
                .filter(|_| arguments.len() == 1)
                .map(|input| ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: INTERNAL_GRAPH_PROPOSE_BRUTE_FORCE_ROUTE.to_owned(),
                    body: serde_json::to_vec(input)
                        .expect("brute-force proposal arguments serialize"),
                    principal: principal.clone(),
                }),
            "conversation.create" => {
                let user_id = arguments
                    .get("user_id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.trim().is_empty());
                let conversation_id = arguments
                    .get("conversation_id")
                    .filter(|value| !value.is_null())
                    .and_then(Value::as_str)
                    .filter(|value| !value.trim().is_empty());
                let start_node_id = arguments
                    .get("start_node_id")
                    .filter(|value| !value.is_null())
                    .and_then(Value::as_str)
                    .filter(|value| !value.trim().is_empty());
                let optional_strings_valid =
                    ["conversation_id", "start_node_id"].iter().all(|key| {
                        arguments.get(*key).is_none_or(|value| {
                            value.is_null()
                                || value.as_str().is_some_and(|value| !value.trim().is_empty())
                        })
                    });
                let known_arguments = ["user_id", "conversation_id", "start_node_id"];
                user_id
                    .filter(|_| {
                        optional_strings_valid
                            && arguments
                                .keys()
                                .all(|argument| known_arguments.contains(&argument.as_str()))
                    })
                    .map(|user_id| {
                        let mut syscall_args = json!({"user_id": user_id});
                        if let Some(conversation_id) = conversation_id {
                            syscall_args["conversation_id"] = json!(conversation_id);
                        }
                        if let Some(start_node_id) = start_node_id {
                            syscall_args["start_node_id"] = json!(start_node_id);
                        }
                        ApiEffectRequest {
                            contract_version: 1,
                            method: "POST".to_owned(),
                            path_and_query: "/api/syscall/v1/create_conversation".to_owned(),
                            body: serde_json::to_vec(&json!({
                                "version": "v1",
                                "op": "create_conversation",
                                "args": syscall_args,
                            }))
                            .expect("conversation create syscall serializes"),
                            principal: principal.clone(),
                        }
                    })
            }
            "conversation.ask" => {
                let conversation_id = arguments
                    .get("conversation_id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.trim().is_empty());
                let text = arguments
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|value| !value.trim().is_empty());
                let user_id_valid = arguments
                    .get("user_id")
                    .is_none_or(|value| value.is_null() || value.is_string());
                let workflow_id = arguments
                    .get("workflow_id")
                    .and_then(Value::as_str)
                    .unwrap_or("agentic_answering.v2");
                let known_arguments = ["conversation_id", "text", "user_id", "workflow_id"];
                conversation_id
                    .zip(text)
                    .filter(|_| {
                        user_id_valid
                            && arguments.get("workflow_id").is_none_or(Value::is_string)
                            && arguments
                                .keys()
                                .all(|argument| known_arguments.contains(&argument.as_str()))
                    })
                    .map(|(conversation_id, text)| {
                        let mut syscall_args = json!({
                            "conversation_id": conversation_id,
                            "text": text,
                            "workflow_id": workflow_id,
                        });
                        if let Some(user_id) = arguments.get("user_id") {
                            syscall_args["user_id"] = user_id.clone();
                        }
                        ApiEffectRequest {
                            contract_version: 1,
                            method: "POST".to_owned(),
                            path_and_query: "/api/syscall/v1/send_message".to_owned(),
                            body: serde_json::to_vec(&json!({
                                "version": "v1",
                                "op": "send_message",
                                "args": syscall_args,
                            }))
                            .expect("conversation ask syscall serializes"),
                            principal: principal.clone(),
                        }
                    })
            }
            "conversation.get_transcript" => {
                let conversation_id = arguments
                    .get("conversation_id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.trim().is_empty());
                conversation_id
                    .filter(|_| arguments.len() == 1)
                    .map(|conversation_id| ApiEffectRequest {
                        contract_version: 1,
                        method: "GET".to_owned(),
                        path_and_query: format!(
                            "/api/conversations/{}/turns",
                            percent_encoding::utf8_percent_encode(
                                conversation_id,
                                percent_encoding::NON_ALPHANUMERIC
                            )
                        ),
                        body: Vec::new(),
                        principal: principal.clone(),
                    })
            }
            "kg_find_edges" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/find-edges".to_owned(),
                body: serde_json::to_vec(&arguments).expect("MCP graph arguments serialize"),
                principal: principal.clone(),
            }),
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/mcp_tools.py#propose_vector; fixture=contracts/golden/graph-vector-proposal-acl-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/in_memory_graph_routes.rs#in_memory_backend_routes_authenticated_graph_write_and_mcp_read; python_test=tests/mcp/test_chat_mcp_tools.py#test_propose_vector_returns_only_acl_visible_pairs
            "propose_vector" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: INTERNAL_GRAPH_PROPOSE_VECTOR_ROUTE.to_owned(),
                body: serde_json::to_vec(
                    arguments
                        .get("inp")
                        .and_then(Value::as_object)
                        .unwrap_or(&arguments),
                )
                .expect("MCP graph arguments serialize"),
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
            "kg_viz_d3_json" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: INTERNAL_GRAPH_VIZ_D3_ROUTE.to_owned(),
                body: serde_json::to_vec(&arguments).expect("MCP graph arguments serialize"),
                principal: principal.clone(),
            }),
            "kg_viz_cytoscape_json" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE.to_owned(),
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
            "wisdom.kg_upsert_graph" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/wisdom/graph/upsert".to_owned(),
                body: serde_json::to_vec(&arguments).expect("MCP wisdom graph arguments serialize"),
                principal: principal.clone(),
            }),
            "wisdom.semantic_seed_then_expand" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: WISDOM_SEMANTIC_SEED_EXPAND_ROUTE.to_owned(),
                body: serde_json::to_vec(&arguments)
                    .expect("MCP wisdom semantic arguments serialize"),
                principal: principal.clone(),
            }),
            "kg_semantic_seed_then_expand_text" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: GRAPH_SEMANTIC_SEED_EXPAND_ROUTE.to_owned(),
                body: serde_json::to_vec(&arguments)
                    .expect("MCP graph semantic arguments serialize"),
                principal: principal.clone(),
            }),
            // PYTHON_PARITY_V1: conversation status must use the conversation-scoped route; rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_dispatch; python=kogwistar/server/chat_api.py#create_chat_router.get_run; fixture=contracts/golden/conversation-run-status-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/conversation_tool_authorization.rs#conversation_run_status_mcp_admission_matches_python_policy_for_all_stores; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_conversation_run_status_matches_shared_contract
            "workflow.run_status" if !run_id.is_empty() => {
                Some(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: format!("/api/workflow/runs/{run_id}"),
                    body: Vec::new(),
                    principal: principal.clone(),
                })
            }
            "conversation.run_status" if !run_id.is_empty() => {
                Some(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: format!("/api/runs/{run_id}"),
                    body: Vec::new(),
                    principal: principal.clone(),
                })
            }
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_run_steps; fixture=contracts/golden/workflow-run-steps-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/mcp_run_observability.rs#mcp_run_lineage_builds_bounded_encoded_read_only_route; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_run_steps_matches_shared_contract
            "workflow.run_steps" if !run_id.is_empty() => {
                let valid_arguments = !arguments
                    .keys()
                    .any(|argument| !matches!(argument.as_str(), "run_id" | "limit"))
                    && arguments
                        .get("limit")
                        .map(Value::as_i64)
                        .unwrap_or(Some(200))
                        .is_some_and(|limit| (1..=1000).contains(&limit));
                valid_arguments.then(|| {
                    let limit = arguments
                        .get("limit")
                        .and_then(Value::as_i64)
                        .unwrap_or(200);
                    let encoded_run_id = percent_encoding::utf8_percent_encode(
                        run_id,
                        percent_encoding::NON_ALPHANUMERIC,
                    );
                    ApiEffectRequest {
                        contract_version: 1,
                        method: "GET".to_owned(),
                        path_and_query: format!(
                            "/api/workflow/runs/{encoded_run_id}/steps?limit={limit}"
                        ),
                        body: Vec::new(),
                        principal: principal.clone(),
                    }
                })
            }
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_run_cancel; fixture=contracts/golden/workflow-run-cancel-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/mcp_workflow_contracts.rs#mcp_workflow_run_cancel_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_run_cancel_matches_shared_contract
            "workflow.run_cancel" if !run_id.is_empty() => {
                Some(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: format!("/api/workflow/runs/{run_id}/cancel"),
                    body: Vec::new(),
                    principal: principal.clone(),
                })
            }
            // PYTHON_PARITY_V1: conversation cancellation must use the conversation-scoped route; rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_dispatch; python=kogwistar/server/chat_api.py#create_chat_router.cancel_run; rust_test=rust/crates/kogwistar-api/src/tests/conversation_mcp.rs#mcp_conversation_run_operations_use_conversation_scope_end_to_end; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_conversation_cancel_run_denies_unauthorized_effects
            "conversation.cancel_run" if !run_id.is_empty() && arguments.len() == 1 => {
                Some(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: format!("/api/runs/{run_id}/cancel"),
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
                    .unwrap_or(200);
                let encoded_run_id = percent_encoding::utf8_percent_encode(
                    run_id,
                    percent_encoding::NON_ALPHANUMERIC,
                );
                Some(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: format!(
                        "/api/workflow/runs/{encoded_run_id}/events/poll?after_seq={after_seq}&limit={limit}"
                    ),
                    body: Vec::new(),
                    principal: principal.clone(),
                })
            }
            "workflow.process_timeline" if !run_id.is_empty() => {
                let valid_arguments = !arguments
                    .keys()
                    .any(|argument| !matches!(argument.as_str(), "run_id" | "after_seq" | "limit"))
                    && arguments
                        .get("after_seq")
                        .map(Value::as_i64)
                        .unwrap_or(Some(0))
                        .is_some_and(|after_seq| after_seq >= 0)
                    && arguments
                        .get("limit")
                        .map(Value::as_i64)
                        .unwrap_or(Some(200))
                        .is_some_and(|limit| (1..=10_000).contains(&limit));
                valid_arguments.then(|| {
                    let after_seq = arguments
                        .get("after_seq")
                        .and_then(Value::as_i64)
                        .unwrap_or(0);
                    let limit = arguments
                        .get("limit")
                        .and_then(Value::as_i64)
                        .unwrap_or(200);
                    let encoded_run_id = percent_encoding::utf8_percent_encode(
                        run_id,
                        percent_encoding::NON_ALPHANUMERIC,
                    );
                    ApiEffectRequest {
                        contract_version: 1,
                        method: "GET".to_owned(),
                        path_and_query: format!(
                            "/api/workflow/runs/{encoded_run_id}/events/poll?after_seq={after_seq}&limit={limit}"
                        ),
                        body: Vec::new(),
                        principal: principal.clone(),
                    }
                })
            }
            "workflow.run_evidence"
                if !run_id.is_empty() && arguments.keys().all(|argument| argument == "run_id") =>
            {
                let encoded_run_id = percent_encoding::utf8_percent_encode(
                    run_id,
                    percent_encoding::NON_ALPHANUMERIC,
                );
                Some(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: format!("/api/workflow/runs/{encoded_run_id}/evidence"),
                    body: Vec::new(),
                    principal: principal.clone(),
                })
            }
            "workflow.run_lineage" if !run_id.is_empty() => {
                let valid_arguments = !arguments
                    .keys()
                    .any(|argument| !matches!(argument.as_str(), "run_id" | "limit"))
                    && arguments
                        .get("limit")
                        .map(Value::as_i64)
                        .unwrap_or(Some(200))
                        .is_some_and(|limit| (1..=1000).contains(&limit));
                valid_arguments.then(|| {
                    let limit = arguments
                        .get("limit")
                        .and_then(Value::as_i64)
                        .unwrap_or(200);
                    let encoded_run_id = percent_encoding::utf8_percent_encode(
                        run_id,
                        percent_encoding::NON_ALPHANUMERIC,
                    );
                    ApiEffectRequest {
                        contract_version: 1,
                        method: "GET".to_owned(),
                        path_and_query: format!(
                            "/api/workflow/runs/{encoded_run_id}/lineage?limit={limit}"
                        ),
                        body: Vec::new(),
                        principal: principal.clone(),
                    }
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
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_run_replay; fixture=contracts/golden/workflow-run-replay-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/mcp_workflow_contracts.rs#mcp_workflow_run_replay_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_run_replay_matches_shared_contract
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
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_run_submit; fixture=contracts/golden/workflow-run-submit-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/mcp_workflow_contracts.rs#mcp_workflow_run_submit_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_run_submit_matches_shared_contract
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
                path_and_query: format!(
                    "/api/workflow/runs/{run_id}/resume?wait_for_completion=true"
                ),
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
            // PYTHON_PARITY_V1: require workflow role/scope plus project_view and read_security_scope before ledger access; python=kogwistar/server/chat_mcp.py#workflow_budget_snapshot+kogwistar/server/chat_service.py#ChatRunService.budget_snapshot; rust_test=rust/crates/kogwistar-api/src/tests/workflow_mutation_authorization.rs#workflow_budget_reads_authorization_denies_before_backend_access; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_budget_snapshot_denies_before_ledger_read; contract=contracts/golden/workflow-budget-http-v1.json
            "workflow.budget_snapshot" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/budget".to_owned(),
                body: Vec::new(),
                principal: principal.clone(),
            }),
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_dispatch; python=kogwistar/server/chat_mcp.py#workflow_budget_history; fixture=contracts/golden/workflow-budget-history-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_service_mcp.rs#mcp_workflow_budget_history_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_budget_history_matches_shared_contract
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
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_dispatch; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_tool_audit; fixture=contracts/golden/workflow-tool-audit-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_service_mcp.rs#mcp_workflow_tool_audit_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_tool_audit_matches_shared_contract
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
            "workflow.process_table" => {
                let valid_names = arguments
                    .keys()
                    .all(|name| matches!(name.as_str(), "status" | "workflow_id" | "conversation_id" | "limit"));
                let optional_text = |name: &str| match arguments.get(name) {
                    None | Some(Value::Null) => Some(None),
                    Some(Value::String(value)) => Some((!value.is_empty()).then_some(value.as_str())),
                    _ => None,
                };
                let status = optional_text("status");
                let workflow_id = optional_text("workflow_id");
                let conversation_id = optional_text("conversation_id");
                let limit = match arguments.get("limit") {
                    None => Some(100),
                    Some(Value::Number(value)) => value.as_i64(),
                    _ => None,
                };
                let (Some(status), Some(workflow_id), Some(conversation_id), Some(limit)) =
                    (status, workflow_id, conversation_id, limit)
                else {
                    return Json(json!({
                        "jsonrpc":"2.0",
                        "id":id,
                        "error":{"code":-32602,"message":"Invalid process-table arguments"}
                    }))
                    .into_response();
                };
                if !valid_names || !(1..=10_000).contains(&limit) {
                    return Json(json!({
                        "jsonrpc":"2.0",
                        "id":id,
                        "error":{"code":-32602,"message":"Invalid process-table arguments"}
                    }))
                    .into_response();
                }
                let mut query = url::form_urlencoded::Serializer::new(String::new());
                query.append_pair("limit", &limit.to_string());
                for (name, value) in [
                    ("status", status),
                    ("workflow_id", workflow_id),
                    ("conversation_id", conversation_id),
                ] {
                    if let Some(value) = value {
                        query.append_pair(name, value);
                    }
                }
                Some(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: format!("/api/workflow/process-table?{}", query.finish()),
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
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_scheduler_timeline; fixture=contracts/golden/workflow-scheduler-timeline-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/mcp_workflow_contracts.rs#mcp_workflow_scheduler_timeline_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_scheduler_timeline_matches_shared_contract
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
            "workflow.blocked_runs" => {
                let limit = match arguments.get("limit") {
                    None => Some(100),
                    Some(Value::Number(value)) => value.as_i64(),
                    _ => None,
                };
                let Some(limit) = limit.filter(|limit| (1..=10_000).contains(limit)) else {
                    return Json(json!({
                        "jsonrpc":"2.0",
                        "id":id,
                        "error":{"code":-32602,"message":"Invalid blocked-runs arguments"}
                    }))
                    .into_response();
                };
                if arguments.keys().any(|name| name != "limit") {
                    return Json(json!({
                        "jsonrpc":"2.0",
                        "id":id,
                        "error":{"code":-32602,"message":"Invalid blocked-runs arguments"}
                    }))
                    .into_response();
                }
                Some(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: format!("/api/workflow/blocked-runs?limit={limit}"),
                    body: Vec::new(),
                    principal: principal.clone(),
                })
            }
            "workflow.operator_inbox" => {
                let valid_names = arguments
                    .keys()
                    .all(|name| matches!(name.as_str(), "inbox_id" | "status" | "limit"));
                let optional_text = |name: &str| match arguments.get(name) {
                    None | Some(Value::Null) => Some(None),
                    Some(Value::String(value)) => Some((!value.is_empty()).then_some(value.as_str())),
                    _ => None,
                };
                let inbox_id = optional_text("inbox_id");
                let status = optional_text("status");
                let limit = match arguments.get("limit") {
                    None => Some(200),
                    Some(Value::Number(value)) => value.as_i64(),
                    _ => None,
                };
                let (Some(inbox_id), Some(status), Some(limit)) = (inbox_id, status, limit) else {
                    return Json(json!({
                        "jsonrpc":"2.0",
                        "id":id,
                        "error":{"code":-32602,"message":"Invalid operator-inbox arguments"}
                    }))
                    .into_response();
                };
                if !valid_names || !(1..=10_000).contains(&limit) {
                    return Json(json!({
                        "jsonrpc":"2.0",
                        "id":id,
                        "error":{"code":-32602,"message":"Invalid operator-inbox arguments"}
                    }))
                    .into_response();
                }
                let mut query = url::form_urlencoded::Serializer::new(String::new());
                query.append_pair("limit", &limit.to_string());
                for (name, value) in [("inbox_id", inbox_id), ("status", status)] {
                    if let Some(value) = value {
                        query.append_pair(name, value);
                    }
                }
                Some(ApiEffectRequest {
                    contract_version: 1,
                    method: "GET".to_owned(),
                    path_and_query: format!("/api/workflow/operator/inbox?{}", query.finish()),
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
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_list; fixture=contracts/golden/workflow-service-list-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/mcp_workflow_contracts.rs#mcp_workflow_service_list_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_service_list_matches_shared_contract
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
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_get; fixture=contracts/golden/workflow-service-get-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/mcp_workflow_contracts.rs#mcp_workflow_service_get_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_service_get_matches_shared_contract
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
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_events; fixture=contracts/golden/workflow-service-events-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/mcp_workflow_contracts.rs#mcp_workflow_service_events_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_service_events_matches_shared_contract
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
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_repair; fixture=contracts/golden/workflow-service-repair-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/mcp_workflow_contracts.rs#mcp_workflow_service_repair_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_service_repair_matches_shared_contract
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
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_enable; fixture=contracts/golden/workflow-service-enable-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/mcp_workflow_contracts.rs#mcp_workflow_service_enable_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_service_enable_matches_shared_contract
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_disable; fixture=contracts/golden/workflow-service-disable-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/mcp_workflow_contracts.rs#mcp_workflow_service_disable_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_service_disable_matches_shared_contract
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
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_heartbeat; fixture=contracts/golden/workflow-service-heartbeat-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/mcp_workflow_contracts.rs#mcp_workflow_service_heartbeat_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_service_heartbeat_matches_shared_contract
            "workflow.service_heartbeat" => arguments
                .get("service_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(|service_id| ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: format!("/api/workflow/services/{service_id}/heartbeat"),
                    body: serde_json::to_vec(&json!({
                        "instance_id": arguments.get("instance_id").cloned(),
                        "payload": arguments
                            .get("payload")
                            .filter(|payload| !payload.is_null())
                            .cloned()
                            .unwrap_or_else(|| json!({})),
                    }))
                    .expect("heartbeat request serializes"),
                    principal: principal.clone(),
                }),
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_trigger; fixture=contracts/golden/workflow-service-trigger-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_service_mcp.rs#mcp_workflow_service_trigger_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_service_trigger_matches_shared_contract
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
                            .filter(|value| !value.is_null())
                            .cloned()
                            .unwrap_or_else(|| json!({})),
                    }))
                    .expect("MCP arguments serialize"),
                    principal: principal.clone(),
                }),
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_declare; fixture=contracts/golden/workflow-service-declare-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/mcp_workflow_contracts.rs#mcp_workflow_service_declare_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_service_declare_matches_shared_contract
            "workflow.service_declare" => Some(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/workflow/services".to_owned(),
                body: serde_json::to_vec(&arguments).expect("MCP arguments serialize"),
                principal: principal.clone(),
            }),
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_services_repair; fixture=contracts/golden/workflow-services-repair-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/workflow_service_mcp.rs#mcp_workflow_services_repair_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_services_repair_matches_shared_contract
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
            // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_effect_request; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_visibility_snapshot; fixture=contracts/golden/workflow-visibility-snapshot-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/mcp_workflow_contracts.rs#mcp_workflow_visibility_snapshot_matches_shared_contract; python_test=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_visibility_snapshot_matches_shared_contract
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
        let mut result = serde_json::from_slice::<Value>(&response.body).unwrap_or(Value::Null);
        if tool_name == "kg_load_persisted"
            && arguments
                .get("inp")
                .and_then(|input| input.get("sid"))
                .and_then(Value::as_bool)
                != Some(false)
        {
            let Some(nodes) = result.get("node_ids").and_then(Value::as_array) else {
                return Json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32000, "message": "Invalid persisted graph response from Rust application service"},
                }))
                .into_response();
            };
            let Some(edges) = result.get("edge_ids").and_then(Value::as_array) else {
                return Json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32000, "message": "Invalid persisted graph response from Rust application service"},
                }))
                .into_response();
            };
            let map_key = crate::short_id_store::short_id_map_key_for_bearer(bearer_token(&headers));
            let mapped_nodes = match map_graph_short_ids(nodes, &map_key) {
                Ok(ids) => ids,
                Err(()) => return Json(json!({"jsonrpc":"2.0","id":id,"error":{"code":-32000,"message":"Could not persist graph ID mappings"}})).into_response(),
            };
            let mapped_edges = match map_graph_short_ids(edges, &map_key) {
                Ok(ids) => ids,
                Err(()) => return Json(json!({"jsonrpc":"2.0","id":id,"error":{"code":-32000,"message":"Could not persist graph ID mappings"}})).into_response(),
            };
            result = json!({"node_ids": mapped_nodes, "edge_ids": mapped_edges});
        }
        if tool_name == "conversation.create" {
            result = result.get("result").cloned().unwrap_or(result);
        } else if tool_name == "conversation.ask" {
            let Some(admission) = result.get("result") else {
                return Json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32000, "message": "Invalid conversation ask response from Rust application service"},
                }))
                .into_response();
            };
            let requested_conversation_id = arguments
                .get("conversation_id")
                .and_then(Value::as_str);
            let requested_workflow_id = arguments
                .get("workflow_id")
                .and_then(Value::as_str)
                .unwrap_or("agentic_answering.v2");
            let run_id = admission.get("run_id").and_then(Value::as_str);
            let turn_node_id = admission.get("turn_node_id").and_then(Value::as_str);
            let accepted = admission.get("status").and_then(Value::as_str) == Some("queued")
                && admission
                    .get("conversation_id")
                    .and_then(Value::as_str)
                    .is_none_or(|value| Some(value) == requested_conversation_id)
                && admission
                    .get("workflow_id")
                    .and_then(Value::as_str)
                    .is_none_or(|value| value == requested_workflow_id);
            let (Some(conversation_id), Some(run_id), Some(turn_node_id)) =
                (requested_conversation_id, run_id, turn_node_id)
            else {
                return Json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32000, "message": "Invalid conversation ask response from Rust application service"},
                }))
                .into_response();
            };
            if !accepted {
                return Json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32000, "message": "Conversation ask admission does not match the request"},
                }))
                .into_response();
            }
            result = json!({
                "run_id": run_id,
                "conversation_id": conversation_id,
                "workflow_id": requested_workflow_id,
                "status": "queued",
                "user_turn_node_id": turn_node_id,
            });
        } else if tool_name == "conversation.get_transcript" {
            let conversation_id = arguments.get("conversation_id").and_then(Value::as_str);
            let transcript_id = result.get("conversation_id").and_then(Value::as_str);
            let turns = result.get("turns").and_then(Value::as_array);
            let (Some(conversation_id), Some(transcript_id), Some(turns)) =
                (conversation_id, transcript_id, turns)
            else {
                return Json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32000, "message": "Invalid transcript response from Rust application service"},
                }))
                .into_response();
            };
            if conversation_id != transcript_id {
                return Json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32000, "message": "Transcript response conversation does not match request"},
                }))
                .into_response();
            }
            let mut turns = turns.clone();
            for turn in &mut turns {
                if let Some(turn) = turn.as_object_mut() {
                    turn.remove("schema_version");
                }
            }
            result = json!({
                "conversation_id": conversation_id,
                "turns": turns,
            });
        // PYTHON_PARITY_V1: preserve filename and return a caller-scoped reversible short ID; rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#mcp_dispatch; python=kogwistar/server/mcp_tools.py#document_id_from_file_name; fixture=contracts/golden/document-id-from-file-name-mcp-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/document_validation.rs#sqlite_document_id_mcp_returns_python_compatible_short_id_mapping; python_test=tests/mcp/test_chat_mcp_tools.py#test_document_id_from_file_name_mcp_matches_shared_contract
        } else if tool_name == "document_id_from_file_name" {
            let Some(ids) = result.get("ids").and_then(Value::as_array) else {
                return Json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32000, "message": "Invalid document lookup response from Rust application service"},
                }))
                .into_response();
            };
            let map_key = crate::short_id_store::short_id_map_key_for_bearer(
                bearer_token(&headers),
            );
            let mut id_mapping = Vec::with_capacity(ids.len());
            for value in ids {
                let Some(file_name) = value.as_str() else {
                    return Json(json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {"code": -32000, "message": "Invalid document ID in Rust application response"},
                    }))
                    .into_response();
                };
                let short_id = match crate::short_id_store::short_id_for(
                    ".shortids",
                    &map_key,
                    file_name,
                ) {
                    Ok(id) => id,
                    Err(_) => {
                        return Json(json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {"code": -32000, "message": "Could not persist document ID mapping"},
                        }))
                        .into_response();
                    }
                };
                id_mapping.push(json!({"file_name": file_name, "id": short_id}));
            }
            result = json!({"id_mapping": id_mapping});
        }
        let result = if standard_mcp {
            mcp_tool_success_result(result)
        } else {
            result
        };
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
    let d3_bundle = method == Method::GET && uri.path() == "/viz/d3.bundle";
    let bundle_path_and_query = uri.to_string();
    let (method, path_and_query, body) = public_visualization_effect(&method, &uri, body);
    let mut path_and_query = path_and_query;
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
    let mut response = state
        .application
        .execute(ApiEffectRequest {
            contract_version: 1,
            method,
            path_and_query,
            body: body.to_vec(),
            principal: request_principal(&headers, &state.api.auth),
        })
        .await;
    if d3_bundle && (200..300).contains(&response.status) {
        response = match serde_json::from_slice::<Value>(&response.body) {
            Ok(payload) => d3_bundle_html_effect(&bundle_path_and_query, payload),
            Err(_) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_INVALID_RESPONSE",
                "Graph visualization returned an invalid D3 payload",
            ),
        };
    }
    let status = StatusCode::from_u16(response.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (
        status,
        [("content-type", response.content_type)],
        Body::from(response.body),
    )
        .into_response()
}

fn public_visualization_effect(
    method: &Method,
    uri: &axum::http::Uri,
    body: Bytes,
) -> (String, String, Bytes) {
    if method != Method::GET {
        return (method.to_string(), uri.to_string(), body);
    }
    let (path, query) = uri.path_and_query().map_or((uri.path(), ""), |value| {
        value
            .as_str()
            .split_once('?')
            .unwrap_or((value.as_str(), ""))
    });
    let internal_path = match path {
        "/api/viz/d3.json" => INTERNAL_GRAPH_VIZ_D3_ROUTE,
        "/viz/d3.bundle" => INTERNAL_GRAPH_VIZ_D3_ROUTE,
        "/api/viz/cytoscape.json" => INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE,
        _ => return (method.to_string(), uri.to_string(), body),
    };
    let mut input = serde_json::Map::new();
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        if matches!(
            key.as_ref(),
            "doc_id" | "mode" | "insertion_method" | "graph_type"
        ) {
            input.insert(key.into_owned(), json!(value.into_owned()));
        }
    }
    (
        "POST".to_owned(),
        internal_path.to_owned(),
        Bytes::from(
            serde_json::to_vec(&Value::Object(input)).expect("visualization query serializes"),
        ),
    )
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

#[cfg(test)]
mod visualization_route_tests {
    use super::*;

    #[test]
    fn public_visualization_get_preserves_python_query_options() {
        let uri = "/api/viz/d3.json?doc_id=doc%3A%2F%2Fone&mode=classic&insertion_method=parser&graph_type=wisdom"
            .parse()
            .expect("valid visualization URI");
        let (method, path, body) = public_visualization_effect(&Method::GET, &uri, Bytes::new());
        assert_eq!(method, "POST");
        assert_eq!(path, INTERNAL_GRAPH_VIZ_D3_ROUTE);
        assert_eq!(
            serde_json::from_slice::<Value>(&body).expect("query becomes JSON"),
            json!({
                "doc_id":"doc://one",
                "mode":"classic",
                "insertion_method":"parser",
                "graph_type":"wisdom"
            })
        );
    }

    #[test]
    fn public_d3_bundle_reuses_authorized_d3_projection_with_query_options() {
        let uri = "/viz/d3.bundle?doc_id=doc%3A%2F%2Fone&mode=classic&insertion_method=parser&graph_type=wisdom"
            .parse()
            .expect("valid D3 bundle URI");
        let (method, path, body) = public_visualization_effect(&Method::GET, &uri, Bytes::new());
        assert_eq!(method, "POST");
        assert_eq!(path, INTERNAL_GRAPH_VIZ_D3_ROUTE);
        assert_eq!(
            serde_json::from_slice::<Value>(&body).expect("query becomes JSON"),
            json!({
                "doc_id":"doc://one",
                "mode":"classic",
                "insertion_method":"parser",
                "graph_type":"wisdom"
            })
        );
    }

    #[test]
    fn unrelated_get_routes_are_not_rewritten() {
        let uri = "/api/search_index_hybrid?q=graph"
            .parse()
            .expect("valid URI");
        let (method, path, body) = public_visualization_effect(&Method::GET, &uri, Bytes::new());
        assert_eq!(method, "GET");
        assert_eq!(path, uri.to_string());
        assert!(body.is_empty());
    }

    #[test]
    fn public_visualization_get_preserves_empty_doc_id_for_python_parity() {
        let uri = "/api/viz/d3.json?doc_id="
            .parse()
            .expect("valid URI with an explicitly empty query value");
        let (method, path, body) = public_visualization_effect(&Method::GET, &uri, Bytes::new());
        assert_eq!(method, "POST");
        assert_eq!(path, INTERNAL_GRAPH_VIZ_D3_ROUTE);
        assert_eq!(
            serde_json::from_slice::<Value>(&body).expect("query becomes JSON"),
            json!({"doc_id": ""})
        );
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
    router_with_application_and_mcp_sessions(
        state,
        application,
        Arc::new(McpSessionRegistry::default()),
    )
}

pub fn router_with_application_and_mcp_sessions(
    state: ApiState,
    application: Arc<dyn ApplicationService>,
    mcp_sessions: Arc<McpSessionRegistry>,
) -> Router {
    let mut router = Router::new()
        .route("/api/events", get(sse_handler))
        .route(
            "/mcp",
            get(mcp_get_handler)
                .post(mcp_handler)
                .delete(mcp_delete_handler),
        )
        .route(
            "/mcp/conversation",
            get(mcp_get_handler)
                .post(mcp_handler)
                .delete(mcp_delete_handler),
        )
        .route(
            "/mcp/workflow",
            get(mcp_get_handler)
                .post(mcp_handler)
                .delete(mcp_delete_handler),
        );
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
            mcp_sessions,
        }))
}

pub async fn serve(
    listener: tokio::net::TcpListener,
    state: ApiState,
) -> Result<(), std::io::Error> {
    axum::serve(listener, router(state)).await
}
