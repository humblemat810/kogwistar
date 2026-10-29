const INTERNAL_DOCUMENT_PROJECTION_ROUTE: &str = "/internal/document/projection";

fn document_projection_input(
    request: &ApiEffectRequest,
) -> Result<DocumentUpsertV1, ApiEffectResponse> {
    let input: DocumentUpsertV1 = serde_json::from_slice(&request.body).map_err(|error| {
        effect_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "KOGWISTAR_INVALID_REQUEST",
            error.to_string(),
        )
    })?;
    input.validate().map_err(|error| {
        effect_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "KOGWISTAR_INVALID_REQUEST",
            error.to_string(),
        )
    })?;
    Ok(input)
}

fn document_projection_write(
    request: &ApiEffectRequest,
    capabilities: &SharedCapabilityState,
) -> Result<DocumentProjectionUpsert, ApiEffectResponse> {
    if !capability_allowed(request, capabilities, "write_graph", &["write_graph"]) {
        return Err(effect_error(
            StatusCode::FORBIDDEN,
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
            "Document projection requires write_graph capability",
        ));
    }
    let input = document_projection_input(request)?;
    if input.doc_type != "text" {
        return Err(effect_error(
            StatusCode::NOT_IMPLEMENTED,
            "KOGWISTAR_DOCUMENT_PROJECTION_UNSUPPORTED",
            "Rust projection handoff currently requires normalized text content",
        ));
    }
    let scope = security_scope_value(&request.principal);
    let mut metadata = input.metadata;
    metadata.insert("doc_type".to_owned(), Value::String(input.doc_type));
    metadata.insert(
        "insertion_method".to_owned(),
        Value::String(input.insertion_method),
    );
    Ok(DocumentProjectionUpsert {
        scope: GraphScope {
            namespace: scope["storage_namespace"]
                .as_str()
                .unwrap_or("docs")
                .to_owned(),
            workspace_id: scope["workspace"].as_str().map(str::to_owned),
            graph_space: scope["graph_space"].as_str().map(str::to_owned),
        },
        record: GraphRecord {
            id: input.doc_id,
            document: Some(input.content),
            metadata,
            embedding: None,
        },
        embedding_dim: 0,
    })
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

pub(crate) fn persisted_directory_bytes(paths: &[std::path::PathBuf]) -> u64 {
    fn directory_bytes(root: &std::path::Path) -> u64 {
        let mut pending = vec![root.to_path_buf()];
        let mut total = 0_u64;
        while let Some(directory) = pending.pop() {
            let Ok(entries) = std::fs::read_dir(directory) else {
                continue;
            };
            for entry in entries.flatten() {
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                if file_type.is_dir() {
                    pending.push(entry.path());
                } else if file_type.is_file() {
                    if let Ok(metadata) = entry.metadata() {
                        total = total.saturating_add(metadata.len());
                    }
                } else if file_type.is_symlink()
                    && entry
                        .metadata()
                        .is_ok_and(|metadata| metadata.is_file())
                {
                    if let Ok(metadata) = entry.metadata() {
                        total = total.saturating_add(metadata.len());
                    }
                }
            }
        }
        total
    }

    paths.iter().fold(0_u64, |total, path| {
        total.saturating_add(directory_bytes(path))
    })
}

pub(crate) fn configured_storage_usage_bytes() -> u64 {
    let configured = |name: &str, default: &str| {
        std::env::var(name).unwrap_or_else(|_| default.to_owned())
    };
    let directories = [
        configured("KOGWISTAR_PERSIST_DIRECTORY", ".kogwistar"),
        configured(
            "KOGWISTAR_CONVERSATION_PERSIST_DIRECTORY",
            ".kogwistar/conversation",
        ),
        configured("KOGWISTAR_WORKFLOW_PERSIST_DIRECTORY", ".kogwistar/workflow"),
    ]
    .into_iter()
    .map(std::path::PathBuf::from)
    .collect::<Vec<_>>();
    persisted_directory_bytes(&directories)
}

fn resource_value(
    runs: &[ServerRun],
    lanes: &[ProjectedLaneMessage],
    usage: &[Value],
    events: &[ServerRunEvent],
    storage_usage_bytes: u64,
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
        "storage_usage_bytes": storage_usage_bytes,
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

#[cfg(test)]
mod storage_usage_tests {
    use super::{persisted_directory_bytes, resource_value};

    #[test]
    fn persisted_directory_bytes_sums_files_and_ignores_missing_paths() {
        let root = std::env::temp_dir().join(format!(
            "kogwistar-storage-usage-{}",
            uuid::Uuid::new_v4()
        ));
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(root.join("one.bin"), [1_u8; 7]).unwrap();
        std::fs::write(nested.join("two.bin"), [2_u8; 11]).unwrap();

        let bytes = persisted_directory_bytes(&[
            root.clone(),
            root.join("missing"),
        ]);

        assert_eq!(bytes, 18);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resource_snapshot_keeps_storage_usage_as_an_integer() {
        let snapshot = resource_value(&[], &[], &[], &[], 18);
        assert_eq!(snapshot["storage_usage_bytes"], 18);
    }
}

fn resource_service_value(services: &[Value]) -> Value {
    let mut by_health = BTreeMap::<String, usize>::new();
    for service in services {
        let health = service
            .get("health_status")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .unwrap_or("unknown");
        *by_health.entry(health.to_owned()).or_default() += 1;
    }
    json!({"total_services": services.len(), "by_health": by_health})
}

fn operational_effect(
    request: &ApiEffectRequest,
    path: &str,
    runs: Vec<ServerRun>,
    lanes: Vec<ProjectedLaneMessage>,
    events: Vec<ServerRunEvent>,
    services: Vec<Value>,
) -> ApiEffectResponse {
    let limit = query_integer(&request.path_and_query, "limit", 200).clamp(0, 10_000) as usize;
    let usage = usage_history(&events, limit.max(1));
    let storage_usage_bytes = if matches!(
        path,
        "/api/workflow/resources" | "/api/workflow/operator/dashboard"
    ) {
        configured_storage_usage_bytes()
    } else {
        0
    };
    let mut resources = resource_value(
        &runs,
        &lanes,
        &usage,
        &events,
        storage_usage_bytes,
    );
    resources["services"] = resource_service_value(&services);
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
    values.sort_by_key(|left| (left.0, left.1));
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
