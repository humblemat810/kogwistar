fn json_effect(status: StatusCode, value: Value) -> ApiEffectResponse {
    ApiEffectResponse {
        status: status.as_u16(),
        content_type: "application/json".to_owned(),
        body: serde_json::to_vec(&value).expect("JSON value always serializes"),
    }
}

fn public_context_snapshot_value(snapshot: ContextSnapshotV1) -> Value {
    let mut payload = serde_json::to_value(snapshot).expect("snapshot serializes");
    payload
        .as_object_mut()
        .expect("typed snapshot serializes as an object")
        .remove("schema_version");
    payload
}

#[cfg(test)]
fn strip_snapshot_implementation_fields(response: &mut ApiEffectResponse) {
    if !(200..300).contains(&response.status) {
        return;
    }
    let Ok(mut payload) = serde_json::from_slice::<Value>(&response.body) else {
        return;
    };
    let Some(object) = payload.as_object_mut() else {
        return;
    };
    object.remove("schema_version");
    object.remove("implementation_authority");
    object.remove("seam_attestation_version");
    object.remove("seam_crossings");
    response.body = serde_json::to_vec(&payload).expect("snapshot response serializes");
}

#[cfg(test)]
fn legacy_receive_message_payload(response: &mut ApiEffectResponse) {
    if !(200..300).contains(&response.status) {
        return;
    }
    let Ok(mut payload) = serde_json::from_slice::<Value>(&response.body) else {
        *response = effect_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "KOGWISTAR_INVALID_RESPONSE",
            "Transcript service returned an invalid response",
        );
        return;
    };
    if let Some(object) = payload.as_object_mut() {
        object.remove("implementation_authority");
        object.remove("seam_attestation_version");
        object.remove("seam_crossings");
    }
    let Ok(transcript) = serde_json::from_value::<ConversationTranscriptV1>(payload) else {
        *response = effect_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "KOGWISTAR_INVALID_RESPONSE",
            "Transcript service returned an invalid response",
        );
        return;
    };
    response.body = serde_json::to_vec(&legacy_receive_message_value(transcript))
        .expect("legacy transcript response serializes");
}

fn legacy_receive_message_value(transcript: ConversationTranscriptV1) -> Value {
    let messages = transcript
        .turns
        .into_iter()
        .map(|turn| {
            json!({
                "node_id": turn.node_id,
                "turn_index": turn.turn_index,
                "role": turn.role,
                "content": turn.content,
                "entity_type": turn.entity_type,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "conversation_id": transcript.conversation_id,
        "messages": messages,
    })
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

fn tool_audit_query(path_and_query: &str) -> Result<(Option<String>, usize), String> {
    let query = path_and_query
        .split_once('?')
        .map(|(_, query)| query)
        .unwrap_or_default();
    let mut conversation_id = None;
    let mut limit = 200usize;
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        match key.as_ref() {
            "conversation_id" => conversation_id = Some(value.into_owned()),
            "limit" => {
                limit = value
                    .parse::<usize>()
                    .map_err(|_| "limit must be a non-negative integer".to_owned())?;
            }
            _ => {}
        }
    }
    if conversation_id.as_deref() == Some("") {
        return Err("conversation_id must not be empty".to_owned());
    }
    if limit > 10_000 {
        return Err("limit must be <= 10000".to_owned());
    }
    Ok((conversation_id, limit))
}

fn latest_snapshot_conversation_id(path_and_query: &str) -> Option<String> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    let encoded_id = path
        .strip_prefix("/api/conversations/")?
        .strip_suffix("/snapshots/latest")?;
    let conversation_id = percent_encoding::percent_decode_str(encoded_id)
        .decode_utf8()
        .ok()?
        .into_owned();
    (!conversation_id.is_empty() && !conversation_id.contains('/')).then_some(conversation_id)
}

fn conversation_detail_id(path_and_query: &str) -> Option<String> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    let encoded_id = path.strip_prefix("/api/conversations/")?;
    if encoded_id.contains('/') {
        return None;
    }
    let conversation_id = percent_encoding::percent_decode_str(encoded_id)
        .decode_utf8()
        .ok()?
        .into_owned();
    (!conversation_id.is_empty() && !conversation_id.contains('/')).then_some(conversation_id)
}

fn is_conversation_list_route(path_and_query: &str) -> bool {
    path_and_query.split('?').next() == Some("/api/conversations")
}

fn conversation_transcript_id(path_and_query: &str) -> Option<String> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    let encoded_id = path
        .strip_prefix("/api/conversations/")?
        .strip_suffix("/turns")?;
    let conversation_id = percent_encoding::percent_decode_str(encoded_id)
        .decode_utf8()
        .ok()?
        .into_owned();
    (!conversation_id.is_empty() && !conversation_id.contains('/')).then_some(conversation_id)
}

fn context_snapshot_query(path_and_query: &str) -> (Option<String>, Option<String>) {
    let mut run_id = None;
    let mut stage = None;
    for (key, value) in url::form_urlencoded::parse(
        path_and_query
            .split_once('?')
            .map(|(_, query)| query)
            .unwrap_or_default()
            .as_bytes(),
    ) {
        match key.as_ref() {
            "run_id" => run_id = Some(value.into_owned()),
            "stage" => stage = Some(value.into_owned()),
            _ => {}
        }
    }
    (run_id, stage)
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

const SQLITE_UNSUPPORTED_HTTP_ROUTES: &[(&str, &str)] = &[
    ("GET", "/api/conversations"),
    ("GET", "/api/workflow/tools/audit"),
    ("GET", "/api/conversations/{conversation_id}"),
    ("GET", "/api/conversations/{conversation_id}/turns"),
    (
        "GET",
        "/api/conversations/{conversation_id}/snapshots/latest",
    ),
];

fn principal_strings(principal: &Value, name: &str) -> Vec<String> {
    match principal.get(name) {
        Some(Value::String(raw)) => raw
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_lowercase)
            .collect(),
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_lowercase)
            .collect(),
        _ => Vec::new(),
    }
}

fn principal_group_set(principal: &Value) -> BTreeSet<String> {
    ["groups", "principal_groups"]
        .into_iter()
        .flat_map(|name| principal_strings(principal, name))
        .collect()
}

fn graph_acl_principal_from_claims(
    principal: &Value,
    security_scope: Option<&str>,
) -> GraphAclPrincipal {
    GraphAclPrincipal {
        principal_id: graph_acl_principal_id(principal),
        groups: principal_group_set(principal),
        security_scope: security_scope.map(str::to_owned),
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

fn authenticated_run_owner_id(principal: &Value) -> Option<&str> {
    ["user_id", "sub", "agent_id"]
        .into_iter()
        .find_map(|name| principal.get(name).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("anonymous"))
}

fn graph_acl_principal_id(principal: &Value) -> String {
    ["agent_id", "actor_id", "sub", "user_id"]
        .into_iter()
        .find_map(|name| principal.get(name).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("system")
        .to_lowercase()
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

/// Build the authority snapshot carried by a delegated workflow run.
///
/// The child may request less authority, but it can never widen the parent's
/// effective set. The result is stored in reserved runtime state rather than
/// trusting a child-supplied capability field during execution.
fn prepare_delegated_run_args(
    args: &Value,
    principal: &Value,
    shared: &SharedCapabilityState,
) -> Result<Value, String> {
    let mut args = args
        .as_object()
        .cloned()
        .ok_or_else(|| "spawn_process args must be a JSON object".to_owned())?;
    let mut initial_state = args
        .remove("initial_state")
        .unwrap_or_else(|| Value::Object(Map::new()))
        .as_object()
        .cloned()
        .ok_or_else(|| "spawn_process initial_state must be a JSON object".to_owned())?;
    let requested = match initial_state.get("_rt_requested_capabilities") {
        None => None,
        Some(Value::Array(values)) => Some(
            values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(str::trim)
                        .filter(|capability| !capability.is_empty())
                        .map(str::to_ascii_lowercase)
                        .ok_or_else(|| {
                            "_rt_requested_capabilities must contain non-empty strings".to_owned()
                        })
                })
                .collect::<Result<BTreeSet<_>, _>>()?,
        ),
        Some(_) => {
            return Err("_rt_requested_capabilities must be an array of strings".to_owned());
        }
    };
    let parent = shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let parent_effective = effective_capabilities(principal, &parent);
    let child_effective = attenuate_capabilities(&parent_effective, requested.as_ref())
        .map_err(|error| error.to_string())?;
    let authority = DelegationAuthorityV1::new(principal_subject(principal), &child_effective)
        .map_err(|error| error.to_string())?;
    initial_state.insert(
        "_rt_effective_capabilities".to_owned(),
        json!(child_effective),
    );
    initial_state.insert(
        "_rt_delegated_by".to_owned(),
        Value::String(principal_subject(principal)),
    );
    initial_state.insert(
        "_rt_delegation".to_owned(),
        serde_json::to_value(authority).expect("delegation authority serializes"),
    );
    args.insert("initial_state".to_owned(), Value::Object(initial_state));
    Ok(Value::Object(args))
}

fn delegated_child_principal(parent: &Value, args: &Value) -> Value {
    let mut child = parent.as_object().cloned().unwrap_or_default();
    if let Some(capabilities) = args
        .get("initial_state")
        .and_then(|state| state.get("_rt_effective_capabilities"))
    {
        child.insert("capabilities".to_owned(), capabilities.clone());
    }
    child.insert(
        "delegated_by".to_owned(),
        Value::String(principal_subject(parent)),
    );
    Value::Object(child)
}

fn inject_authoritative_capability_state(
    initial_state: &mut Map<String, Value>,
    principal: &Value,
    shared: &SharedCapabilityState,
) {
    for key in [
        "effective_capabilities",
        "_rt_effective_capabilities",
        "_rt_requested_capabilities",
        "_rt_delegated_by",
        "_rt_delegation",
    ] {
        initial_state.remove(key);
    }
    let state = shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let capabilities = effective_capabilities(principal, &state);
    initial_state.insert("effective_capabilities".to_owned(), json!(capabilities));
    initial_state.insert("_rt_effective_capabilities".to_owned(), json!(capabilities));
    if let Some(delegated_by) = principal.get("delegated_by").and_then(Value::as_str)
        && let Ok(authority) = DelegationAuthorityV1::new(delegated_by, &capabilities)
    {
        initial_state.insert(
            "_rt_delegated_by".to_owned(),
            Value::String(delegated_by.to_owned()),
        );
        initial_state.insert(
            "_rt_delegation".to_owned(),
            serde_json::to_value(authority).expect("delegation authority serializes"),
        );
    }
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
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_lowercase)
        .unwrap_or_else(|| default_namespace.clone());
    let execution_namespace = principal["execution_ns"]
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_lowercase)
        .unwrap_or_else(|| default_namespace.clone());
    let normalized_component = |name: &str| {
        principal[name]
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_lowercase)
            .unwrap_or_default()
    };
    let tenant = normalized_component("tenant");
    let workspace = normalized_component("workspace");
    let project = normalized_component("project");
    let scope_path = [tenant.as_str(), workspace.as_str(), project.as_str()]
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join("/");
    let security_scope_path = principal["security_scope"]
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != "*")
        .map(str::to_lowercase)
        .unwrap_or_else(|| scope_path.clone());
    let selected_security_scope = ["security_scope", "tenant", "scope"]
        .into_iter()
        .find_map(|name| {
            principal[name]
                .as_str()
                .filter(|value| !value.is_empty())
                .map(str::trim)
                .map(str::to_lowercase)
        })
        .filter(|value| !value.is_empty() && value != "*");
    let security_scope = selected_security_scope
        .or_else(|| (!security_scope_path.is_empty()).then(|| security_scope_path.clone()))
        .unwrap_or_else(|| default_namespace.clone());
    json!({
        "storage_namespace": storage_namespace,
        "execution_namespace": execution_namespace,
        "security_scope": security_scope,
        "security_scope_path": security_scope_path,
        "tenant": tenant,
        "workspace": workspace,
        "project": project,
        "graph_space": principal["graph_space"]
            .as_str()
            .unwrap_or_default()
            .to_ascii_lowercase(),
    })
}

fn conversation_graph_scope_and_acl_principal(
    request_principal: &Value,
) -> (GraphScope, GraphAclPrincipal) {
    let scope_value = security_scope_value(request_principal);
    let scope = GraphScope {
        namespace: scope_value["storage_namespace"]
            .as_str()
            .unwrap_or("default")
            .to_owned(),
        workspace_id: request_principal
            .get("workspace_id")
            .or_else(|| request_principal.get("workspace"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        graph_space: request_principal
            .get("graph_space")
            .and_then(Value::as_str)
            .map(str::to_owned),
    };
    let acl_principal = graph_acl_principal_from_claims(
        request_principal,
        scope_value["security_scope"].as_str(),
    );
    (scope, acl_principal)
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
    syscall_allowed_in_namespace(request, shared, op, write, required, "workflow")
}

fn syscall_allowed_in_namespace(
    request: &ApiEffectRequest,
    shared: &SharedCapabilityState,
    op: &str,
    write: bool,
    required: &[&str],
    expected_namespace: &str,
) -> bool {
    let role = principal_role(&request.principal);
    let role_allowed = !write || role == "rw";
    let namespace_allowed = principal_strings(&request.principal, "ns")
        .into_iter()
        .any(|namespace| namespace == expected_namespace || namespace == "*");
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

/// Narrow internal graph-authority contract for callers that already own
/// deterministic embeddings.  This is deliberately separate from the public
/// document upsert contract: indexing and document parsing remain outside this
/// Rust cutover until their behavior is parity-tested.
fn graph_mutation_into_store(input: GraphMutationInput) -> GraphMutation {
    GraphMutation {
        scope: GraphScope {
            namespace: input.scope.namespace,
            workspace_id: input.scope.workspace_id,
            graph_space: input.scope.graph_space,
        },
        table: input.table,
        entity_kind: input.entity_kind,
        event_id: input.event_id,
        op: input.op,
        payload: input.payload,
        record: GraphRecord {
            id: input.record.id,
            document: input.record.document,
            metadata: input.record.metadata,
            embedding: input.record.embedding,
        },
        embedding_dim: input.embedding_dim,
    }
}

fn graph_mutation_batch_value(
    input: GraphMutationBatchInput,
) -> Result<Vec<GraphMutation>, String> {
    input.validate().map_err(|error| error.to_string())?;
    if input.contract_version != 1 {
        return Err(format!(
            "unsupported graph mutation contract version {}",
            input.contract_version
        ));
    }
    if !input.atomic {
        return Err("graph mutation batches must set atomic=true".to_owned());
    }
    if input.mutations.is_empty() {
        return Err("graph mutation batches must not be empty".to_owned());
    }
    let mutations = input
        .mutations
        .into_iter()
        .map(graph_mutation_into_store)
        .collect::<Vec<_>>();
    let first_scope = mutations[0].scope.clone();
    for mutation in &mutations {
        if mutation.scope.namespace.is_empty() {
            return Err("graph mutation namespace must not be empty".to_owned());
        }
        if mutation.scope != first_scope {
            return Err("all atomic graph mutations must share one graph scope".to_owned());
        }
        let expected_table = match mutation.entity_kind.as_str() {
            "node" => "gke_nodes",
            "edge" => "gke_edges",
            other => return Err(format!("unsupported graph entity kind {other:?}")),
        };
        if mutation.table != expected_table {
            return Err(format!(
                "graph entity kind {:?} must use table {:?}",
                mutation.entity_kind, expected_table
            ));
        }
        if let Some(embedding) = &mutation.record.embedding
            && embedding.len() != mutation.embedding_dim
        {
            return Err(format!(
                "embedding dimension {} does not match vector length {}",
                mutation.embedding_dim,
                embedding.len()
            ));
        }
    }
    Ok(mutations)
}

fn validate_graph_mutation_envelopes(
    input: &GraphMutationBatchInput,
    mutations: &[GraphMutation],
) -> Result<(), String> {
    for (raw, mutation) in input.mutations.iter().zip(mutations) {
        if let Some(envelope) = &raw.model_envelope {
            validate_typed_graph_mutation(mutation, envelope)
                .map_err(|error| format!("typed graph mutation is invalid: {error}"))?;
        }
    }
    Ok(())
}

fn graph_index_jobs_value(
    input: &GraphMutationBatchInput,
    mutations: &[GraphMutation],
) -> Result<Vec<NewIndexJob>, String> {
    input.validate().map_err(|error| error.to_string())?;
    let Some(first) = mutations.first() else {
        return Err("graph mutation batches must not be empty".to_owned());
    };
    input
        .index_jobs
        .iter()
        .map(|job| {
            if job.job_id.is_empty()
                || job.namespace.is_empty()
                || job.entity_kind.is_empty()
                || job.entity_id.is_empty()
                || job.index_kind.is_empty()
            {
                return Err("index job identifiers must not be empty".to_owned());
            }
            if job.namespace != first.scope.namespace {
                return Err("index job namespace must match graph mutation namespace".to_owned());
            }
            if !matches!(job.op.as_str(), "UPSERT" | "DELETE") {
                return Err("index job op must be UPSERT or DELETE".to_owned());
            }
            if job.max_retries < 0 {
                return Err("index job max_retries must not be negative".to_owned());
            }
            Ok(NewIndexJob {
                job_id: job.job_id.clone(),
                namespace: job.namespace.clone(),
                entity_kind: job.entity_kind.clone(),
                entity_id: job.entity_id.clone(),
                index_kind: job.index_kind.clone(),
                op: job.op.clone(),
                payload_json: job.payload_json.clone(),
                max_retries: job.max_retries,
            })
        })
        .collect()
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
