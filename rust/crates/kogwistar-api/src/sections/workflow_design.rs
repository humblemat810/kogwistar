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

fn graph_mutation_scope_allowed(scope: &GraphScope, security_scope: &Value) -> bool {
    if scope.namespace
        != security_scope
            .get("storage_namespace")
            .and_then(Value::as_str)
            .unwrap_or_default()
    {
        return false;
    }
    if let Some(workspace) = security_scope.get("workspace").and_then(Value::as_str)
        && !workspace.is_empty()
        && scope.workspace_id.as_deref() != Some(workspace)
    {
        return false;
    }
    if let Some(graph_space) = security_scope.get("graph_space").and_then(Value::as_str)
        && !graph_space.is_empty()
        && scope.graph_space.as_deref() != Some(graph_space)
    {
        return false;
    }
    true
}
