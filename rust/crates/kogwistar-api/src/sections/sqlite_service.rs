// SQLite service composition, workflow-design reads, projection reconciliation, and setup.
// Python parity: kogwistar/engine_core/engine_sqlite.py, engine_core/two_stage_async.py,
// and kogwistar/server/runtime_api.py (service setup and workflow runtime routes).
#[derive(Clone)]
pub struct SqliteRunApplicationService {
    store: SqliteStore,
    graph_role: String,
    chroma_graph: Option<kogwistar_store_chroma::ChromaGraphStore>,
    chroma_node_index: Option<kogwistar_store_chroma::ChromaVectorCollectionStore>,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    search_index_fts: Option<Arc<Mutex<rusqlite::Connection>>>,
    chat_completion_provider: Option<crate::native_answer_generation::OpenAiCompatibleChatConfig>,
    owner_auth_store: Option<AuthStore>,
    max_queue: usize,
    capabilities: SharedCapabilityState,
    native_operations: kogwistar_runtime::NativeOperationRegistry,
    transient_run_deltas: crate::transient_deltas::TransientRunDeltas,
}

#[derive(Clone)]
struct SqliteChromaGraphConfig {
    endpoint: String,
    metric: DistanceMetric,
    node_collection: String,
    edge_collection: String,
}

async fn open_sqlite_application_service(
    path: impl AsRef<std::path::Path>,
    chroma: Option<SqliteChromaGraphConfig>,
) -> Result<SqliteRunApplicationService, String> {
    let mut service = SqliteRunApplicationService::open(path)?;
    if let Some(chroma) = chroma {
        let common = kogwistar_store_chroma::ChromaStoreConfig {
            endpoint: chroma.endpoint,
            metric: chroma.metric,
            ..Default::default()
        };
        let node_config = kogwistar_store_chroma::ChromaStoreConfig {
            collection: chroma.node_collection,
            ..common.clone()
        };
        let edge_config = kogwistar_store_chroma::ChromaStoreConfig {
            collection: chroma.edge_collection,
            ..common.clone()
        };
        let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
            node_config,
            edge_config,
        )
        .await
        .map_err(|error| error.to_string())?;
        let node_index = kogwistar_store_chroma::ChromaVectorCollectionStore::connect(
            kogwistar_store_chroma::ChromaStoreConfig {
                collection: "nodes_index".to_owned(),
                ..common
            },
        )
        .await
        .map_err(|error| error.to_string())?;
        service = service.with_chroma_graph(graph).with_chroma_node_index(node_index);
    }
    Ok(service)
}

fn runtime_max_queue() -> usize {
    std::env::var("KOGWISTAR_RUNTIME_MAX_QUEUE")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(128)
}

fn sqlite_canonical_entity_revision(
    store: &SqliteStore,
    namespace: &str,
    entity_kind: &str,
    entity_id: &str,
) -> Result<Option<kogwistar_engine::canonical_revision::CanonicalEntityRevision>, StoreError> {
    let mut latest = store
        .latest_entity_events(namespace, entity_kind, &[entity_id.to_owned()])
        .map_err(|error| StoreError::Backend {
            backend: "sqlite-chroma-projection".to_owned(),
            message: error.to_string(),
        })?;
    let Some(raw) = latest.remove(entity_id) else {
        return Ok(None);
    };
    let payload = serde_json::from_str::<Value>(&raw.payload_json).map_err(|error| {
        StoreError::InvalidEntityEventPayload {
            message: error.to_string(),
        }
    })?;
    let event = EntityEvent {
        namespace: raw.namespace,
        seq: raw.seq,
        event_id: raw.event_id,
        entity_kind: raw.entity_kind,
        entity_id: raw.entity_id,
        op: raw.op,
        payload,
    };
    kogwistar_engine::canonical_revision::read_canonical_entity_revision(
        &[event],
        namespace,
        entity_kind,
        entity_id,
    )
}

impl SqliteRunApplicationService {
    fn stage2_record_is_current(
        record: &GraphRecord,
        latest: Option<&SqliteRawEntityEvent>,
    ) -> bool {
        match record.metadata.get("_kogwistar_stage2_ready") {
            None => true,
            Some(Value::Bool(true)) => {
                let Some(source_revision) = record
                    .metadata
                    .get("_kogwistar_source_revision")
                    .and_then(Value::as_i64)
                else {
                    return false;
                };
                let Some(event) = latest else {
                    return false;
                };
                if event.seq != source_revision {
                    return false;
                }
                let Ok(payload) = serde_json::from_str::<Value>(&event.payload_json) else {
                    return false;
                };
                if !payload.is_object() {
                    return false;
                }
                let event = EntityEvent {
                    namespace: event.namespace.clone(),
                    seq: event.seq,
                    event_id: event.event_id.clone(),
                    entity_kind: event.entity_kind.clone(),
                    entity_id: event.entity_id.clone(),
                    op: event.op.clone(),
                    payload,
                };
                let namespace = event.namespace.clone();
                let entity_kind = event.entity_kind.clone();
                read_canonical_entity_revision(&[event], &namespace, &entity_kind, &record.id)
                    .ok()
                    .flatten()
                    .is_some_and(|revision| {
                        revision.revision == source_revision && !revision.is_deleted()
                    })
            }
            Some(Value::Bool(false)) | Some(_) => false,
        }
    }

    fn filter_current_chroma_stage2(
        &self,
        namespace: &str,
        nodes: &mut Vec<GraphRecord>,
        edges: &mut Vec<GraphRecord>,
    ) -> Result<(), String> {
        for (entity_kind, records) in [("node", nodes), ("edge", edges)] {
            let ids = records
                .iter()
                .filter(|record| {
                    record
                        .metadata
                        .get("_kogwistar_stage2_ready")
                        .and_then(Value::as_bool)
                        == Some(true)
                })
                .map(|record| record.id.clone())
                .collect::<Vec<_>>();
            let latest = self
                .store
                .latest_entity_events(namespace, entity_kind, &ids)
                .map_err(|error| error.to_string())?;
            records.retain(|record| Self::stage2_record_is_current(record, latest.get(&record.id)));
        }
        Ok(())
    }

    fn merge_current_stage1_records(
        &self,
        namespace: &str,
        entity_kind: &str,
        filter: &MetadataFilter,
        records: &mut Vec<GraphRecord>,
        limit: usize,
    ) -> Result<(), String> {
        let remaining = limit.saturating_sub(records.len());
        if remaining == 0 {
            return Ok(());
        }
        let prefix = format!("{entity_kind}:");
        let existing_ids = records
            .iter()
            .map(|record| record.id.as_str())
            .collect::<std::collections::HashSet<_>>();
        let rows = self
            .store
            .list_stage1_node_projections(namespace)
            .map_err(|error| error.to_string())?;
        let mut candidates = Vec::new();
        for row in rows {
            let Some(entity_id) = row.key.strip_prefix(&prefix) else {
                continue;
            };
            let Some(metadata) = row.payload.get("metadata").and_then(Value::as_object) else {
                continue;
            };
            if !filter.matches(metadata) {
                continue;
            }
            if existing_ids.contains(entity_id) {
                continue;
            }
            let entity_id = entity_id.to_owned();
            candidates.push((row, entity_id));
            if candidates.len() >= remaining {
                break;
            }
        }
        let ids = candidates
            .iter()
            .map(|(_, entity_id)| entity_id.clone())
            .collect::<Vec<_>>();
        let latest_events = self
            .store
            .latest_entity_events(namespace, entity_kind, &ids)
            .map_err(|error| error.to_string())?;
        let mut stage1 = Vec::new();
        for (row, entity_id) in candidates {
            if row.materialization_status != "pending" {
                continue;
            }
            let Some(event) = latest_events.get(&entity_id) else {
                continue;
            };
            let payload = serde_json::from_str::<Value>(&event.payload_json)
                .map_err(|error| error.to_string())?;
            let canonical = read_canonical_entity_revision(
                &[EntityEvent {
                    namespace: event.namespace.clone(),
                    seq: event.seq,
                    event_id: event.event_id.clone(),
                    entity_kind: event.entity_kind.clone(),
                    entity_id: event.entity_id.clone(),
                    op: event.op.clone(),
                    payload,
                }],
                namespace,
                entity_kind,
                &entity_id,
            )
            .map_err(|error| error.to_string())?;
            let Some(canonical) = canonical.filter(|revision| {
                !revision.is_deleted() && revision.revision == row.last_authoritative_seq
            }) else {
                continue;
            };
            let expected_fingerprint =
                RevisionFencedChromaProjectionStore::canonical_fingerprint(&canonical)
                    .map_err(|error| error.to_string())?;
            if row
                .payload
                .get("source_fingerprint")
                .and_then(Value::as_str)
                != Some(expected_fingerprint.as_str())
            {
                continue;
            }
            let Some(document) = row.payload.get("document").and_then(Value::as_str) else {
                continue;
            };
            let Some(metadata) = row.payload.get("metadata").and_then(Value::as_object) else {
                continue;
            };
            stage1.push(GraphRecord {
                id: entity_id,
                document: Some(document.to_owned()),
                metadata: metadata.clone(),
                embedding: None,
            });
        }

        for staged in stage1 {
            if records.len() < limit {
                records.push(staged);
            }
        }
        Ok(())
    }

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
        if matches!(&route, DesignRoute::History)
            && (!matches!(request.principal.get("role").and_then(Value::as_str), Some("ro" | "rw"))
                || !principal_strings(&request.principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "Inspecting workflow design history requires ro/rw role and workflow namespace",
            );
        }
        if matches!(
            &route,
            DesignRoute::Undo
                | DesignRoute::Redo
                | DesignRoute::NodeUpsert
                | DesignRoute::NodeDelete(_)
                | DesignRoute::EdgeUpsert
                | DesignRoute::EdgeDelete(_)
        )
            && (principal_role(&request.principal) != "rw"
                || !principal_strings(&request.principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "Mutating workflow design requires rw role and workflow namespace",
            );
        }
        let required = if matches!(route, DesignRoute::Graph | DesignRoute::History) {
            "workflow.design.inspect"
        } else {
            "workflow.design.write"
        };
        let required_capabilities: &[&str] = if matches!(
            &route,
            DesignRoute::Undo | DesignRoute::Redo
        ) {
            &["workflow.design.write", "approve_action"]
        } else if matches!(
            &route,
            DesignRoute::NodeUpsert
                | DesignRoute::NodeDelete(_)
                | DesignRoute::EdgeUpsert
                | DesignRoute::EdgeDelete(_)
        ) {
            &["workflow.design.write", "write_graph"]
        } else {
            &[required]
        };
        if !capability_allowed(
            request,
            &self.capabilities,
            required,
            required_capabilities,
        ) {
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
                        let mut value = history.clone();
                        let _ = value
                            .as_object_mut()
                            .and_then(|fields| fields.remove("materialization_status"));
                        value["status"] = json!("noop");
                        return Ok(value);
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
                    let snapshot_graph = graph_at_version(&events, target);
                    let snapshot_seq = history["commits"][target.to_string()]["target_seq"]
                        .as_i64()
                        .unwrap_or(0);
                    uow.put_workflow_design_snapshot(workflow_id, WorkflowDesignSnapshotWrite {
                        version: target,
                        seq: snapshot_seq,
                        payload_json: snapshot_graph.to_string(),
                        schema_version: 1,
                    })?;
                    uow.replace_named_projection("workflow_design", workflow_id, workflow_design_projection_write(&folded))?;
                    // PYTHON_REFERENCE_V1: rust=rust/crates/kogwistar-api/src/sections/sqlite_service.rs#workflow_design_effect; python=kogwistar/server/chat_service_workflow_design.py#workflow_design_undo+workflow_design_redo; fixture=contracts/golden/workflow-design-redo-http-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_03.rs#sqlite_design_capability_mutates_graph_and_undoes_atomically; python_test=tests/server/test_chat_server_api.py#test_workflow_design_redo_matches_shared_contract; note=internal projection status is not part of Python's undo/redo mutation response.
                    let _ = folded
                        .as_object_mut()
                        .and_then(|value| value.remove("materialization_status"));
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
                    uow.put_workflow_design_snapshot(workflow_id, WorkflowDesignSnapshotWrite {
                        version, seq: data_event.event.seq, payload_json: graph.to_string(), schema_version: 1,
                    })?;
                    let folded_events = uow.replay_raw_events(&namespace, 0, usize::MAX)?.into_iter().map(|event| EntityEvent {
                        namespace: event.namespace, seq: event.seq, event_id: event.event_id, entity_kind: event.entity_kind,
                        entity_id: event.entity_id, op: event.op, payload: serde_json::from_str(&event.payload_json).unwrap_or(Value::Null),
                    }).collect::<Vec<_>>();
                    let folded = workflow_design_history_value(workflow_id, folded_events, "ready")
                        .map_err(kogwistar_store_sqlite::SqliteStoreError::TransactionAborted)?;
                    uow.replace_named_projection("workflow_design", workflow_id, workflow_design_projection_write(&folded))?;
                    // PYTHON_REFERENCE_V1: rust=rust/crates/kogwistar-api/src/sections/sqlite_service.rs#workflow_design_effect; python=kogwistar/server/chat_service_workflow_design.py#workflow_design_upsert_edge+workflow_design_delete_edge; fixture=contracts/golden/workflow-design-edge-upsert-http-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_03.rs#sqlite_design_capability_mutates_graph_and_undoes_atomically; python_test=tests/server/test_chat_server_api.py#test_workflow_design_edge_upsert_matches_shared_contract; note=deleted is emitted only for delete responses, matching Python's mutation envelope.
                    let mut response = json!({"workflow_id": workflow_id, "namespace": namespace,
                        if deleted { if action == "node_delete" { "node_id" } else { "edge_id" } } else if action == "node_upsert" { "node_id" } else { "edge_id" }: entity_id,
                        "designer_id": designer, "version": folded["current_version"],
                        "seq": data_event.event.seq, "can_undo": folded["can_undo"], "can_redo": folded["can_redo"]});
                    if deleted {
                        response["deleted"] = json!(true);
                    }
                    Ok(response)
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
        let is_service_read = path == "/api/workflow/services"
            || service_route(&request.path_and_query)
                .is_some_and(|(_, route)| matches!(route, ServiceRoute::Get | ServiceRoute::Events));
        // PYTHON_PARITY_V1: service get/list/events REST effects enforce MCP-declared role and namespace before storage reads; python=kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_get+workflow_service_list+workflow_service_events; rust_test=rust/crates/kogwistar-api/src/tests/part_05.rs#workflow_service_read_mcp_admission_matches_python_policy_for_all_stores+workflow_service_events_mcp_admission_matches_python_policy_for_all_stores.
        if is_service_read
            && (!matches!(principal_role(&request.principal).as_str(), "ro" | "rw")
                || !principal_strings(&request.principal, "ns")
                    .iter()
                    .any(|namespace| namespace == "workflow" || namespace == "*"))
        {
            return Some(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "Inspecting service requires ro/rw role and workflow namespace",
            ));
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
        Some(match self.store.get_named_projection("service_registry", &service_id) {
            Ok(Some(row)) => json_effect(StatusCode::OK, Value::Object(row.payload)),
            Ok(None) => match self.store.replay_raw_events("workflow", 0, usize::MAX) {
                Ok(events) => match service_projection_from_history(
                    events.into_iter().map(|event| (
                        event.seq,
                        event.entity_kind,
                        event.entity_id,
                        event.op,
                        event.payload_json,
                    )),
                    &service_id,
                ) {
                    Ok(Some((seq, value))) => match self.store.replace_named_projection(
                        "service_registry",
                        &service_id,
                        service_projection_write(seq, &value),
                    ) {
                        Ok(()) => json_effect(StatusCode::OK, value),
                        Err(error) => effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        ),
                    },
                    Ok(None) => effect_error(
                        StatusCode::NOT_FOUND,
                        "KOGWISTAR_SERVICE_NOT_FOUND",
                        format!("Unknown service_id: {service_id}"),
                    ),
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_SERVICE_HISTORY_INVALID",
                        error,
                    ),
                },
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
        })
    }

    fn repair_orphaned_messages(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        // PYTHON_REFERENCE_V1: rust=rust/crates/kogwistar-api/src/sections/sqlite_service.rs#repair_orphaned_messages; python=kogwistar/server/runtime_api.py#repair_orphaned_claimed_messages+kogwistar/server/chat_service.py#repair_orphaned_claimed_messages; fixture=contracts/golden/workflow-repair-orphans-http-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_05.rs#sqlite_run_service_handles_get_events_and_cancel; python_test=tests/server/test_chat_server_api.py#test_workflow_repair_orphans_matches_shared_contract; note=only expired claims in the selected inbox are requeued; active leases remain claimed.
        if principal_role(&request.principal) != "rw"
            || !principal_strings(&request.principal, "ns")
                .iter()
                .any(|namespace| namespace == "workflow" || namespace == "*")
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "Repairing claimed workflow messages requires rw role and workflow namespace",
            );
        }
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
        if principal_role(&request.principal) != "rw"
            || !principal_strings(&request.principal, "ns")
                .iter()
                .any(|namespace| namespace == "workflow" || namespace == "*")
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "Replaying dead letters requires rw role and workflow namespace",
            );
        }
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
        let native_operations = crate::native_answer_persist::registry().map_err(|error| {
            format!("could not register native conversation operations: {error}")
        })?;
        SqliteStore::open(path)
            .map(|store| Self {
                store,
                graph_role: "knowledge".to_owned(),
                chroma_graph: None,
                chroma_node_index: None,
                embedding_provider: None,
                search_index_fts: None,
                chat_completion_provider: None,
                owner_auth_store: None,
                max_queue: runtime_max_queue(),
                capabilities: Arc::new(Mutex::new(CapabilityState::default())),
                native_operations,
                transient_run_deltas: crate::transient_deltas::TransientRunDeltas::default(),
            })
            .map_err(|error| error.to_string())
    }

    /// Compose trusted application-owned Rust operations into the opt-in native
    /// workflow lane. The default registry remains unchanged for existing users.
    pub fn with_native_operations(
        mut self,
        native_operations: kogwistar_runtime::NativeOperationRegistry,
    ) -> Self {
        self.native_operations = native_operations;
        self
    }

    pub fn with_max_queue(mut self, max_queue: usize) -> Self {
        self.max_queue = max_queue.max(1);
        self
    }

    /// Select the host-owned graph role used by generic graph-upsert and ACL
    /// truth-graph operations. The request payload cannot override this value.
    pub fn with_graph_role(mut self, role: impl AsRef<str>) -> Result<Self, String> {
        self.graph_role = normalize_graph_role(role.as_ref())?;
        Ok(self)
    }

    /// Compose Chroma graph reads with SQLite's canonical event journal and
    /// durable projection queue. Graph mutations commit only to SQLite; Chroma
    /// remains an eventual, revision-fenced projection.
    pub fn with_chroma_graph(mut self, graph: kogwistar_store_chroma::ChromaGraphStore) -> Self {
        self.chroma_graph = Some(graph);
        self
    }

    fn with_chroma_node_index(
        mut self,
        index: kogwistar_store_chroma::ChromaVectorCollectionStore,
    ) -> Self {
        self.chroma_node_index = Some(index);
        self
    }

    pub fn with_embedding_provider(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(provider);
        self
    }

    /// Configure the local FTS sidecar used by the hybrid search index.
    /// Vector data remains in the selected graph backend's node-index store.
    ///
    pub fn with_search_index_path(mut self, path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let connection = crate::search_index_fts::open_index(path)
            .map_err(|error| format!("could not open search-index FTS database: {error}"))?;
        self.search_index_fts = Some(Arc::new(Mutex::new(connection)));
        Ok(self)
    }

    /// Configure the explicit Rust-native structured-answer provider.
    /// Missing configuration never falls back to a Python worker.
    pub(crate) fn with_chat_completion_provider(
        mut self,
        provider: crate::native_answer_generation::OpenAiCompatibleChatConfig,
    ) -> Self {
        self.chat_completion_provider = Some(provider);
        self
    }

    pub async fn with_owner_auth_config(mut self, config: &AuthConfig) -> Result<Self, String> {
        self.owner_auth_store = Some(AuthStore::from_config(config).await?);
        Ok(self)
    }

    async fn resolve_inline_native_owner_principal(
        &self,
        run_id: &str,
        required_capability: &str,
    ) -> Result<Value, InlineOwnerPrincipalError> {
        let run = self
            .store
            .get_server_run(run_id)
            .map_err(|error| InlineOwnerPrincipalError::Unavailable(error.to_string()))?
            .ok_or_else(|| {
                InlineOwnerPrincipalError::Denied("native worker run is missing".to_owned())
            })?;
        let owner_id = run
            .user_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                InlineOwnerPrincipalError::Denied(
                    "native worker run has no authenticated owner".to_owned(),
                )
            })?;
        let admission = self
            .store
            .list_server_run_events(run_id, 0, 1)
            .map_err(|error| InlineOwnerPrincipalError::Unavailable(error.to_string()))?
            .into_iter()
            .find(|event| event.event_type == "run.created")
            .ok_or_else(|| {
                InlineOwnerPrincipalError::Denied(
                    "native worker run has no admission event".to_owned(),
                )
            })?;
        let admission_payload: Value =
            serde_json::from_str(&admission.payload_json).map_err(|_| {
                InlineOwnerPrincipalError::Denied(
                    "native worker admission event is invalid".to_owned(),
                )
            })?;
        let binding: RuntimeOwnerBindingV1 = serde_json::from_value(
            admission_payload
                .get("runtime_owner_binding")
                .cloned()
                .ok_or_else(|| {
                    InlineOwnerPrincipalError::Denied(
                        "native worker run has no trusted owner binding".to_owned(),
                    )
                })?,
        )
        .map_err(|_| {
            InlineOwnerPrincipalError::Denied("native worker owner binding is invalid".to_owned())
        })?;
        if binding.user_id != owner_id {
            return Err(InlineOwnerPrincipalError::Denied(
                "native worker owner binding does not match the durable run owner".to_owned(),
            ));
        }
        let auth_store = self.owner_auth_store.as_ref().ok_or_else(|| {
            InlineOwnerPrincipalError::Unavailable(
                "native worker owner revalidation is not configured".to_owned(),
            )
        })?;
        let user = auth_store
            .auth_user(owner_id)
            .await
            .map_err(InlineOwnerPrincipalError::Unavailable)?
            .filter(|user| user.is_active)
            .ok_or_else(|| {
                InlineOwnerPrincipalError::Denied(
                    "native worker run owner is missing or inactive".to_owned(),
                )
            })?;
        revalidated_runtime_owner_principal(
            &binding,
            &user,
            &self.capabilities,
            required_capability,
        )
        .map_err(InlineOwnerPrincipalError::Denied)
    }

    /// Reconcile one SQLite/Chroma projection namespace against canonical
    /// SQLite events. Chroma is eventually consistent, so callers may rerun
    /// this bounded-page scan; it is not a multi-writer transaction.
    pub async fn reconcile_chroma_projection(
        &self,
        namespace: &str,
        page_size: usize,
    ) -> Result<usize, String> {
        if namespace.trim().is_empty() || page_size == 0 || page_size > 500 {
            return Err("namespace must be non-empty and page_size must be 1..=500".to_owned());
        }
        let graph = self
            .chroma_graph
            .as_ref()
            .cloned()
            .ok_or_else(|| "Rust Chroma graph projection is not configured".to_owned())?;
        let projection_store = RevisionFencedChromaProjectionStore {
            queue: self.store.clone(),
            graph: graph.clone(),
        };
        let filter = kogwistar_store::MetadataFilter::default();
        let mut affected = 0_usize;
        let mut deleted = Vec::new();
        let mut restored_stage1 = Vec::new();

        // Repair stale durable Stage-1 rows even when their Stage-2 Chroma row
        // is absent. A Stage-2-only scan cannot discover that failure mode.
        let stage1_rows = self
            .store
            .list_stage1_node_projections(namespace)
            .map_err(|error| error.to_string())?;
        for kind_name in ["node", "edge"] {
            let prefix = format!("{kind_name}:");
            let rows = stage1_rows
                .iter()
                .filter_map(|row| {
                    let entity_id = row.key.strip_prefix(&prefix)?;
                    (!entity_id.is_empty()).then_some((row, entity_id.to_owned()))
                })
                .collect::<Vec<_>>();
            let entity_ids = rows.iter().map(|(_, id)| id.clone()).collect::<Vec<_>>();
            let latest_events = self
                .store
                .latest_entity_events(namespace, kind_name, &entity_ids)
                .map_err(|error| error.to_string())?;
            for (row, entity_id) in rows {
                let current = latest_events
                    .get(&entity_id)
                    .map(|raw| {
                        let payload = serde_json::from_str::<Value>(&raw.payload_json)
                            .map_err(|error| error.to_string())?;
                        let event = kogwistar_store::EntityEvent {
                            namespace: raw.namespace.clone(),
                            seq: raw.seq,
                            event_id: raw.event_id.clone(),
                            entity_kind: raw.entity_kind.clone(),
                            entity_id: raw.entity_id.clone(),
                            op: raw.op.clone(),
                            payload,
                        };
                        read_canonical_entity_revision(&[event], namespace, kind_name, &entity_id)
                            .map_err(|error| error.to_string())
                    })
                    .transpose()?
                    .flatten();
                let Some(current) = current.filter(|revision| !revision.is_deleted()) else {
                    if self
                        .store
                        .clear_stage1_node_projection_if_current(
                            namespace,
                            &row.key,
                            row.last_authoritative_seq,
                        )
                        .map_err(|error| error.to_string())?
                    {
                        affected += 1;
                    }
                    continue;
                };
                let fingerprint =
                    RevisionFencedChromaProjectionStore::canonical_fingerprint(&current)
                        .map_err(|error| error.to_string())?;
                let staged_fingerprint = row
                    .payload
                    .get("source_fingerprint")
                    .and_then(Value::as_str);
                if row.last_authoritative_seq == current.revision
                    && staged_fingerprint == Some(fingerprint.as_str())
                {
                    continue;
                }

                let recovered = RevisionFencedChromaProjectionStore::restore_stage1_from_revision(
                    &current,
                    &fingerprint,
                    row.payload
                        .get("metadata")
                        .and_then(Value::as_object)
                        .and_then(|metadata| metadata.get("_class_name"))
                        .and_then(Value::as_str),
                )
                .map_err(|error| error.to_string())?;
                self.store
                    .replace_stage1_node_projection(namespace, &row.key, recovered)
                    .map_err(|error| error.to_string())?;
                let repaired = self
                    .store
                    .get_stage1_node_projection(namespace, &row.key)
                    .map_err(|error| error.to_string())?;
                if repaired.as_ref().is_none_or(|projection| {
                    projection.last_authoritative_seq != current.revision
                        || projection
                            .payload
                            .get("source_fingerprint")
                            .and_then(Value::as_str)
                            != Some(fingerprint.as_str())
                }) {
                    return Err(
                        "Stage-1 recovery did not persist the current canonical revision"
                            .to_owned(),
                    );
                }
                // Never enqueue a revision that changed while the repair was
                // being written; a later reconciliation can repair its row.
                let latest =
                    sqlite_canonical_entity_revision(&self.store, namespace, kind_name, &entity_id)
                        .map_err(|error| error.to_string())?;
                if latest.as_ref().map(|revision| revision.revision) != Some(current.revision) {
                    continue;
                }
                projection_store
                    .enqueue_current_revision(&current)
                    .map_err(|error| error.to_string())?;
                restored_stage1.push((kind_name, entity_id));
                affected += 1;
            }
        }

        for (kind_name, kind) in [
            ("node", kogwistar_store::GraphEntityKind::Node),
            ("edge", kogwistar_store::GraphEntityKind::Edge),
        ] {
            let mut offset = 0_u32;
            loop {
                let page = graph
                    .graph_records_of_kind_page(namespace, kind, &filter, offset, page_size as u32)
                    .await
                    .map_err(|error| error.to_string())?;
                let count = page.len();
                let record_ids = page
                    .iter()
                    .map(|record| record.id.clone())
                    .collect::<Vec<_>>();
                let latest_events = self
                    .store
                    .latest_entity_events(namespace, kind_name, &record_ids)
                    .map_err(|error| error.to_string())?;
                for record in page {
                    if record
                        .metadata
                        .get("_kogwistar_stage2_ready")
                        .and_then(Value::as_bool)
                        != Some(true)
                    {
                        continue;
                    }
                    let Some(projected_revision) = record
                        .metadata
                        .get("_kogwistar_source_revision")
                        .and_then(Value::as_i64)
                        .filter(|revision| *revision > 0)
                    else {
                        continue;
                    };
                    let Some(projected_fingerprint) = record
                        .metadata
                        .get("_kogwistar_source_fingerprint")
                        .and_then(Value::as_str)
                        .filter(|fingerprint| !fingerprint.is_empty())
                    else {
                        continue;
                    };
                    let current = latest_events
                        .get(&record.id)
                        .map(|raw| {
                            let payload = serde_json::from_str::<Value>(&raw.payload_json)
                                .map_err(|error| error.to_string())?;
                            let event = kogwistar_store::EntityEvent {
                                namespace: raw.namespace.clone(),
                                seq: raw.seq,
                                event_id: raw.event_id.clone(),
                                entity_kind: raw.entity_kind.clone(),
                                entity_id: raw.entity_id.clone(),
                                op: raw.op.clone(),
                                payload,
                            };
                            read_canonical_entity_revision(
                                &[event],
                                namespace,
                                kind_name,
                                &record.id,
                            )
                            .map_err(|error| error.to_string())
                        })
                        .transpose()?
                        .flatten();
                    let Some(current) = current.filter(|revision| !revision.is_deleted()) else {
                        deleted.push((kind_name, record.id));
                        continue;
                    };
                    let current_fingerprint =
                        RevisionFencedChromaProjectionStore::canonical_fingerprint(&current)
                            .map_err(|error| error.to_string())?;
                    if projected_revision != current.revision
                        || projected_fingerprint != current_fingerprint
                    {
                        let stage1 = self
                            .store
                            .get_stage1_node_projection(
                                namespace,
                                &format!("{kind_name}:{}", current.entity_id),
                            )
                            .map_err(|error| error.to_string())?;
                        if stage1.as_ref().is_some_and(|projection| {
                            projection.last_authoritative_seq == current.revision
                                && projection
                                    .payload
                                    .get("source_fingerprint")
                                    .and_then(Value::as_str)
                                    == Some(current_fingerprint.as_str())
                        }) {
                            continue;
                        }
                        if restored_stage1.iter().any(|(restored_kind, restored_id)| {
                            *restored_kind == kind_name && restored_id == &record.id
                        }) {
                            continue;
                        }
                        let stage1 =
                            RevisionFencedChromaProjectionStore::restore_stage1_from_revision(
                                &current,
                                &current_fingerprint,
                                record
                                    .metadata
                                    .get("_class_name")
                                    .and_then(Value::as_str),
                            )
                            .map_err(|error| error.to_string())?;
                        self.store
                            .replace_stage1_node_projection(
                                namespace,
                                &format!("{kind_name}:{}", current.entity_id),
                                stage1,
                            )
                            .map_err(|error| error.to_string())?;
                        projection_store
                            .enqueue_current_revision(&current)
                            .map_err(|error| error.to_string())?;
                        affected += 1;
                    }
                }
                if count < page_size {
                    break;
                }
                offset = offset
                    .checked_add(count as u32)
                    .ok_or_else(|| "Chroma reconciliation offset overflow".to_owned())?;
            }
        }

        // Defer deletes until paging is complete; deleting in-place would shift
        // Chroma offsets and could silently skip later rows.
        for (kind, id) in deleted {
            let (_, outcome) = projection_store
                .delete_with_revision_fence(namespace, kind, &id)
                .await
                .map_err(|error| error.to_string())?;
            if outcome == kogwistar_store::ProjectionWriteOutcome::Applied {
                affected += 1;
            }
        }
        Ok(affected)
    }

    /// Project one bounded batch from the existing SQLite Stage-1 rows into
    /// Chroma. SQLite owns queue/revision state; Chroma remains an eventual
    /// projection and each read is still revision-fenced.
    pub async fn run_embedding_projection_jobs_once(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> Result<kogwistar_engine::projection_worker::ProjectionWorkerSummary, String> {
        if limit == 0 || limit > 1000 || lease_seconds <= 0 {
            return Err("limit must be 1..=1000 and lease_seconds must be positive".to_owned());
        }
        if namespace.is_some_and(|value| value.trim().is_empty()) {
            return Err("namespace must not be empty".to_owned());
        }
        let graph = self
            .chroma_graph
            .as_ref()
            .cloned()
            .ok_or_else(|| "Rust Chroma graph projection is not configured".to_owned())?;
        let provider = self
            .embedding_provider
            .as_ref()
            .cloned()
            .ok_or_else(|| "Rust embedding provider is not configured".to_owned())?;
        provider
            .profile()
            .validate()
            .map_err(|error| error.to_string())?;
        let store = self.store.clone();
        let projection_store = RevisionFencedChromaProjectionStore {
            queue: self.store.clone(),
            graph,
        };
        let index_filter = IndexJobClaimFilter {
            namespace: namespace.map(str::to_owned),
            index_kinds: vec!["node_embedding".to_owned()],
        };
        let summary = kogwistar_engine::projection_worker::run_graph_projection_jobs_once_filtered_with_batch_provider_input(
            &self.store,
            &projection_store,
            limit,
            lease_seconds,
            index_filter,
            move |inputs| {
                let store = store.clone();
                let provider = provider.clone();
                async move {
                    let mut results = Vec::with_capacity(inputs.len());
                    let mut prepared = Vec::new();
                    for input in inputs {
                        let job_id = input.job_id.clone();
                        let prepared_input = async {
                    if input.schema_version != 1
                        || input.index_kind != "node_embedding"
                        || !matches!(input.entity_kind.as_str(), "node" | "edge")
                    {
                        return Err(StoreError::Backend {
                            backend: "sqlite-chroma-projection".to_owned(),
                            message: "unsupported embedding job contract".to_owned(),
                        });
                    }
                    let current = sqlite_canonical_entity_revision(
                        &store,
                        &input.namespace,
                        &input.entity_kind,
                        &input.entity_id,
                    )?
                    .filter(|revision| !revision.is_deleted())
                    .ok_or_else(|| StoreError::Backend {
                        backend: "sqlite-chroma-projection".to_owned(),
                        message: "canonical entity is missing or deleted".to_owned(),
                    })?;
                    let current_job_payload =
                        kogwistar_engine::canonical_revision::canonical_revision_job_payload(
                            &current,
                        );
                    let current_fingerprint = serde_json::from_str::<Value>(&current_job_payload)
                        .ok()
                        .and_then(|value| {
                            value
                                .get("source_fingerprint")
                                .and_then(Value::as_str)
                                .map(str::to_owned)
                        })
                        .ok_or_else(|| StoreError::Backend {
                            backend: "sqlite-chroma-projection".to_owned(),
                            message: "canonical revision fingerprint is unavailable".to_owned(),
                        })?;
                    let queued_fingerprint = input
                        .payload_json
                        .as_deref()
                        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                        .and_then(|value| {
                            value
                                .get("source_fingerprint")
                                .and_then(Value::as_str)
                                .map(str::to_owned)
                        })
                        .ok_or_else(|| StoreError::Backend {
                            backend: "sqlite-chroma-projection".to_owned(),
                            message: "embedding job lacks a canonical source fingerprint".to_owned(),
                        })?;
                    let queued_class_name = input
                        .payload_json
                        .as_deref()
                        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                        .and_then(|value| {
                            value
                                .get("projection_metadata")
                                .and_then(|metadata| metadata.get("_class_name"))
                                .and_then(Value::as_str)
                                .map(str::to_owned)
                        });
                    let stage1_key = format!("{}:{}", input.entity_kind, input.entity_id);
                    let stage1 = store
                        .get_stage1_node_projection(&input.namespace, &stage1_key)
                        .map_err(|error| StoreError::Backend {
                            backend: "sqlite-chroma-projection".to_owned(),
                            message: error.to_string(),
                        })?;
                    let stage1_matches_revision = |stage1: &NamedProjection| {
                        stage1
                            .payload
                            .get("source_fingerprint")
                            .and_then(Value::as_str)
                            == Some(current_fingerprint.as_str())
                            && stage1.last_authoritative_seq == current.revision
                    };
                    let stage1 = if let Some(stage1) = stage1 {
                        if !stage1_matches_revision(&stage1) {
                            return Err(StoreError::Backend {
                                backend: "sqlite-chroma-projection".to_owned(),
                                message: "queued, staged, and canonical revisions do not match".to_owned(),
                            });
                        }
                        stage1
                    } else {
                        let recovered =
                            RevisionFencedChromaProjectionStore::restore_stage1_from_revision(
                                &current,
                                &current_fingerprint,
                                queued_class_name.as_deref(),
                            )?;
                        store
                            .replace_stage1_node_projection(
                                &input.namespace,
                                &stage1_key,
                                recovered,
                            )
                            .map_err(|error| StoreError::Backend {
                                backend: "sqlite-chroma-projection".to_owned(),
                                message: error.to_string(),
                            })?;
                        let latest = sqlite_canonical_entity_revision(
                            &store,
                            &input.namespace,
                            &input.entity_kind,
                            &input.entity_id,
                        )?;
                        if latest.as_ref().map(|revision| revision.revision)
                            != Some(current.revision)
                        {
                            store
                                .clear_stage1_node_projection_if_current(
                                    &input.namespace,
                                    &stage1_key,
                                    current.revision,
                                )
                                .map_err(|error| StoreError::Backend {
                                    backend: "sqlite-chroma-projection".to_owned(),
                                    message: error.to_string(),
                                })?;
                            return Err(StoreError::Backend {
                                backend: "sqlite-chroma-projection".to_owned(),
                                message: "canonical revision changed during Stage-1 recovery".to_owned(),
                            });
                        }
                        store
                            .get_stage1_node_projection(&input.namespace, &stage1_key)
                            .map_err(|error| StoreError::Backend {
                                backend: "sqlite-chroma-projection".to_owned(),
                                message: error.to_string(),
                            })?
                            .filter(|projection| stage1_matches_revision(projection))
                            .ok_or_else(|| StoreError::Backend {
                                backend: "sqlite-chroma-projection".to_owned(),
                                message: "SQLite Stage-1 source row could not be restored".to_owned(),
                            })?
                    };
                    let stage1_fingerprint = stage1
                        .payload
                        .get("source_fingerprint")
                        .and_then(Value::as_str);
                    if queued_fingerprint != current_fingerprint
                        || stage1_fingerprint != Some(current_fingerprint.as_str())
                        || stage1.last_authoritative_seq != current.revision
                    {
                        return Err(StoreError::Backend {
                            backend: "sqlite-chroma-projection".to_owned(),
                            message: "queued, staged, and canonical revisions do not match".to_owned(),
                        });
                    }
                    let document = stage1
                        .payload
                        .get("document")
                        .and_then(Value::as_str)
                        .filter(|value| !value.is_empty())
                        .ok_or_else(|| StoreError::Backend {
                            backend: "sqlite-chroma-projection".to_owned(),
                            message: "SQLite Stage-1 document is missing".to_owned(),
                        })?
                        .to_owned();
                    let metadata = stage1
                        .payload
                        .get("metadata")
                        .and_then(Value::as_object)
                        .cloned()
                        .ok_or_else(|| StoreError::Backend {
                            backend: "sqlite-chroma-projection".to_owned(),
                            message: "SQLite Stage-1 metadata is malformed".to_owned(),
                        })?;
                    Ok((input.entity_id, document, metadata, current.revision, current_fingerprint))
                        }
                        .await;
                        match prepared_input {
                            Ok(prepared_input) => prepared.push((job_id, prepared_input)),
                            Err(error) => results.push((job_id, Err(error))),
                        }
                    }

                    if prepared.is_empty() {
                        return Ok(results);
                    }
                    let documents = prepared
                        .iter()
                        .map(|(_, (_, document, _, _, _))| document.clone())
                        .collect::<Vec<_>>();
                    let vectors = match provider.embed(&documents).await {
                        Ok(vectors) if vectors.len() == prepared.len() => vectors,
                        Ok(_) => {
                            let message = "embedding provider returned wrong batch length";
                            results.extend(prepared.into_iter().map(|(id, _)| (id, Err(StoreError::Backend {
                                backend: "sqlite-chroma-projection".to_owned(),
                                message: message.to_owned(),
                            }))));
                            return Ok(results);
                        }
                        Err(_) => {
                            let message = "embedding provider failed";
                            results.extend(prepared.into_iter().map(|(id, _)| (id, Err(StoreError::Backend {
                                backend: "sqlite-chroma-projection".to_owned(),
                                message: message.to_owned(),
                            }))));
                            return Ok(results);
                        }
                    };
                    for ((job_id, (entity_id, document, mut metadata, revision, fingerprint)), vector) in
                        prepared.into_iter().zip(vectors)
                    {
                        if vector.len() != provider.profile().dimension
                            || vector.iter().any(|value| !value.is_finite())
                        {
                            results.push((job_id, Err(StoreError::Backend {
                                backend: "sqlite-chroma-projection".to_owned(),
                                message: "embedding provider returned an invalid vector".to_owned(),
                            })));
                            continue;
                        }
                        let embedding = vector.iter().map(|value| *value as f32).collect::<Vec<_>>();
                        if embedding.iter().any(|value| !value.is_finite()) {
                            results.push((job_id, Err(StoreError::NonFiniteVector)));
                            continue;
                        }
                    metadata.insert("_kogwistar_stage2_ready".to_owned(), Value::Bool(true));
                    metadata.insert(
                        "_kogwistar_source_revision".to_owned(),
                        Value::from(revision),
                    );
                    metadata.insert(
                        "_kogwistar_source_fingerprint".to_owned(),
                        Value::String(fingerprint),
                    );
                        results.push((job_id, Ok(GraphRecord {
                        id: entity_id,
                        document: Some(document),
                        metadata,
                        embedding: Some(embedding),
                        })));
                    }
                    Ok(results)
                }
            },
        )
        .await
        .map_err(|error| error.to_string())?;
        Ok(summary)
    }

    async fn chroma_graph_snapshot(
        &self,
        request: &ApiEffectRequest,
        truth_graph: &str,
    ) -> Result<GraphAclProjection, ApiEffectResponse> {
        let scope_value = security_scope_value(&request.principal);
        let namespace = scope_value["storage_namespace"]
            .as_str()
            .unwrap_or("default");
        let mut filter = MetadataFilter::default();
        if let Some(workspace) = request
            .principal
            .get("workspace_id")
            .or_else(|| request.principal.get("workspace"))
            .and_then(Value::as_str)
        {
            filter.equals.insert(
                "workspace_id".to_owned(),
                Value::String(workspace.to_owned()),
            );
        }
        if let Some(graph_space) = request.principal.get("graph_space").and_then(Value::as_str) {
            filter.equals.insert(
                "graph_space".to_owned(),
                Value::String(graph_space.to_owned()),
            );
        }
        let Some(graph) = &self.chroma_graph else {
            return Err(effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "The SQLite service has no configured graph projection",
            ));
        };
        const MAX_GRAPH_SNAPSHOT_RECORDS: usize = 10_000;
        let (mut nodes, mut edges) = match tokio::try_join!(
            graph.graph_records_of_kind_bounded(
                namespace,
                GraphEntityKind::Node,
                &filter,
                MAX_GRAPH_SNAPSHOT_RECORDS,
            ),
            graph.graph_records_of_kind_bounded(
                namespace,
                GraphEntityKind::Edge,
                &filter,
                MAX_GRAPH_SNAPSHOT_RECORDS,
            ),
        ) {
            Ok(records) => records,
            Err(error) => {
                return Err(effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ));
            }
        };
        nodes.retain(|record| chroma_record_matches_principal_scope(record, &request.principal));
        edges.retain(|record| chroma_record_matches_principal_scope(record, &request.principal));
        if let Err(error) = self.filter_current_chroma_stage2(namespace, &mut nodes, &mut edges) {
            return Err(effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error,
            ));
        }
        for (entity_kind, records) in [("node", &mut nodes), ("edge", &mut edges)] {
            if let Err(error) = self.merge_current_stage1_records(
                namespace,
                entity_kind,
                &filter,
                records,
                MAX_GRAPH_SNAPSHOT_RECORDS,
            ) {
                return Err(effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                ));
            }
        }
        nodes.retain(|record| chroma_record_matches_principal_scope(record, &request.principal));
        edges.retain(|record| chroma_record_matches_principal_scope(record, &request.principal));
        let entity_ids = nodes
            .iter()
            .chain(edges.iter())
            .map(|record| record.id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if entity_ids.len() > MAX_GRAPH_SNAPSHOT_RECORDS {
            return Err(effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_GRAPH_QUERY_ERROR",
                "graph snapshot ACL set exceeds the 10000-record limit",
            ));
        }
        let acl_scope = kogwistar_store::GraphScope {
            namespace: namespace.to_owned(),
            workspace_id: request
                .principal
                .get("workspace_id")
                .or_else(|| request.principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: request
                .principal
                .get("graph_space")
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        let acl_records = self
            .store
            .immediate_transaction(|uow| {
                uow.graph_acl_records_for_entities(&acl_scope, truth_graph, &entity_ids)
            })
            .map_err(|error| {
                effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                )
            })?;
        let mut acl_index = GraphAclIndex::default();
        acl_index.extend_from_records(&acl_records, truth_graph);
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        Ok(GraphAclProjection::from_indexed_records(
            &nodes,
            &edges,
            acl_index,
            &principal,
        ))
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/sqlite_service.rs#sqlite_search_index_hybrid_effect; python=kogwistar/server_mcp_with_admin.py#search_index_hybrid; fixture=contracts/golden/search-index-hybrid-v1.json; rust_test=rust/crates/kogwistar-api/src/search_index_fts.rs#hybrid_fusion_matches_shared_python_golden_fixture; python_test=tests/core/test_search_index_contract.py#test_hybrid_search_fusion_matches_shared_rust_golden_fixture
    async fn sqlite_search_index_hybrid_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Search-index query requires read_graph capability",
            );
        }
        let (query, limit, resolve_node) = match crate::search_index_fts::parse_hybrid_search_query(
            &request.path_and_query,
        ) {
            Ok(parsed) => parsed,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let (Some(provider), Some(fts), Some(index_store)) = (
            self.embedding_provider.as_ref(),
            self.search_index_fts.as_ref(),
            self.chroma_node_index.as_ref(),
        ) else {
            return effect_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "KOGWISTAR_SEARCH_INDEX_UNAVAILABLE",
                "Search-index embedding, FTS, or Chroma node-index storage is not configured",
            );
        };
        let hits = match crate::search_index_fts::hybrid_search(
            fts,
            index_store,
            "nodes_index",
            provider.as_ref(),
            &query,
            limit,
        )
        .await
        {
            Ok(hits) => hits,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_SEARCH_INDEX_ERROR",
                    error,
                );
            }
        };
        let projection = match self.chroma_graph_snapshot(request, &self.graph_role).await {
            Ok(projection) => projection,
            Err(response) => return response,
        };
        let candidate_ids = hits
            .iter()
            .map(|hit| hit.node_id.as_str())
            .collect::<BTreeSet<_>>();
        let canonical_nodes = projection
            .nodes()
            .iter()
            .filter(|record| candidate_ids.contains(record.id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        json_effect(
            StatusCode::OK,
            crate::search_index_fts::project_hybrid_search_result(
                &query,
                &hits,
                &canonical_nodes,
                resolve_node,
                limit,
            ),
        )
    }


}
