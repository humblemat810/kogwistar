#[derive(Clone)]
pub struct SqliteRunApplicationService {
    store: SqliteStore,
    graph_role: String,
    chroma_graph: Option<kogwistar_store_chroma::ChromaGraphStore>,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
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
            ..common
        };
        let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
            node_config,
            edge_config,
        )
        .await
        .map_err(|error| error.to_string())?;
        service = service.with_chroma_graph(graph);
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

#[derive(Clone)]
struct RevisionFencedChromaProjectionStore {
    queue: SqliteStore,
    graph: kogwistar_store_chroma::ChromaGraphStore,
}

impl RevisionFencedChromaProjectionStore {
    fn current_revision(
        &self,
        namespace: &str,
        entity_kind: &str,
        entity_id: &str,
    ) -> Result<Option<kogwistar_engine::canonical_revision::CanonicalEntityRevision>, StoreError>
    {
        sqlite_canonical_entity_revision(&self.queue, namespace, entity_kind, entity_id)
    }

    fn enqueue_current_revision(
        &self,
        revision: &kogwistar_engine::canonical_revision::CanonicalEntityRevision,
    ) -> Result<(), StoreError> {
        let payload_json =
            kogwistar_engine::canonical_revision::canonical_revision_job_payload(revision);
        self.queue
            .enqueue_index_job(NewIndexJob {
                job_id: format!("revision-fence-{}", uuid::Uuid::new_v4()),
                namespace: revision.namespace.clone(),
                entity_kind: revision.entity_kind.clone(),
                entity_id: revision.entity_id.clone(),
                index_kind: "node_embedding".to_owned(),
                op: if revision.is_deleted() {
                    "DELETE".to_owned()
                } else {
                    "UPSERT".to_owned()
                },
                payload_json: Some(payload_json),
                max_retries: 10,
            })
            .map_err(|error| StoreError::Backend {
                backend: "sqlite-chroma-projection".to_owned(),
                message: error.to_string(),
            })?;
        Ok(())
    }

    fn clear_stage1_if_current(
        &self,
        revision: &kogwistar_engine::canonical_revision::CanonicalEntityRevision,
    ) -> Result<(), StoreError> {
        self.queue
            .clear_stage1_node_projection_if_current(
                &revision.namespace,
                &format!("{}:{}", revision.entity_kind, revision.entity_id),
                revision.revision,
            )
            .map_err(|error| StoreError::Backend {
                backend: "sqlite-chroma-projection".to_owned(),
                message: error.to_string(),
            })?;
        Ok(())
    }

    fn clear_stage1_through_deleted_revision(
        &self,
        revision: &kogwistar_engine::canonical_revision::CanonicalEntityRevision,
    ) -> Result<(), StoreError> {
        let key = format!("{}:{}", revision.entity_kind, revision.entity_id);
        let stage1 = self
            .queue
            .get_stage1_node_projection(&revision.namespace, &key)
            .map_err(|error| StoreError::Backend {
                backend: "sqlite-chroma-projection".to_owned(),
                message: error.to_string(),
            })?;
        if let Some(stage1) = stage1
            && stage1.last_authoritative_seq <= revision.revision
        {
            self.queue
                .clear_stage1_node_projection_if_current(
                    &revision.namespace,
                    &key,
                    stage1.last_authoritative_seq,
                )
                .map_err(|error| StoreError::Backend {
                    backend: "sqlite-chroma-projection".to_owned(),
                    message: error.to_string(),
                })?;
        }
        Ok(())
    }

    fn canonical_fingerprint(
        revision: &kogwistar_engine::canonical_revision::CanonicalEntityRevision,
    ) -> Result<String, StoreError> {
        let payload =
            kogwistar_engine::canonical_revision::canonical_revision_job_payload(revision);
        serde_json::from_str::<Value>(&payload)
            .ok()
            .and_then(|value| {
                value
                    .get("source_fingerprint")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .ok_or_else(|| StoreError::InvalidEntityEventPayload {
                message: "canonical revision fingerprint is unavailable".to_owned(),
            })
    }

    fn restore_stage1_from_revision(
        revision: &kogwistar_engine::canonical_revision::CanonicalEntityRevision,
        fingerprint: &str,
    ) -> Result<NamedProjectionWrite, StoreError> {
        kogwistar_store_sqlite::stage1_projection_from_canonical_event(
            &revision.namespace,
            &revision.entity_kind,
            &revision.entity_id,
            revision.revision,
            &revision.payload,
            fingerprint,
        )
    }

    fn graph_kind(entity_kind: &str) -> Result<kogwistar_store::GraphEntityKind, StoreError> {
        match entity_kind {
            "node" => Ok(kogwistar_store::GraphEntityKind::Node),
            "edge" => Ok(kogwistar_store::GraphEntityKind::Edge),
            _ => Err(StoreError::Backend {
                backend: "sqlite-chroma-projection".to_owned(),
                message: "projection entity kind must be 'node' or 'edge'".to_owned(),
            }),
        }
    }

    async fn upsert_with_revision_fence(
        &self,
        namespace: &str,
        entity_kind: &str,
        record: GraphRecord,
    ) -> Result<kogwistar_store::ProjectionWriteOutcome, StoreError> {
        use kogwistar_store::ProjectionWriteOutcome;

        let kind = Self::graph_kind(entity_kind)?;
        let Some(current) = self.current_revision(namespace, entity_kind, &record.id)? else {
            kogwistar_store::GraphWriteStore::delete_graph_record_of_kind(
                &self.graph,
                namespace,
                kind,
                &record.id,
            )
            .await?;
            return Ok(ProjectionWriteOutcome::Applied);
        };
        if current.is_deleted() {
            self.enqueue_current_revision(&current)?;
            return Ok(ProjectionWriteOutcome::Superseded);
        }
        let current_fingerprint = Self::canonical_fingerprint(&current)?;
        let record_revision = record
            .metadata
            .get("_kogwistar_source_revision")
            .and_then(Value::as_i64);
        let record_fingerprint = record
            .metadata
            .get("_kogwistar_source_fingerprint")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let stage2_ready = record
            .metadata
            .get("_kogwistar_stage2_ready")
            .and_then(Value::as_bool)
            == Some(true);
        if !stage2_ready
            || record_revision != Some(current.revision)
            || record_fingerprint.as_deref() != Some(current_fingerprint.as_str())
        {
            self.enqueue_current_revision(&current)?;
            return Ok(ProjectionWriteOutcome::Superseded);
        }
        let projected_revision = record_revision.expect("stage2-ready revision was validated");
        let projected_fingerprint =
            record_fingerprint.expect("stage2-ready fingerprint was validated");
        let projected_id = record.id.clone();
        kogwistar_store::GraphWriteStore::upsert_graph_record_of_kind(
            &self.graph,
            namespace,
            kind,
            record,
        )
        .await?;
        match self.current_revision(namespace, entity_kind, &projected_id)? {
            Some(latest)
                if !latest.is_deleted()
                    && latest.revision == projected_revision
                    && Self::canonical_fingerprint(&latest)? == projected_fingerprint =>
            {
                self.clear_stage1_if_current(&latest)?;
                Ok(ProjectionWriteOutcome::Applied)
            }
            Some(latest) => {
                self.enqueue_current_revision(&latest)?;
                Ok(ProjectionWriteOutcome::Superseded)
            }
            None => {
                kogwistar_store::GraphWriteStore::delete_graph_record_of_kind(
                    &self.graph,
                    namespace,
                    kind,
                    &projected_id,
                )
                .await?;
                Ok(ProjectionWriteOutcome::Superseded)
            }
        }
    }

    async fn delete_with_revision_fence(
        &self,
        namespace: &str,
        entity_kind: &str,
        id: &str,
    ) -> Result<(bool, kogwistar_store::ProjectionWriteOutcome), StoreError> {
        use kogwistar_store::ProjectionWriteOutcome;

        let kind = Self::graph_kind(entity_kind)?;
        if let Some(current) = self.current_revision(namespace, entity_kind, id)?
            && !current.is_deleted()
        {
            self.enqueue_current_revision(&current)?;
            return Ok((false, ProjectionWriteOutcome::Superseded));
        }
        let deleted = kogwistar_store::GraphWriteStore::delete_graph_record_of_kind(
            &self.graph,
            namespace,
            kind,
            id,
        )
        .await?;
        if let Some(current) = self.current_revision(namespace, entity_kind, id)? {
            if !current.is_deleted() {
                self.enqueue_current_revision(&current)?;
                return Ok((deleted, ProjectionWriteOutcome::Superseded));
            }
            self.clear_stage1_through_deleted_revision(&current)?;
        }
        Ok((deleted, ProjectionWriteOutcome::Applied))
    }
}

impl kogwistar_store::GraphProjectionWriteStore for RevisionFencedChromaProjectionStore {
    // Preserve the trait's explicit opaque-future + Send contract at this adapter.
    #[allow(clippy::manual_async_fn)]
    fn upsert_graph_projection_record(
        &self,
        namespace: &str,
        entity_kind: &str,
        record: GraphRecord,
    ) -> impl std::future::Future<Output = Result<(), StoreError>> + Send {
        async move {
            self.upsert_with_revision_fence(namespace, entity_kind, record)
                .await
                .map(|_| ())
        }
    }

    // Preserve the trait's explicit opaque-future + Send contract at this adapter.
    #[allow(clippy::manual_async_fn)]
    fn delete_graph_projection_record(
        &self,
        namespace: &str,
        entity_kind: &str,
        id: &str,
    ) -> impl std::future::Future<Output = Result<bool, StoreError>> + Send {
        async move {
            self.delete_with_revision_fence(namespace, entity_kind, id)
                .await
                .map(|(deleted, _)| deleted)
        }
    }

    // Preserve the trait's explicit opaque-future + Send contract at this adapter.
    #[allow(clippy::manual_async_fn)]
    fn upsert_graph_projection_record_with_outcome(
        &self,
        namespace: &str,
        entity_kind: &str,
        record: GraphRecord,
    ) -> impl std::future::Future<
        Output = Result<kogwistar_store::ProjectionWriteOutcome, StoreError>,
    > + Send {
        async move {
            self.upsert_with_revision_fence(namespace, entity_kind, record)
                .await
        }
    }

    // Preserve the trait's explicit opaque-future + Send contract at this adapter.
    #[allow(clippy::manual_async_fn)]
    fn delete_graph_projection_record_with_outcome(
        &self,
        namespace: &str,
        entity_kind: &str,
        id: &str,
    ) -> impl std::future::Future<
        Output = Result<kogwistar_store::ProjectionWriteOutcome, StoreError>,
    > + Send {
        async move {
            self.delete_with_revision_fence(namespace, entity_kind, id)
                .await
                .map(|(_, outcome)| outcome)
        }
    }
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
        let native_operations = crate::native_answer_persist::registry().map_err(|error| {
            format!("could not register native conversation operations: {error}")
        })?;
        SqliteStore::open(path)
            .map(|store| Self {
                store,
                graph_role: "knowledge".to_owned(),
                chroma_graph: None,
                embedding_provider: None,
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

    async fn sqlite_chroma_invoke_builtin_tool_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let input = match syscall_input(request, "invoke_tool") {
            Ok(input) => input,
            Err(response) => return response,
        };
        if !syscall_allowed_in_namespace(
            request,
            &self.capabilities,
            "invoke_tool",
            true,
            &[],
            "conversation",
        ) {
            return syscall_forbidden("invoke_tool");
        }
        let Some(tool_name) = input
            .args
            .get("tool_name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_ascii_lowercase)
        else {
            return effect_error(
                StatusCode::BAD_REQUEST,
                "KOGWISTAR_INVALID_REQUEST",
                "tool_name is required",
            );
        };

        let result = match tool_name.as_str() {
            "list_transcript" | "latest_snapshot" => {
                let Some(conversation_id) = input
                    .args
                    .get("conversation_id")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                else {
                    return effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        format!("conversation_id is required for {tool_name}"),
                    );
                };
                let operation = if tool_name == "list_transcript" {
                    "receive_message"
                } else {
                    "mount_memory"
                };
                let mut args = serde_json::Map::new();
                args.insert(
                    "conversation_id".to_owned(),
                    Value::String(conversation_id.to_owned()),
                );
                if tool_name == "latest_snapshot" {
                    for field in ["run_id", "stage"] {
                        if let Some(value) = input.args.get(field) {
                            args.insert(field.to_owned(), value.clone());
                        }
                    }
                }
                let nested_request = ApiEffectRequest {
                    contract_version: request.contract_version,
                    method: "POST".to_owned(),
                    path_and_query: format!("/api/syscall/v1/{operation}"),
                    body: serde_json::to_vec(&json!({
                        "version": input.version,
                        "args": args,
                    }))
                    .expect("nested syscall args serialize"),
                    principal: request.principal.clone(),
                };
                let response = if tool_name == "list_transcript" {
                    self.sqlite_chroma_receive_message_effect(&nested_request)
                        .await
                } else {
                    self.sqlite_chroma_mount_memory_effect(&nested_request)
                        .await
                };
                if !(200..300).contains(&response.status) {
                    return response;
                }
                let payload: Value = match serde_json::from_slice(&response.body) {
                    Ok(payload) => payload,
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_INVALID_INTERNAL_RESPONSE",
                            error.to_string(),
                        );
                    }
                };
                let Some(mut value) = payload.get("result").cloned() else {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_INVALID_INTERNAL_RESPONSE",
                        "built-in graph tool returned no result",
                    );
                };
                if tool_name == "list_transcript" {
                    value["tool_name"] = Value::String(tool_name.clone());
                    value
                } else {
                    json!({
                        "tool_name": tool_name,
                        "conversation_id": conversation_id,
                        "snapshot": value,
                    })
                }
            }
            "capability_snapshot" => {
                if !capability_allowed(
                    request,
                    &self.capabilities,
                    "project_view",
                    &["project_view", "read_security_scope"],
                ) {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "built-in tool \"capability_snapshot\" requires project_view and read_security_scope",
                    );
                }
                let state = self
                    .capabilities
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                json!({
                    "tool_name": tool_name,
                    "snapshot": capability_snapshot(&request.principal, &state),
                })
            }
            "resource_snapshot" => {
                if !capability_allowed(
                    request,
                    &self.capabilities,
                    "project_view",
                    &["project_view", "read_security_scope"],
                ) {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "built-in tool \"resource_snapshot\" requires project_view and read_security_scope",
                    );
                }
                let runs = match self.store.list_server_runs(None, None, None, 10_000) {
                    Ok(rows) => rows,
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
                    Ok(rows) => rows,
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
                    match self.store.list_server_run_events(&run.run_id, 0, 200) {
                        Ok(mut rows) => events.append(&mut rows),
                        Err(error) => {
                            return effect_error(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                "KOGWISTAR_STORE_ERROR",
                                error.to_string(),
                            );
                        }
                    }
                }
                let service_rows = match self.store.list_named_projections("service_registry") {
                    Ok(rows) => rows,
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        );
                    }
                };
                let services = service_rows
                    .into_iter()
                    .map(|row| Value::Object(row.payload))
                    .collect::<Vec<_>>();
                let usage = usage_history(&events, 10_000);
                let mut snapshot = resource_value(
                    &runs,
                    &lanes,
                    &usage,
                    &events,
                    configured_storage_usage_bytes(),
                );
                snapshot["services"] = resource_service_value(&services);
                snapshot
                    .as_object_mut()
                    .expect("resource snapshot is an object")
                    .remove("migration");
                json!({"tool_name": tool_name, "snapshot": snapshot})
            }
            _ => {
                return effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_TOOL_NOT_FOUND",
                    format!("unknown built-in tool: {tool_name}"),
                );
            }
        };
        syscall_result(
            &input.version,
            "invoke_tool",
            json_effect(StatusCode::OK, result),
        )
    }

    async fn sqlite_document_projection_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !self
            .document_authority_capability()
            .supports_projection_upsert()
        {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_DOCUMENT_AUTHORITY_UNSUPPORTED",
                "The selected SQLite+Chroma service does not own source-document projections",
            );
        }
        let write = match document_projection_write(request, &self.capabilities) {
            Ok(write) => write,
            Err(response) => return response,
        };
        match self.store.upsert_document_projection(write).await {
            Ok(()) => json_effect(StatusCode::OK, json!({"ok": true})),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_DOCUMENT_PROJECTION_FAILED",
                error.to_string(),
            ),
        }
    }

    async fn sqlite_chroma_graph_mutation_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if self.chroma_graph.is_none() {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "SQLite graph mutations require the configured eventual Chroma projection",
            );
        }
        if !capability_allowed_for_principal(
            &request.principal,
            &self.capabilities,
            "write_graph",
            &["write_graph"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph mutation requires write_graph capability",
            );
        }
        let input: GraphMutationBatchInput = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if let Err(error) = input.validate() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error.to_string(),
            );
        }
        let mutations = match graph_mutation_batch_value(input.clone()) {
            Ok(mutations) => mutations,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        if let Err(error) = validate_graph_mutation_envelopes(&input, &mutations) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error,
            );
        }
        let security_scope = security_scope_value(&request.principal);
        if mutations
            .iter()
            .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "Graph mutation scope is outside the principal storage scope",
            );
        }
        let index_jobs = match graph_index_jobs_value(&input, &mutations) {
            Ok(jobs) => jobs,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let committed = self.store.immediate_transaction(|uow| {
            uow.apply_canonical_graph_mutation_batch(GraphMutationBatch {
                mutations: mutations.clone(),
                index_jobs: index_jobs.clone(),
            })
        });
        match committed {
            Ok(results) => json_effect(
                StatusCode::OK,
                rust_native_attested(json!({
                    "contract_version": 1,
                    "atomic": true,
                    "projection_consistency": "eventual",
                    "results": results.into_iter().map(|result| json!({
                        "event": {
                            "namespace": result.event.namespace,
                            "seq": result.event.seq,
                            "event_id": result.event.event_id,
                            "entity_kind": result.event.entity_kind,
                            "entity_id": result.event.entity_id,
                            "op": result.event.op,
                            "payload": result.event.payload,
                        },
                        "inserted": result.inserted,
                        "mutated": result.mutated,
                    })).collect::<Vec<_>>(),
                })),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_GRAPH_MUTATION_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn sqlite_chroma_create_conversation_effect(
        &self,
        request: &ApiEffectRequest,
        input: &SyscallInput,
    ) -> ApiEffectResponse {
        if !syscall_allowed_in_namespace(
            request,
            &self.capabilities,
            "create_conversation",
            true,
            &["spawn_process"],
            "conversation",
        ) {
            return syscall_forbidden("create_conversation");
        }
        let Some(owner_id) = authenticated_run_owner_id(&request.principal) else {
            return effect_error(
                StatusCode::UNAUTHORIZED,
                "KOGWISTAR_AUTHENTICATION_REQUIRED",
                "conversation creation requires an authenticated principal",
            );
        };
        let Some(requested_user_id) = input
            .args
            .get("user_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return effect_error(
                StatusCode::BAD_REQUEST,
                "KOGWISTAR_INVALID_REQUEST",
                "user_id is required",
            );
        };
        if !requested_user_id.eq_ignore_ascii_case(owner_id) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "user_id must match the authenticated principal",
            );
        }
        let optional_id = |key: &str| -> Result<Option<String>, ApiEffectResponse> {
            match input.args.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::String(value)) if !value.trim().is_empty() => {
                    Ok(Some(value.trim().to_owned()))
                }
                _ => Err(effect_error(
                    StatusCode::BAD_REQUEST,
                    "KOGWISTAR_INVALID_REQUEST",
                    format!("{key} must be a non-empty string or null"),
                )),
            }
        };
        let conversation_id = match optional_id("conversation_id") {
            Ok(Some(value)) => value,
            Ok(None) => uuid::Uuid::new_v4().to_string(),
            Err(response) => return response,
        };
        let requested_start_id = match optional_id("start_node_id") {
            Ok(value) => value,
            Err(response) => return response,
        };
        let user_id = owner_id.to_owned();
        let start_id = requested_start_id.unwrap_or_else(|| {
            kogwistar_contracts::stable_id(
                "conversation.node",
                &[
                    user_id.clone(),
                    conversation_id.clone(),
                    "Start of conversation".to_owned(),
                    "-1".to_owned(),
                    "system".to_owned(),
                    "conversation_summary".to_owned(),
                    "True".to_owned(),
                ],
            )
            .to_string()
        });
        let scope_values = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: scope_values["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: scope_values["workspace"].as_str().map(str::to_owned),
            graph_space: Some("conversation".to_owned()),
        };
        if scope.namespace != "conversation" || !graph_mutation_scope_allowed(&scope, &scope_values)
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "conversation creation requires the authenticated conversation storage scope",
            );
        }

        let metadata = json!({
            "level_from_root": 0,
            "entity_type": "conversation_start",
            "turn_index": -1,
            "in_conversation_chain": true,
            "in_ui_chain": true,
            "lifecycle_status": "active",
            "redirect_to_id": null,
            "role": "system",
            "conversation_id": conversation_id,
            "user_id": user_id,
        });
        let mentions = json!([{
            "spans": [{
                "collection_page_url": "conversation/_conv:_dummy",
                "document_page_url": "conversation/_conv:_dummy",
                "doc_id": "_conv:_dummy",
                "insertion_method": "system",
                "page_number": 1,
                "start_char": 0,
                "end_char": 1,
                "excerpt": "",
                "context_before": "",
                "context_after": "",
                "source_cluster_id": null,
                "verification": {"method": "system", "is_verified": true, "score": 1.0, "notes": ""}
            }],
            "multimodal_spans": []
        }]);
        let payload = json!({
            "id": start_id,
            "label": "conversation start",
            "type": "entity",
            "summary": "Start of conversation",
            "domain_id": null,
            "canonical_entity_id": null,
            "properties": {"status": "active"},
            "mentions": mentions,
            "embedding": null,
            "doc_id": null,
            "metadata": metadata,
            "level_from_root": 0,
            "role": "system",
            "turn_index": -1,
            "conversation_id": conversation_id,
            "user_id": user_id,
        });
        let envelope = kogwistar_contracts::GraphModelEnvelope::new(
            kogwistar_contracts::GraphModelClass::ConversationNode,
            payload.clone(),
        );
        if let Err(error) = envelope.validate_family_v1() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_GRAPH_MODEL",
                error.to_string(),
            );
        }
        let document = {
            let mut document_value = payload.clone();
            if let Some(object) = document_value.as_object_mut() {
                object.remove("metadata");
                object.remove("embedding");
            }
            match serde_json::to_string(&document_value) {
                Ok(document) => document,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_SERIALIZATION_FAILED",
                        error.to_string(),
                    );
                }
            }
        };
        let mut record_metadata = metadata.as_object().cloned().unwrap_or_default();
        record_metadata.insert("namespace".to_owned(), json!(scope.namespace));
        record_metadata.insert("_class_name".to_owned(), json!("ConversationNode"));
        record_metadata.insert("label".to_owned(), json!("conversation start"));
        record_metadata.insert("type".to_owned(), json!("entity"));
        record_metadata.insert("summary".to_owned(), json!("Start of conversation"));
        record_metadata.insert("properties".to_owned(), json!("{\"status\":\"active\"}"));
        record_metadata.insert("mentions".to_owned(), json!(mentions.to_string()));
        if let Some(graph_space) = &scope.graph_space {
            record_metadata.insert("graph_space".to_owned(), json!(graph_space));
        }
        if let Some(workspace_id) = &scope.workspace_id {
            record_metadata.insert("workspace_id".to_owned(), json!(workspace_id));
        }
        let event_id = kogwistar_contracts::stable_id(
            "conversation.create.event",
            &[conversation_id.clone(), start_id.clone()],
        )
        .to_string();
        let mutation = GraphMutation {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: event_id.clone(),
            op: "ADD".to_owned(),
            payload: payload.clone(),
            record: GraphRecord {
                id: start_id.clone(),
                document: Some(document),
                metadata: record_metadata,
                embedding: None,
            },
            embedding_dim: 0,
        };
        let owner_acl_id = user_id.to_ascii_lowercase();
        let acl_scope = scope_values["security_scope"]
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .map(str::to_ascii_lowercase);
        let response_conversation_id = conversation_id.clone();
        let response_start_id = start_id.clone();
        let transaction = self.store.immediate_transaction(|uow| {
            if !uow
                .latest_entity_events_by_metadata(
                    &scope.namespace,
                    "node",
                    "conversation_id",
                    &conversation_id,
                )?
                .is_empty()
            {
                return Err(SqliteStoreError::TransactionAborted(
                    "conversation_id already exists".to_owned(),
                ));
            }
            if uow
                .latest_entity_events(&scope.namespace, "node", std::slice::from_ref(&start_id))?
                .contains_key(&start_id)
            {
                return Err(SqliteStoreError::TransactionAborted(
                    "start_node_id already exists".to_owned(),
                ));
            }
            let targets = acl_targets_for_graph_mutations(std::slice::from_ref(&mutation)).0;
            let target_ids = acl_target_entity_ids(&targets);
            let existing =
                uow.graph_acl_records_for_entities(&scope, "conversation", &target_ids)?;
            let (acl_mutations, mut index_jobs) = derive_default_acl_graph_effects_from_records(
                &scope,
                std::slice::from_ref(&mutation),
                targets,
                existing,
                "conversation",
                &owner_acl_id,
                acl_scope.as_deref(),
                &event_id,
            )
            .map_err(SqliteStoreError::TransactionAborted)?;
            index_jobs.push(NewIndexJob {
                job_id: kogwistar_contracts::stable_id(
                    "conversation.create.index_job",
                    &[event_id.clone(), start_id.clone()],
                )
                .to_string(),
                namespace: scope.namespace.clone(),
                entity_kind: "node".to_owned(),
                entity_id: start_id.clone(),
                index_kind: "node_embedding".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: None,
                max_retries: 10,
            });
            let mut mutations = vec![mutation.clone()];
            mutations.extend(acl_mutations);
            uow.apply_canonical_graph_mutation_batch(GraphMutationBatch {
                mutations,
                index_jobs,
            })?;
            Ok(())
        });
        match transaction {
            Ok(()) => syscall_result(
                &input.version,
                "create_conversation",
                json_effect(
                    StatusCode::CREATED,
                    json!({
                        "conversation_id": response_conversation_id,
                        "user_id": user_id,
                        "status": "active",
                        "start_node_id": response_start_id,
                        "tail_node_id": response_start_id,
                        "turn_count": 0,
                    }),
                ),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_CONVERSATION_CREATE_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn sqlite_chroma_send_message_effect(
        &self,
        request: &ApiEffectRequest,
        input: &SyscallInput,
    ) -> ApiEffectResponse {
        if !syscall_allowed_in_namespace(
            request,
            &self.capabilities,
            "send_message",
            true,
            &["send_message"],
            "conversation",
        ) {
            return syscall_forbidden("send_message");
        }
        let conversation_id = input
            .args
            .get("conversation_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(conversation_id) = conversation_id else {
            return effect_error(
                StatusCode::BAD_REQUEST,
                "KOGWISTAR_INVALID_REQUEST",
                "conversation_id is required",
            );
        };
        let text = input
            .args
            .get("text")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(text) = text else {
            return effect_error(
                StatusCode::BAD_REQUEST,
                "KOGWISTAR_INVALID_REQUEST",
                "text must be non-empty",
            );
        };
        let workflow_id = input
            .args
            .get("workflow_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("agentic_answering.v2");
        let requested_user_id = match input.args.get("user_id") {
            None | Some(Value::Null) => None,
            Some(Value::String(value)) if value.trim().is_empty() => None,
            Some(Value::String(value)) => Some(value.trim()),
            Some(_) => {
                return effect_error(
                    StatusCode::BAD_REQUEST,
                    "KOGWISTAR_INVALID_REQUEST",
                    "user_id must be a string when provided",
                );
            }
        };
        let (scope, acl_principal) = conversation_graph_scope_and_acl_principal(&request.principal);
        if scope.namespace != "conversation" {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "conversation message admission requires conversation storage scope",
            );
        }
        let graph_snapshot = match self.chroma_graph_snapshot(request, "conversation").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        // Keep the typed projection alive while selecting this conversation;
        // it carries the ACL index derived from the complete snapshot.
        let conversation_nodes = graph_snapshot
            .nodes()
            .iter()
            .filter(|record| {
                conversation_record_string(record, "conversation_id").as_deref()
                    == Some(conversation_id)
                    && !matches!(
                        conversation_record_string(record, "lifecycle_status").as_deref(),
                        Some("tombstoned" | "deleted")
                    )
            })
            .collect::<Vec<_>>();
        let starts = conversation_nodes
            .iter()
            .filter(|record| {
                conversation_record_string(record, "entity_type").as_deref()
                    == Some("conversation_start")
            })
            .collect::<Vec<_>>();
        let Some(start) = starts.first() else {
            return effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_CONVERSATION_NOT_FOUND",
                format!("Unknown conversation_id: {conversation_id}"),
            );
        };
        if starts.len() != 1 {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_CONVERSATION_INVALID",
                "conversation must have exactly one visible conversation_start node",
            );
        }
        let Some(owner_id) =
            conversation_record_string(start, "user_id").filter(|value| !value.trim().is_empty())
        else {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_CONVERSATION_INVALID",
                "conversation owner is missing",
            );
        };
        let actor_id = authenticated_run_owner_id(&request.principal);
        if requested_user_id.or(actor_id) != Some(owner_id.as_str()) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "user_id must match the owner of the target conversation",
            );
        }
        let Some(auth_store) = self.owner_auth_store.as_ref() else {
            return effect_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "KOGWISTAR_OWNER_REVALIDATION_UNAVAILABLE",
                "conversation owner revalidation is not configured",
            );
        };
        let owner_user = match auth_store.auth_user(&owner_id).await {
            Ok(Some(user)) if user.is_active => user,
            Ok(_) => {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                    "conversation owner is missing or inactive",
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "KOGWISTAR_OWNER_REVALIDATION_UNAVAILABLE",
                    error,
                );
            }
        };
        let mut owner_principal = runtime_principal_from_auth_user(&owner_user);
        if !principal_strings(&owner_principal, "ns")
            .iter()
            .any(|namespace| namespace == &scope.namespace || namespace == "*")
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "conversation owner is not authorized for the conversation namespace",
            );
        }
        owner_principal["ns"] = json!([scope.namespace]);
        owner_principal["storage_ns"] = json!(scope.namespace);
        if let Some(workspace_id) = &scope.workspace_id {
            owner_principal["workspace"] = json!(workspace_id);
        }
        if let Some(graph_space) = &scope.graph_space {
            owner_principal["graph_space"] = json!(graph_space);
        }
        if let Some(security_scope) = &acl_principal.security_scope {
            owner_principal["security_scope"] = json!(security_scope);
        }
        let Some(owner_binding) = runtime_owner_binding_v1(&owner_principal, &self.capabilities)
        else {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "conversation owner has no authenticated run identity",
            );
        };
        owner_principal = match revalidated_runtime_owner_principal(
            &owner_binding,
            &owner_user,
            &self.capabilities,
            "write_graph",
        ) {
            Ok(principal) => principal,
            Err(error) => {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                    error,
                );
            }
        };
        let Some(provider) = self.embedding_provider.as_ref() else {
            return effect_error(
                StatusCode::BAD_GATEWAY,
                "KOGWISTAR_EMBEDDING_FAILED",
                "conversation message admission requires a configured embedding provider",
            );
        };
        let profile = provider.profile();
        if let Err(error) = profile.validate() {
            return effect_error(
                StatusCode::BAD_GATEWAY,
                "KOGWISTAR_EMBEDDING_FAILED",
                error.to_string(),
            );
        }
        let turn_id = uuid::Uuid::new_v4().to_string();
        let run_id = uuid::Uuid::new_v4().to_string();
        let mut tail = conversation_nodes
            .iter()
            .filter(|record| {
                matches!(
                    conversation_record_string(record, "entity_type").as_deref(),
                    Some(
                        "conversation_start"
                            | "conversation_turn"
                            | "conversation_summary"
                            | "assistant_turn"
                    )
                ) && conversation_record_value(record, "in_conversation_chain")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(true)
            })
            .cloned()
            .collect::<Vec<_>>();
        tail.sort_by_key(|record| {
            (
                conversation_record_i64(record, "turn_index").unwrap_or(-1),
                record.id.clone(),
            )
        });
        let previous = tail.last();
        let turn_index = previous
            .and_then(|record| conversation_record_i64(record, "turn_index"))
            .map(|index| index + 1)
            .unwrap_or(0);
        let step_seq = previous
            .and_then(|record| conversation_record_i64(record, "run_step_seq"))
            .unwrap_or(0)
            + 1;
        let node_metadata = json!({
            "entity_type": "conversation_turn",
            "turn_index": turn_index,
            "level_from_root": 0,
            "in_conversation_chain": true,
            "in_ui_chain": true,
            "run_id": conversation_id,
            "run_step_seq": step_seq,
            "attempt_seq": 0,
            "lifecycle_status": "active",
            "redirect_to_id": null,
            "role": "user",
            "conversation_id": conversation_id,
            "user_id": owner_id,
        });
        let node_mentions = json!([{"spans": [conversation_span(conversation_id, &turn_id, text)], "multimodal_spans": []}]);
        let node_payload = json!({
            "id": turn_id,
            "label": format!("Turn {turn_index} (user)"),
            "type": "entity",
            "summary": text,
            "domain_id": null,
            "canonical_entity_id": null,
            "properties": {},
            "mentions": node_mentions,
            "doc_id": turn_id,
            "metadata": node_metadata,
            "level_from_root": 0,
            "role": "user",
            "turn_index": turn_index,
            "conversation_id": conversation_id,
            "user_id": owner_id,
        });
        let node_envelope = kogwistar_contracts::GraphModelEnvelope::new(
            kogwistar_contracts::GraphModelClass::ConversationNode,
            node_payload.clone(),
        );
        if let Err(error) = node_envelope.validate_family_v1() {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_CONVERSATION_MODEL_INVALID",
                error.to_string(),
            );
        }
        let mut node_document_payload = node_payload.clone();
        node_document_payload
            .as_object_mut()
            .expect("conversation turn payload is an object")
            .remove("metadata");
        let node_document = serde_json::to_string(&node_document_payload)
            .expect("conversation turn document serializes");
        let mut node_projection = node_metadata.as_object().unwrap().clone();
        node_projection.extend([
            ("doc_id".to_owned(), json!(turn_id)),
            (
                "label".to_owned(),
                json!(format!("Turn {turn_index} (user)")),
            ),
            ("type".to_owned(), json!("entity")),
            ("summary".to_owned(), json!(text)),
            ("properties".to_owned(), json!("{}")),
            ("mentions".to_owned(), json!(node_mentions.to_string())),
        ]);
        if let Some(graph_space) = &scope.graph_space {
            node_projection.insert("graph_space".to_owned(), json!(graph_space));
        }
        if let Some(workspace_id) = &scope.workspace_id {
            node_projection.insert("workspace_id".to_owned(), json!(workspace_id));
        }
        let mut mutations = vec![(
            GraphMutation {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: uuid::Uuid::new_v4().to_string(),
                op: "ADD".to_owned(),
                payload: node_payload,
                record: GraphRecord {
                    id: turn_id.clone(),
                    document: Some(node_document),
                    metadata: node_projection,
                    embedding: None,
                },
                embedding_dim: profile.dimension,
            },
            node_envelope,
        )];
        let mut index_jobs = conversation_index_job_kinds("node")
            .expect("node conversation indexing contract exists")
            .iter()
            .map(|kind| conversation_index_job(&scope.namespace, "node", &turn_id, kind))
            .collect::<Vec<_>>();
        let mut edge_payload_and_document = None;
        if let Some(previous) = previous {
            let previous_id = previous.id.clone();
            let (char_distance, turn_distance, tail_turn_index) =
                conversation_next_turn_distances(previous, text, turn_index);
            let edge_id = kogwistar_contracts::stable_id(
                "conversation.edge",
                &[
                    owner_id.clone(),
                    conversation_id.to_owned(),
                    "next_turn".to_owned(),
                    (turn_index + 1).to_string(),
                    format!("['{previous_id}']"),
                    format!("['{turn_id}']"),
                    "[]".to_owned(),
                    "[]".to_owned(),
                    "conversation_edge".to_owned(),
                ],
            )
            .to_string();
            let edge_payload = json!({
                "id": edge_id,
                "source_ids": [previous_id],
                "target_ids": [turn_id],
                "source_edge_ids": [],
                "target_edge_ids": [],
                "relation": "next_turn",
                "label": "next_turn",
                "type": "relationship",
                "summary": "Sequential flow",
                "doc_id": format!("conv:{conversation_id}"),
                "mentions": node_mentions,
                "domain_id": null,
                "canonical_entity_id": null,
                "properties": {"entity_type": "conversation_edge"},
                "metadata": {
                    "relation": "next_turn",
                    "target_id": turn_id,
                    "char_distance_from_last_summary": char_distance,
                    "turn_distance_from_last_summary": turn_distance,
                    "tail_turn_index": tail_turn_index,
                    "causal_type": "chain",
                    "run_id": conversation_id,
                    "run_step_seq": step_seq,
                    "attempt_seq": 0,
                    "lifecycle_status": "active",
                    "redirect_to_id": null,
                },
            });
            let edge_envelope = kogwistar_contracts::GraphModelEnvelope::new(
                kogwistar_contracts::GraphModelClass::ConversationEdge,
                edge_payload.clone(),
            );
            if let Err(error) = edge_envelope.validate_family_v1() {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_CONVERSATION_MODEL_INVALID",
                    error.to_string(),
                );
            }
            let edge_document = serde_json::to_string(&edge_payload)
                .expect("conversation edge document serializes");
            edge_payload_and_document = Some((edge_payload, edge_document, edge_envelope));
        }
        let embedding_texts = edge_payload_and_document
            .as_ref()
            .map(|(_, document, _)| vec![format!("user: {text}"), document.clone()])
            .unwrap_or_else(|| vec![format!("user: {text}")]);
        let embeddings = match provider.embed(&embedding_texts).await {
            Ok(vectors)
                if vectors.len() == embedding_texts.len()
                    && vectors.iter().all(|vector| {
                        vector.len() == profile.dimension && vector.iter().all(|v| v.is_finite())
                    }) =>
            {
                vectors
            }
            Ok(_) => {
                return effect_error(
                    StatusCode::BAD_GATEWAY,
                    "KOGWISTAR_EMBEDDING_FAILED",
                    "embedding provider returned an invalid conversation vector",
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::BAD_GATEWAY,
                    "KOGWISTAR_EMBEDDING_FAILED",
                    error.to_string(),
                );
            }
        };
        mutations[0].0.record.embedding =
            Some(embeddings[0].iter().map(|value| *value as f32).collect());
        if let Some((edge_payload, edge_document, edge_envelope)) = edge_payload_and_document {
            let edge_id = edge_payload["id"].as_str().unwrap().to_owned();
            let mut metadata = edge_payload["metadata"].as_object().unwrap().clone();
            metadata.extend([
                ("doc_id".to_owned(), edge_payload["doc_id"].clone()),
                ("relation".to_owned(), json!("next_turn")),
                (
                    "source_ids".to_owned(),
                    json!(edge_payload["source_ids"].to_string()),
                ),
                (
                    "target_ids".to_owned(),
                    json!(edge_payload["target_ids"].to_string()),
                ),
                ("source_edge_ids".to_owned(), json!("[]")),
                ("target_edge_ids".to_owned(), json!("[]")),
                ("type".to_owned(), json!("relationship")),
                ("summary".to_owned(), json!("Sequential flow")),
                (
                    "properties".to_owned(),
                    json!(edge_payload["properties"].to_string()),
                ),
                (
                    "mentions".to_owned(),
                    json!(edge_payload["mentions"].to_string()),
                ),
                ("node_endpoint_count".to_owned(), json!(2)),
                ("edge_endpoint_count".to_owned(), json!(0)),
                ("total_endpoint_count".to_owned(), json!(2)),
            ]);
            if let Some(graph_space) = &scope.graph_space {
                metadata.insert("graph_space".to_owned(), json!(graph_space));
            }
            if let Some(workspace_id) = &scope.workspace_id {
                metadata.insert("workspace_id".to_owned(), json!(workspace_id));
            }
            mutations.push((
                GraphMutation {
                    scope: scope.clone(),
                    table: "gke_edges".to_owned(),
                    entity_kind: "edge".to_owned(),
                    event_id: uuid::Uuid::new_v4().to_string(),
                    op: "ADD".to_owned(),
                    payload: edge_payload,
                    record: GraphRecord {
                        id: edge_id.clone(),
                        document: Some(edge_document),
                        metadata,
                        embedding: Some(embeddings[1].iter().map(|value| *value as f32).collect()),
                    },
                    embedding_dim: profile.dimension,
                },
                edge_envelope,
            ));
            for kind in conversation_index_job_kinds("edge")
                .expect("edge conversation indexing contract exists")
            {
                index_jobs.push(conversation_index_job(
                    &scope.namespace,
                    "edge",
                    &edge_id,
                    kind,
                ));
            }
        }
        let prepared = PreparedConversationAdmission {
            conversation_id: conversation_id.to_owned(),
            owner_id: owner_id.clone(),
            expected_tail_id: previous.map(|record| record.id.clone()),
            scope: scope.clone(),
            acl_principal,
            mutations,
            index_jobs,
            actor_user_id: actor_id.unwrap_or(&owner_id).to_owned(),
        };
        let initial_state = serde_json::Map::from_iter([
            ("conversation_id".to_owned(), json!(conversation_id)),
            ("user_text".to_owned(), json!(text)),
            ("turn_node_id".to_owned(), json!(turn_id)),
            ("turn_index".to_owned(), json!(turn_index)),
            ("user_id".to_owned(), json!(owner_id)),
            ("max_retrieval_level".to_owned(), json!(4)),
        ]);
        let nested = ApiEffectRequest {
            contract_version: request.contract_version,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": run_id,
                "workflow_id": workflow_id,
                "conversation_id": conversation_id,
                "turn_node_id": turn_id,
                "user_id": owner_id,
                "initial_state": initial_state,
            }))
            .expect("conversation message workflow request serializes"),
            principal: owner_principal,
        };
        let response = self.submit_run_with_conversation_admission(&nested, Some(prepared));
        syscall_result(&input.version, "send_message", response)
    }

    pub fn with_embedding_provider(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(provider);
        self
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
            .ok_or_else(|| InlineOwnerPrincipalError::Denied("native worker run is missing".to_owned()))?;
        let owner_id = run
            .user_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| InlineOwnerPrincipalError::Denied("native worker run has no authenticated owner".to_owned()))?;
        let admission = self
            .store
            .list_server_run_events(run_id, 0, 1)
            .map_err(|error| InlineOwnerPrincipalError::Unavailable(error.to_string()))?
            .into_iter()
            .find(|event| event.event_type == "run.created")
            .ok_or_else(|| InlineOwnerPrincipalError::Denied("native worker run has no admission event".to_owned()))?;
        let admission_payload: Value = serde_json::from_str(&admission.payload_json)
            .map_err(|_| InlineOwnerPrincipalError::Denied("native worker admission event is invalid".to_owned()))?;
        let binding: RuntimeOwnerBindingV1 = serde_json::from_value(
            admission_payload
                .get("runtime_owner_binding")
                .cloned()
                .ok_or_else(|| InlineOwnerPrincipalError::Denied("native worker run has no trusted owner binding".to_owned()))?,
        )
        .map_err(|_| InlineOwnerPrincipalError::Denied("native worker owner binding is invalid".to_owned()))?;
        if binding.user_id != owner_id {
            return Err(InlineOwnerPrincipalError::Denied(
                "native worker owner binding does not match the durable run owner".to_owned(),
            ));
        }
        let auth_store = self
            .owner_auth_store
            .as_ref()
            .ok_or_else(|| InlineOwnerPrincipalError::Unavailable("native worker owner revalidation is not configured".to_owned()))?;
        let user = auth_store
            .auth_user(owner_id)
            .await
            .map_err(InlineOwnerPrincipalError::Unavailable)?
            .filter(|user| user.is_active)
            .ok_or_else(|| InlineOwnerPrincipalError::Denied("native worker run owner is missing or inactive".to_owned()))?;
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
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        Ok(GraphAclProjection::from_complete_snapshot(
            &nodes,
            &edges,
            truth_graph,
            &principal,
        ))
    }

    async fn native_answer_view_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        caller: Option<&Value>,
    ) -> Result<Value, NativeEvidenceContextError> {
        let run = self
            .store
            .get_server_run(&request.run_id)
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        let owner_id = run
            .user_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        if let Some(caller) = caller {
            if authenticated_run_owner_id(caller) != Some(owner_id) {
                return Err(NativeEvidenceContextError::CallerNotRunOwner);
            }
        }
        let mut principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "read_graph")
            .await
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?;
        principal["actor_id"] = json!(owner_id);

        let turn_id = request
            .turn_node_id
            .as_deref()
            .or_else(|| request.state.get("turn_node_id").and_then(Value::as_str))
            .filter(|id| !id.trim().is_empty())
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        if request
            .state
            .get("turn_node_id")
            .and_then(Value::as_str)
            .is_some_and(|state_turn_id| state_turn_id != turn_id)
        {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }

        let graph_request = ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: String::new(),
            body: Vec::new(),
            principal,
        };
        let graph_snapshot = self
            .chroma_graph_snapshot(&graph_request, "conversation")
            .await
            .map_err(|response| {
                if response.status >= StatusCode::INTERNAL_SERVER_ERROR.as_u16() {
                    NativeEvidenceContextError::GraphReadUnavailable
                } else {
                    NativeEvidenceContextError::GraphReadRejected
                }
            })?;
        sqlite_native_answer_view_from_visible_records(
            graph_snapshot.nodes(),
            graph_snapshot.edges(),
            &request.conversation_id,
            turn_id,
        )
    }

    async fn native_candidate_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        caller: Option<&Value>,
    ) -> Result<Value, NativeEvidenceContextError> {
        let run = self
            .store
            .get_server_run(&request.run_id)
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        let owner_id = run
            .user_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        if let Some(caller) = caller {
            if authenticated_run_owner_id(caller) != Some(owner_id) {
                return Err(NativeEvidenceContextError::CallerNotRunOwner);
            }
        }
        let mut principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "read_graph")
            .await
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?;
        principal["actor_id"] = json!(owner_id);

        let question = request
            .state
            .get("question")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty() && value.len() <= 100_000)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let max_candidates = request
            .state
            .get("max_candidates")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value <= 200)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let max_retrieval_level = request
            .state
            .get("max_retrieval_level")
            .and_then(Value::as_i64)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        if max_candidates == 0 {
            return Ok(Value::Array(Vec::new()));
        }
        // Chroma numeric filters use f64; reject bounds that cannot be represented exactly.
        if max_retrieval_level.unsigned_abs() > (1_u64 << 53) {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let provider = self
            .embedding_provider
            .as_ref()
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let profile = provider.profile();
        profile
            .validate()
            .map_err(|_| NativeEvidenceContextError::GraphReadRejected)?;
        if profile.dimension > 8192 {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let mut vectors = provider
            .embed(&[question.to_owned()])
            .await
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        if vectors.len() != 1 || vectors[0].len() != profile.dimension {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let metric = match profile.similarity_metric.to_ascii_lowercase().as_str() {
            "cosine" => DistanceMetric::Cosine,
            "l2" => DistanceMetric::L2,
            "ip" => DistanceMetric::InnerProduct,
            _ => return Err(NativeEvidenceContextError::GraphReadRejected),
        };
        let embedding = vectors
            .pop()
            .expect("the embedding count was validated")
            .into_iter()
            .map(|value| {
                let value = value as f32;
                value
                    .is_finite()
                    .then_some(value)
                    .ok_or(NativeEvidenceContextError::GraphReadRejected)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let scope_value = security_scope_value(&principal);
        let namespace = scope_value["storage_namespace"]
            .as_str()
            .unwrap_or("default")
            .to_owned();
        let mut metadata = MetadataFilter::default();
        if let Some(workspace) = principal
            .get("workspace_id")
            .or_else(|| principal.get("workspace"))
            .and_then(Value::as_str)
        {
            metadata
                .equals
                .insert("workspace_id".to_owned(), json!(workspace));
        }
        if let Some(graph_space) = principal.get("graph_space").and_then(Value::as_str) {
            metadata
                .equals
                .insert("graph_space".to_owned(), json!(graph_space));
        }
        let graph = self
            .chroma_graph
            .as_ref()
            .ok_or(NativeEvidenceContextError::GraphReadUnavailable)?;
        let candidates = graph
            .vector_query_of_kind_numeric_lte(
                &namespace,
                GraphEntityKind::Node,
                &VectorQuery {
                    embedding,
                    limit: max_candidates,
                    metadata,
                    metric,
                },
                "level_from_root",
                max_retrieval_level as f64,
            )
            .await
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        let mut candidate_records = candidates
            .into_iter()
            .map(|matched| matched.record)
            .collect::<Vec<_>>();
        let mut no_edges = Vec::new();
        self.filter_current_chroma_stage2(&namespace, &mut candidate_records, &mut no_edges)
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;

        let mut acl_filter = MetadataFilter {
            equals: BTreeMap::from([
                ("entity_type".to_owned(), json!("acl_record")),
                ("acl_truth_graph".to_owned(), json!("knowledge")),
            ]),
        };
        if let Some(workspace) = principal
            .get("workspace_id")
            .or_else(|| principal.get("workspace"))
            .and_then(Value::as_str)
        {
            acl_filter
                .equals
                .insert("workspace_id".to_owned(), json!(workspace));
        }
        if let Some(graph_space) = principal.get("graph_space").and_then(Value::as_str) {
            acl_filter
                .equals
                .insert("graph_space".to_owned(), json!(graph_space));
        }
        let mut acl_records = graph
            .graph_records_of_kind_bounded(&namespace, GraphEntityKind::Node, &acl_filter, 10_000)
            .await
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        self.filter_current_chroma_stage2(&namespace, &mut acl_records, &mut Vec::new())
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        let acl_principal =
            graph_acl_principal_from_claims(&principal, scope_value["security_scope"].as_str());
        Ok(sqlite_native_candidate_values(
            &candidate_records,
            &acl_records,
            "knowledge",
            &acl_principal,
        ))
    }

    async fn native_evidence_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        caller: Option<&Value>,
    ) -> Result<Value, NativeEvidenceContextError> {
        let run = self
            .store
            .get_server_run(&request.run_id)
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        let owner_id = run
            .user_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        if caller.is_some_and(|caller| authenticated_run_owner_id(caller) != Some(owner_id)) {
            return Err(NativeEvidenceContextError::CallerNotRunOwner);
        }
        let mut principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "read_graph")
            .await
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?;
        principal["actor_id"] = json!(owner_id);

        let read_ids = |field: &str| -> Result<Vec<String>, NativeEvidenceContextError> {
            let Some(value) = request.state.get(field) else {
                return Ok(Vec::new());
            };
            let values = value
                .as_array()
                .filter(|values| values.len() <= 100)
                .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
            let ids = values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .filter(|id| !id.trim().is_empty())
                        .map(str::to_owned)
                        .ok_or(NativeEvidenceContextError::GraphReadRejected)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
                return Err(NativeEvidenceContextError::GraphReadRejected);
            }
            Ok(ids)
        };
        let node_ids = read_ids("used_node_ids")?;
        let edge_ids = read_ids("used_edge_ids")?;
        if node_ids.len().saturating_add(edge_ids.len()) > 100 {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }

        let graph_request = ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: String::new(),
            body: Vec::new(),
            principal,
        };
        let snapshot = self
            .chroma_graph_snapshot(&graph_request, "knowledge")
            .await
            .map_err(|response| {
                if response.status >= StatusCode::INTERNAL_SERVER_ERROR.as_u16() {
                    NativeEvidenceContextError::GraphReadUnavailable
                } else {
                    NativeEvidenceContextError::GraphReadRejected
                }
            })?;
        let selected_nodes = snapshot
            .nodes()
            .iter()
            .filter(|record| node_ids.contains(&record.id))
            .collect::<Vec<_>>();
        let selected_edges = snapshot
            .edges()
            .iter()
            .filter(|record| edge_ids.contains(&record.id))
            .collect::<Vec<_>>();
        if selected_nodes.len() != node_ids.len() || selected_edges.len() != edge_ids.len() {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let project = |records: Vec<&GraphRecord>| {
            records
                .into_iter()
                .map(|record| {
                    json!({"id":record.id,"document":record.document,"metadata":record.metadata})
                })
                .collect::<Vec<_>>()
        };
        Ok(json!({"nodes":project(selected_nodes),"edges":project(selected_edges)}))
    }

    async fn native_pointer_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        caller: Option<&Value>,
    ) -> Result<Value, NativeEvidenceContextError> {
        let run = self
            .store
            .get_server_run(&request.run_id)
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        let owner_id = run
            .user_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        if caller.is_some_and(|caller| authenticated_run_owner_id(caller) != Some(owner_id)) {
            return Err(NativeEvidenceContextError::CallerNotRunOwner);
        }
        let mut principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "read_graph")
            .await
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?;
        principal["actor_id"] = json!(owner_id);

        let read_ids = |field: &str| -> Result<Vec<String>, NativeEvidenceContextError> {
            let Some(value) = request.state.get(field) else {
                return Ok(Vec::new());
            };
            value
                .as_array()
                .filter(|values| values.len() <= 100)
                .ok_or(NativeEvidenceContextError::GraphReadRejected)?
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .filter(|id| !id.trim().is_empty())
                        .map(str::to_owned)
                        .ok_or(NativeEvidenceContextError::GraphReadRejected)
                })
                .collect()
        };
        let used_ids = read_ids("used_node_ids")?;
        let conversation_id = request
            .state
            .get("conversation_id")
            .and_then(Value::as_str)
            .filter(|id| *id == request.conversation_id)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let run_node_id = request
            .state
            .get("run_node_id")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let pointer_ids = used_ids
            .iter()
            .map(|id| {
                crate::native_answer_persist::stable_conversation_id(
                    "ptr",
                    &[
                        ("scope", &format!("conv:{conversation_id}")),
                        ("pointer_kind", "kg_node"),
                        ("target_kind", "node"),
                        ("target_id", id),
                    ],
                )
            })
            .collect::<BTreeSet<_>>();
        let mut wanted_nodes = pointer_ids.clone();
        wanted_nodes.insert(run_node_id.to_owned());
        let wanted_edges = pointer_ids
            .iter()
            .map(|pointer_id| {
                crate::native_answer_persist::stable_conversation_id(
                    "e",
                    &[
                        ("scope", &format!("conv:{conversation_id}")),
                        ("rel", "used_evidence"),
                        ("src", run_node_id),
                        ("dst", pointer_id),
                    ],
                )
            })
            .collect::<BTreeSet<_>>();

        let scope = security_scope_value(&principal);
        let namespace = scope["storage_namespace"].as_str().unwrap_or("default");
        let mut filter = MetadataFilter::default();
        if let Some(workspace) = principal
            .get("workspace_id")
            .or_else(|| principal.get("workspace"))
            .and_then(Value::as_str)
        {
            filter
                .equals
                .insert("workspace_id".to_owned(), json!(workspace));
        }
        if let Some(graph_space) = principal.get("graph_space").and_then(Value::as_str) {
            filter
                .equals
                .insert("graph_space".to_owned(), json!(graph_space));
        }
        let graph = self
            .chroma_graph
            .as_ref()
            .ok_or(NativeEvidenceContextError::GraphReadUnavailable)?;
        const MAX_POINTER_SNAPSHOT_RECORDS: usize = 10_000;
        let (mut nodes, mut edges) = tokio::try_join!(
            graph.graph_records_of_kind_bounded(
                namespace,
                GraphEntityKind::Node,
                &filter,
                MAX_POINTER_SNAPSHOT_RECORDS,
            ),
            graph.graph_records_of_kind_bounded(
                namespace,
                GraphEntityKind::Edge,
                &filter,
                MAX_POINTER_SNAPSHOT_RECORDS,
            ),
        )
        .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        self.filter_current_chroma_stage2(namespace, &mut nodes, &mut edges)
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        for (kind, records) in [("node", &mut nodes), ("edge", &mut edges)] {
            self.merge_current_stage1_records(
                namespace,
                kind,
                &filter,
                records,
                MAX_POINTER_SNAPSHOT_RECORDS,
            )
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        }
        let acl_principal =
            graph_acl_principal_from_claims(&principal, scope["security_scope"].as_str());
        let knowledge =
            GraphAclProjection::from_complete_snapshot(&nodes, &edges, "knowledge", &acl_principal);
        let conversation = GraphAclProjection::from_complete_snapshot(
            &nodes,
            &edges,
            "conversation",
            &acl_principal,
        );
        let selected = used_ids.iter().cloned().collect::<BTreeSet<_>>();
        let knowledge_records = knowledge
            .nodes()
            .iter()
            .filter(|record| selected.contains(&record.id))
            .collect::<Vec<_>>();
        let visible_knowledge = knowledge_records
            .iter()
            .filter(|record| {
                !matches!(
                    record
                        .metadata
                        .get("lifecycle_status")
                        .and_then(Value::as_str),
                    Some("tombstoned" | "deleted")
                )
            })
            .map(|record| record.id.clone())
            .collect::<BTreeSet<_>>();
        if visible_knowledge != selected {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let conversation_nodes = conversation
            .nodes()
            .iter()
            .filter(|record| wanted_nodes.contains(&record.id))
            .collect::<Vec<_>>();
        let conversation_edges = conversation
            .edges()
            .iter()
            .filter(|record| wanted_edges.contains(&record.id))
            .collect::<Vec<_>>();
        let raw_node_ids = nodes
            .iter()
            .filter(|record| wanted_nodes.contains(&record.id))
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();
        let raw_edge_ids = edges
            .iter()
            .filter(|record| wanted_edges.contains(&record.id))
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();
        if conversation_nodes
            .iter()
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>()
            != raw_node_ids
            || conversation_edges
                .iter()
                .map(|record| record.id.as_str())
                .collect::<BTreeSet<_>>()
                != raw_edge_ids
        {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let run_record = conversation_nodes
            .iter()
            .find(|record| record.id == run_node_id)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        if run_record
            .metadata
            .get("entity_type")
            .and_then(Value::as_str)
            != Some("agent_run")
            || run_record
                .metadata
                .get("conversation_id")
                .and_then(Value::as_str)
                != Some(conversation_id)
            || matches!(
                run_record
                    .metadata
                    .get("lifecycle_status")
                    .and_then(Value::as_str),
                Some("tombstoned" | "deleted")
            )
        {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let project = |records: Vec<&GraphRecord>| {
            records
                .into_iter()
                .map(|record| {
                    json!({"id": record.id, "document": record.document, "metadata": record.metadata})
                })
                .collect::<Vec<_>>()
        };
        Ok(json!({
            "knowledge_nodes": project(knowledge_records),
            "conversation_nodes": project(conversation_nodes),
            "conversation_edges": project(conversation_edges)
        }))
    }

    async fn native_turn_link_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        caller: Option<&Value>,
    ) -> Result<Value, NativeEvidenceContextError> {
        let response_id = request
            .state
            .get("answer")
            .and_then(Value::as_object)
            .and_then(|answer| {
                answer
                    .get("response_node_id")
                    .or_else(|| answer.get("assistant_turn_node_id"))
            })
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
            .or_else(|| {
                request
                    .state
                    .get("agentic_answering_result")
                    .and_then(Value::as_object)
                    .and_then(|result| {
                        result
                            .get("assistant_turn_node_id")
                            .or_else(|| result.get("response_node_id"))
                    })
                    .and_then(Value::as_str)
                    .filter(|id| !id.trim().is_empty())
            });
        let Some(response_id) = response_id else {
            return Ok(json!({"nodes": [], "edges": []}));
        };
        let user_turn_id = request
            .state
            .get("turn_node_id")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let conversation_id = request
            .state
            .get("conversation_id")
            .and_then(Value::as_str)
            .filter(|id| *id == request.conversation_id)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let run = self
            .store
            .get_server_run(&request.run_id)
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        let owner_id = run
            .user_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        if caller.is_some_and(|caller| authenticated_run_owner_id(caller) != Some(owner_id)) {
            return Err(NativeEvidenceContextError::CallerNotRunOwner);
        }
        let mut principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "read_graph")
            .await
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?;
        principal["actor_id"] = json!(owner_id);
        let graph_request = ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: String::new(),
            body: Vec::new(),
            principal,
        };
        let snapshot = self
            .chroma_graph_snapshot(&graph_request, "conversation")
            .await
            .map_err(|response| {
                if response.status >= StatusCode::INTERNAL_SERVER_ERROR.as_u16() {
                    NativeEvidenceContextError::GraphReadUnavailable
                } else {
                    NativeEvidenceContextError::GraphReadRejected
                }
            })?;
        let nodes = snapshot
            .nodes()
            .iter()
            .filter(|record| record.id == user_turn_id || record.id == response_id)
            .collect::<Vec<_>>();
        if nodes.len() != 2 {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        for (expected_id, expected_type) in [
            (user_turn_id, "conversation_turn"),
            (response_id, "assistant_turn"),
        ] {
            let record = nodes
                .iter()
                .find(|record| record.id == expected_id)
                .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
            let actual_type = record.metadata.get("entity_type").and_then(Value::as_str);
            let type_matches = actual_type == Some(expected_type)
                || (expected_type == "assistant_turn" && actual_type == Some("conversation_turn"));
            if !type_matches
                || record
                    .metadata
                    .get("conversation_id")
                    .and_then(Value::as_str)
                    != Some(conversation_id)
                || matches!(
                    record
                        .metadata
                        .get("lifecycle_status")
                        .and_then(Value::as_str),
                    Some("tombstoned" | "deleted")
                )
            {
                return Err(NativeEvidenceContextError::GraphReadRejected);
            }
        }
        Ok(json!({
            "nodes": nodes.into_iter().map(|record| json!({
                "id": record.id,
                "document": record.document,
                "metadata": record.metadata
            })).collect::<Vec<_>>(),
            "edges": []
        }))
    }

    async fn persist_sqlite_native_answer_snapshot(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> Result<(), String> {
        if self.chroma_graph.is_none() {
            return Err("native answer snapshots require SQLite canonical graph events with Chroma projection".to_owned());
        }
        let model = self
            .chat_completion_provider
            .as_ref()
            .ok_or_else(|| "Rust-native structured-answer provider is not configured".to_owned())?
            .model()
            .to_owned();
        let batch =
            crate::native_answer_generation::answer_context_snapshot_batch(request, &model)?;
        self.persist_sqlite_native_context_snapshot_batch(request, batch)
            .await
    }

    async fn persist_sqlite_native_selection_context_snapshot(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> Result<(), String> {
        kogwistar_runtime::validate_native_step_registration(request, &self.native_operations)
            .map_err(|_| {
                "native evidence selection is not registered for this operation version".to_owned()
            })?;
        if self.chroma_graph.is_none() {
            return Err("native selection snapshots require SQLite canonical graph events with Chroma projection".to_owned());
        }
        let batch = crate::native_answer_generation::selection_context_snapshot_batch(request)?;
        self.persist_sqlite_native_context_snapshot_batch(request, batch)
            .await
    }

    async fn persist_sqlite_native_context_snapshot_batch(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        batch: kogwistar_contracts::GraphMutationBatchV2,
    ) -> Result<(), String> {
        let (mutations, mut index_jobs) = graph_mutation_batch_v2_value(batch)
            .map_err(|_| "native context snapshot graph mutation is invalid".to_owned())?;
        validate_runtime_graph_mutations(&mutations)
            .map_err(|_| "native context snapshot graph mutation failed validation".to_owned())?;
        let principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "write_graph")
            .await?;
        if !capability_allowed_for_principal(
            &principal,
            &self.capabilities,
            "write_graph.native_context_snapshot",
            &["write_graph"],
        ) || !principal_strings(&principal, "capabilities")
            .iter()
            .any(|capability| capability == "write_graph")
        {
            return Err(
                "native context snapshot requires the run owner's current write_graph capability"
                    .to_owned(),
            );
        }
        let security_scope = security_scope_value(&principal);
        if mutations
            .iter()
            .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
        {
            return Err(
                "native context snapshot scope is outside the run owner's admitted scope"
                    .to_owned(),
            );
        }
        let Some(scope) = mutations.first().map(|mutation| mutation.scope.clone()) else {
            return Err("native context snapshot has no graph mutations".to_owned());
        };
        let owner_id = authenticated_run_owner_id(&principal)
            .ok_or_else(|| "native context snapshot requires an authenticated owner".to_owned())?
            .to_owned();
        let owner_acl_id = owner_id.to_ascii_lowercase();
        let acl_scope = principal
            .get("security_scope")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_ascii_lowercase);
        let effect_id = format!(
            "native-context-snapshot|{}|{}|{}",
            request.run_id, request.step_seq, request.token_id
        );
        let run_id = request.run_id.clone();
        let capabilities = self.capabilities.clone();
        self.store
            .immediate_transaction(move |uow| {
                let run = uow.get_server_run(&run_id)?.ok_or_else(|| {
                    SqliteStoreError::RecordedRuntimeConflict(
                        "native context snapshot run is missing".to_owned(),
                    )
                })?;
                if run.user_id.as_deref() != Some(owner_id.as_str()) {
                    return Err(SqliteStoreError::RecordedRuntimeConflict(
                        "native context snapshot run owner changed before commit".to_owned(),
                    ));
                }
                if !capability_allowed_for_principal(
                    &principal,
                    &capabilities,
                    "write_graph.native_context_snapshot.commit",
                    &["write_graph"],
                ) || !principal_strings(&principal, "capabilities")
                    .iter()
                    .any(|capability| capability == "write_graph")
                {
                    return Err(SqliteStoreError::RecordedRuntimeConflict(
                        "native context snapshot write_graph capability was revoked".to_owned(),
                    ));
                }
                if mutations
                    .iter()
                    .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
                {
                    return Err(SqliteStoreError::RecordedRuntimeConflict(
                        "native context snapshot scope changed before commit".to_owned(),
                    ));
                }
                let acl_targets = acl_targets_for_graph_mutations(&mutations).0;
                let target_ids = acl_target_entity_ids(&acl_targets);
                let existing =
                    uow.graph_acl_records_for_entities(&scope, "conversation", &target_ids)?;
                let (acl_mutations, acl_jobs) = derive_default_acl_graph_effects_from_records(
                    &scope,
                    &mutations,
                    acl_targets,
                    existing,
                    "conversation",
                    &owner_acl_id,
                    acl_scope.as_deref(),
                    &effect_id,
                )
                .map_err(SqliteStoreError::TransactionAborted)?;
                let mut all_mutations = mutations;
                all_mutations.extend(acl_mutations);
                index_jobs.extend(acl_jobs);
                uow.apply_canonical_graph_mutation_batch(GraphMutationBatch {
                    mutations: all_mutations,
                    index_jobs,
                })?;
                Ok(())
            })
            .map_err(|_| "native answer context snapshot could not be committed".to_owned())
    }

    async fn persist_sqlite_native_provider_event(
        &self,
        event: kogwistar_contracts::DurableProviderEventV1,
        conversation_id: String,
        trace_payload: Value,
    ) -> Result<(), String> {
        let run_id = event.trace.run_id.clone();
        let principal = self
            .resolve_inline_native_owner_principal(&run_id, "write_graph")
            .await?;
        if !capability_allowed_for_principal(
            &principal,
            &self.capabilities,
            "write_graph.provider_event",
            &["write_graph"],
        ) || !principal_strings(&principal, "capabilities")
            .iter()
            .any(|capability| capability == "write_graph")
        {
            return Err("provider event publication requires the run owner's current write_graph capability".to_owned());
        }
        if self.chroma_graph.is_none() {
            return Err("provider event publication requires SQLite canonical graph events with Chroma projection".to_owned());
        }
        let security_scope = security_scope_value(&principal);
        let scope = GraphScope {
            namespace: security_scope["storage_namespace"]
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| "provider event owner has no storage namespace".to_owned())?
                .to_owned(),
            workspace_id: security_scope["workspace"]
                .as_str()
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
            graph_space: Some("conversation".to_owned()),
        };
        let publication = kogwistar_contracts::DurableProviderEventPublicationV1 {
            namespace: scope.namespace.clone(),
            conversation_id: conversation_id.clone(),
            event,
        };
        let mutations = provider_event_graph_mutations(
            &publication,
            &scope,
            security_scope["security_scope"]
                .as_str()
                .unwrap_or_default(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|_| "system clock is before Unix epoch".to_owned())?
                .as_secs() as i64,
        );
        validate_provider_event_publications(
            std::slice::from_ref(&publication),
            &mutations,
            &run_id,
        )?;
        if mutations
            .iter()
            .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
        {
            return Err(
                "provider event publication is outside the run owner's admitted scope".to_owned(),
            );
        }
        let owner_id = authenticated_run_owner_id(&principal)
            .ok_or_else(|| "provider event publication requires an authenticated owner".to_owned())?
            .to_owned();
        let owner_acl_id = owner_id.to_ascii_lowercase();
        let acl_scope = principal
            .get("security_scope")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_ascii_lowercase);
        let message_id = format!("msg:{}", publication.event.event_id);
        let event_payload = serde_json::to_string(&publication.event)
            .map_err(|_| "provider event could not be serialized".to_owned())?;
        let run_event_payload = serde_json::to_string(&trace_payload)
            .map_err(|_| "provider event trace could not be serialized".to_owned())?;
        let run_event_type = "workflow.provider_event.v1";
        let capabilities = self.capabilities.clone();
        let event_run_id = run_id.clone();
        let event_conversation_id = conversation_id.clone();
        self.store
            .immediate_transaction(move |uow| {
                let run = uow.get_server_run(&event_run_id)?.ok_or_else(|| {
                    SqliteStoreError::RecordedRuntimeConflict(
                        "provider event run is missing".to_owned(),
                    )
                })?;
                if run.user_id.as_deref() != Some(owner_id.as_str())
                    || run.conversation_id != event_conversation_id
                {
                    return Err(SqliteStoreError::RecordedRuntimeConflict(
                        "provider event owner or conversation changed before commit".to_owned(),
                    ));
                }
                if !capability_allowed_for_principal(
                    &principal,
                    &capabilities,
                    "write_graph.provider_event.commit",
                    &["write_graph"],
                ) || !principal_strings(&principal, "capabilities")
                    .iter()
                    .any(|capability| capability == "write_graph")
                {
                    return Err(SqliteStoreError::RecordedRuntimeConflict(
                        "provider event write_graph capability was revoked before commit"
                            .to_owned(),
                    ));
                }
                if mutations
                    .iter()
                    .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
                {
                    return Err(SqliteStoreError::RecordedRuntimeConflict(
                        "provider event scope changed before commit".to_owned(),
                    ));
                }
                if let Some(existing) = uow.get_projected_lane_message(&message_id)? {
                    let identical = existing.namespace == publication.namespace
                        && existing.inbox_id == "agent-provider-events"
                        && existing.conversation_id == publication.conversation_id
                        && existing.run_id.as_deref() == Some(event_run_id.as_str())
                        && existing.correlation_id.as_deref()
                            == Some(publication.event.event_id.as_str())
                        && existing.payload_json.as_deref() == Some(event_payload.as_str());
                    if identical {
                        return Ok(());
                    }
                    return Err(SqliteStoreError::RecordedRuntimeConflict(
                        "provider event lane identity conflicts with existing content".to_owned(),
                    ));
                }
                let message_mutation = mutations
                    .iter()
                    .find(|mutation| {
                        mutation.entity_kind == "node" && mutation.record.id == message_id
                    })
                    .cloned()
                    .ok_or_else(|| {
                        SqliteStoreError::TransactionAborted(
                            "provider event message node is missing".to_owned(),
                        )
                    })?;
                let (acl_mutations, mut acl_jobs) = derive_default_acl_graph_effects_from_records(
                    &scope,
                    std::slice::from_ref(&message_mutation),
                    acl_targets_for_graph_mutations(std::slice::from_ref(&message_mutation)).0,
                    uow.graph_acl_records_for_entities(
                        &scope,
                        "conversation",
                        &[message_id.clone()],
                    )?,
                    "conversation",
                    &owner_acl_id,
                    acl_scope.as_deref(),
                    &publication.event.event_id,
                )
                .map_err(SqliteStoreError::TransactionAborted)?;
                let mut graph_mutations = mutations;
                graph_mutations.extend(acl_mutations);
                uow.apply_canonical_graph_mutation_batch(GraphMutationBatch {
                    mutations: graph_mutations,
                    index_jobs: std::mem::take(&mut acl_jobs),
                })?;
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|_| {
                        SqliteStoreError::TransactionAborted(
                            "system clock is before Unix epoch".to_owned(),
                        )
                    })?
                    .as_secs() as i64;
                uow.project_lane_message(NewProjectedLaneMessage {
                    message_id,
                    namespace: publication.namespace,
                    purpose: "agent_event".to_owned(),
                    inbox_id: "agent-provider-events".to_owned(),
                    conversation_id: publication.conversation_id,
                    recipient_id: "agent-event-consumer".to_owned(),
                    sender_id: "agent-runtime".to_owned(),
                    msg_type: "agent.provider_event".to_owned(),
                    status: "pending".to_owned(),
                    created_at: now,
                    available_at: now,
                    run_id: Some(event_run_id.clone()),
                    step_id: Some(publication.event.trace.step_id.clone()),
                    correlation_id: Some(publication.event.event_id.clone()),
                    payload_json: Some(event_payload),
                    error_json: None,
                })?;
                uow.append_server_run_event(&event_run_id, run_event_type, run_event_payload)?;
                Ok(())
            })
            .map_err(|_| "provider event SQLite transaction failed".to_owned())
    }

    async fn execute_sqlite_native_answer_generation(
        &self,
        handoff: RecordedWorkerHandoff,
        mut request: kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> Result<RuntimeWorkerResultRequest, String> {
        kogwistar_runtime::validate_native_step_registration(&request, &self.native_operations)
            .map_err(|error| format!("native execution rejected: {error}"))?;
        let Some(provider) = self.chat_completion_provider.as_ref() else {
            return Ok(failed_native_result(
                handoff,
                &request,
                "KOGWISTAR_CHAT_PROVIDER_NOT_CONFIGURED",
                "Rust-native answer generation requires a configured structured chat provider",
            ));
        };
        self.persist_sqlite_native_answer_snapshot(&request).await?;
        let runtime = request
            .state
            .get("_rt")
            .and_then(Value::as_object)
            .ok_or_else(|| "native answer generation requires runtime context".to_owned())?;
        let evidence_pack = runtime.get("evidence_pack").ok_or_else(|| {
            "native answer generation requires a materialized evidence pack".to_owned()
        })?;
        let question = request
            .state
            .get("question")
            .and_then(Value::as_str)
            .ok_or_else(|| "native answer generation requires a question".to_owned())?;
        let system_prompt = request
            .state
            .get("system_prompt")
            .and_then(Value::as_str)
            .ok_or_else(|| "native answer generation requires a system prompt".to_owned())?;
        let used_node_ids = request
            .state
            .get("used_node_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| "native answer generation requires selected evidence IDs".to_owned())?
            .iter()
            .map(|id| {
                id.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "selected evidence IDs must be strings".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let run_id = request.run_id.clone();
        let workflow_id = request.workflow_id.clone();
        let conversation_id = request.conversation_id.clone();
        let node_id = request.node_id.clone();
        let step_seq = request.step_seq;
        let turn_node_id = request
            .turn_node_id
            .clone()
            .or_else(|| {
                request
                    .state
                    .get("turn_node_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .ok_or_else(|| "native answer generation requires a turn node ID".to_owned())?;
        let trace_context = kogwistar_runtime::native_trace_context_fields(&request);
        let model = provider.model().to_owned();
        let event_model = model.clone();
        let service = self;
        let (answer, event_summaries) = provider
            .answer_with_event_sink(
                system_prompt,
                question,
                evidence_pack,
                &used_node_ids,
                move |event| {
                    let trace_context = trace_context.clone();
                    let service = service;
                    let run_id = run_id.clone();
                    let workflow_id = workflow_id.clone();
                    let conversation_id = conversation_id.clone();
                    let node_id = node_id.clone();
                    let turn_node_id = turn_node_id.clone();
                    let model = event_model.clone();
                    async move {
                        if let kogwistar_contracts::ProviderEventV1::TextDelta { delta, .. } =
                            &event
                        {
                            if delta.chars().count() > 4096 {
                                return Err(
                                    "transient text delta exceeds the configured size limit"
                                        .to_owned(),
                                );
                            }
                            let _ = service.transient_run_deltas.publish_next(&run_id, delta);
                            return Ok(());
                        }
                        let Some(summary) =
                            crate::native_answer_generation::semantic_event_trace_summary(&event)
                        else {
                            return Ok(());
                        };
                        let event_value = serde_json::to_value(&event)
                            .map_err(|_| "provider event could not be encoded".to_owned())?;
                        let attempt = trace_context
                            .get("attempt")
                            .and_then(Value::as_i64)
                            .filter(|value| *value > 0)
                            .unwrap_or(1);
                        let client_sse_sequence = service
                            .transient_run_deltas
                            .last_sequence(&run_id)
                            .unwrap_or_default();
                        let trace = kogwistar_contracts::EventTraceMetadataV1 {
                            run_id: run_id.clone(),
                            turn_id: turn_node_id.clone(),
                            workflow_node_id: node_id.clone(),
                            step_id: step_seq.to_string(),
                            attempt,
                            producer_step_seq: Some(step_seq),
                            producer_attempt: Some(attempt),
                            trace_id: trace_context
                                .get("trace_id")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                            span_id: trace_context
                                .get("span_id")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                            parent_span_id: trace_context
                                .get("parent_span_id")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                            provider_id: "openai_compatible_chat_completions".to_owned(),
                            provider_response_id: event_value
                                .get("response_id")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                            provider_sequence: event_value
                                .get("provider_sequence")
                                .and_then(Value::as_i64),
                            client_sse_sequence: Some(client_sse_sequence),
                        };
                        let durable = kogwistar_contracts::durable_provider_event(&trace, event);
                        let mut payload = json!({
                            "run_id": run_id,
                            "trace_id": run_id,
                            "workflow_id": workflow_id,
                            "conversation_id": conversation_id,
                            "turn_node_id": turn_node_id,
                            "step_id": node_id,
                            "step_seq": step_seq,
                            "executor_target": "rust-native",
                            "provider_id": "openai_compatible_chat_completions",
                            "provider_model": model,
                            "client_sse_sequence": client_sse_sequence,
                            "provider_event": summary,
                        });
                        if let Some(fields) = payload.as_object_mut() {
                            fields.extend(trace_context);
                        }
                        service
                            .persist_sqlite_native_provider_event(durable, conversation_id, payload)
                            .await
                    }
                },
            )
            .await?;
        request
            .ephemeral_context
            .insert("provider_answer".to_owned(), answer);
        request.ephemeral_context.insert(
            "_provider_event_trace_summaries".to_owned(),
            Value::Array(event_summaries),
        );
        request
            .ephemeral_context
            .insert("provider_model".to_owned(), json!(model));
        native_result_from_request(handoff, request, &self.native_operations)
    }

    async fn sqlite_chroma_receive_message_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let input = match syscall_input(request, "receive_message") {
            Ok(input) => input,
            Err(response) => return response,
        };
        if !syscall_allowed_in_namespace(
            request,
            &self.capabilities,
            "receive_message",
            false,
            &["read_graph"],
            "conversation",
        ) {
            return syscall_forbidden("receive_message");
        }
        let Some(conversation_id) = input
            .args
            .get("conversation_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
        else {
            return effect_error(
                StatusCode::BAD_REQUEST,
                "KOGWISTAR_INVALID_REQUEST",
                "conversation_id is required",
            );
        };
        let graph_snapshot = match self.chroma_graph_snapshot(request, "conversation").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let transcript =
            match conversation_transcript_from_visible_records(&nodes, conversation_id, 10_000) {
                Ok(transcript) => transcript,
                Err(error) => {
                    return effect_error(
                        StatusCode::NOT_FOUND,
                        "KOGWISTAR_CONVERSATION_NOT_FOUND",
                        error.to_string(),
                    );
                }
            };
        syscall_result(
            &input.version,
            "receive_message",
            json_effect(StatusCode::OK, legacy_receive_message_value(transcript)),
        )
    }

    async fn sqlite_chroma_mount_memory_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let input = match syscall_input(request, "mount_memory") {
            Ok(input) => input,
            Err(response) => return response,
        };
        if !syscall_allowed_in_namespace(
            request,
            &self.capabilities,
            "mount_memory",
            false,
            &["project_view"],
            "conversation",
        ) {
            return syscall_forbidden("mount_memory");
        }
        let Some(conversation_id) = input
            .args
            .get("conversation_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
        else {
            return effect_error(
                StatusCode::BAD_REQUEST,
                "KOGWISTAR_INVALID_REQUEST",
                "conversation_id is required",
            );
        };
        let run_id = match context_snapshot_syscall_field(&input.args, "run_id") {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let stage = match context_snapshot_syscall_field(&input.args, "stage") {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let graph_snapshot = match self.chroma_graph_snapshot(request, "conversation").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let snapshot = match kogwistar_store::context_snapshot_from_visible_records(
            &nodes,
            conversation_id,
            run_id.as_deref(),
            stage.as_deref(),
        ) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                return effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_CONTEXT_SNAPSHOT_NOT_FOUND",
                    error.to_string(),
                );
            }
        };
        syscall_result(
            &input.version,
            "mount_memory",
            json_effect(StatusCode::OK, public_context_snapshot_value(snapshot)),
        )
    }

    async fn sqlite_chroma_conversation_transcript_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let namespaces = principal_strings(&request.principal, "ns");
        let namespaces = if namespaces.is_empty() {
            vec!["docs".to_owned()]
        } else {
            namespaces
        };
        if !namespaces
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "conversation"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Conversation reads require conversation namespace",
            );
        }
        let Some(conversation_id) = conversation_transcript_id(&request.path_and_query) else {
            return effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_ROUTE_NOT_FOUND",
                "Conversation transcript route was not found",
            );
        };
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Conversation transcript reads require read_graph capability",
            );
        }
        let graph_snapshot = match self.chroma_graph_snapshot(request, "conversation").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let transcript =
            match conversation_transcript_from_visible_records(&nodes, &conversation_id, 10_000) {
                Ok(transcript) => transcript,
                Err(error) => {
                    return effect_error(
                        StatusCode::NOT_FOUND,
                        "KOGWISTAR_CONVERSATION_NOT_FOUND",
                        error.to_string(),
                    );
                }
            };
        let mut payload = serde_json::to_value(transcript).expect("transcript serializes");
        if let Some(object) = payload.as_object_mut() {
            object.remove("schema_version");
            if let Some(turns) = object.get_mut("turns").and_then(Value::as_array_mut) {
                for turn in turns {
                    if let Some(turn) = turn.as_object_mut() {
                        turn.remove("schema_version");
                    }
                }
            }
        }
        json_effect(StatusCode::OK, payload)
    }

    async fn sqlite_chroma_conversation_detail_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let namespaces = principal_strings(&request.principal, "ns");
        let namespaces = if namespaces.is_empty() {
            vec!["docs".to_owned()]
        } else {
            namespaces
        };
        if !namespaces
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "conversation"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Conversation reads require conversation namespace",
            );
        }
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Conversation reads require read_graph capability",
            );
        }
        let Some(conversation_id) = conversation_detail_id(&request.path_and_query) else {
            return effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_ROUTE_NOT_FOUND",
                "Conversation route was not found",
            );
        };
        let graph_snapshot = match self.chroma_graph_snapshot(request, "conversation").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let summary = match conversation_summary_from_visible_records(&nodes, &conversation_id) {
            Ok(summary) => summary,
            Err(error) => {
                return effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_CONVERSATION_NOT_FOUND",
                    error.to_string(),
                );
            }
        };
        let mut payload = serde_json::to_value(summary).expect("conversation summary serializes");
        if let Some(object) = payload.as_object_mut() {
            object.remove("schema_version");
        }
        json_effect(StatusCode::OK, payload)
    }

    async fn sqlite_chroma_conversation_list_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let namespaces = principal_strings(&request.principal, "ns");
        let namespaces = if namespaces.is_empty() {
            vec!["docs".to_owned()]
        } else {
            namespaces
        };
        if !namespaces
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "conversation"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Conversation reads require conversation namespace",
            );
        }
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Conversation reads require read_graph capability",
            );
        }
        let user_id = request
            .principal
            .get("sub")
            .or_else(|| request.principal.get("user_id"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(user_id) = user_id else {
            return effect_error(
                StatusCode::UNAUTHORIZED,
                "KOGWISTAR_USER_IDENTITY_REQUIRED",
                "User Identity required",
            );
        };
        let graph_snapshot = match self.chroma_graph_snapshot(request, "conversation").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let candidates = compact_conversation_list_candidates(&nodes);
        let mut conversations = ConversationListAccumulator::new(user_id);
        conversations.extend_visible_records(&candidates);
        let conversations = conversations
            .finish()
            .into_iter()
            .map(|entry| {
                json!({
                    "id": entry.id,
                    "start_node_id": entry.start_node_id,
                    "status": entry.status,
                    "turn_count": entry.turn_count,
                })
            })
            .collect::<Vec<_>>();
        json_effect(StatusCode::OK, json!({"conversations": conversations}))
    }

    async fn chroma_graph_neighbors_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            );
        }
        let input = match parse_graph_neighbors_request(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let graph_snapshot = match self
            .chroma_graph_snapshot(request, input.truth_graph.trim())
            .await
        {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let edges = graph_snapshot.edges();
        let neighbors =
            graph_neighbors_from_records(input.rid.trim(), &nodes, &edges, input.doc_id.as_deref());
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({
                "nodes": neighbors.nodes.into_iter().collect::<Vec<_>>(),
                "edges": neighbors.edges.into_iter().collect::<Vec<_>>(),
            })),
        )
    }

    async fn chroma_graph_find_edges_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            );
        }
        let input = match parse_graph_find_edges_request(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let graph_snapshot = match self.chroma_graph_snapshot(request, "knowledge").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let edges = graph_snapshot.edges();
        let query = GraphFindEdgesQuery {
            relation: input.relation,
            src_label_contains: input.src_label_contains,
            tgt_label_contains: input.tgt_label_contains,
            doc_id: input.doc_id,
        };
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({
                "edges": graph_find_edges_from_records(&query, &nodes, &edges),
            })),
        )
    }

    async fn chroma_graph_shortest_path_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            );
        }
        let input: GraphShortestPathRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if input.src_id.trim().is_empty() || input.dst_id.trim().is_empty() || input.max_depth > 128
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "src_id and dst_id are required and max_depth must be <= 128",
            );
        }
        let graph_snapshot = match self.chroma_graph_snapshot(request, "knowledge").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let edges = graph_snapshot.edges();
        let query = GraphShortestPathQuery {
            src_id: input.src_id,
            dst_id: input.dst_id,
            doc_id: input.doc_id,
            max_depth: input.max_depth,
        };
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({
                "path": graph_shortest_path_from_records(&query, &nodes, &edges),
            })),
        )
    }

    async fn chroma_graph_k_hop_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            );
        }
        let input: GraphKHopRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if input.start_ids.is_empty() || input.start_ids.iter().any(|id| id.trim().is_empty()) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "start_ids must contain at least one non-empty ID",
            );
        }
        if input.k > 32 {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "k must be <= 32",
            );
        }
        let graph_snapshot = match self.chroma_graph_snapshot(request, "knowledge").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let edges = graph_snapshot.edges();
        let query = GraphKHopQuery {
            start_ids: input.start_ids,
            k: input.k,
            doc_id: input.doc_id,
        };
        let layers = graph_k_hop_from_records(&query, &nodes, &edges)
            .into_iter()
            .map(|layer| {
                json!({
                    "nodes": layer.nodes.into_iter().collect::<Vec<_>>(),
                    "edges": layer.edges.into_iter().collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>();
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({"layers": layers})),
        )
    }

    async fn chroma_graph_visualization_effect(
        &self,
        request: &ApiEffectRequest,
        cytoscape: bool,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            );
        }
        let input: GraphVisualizationRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let graph_snapshot = match self.chroma_graph_snapshot(request, "knowledge").await {
            Ok(records) => records,
            Err(response) => return response,
        };
        let nodes = graph_snapshot.nodes();
        let edges = graph_snapshot.edges();
        let result = if cytoscape {
            graph_viz_cytoscape_from_records(&nodes, &edges, input.doc_id.as_deref(), &input.mode)
        } else {
            graph_viz_d3_from_records(&nodes, &edges, input.doc_id.as_deref(), &input.mode)
        };
        json_effect(StatusCode::OK, rust_native_attested(result))
    }

    async fn chroma_graph_vector_query_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph vector query requires read_graph capability",
            );
        }
        let input = match parse_graph_vector_query_request(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };

        let kind = if input.entity_kind == "node" {
            GraphEntityKind::Node
        } else {
            GraphEntityKind::Edge
        };
        let scope_value = security_scope_value(&request.principal);
        let namespace = scope_value["storage_namespace"]
            .as_str()
            .unwrap_or("default");
        let mut metadata = input.where_filter.clone();
        if let Some(workspace) = request
            .principal
            .get("workspace_id")
            .or_else(|| request.principal.get("workspace"))
            .and_then(Value::as_str)
        {
            metadata.equals.insert(
                "workspace_id".to_owned(),
                Value::String(workspace.to_owned()),
            );
        }
        if let Some(graph_space) = request.principal.get("graph_space").and_then(Value::as_str) {
            metadata.equals.insert(
                "graph_space".to_owned(),
                Value::String(graph_space.to_owned()),
            );
        }

        let Some(graph) = &self.chroma_graph else {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "The SQLite service has no configured graph projection",
            );
        };
        let mut matches_by_query = Vec::with_capacity(input.query_embeddings.len());
        for embedding in &input.query_embeddings {
            match graph
                .vector_query_of_kind(
                    namespace,
                    kind,
                    &VectorQuery {
                        embedding: embedding.clone(),
                        limit: input.n_results,
                        metadata: metadata.clone(),
                        metric: input.metric,
                    },
                )
                .await
            {
                Ok(matches) => matches_by_query.push(matches),
                Err(error) => {
                    return effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_VECTOR_QUERY_ERROR",
                        error.to_string(),
                    );
                }
            }
        }
        for matches in &mut matches_by_query {
            matches.retain(|matched| {
                chroma_record_matches_principal_scope(&matched.record, &request.principal)
            });
        }

        let projected_hits = matches_by_query
            .iter()
            .flatten()
            .map(|matched| matched.record.clone())
            .collect::<Vec<_>>();
        let (mut hit_nodes, mut hit_edges) = if kind == GraphEntityKind::Node {
            (projected_hits, Vec::new())
        } else {
            (Vec::new(), projected_hits)
        };
        if let Err(error) =
            self.filter_current_chroma_stage2(namespace, &mut hit_nodes, &mut hit_edges)
        {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error,
            );
        }
        let current_hit_ids = hit_nodes
            .iter()
            .chain(hit_edges.iter())
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();
        for matches in &mut matches_by_query {
            matches.retain(|matched| current_hit_ids.contains(matched.record.id.as_str()));
        }

        let mut candidate_by_id = BTreeMap::new();
        for matched in matches_by_query.iter().flatten() {
            candidate_by_id
                .entry(matched.record.id.clone())
                .or_insert_with(|| matched.record.clone());
        }
        if candidate_by_id.len() > 10_000 {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_VECTOR_QUERY_ERROR",
                "vector candidate ACL set exceeds the 10000-record limit",
            );
        }
        let candidate_records = candidate_by_id.into_values().collect::<Vec<_>>();
        let (candidate_nodes, candidate_edges) = if kind == GraphEntityKind::Node {
            (candidate_records, Vec::new())
        } else {
            let mut edges = candidate_records;
            let mut known_edge_ids = edges
                .iter()
                .map(|record| record.id.clone())
                .collect::<BTreeSet<_>>();
            let mut frontier = edges
                .iter()
                .flat_map(|record| {
                    graph_record_string_ids(record, "source_edge_ids")
                        .into_iter()
                        .chain(graph_record_string_ids(record, "target_edge_ids"))
                })
                .collect::<BTreeSet<_>>();
            while !frontier.is_empty() {
                let pending = frontier
                    .iter()
                    .filter(|edge_id| !known_edge_ids.contains(*edge_id))
                    .cloned()
                    .collect::<Vec<_>>();
                frontier.clear();
                if known_edge_ids.len().saturating_add(pending.len()) > 10_000 {
                    return effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_VECTOR_QUERY_ERROR",
                        "edge ACL dependency closure exceeds 10000 records",
                    );
                }
                let referenced = match graph
                    .graph_records_by_ids_of_kind(namespace, GraphEntityKind::Edge, &pending)
                    .await
                {
                    Ok(records) => records,
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        );
                    }
                };
                for record in referenced {
                    if !chroma_record_matches_principal_scope(&record, &request.principal) {
                        continue;
                    }
                    if !known_edge_ids.insert(record.id.clone()) {
                        continue;
                    }
                    frontier.extend(graph_record_string_ids(&record, "source_edge_ids"));
                    frontier.extend(graph_record_string_ids(&record, "target_edge_ids"));
                    edges.push(record);
                }
            }
            let node_ids = edges
                .iter()
                .flat_map(|record| {
                    graph_record_string_ids(record, "source_ids")
                        .into_iter()
                        .chain(graph_record_string_ids(record, "target_ids"))
                })
                .collect::<BTreeSet<_>>();
            let node_ids = node_ids.into_iter().collect::<Vec<_>>();
            if node_ids.len() > 10_000 {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_VECTOR_QUERY_ERROR",
                    "edge endpoint ACL set exceeds the 10000-record limit",
                );
            }
            let mut nodes = match graph
                .graph_records_by_ids_of_kind(namespace, GraphEntityKind::Node, &node_ids)
                .await
            {
                Ok(records) => records,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    );
                }
            };
            nodes
                .retain(|record| chroma_record_matches_principal_scope(record, &request.principal));
            (nodes, edges)
        };
        let mut acl_filter = MetadataFilter {
            equals: BTreeMap::from([
                ("entity_type".to_owned(), json!("acl_record")),
                ("acl_truth_graph".to_owned(), json!(input.truth_graph)),
            ]),
        };
        if let Some(workspace) = request
            .principal
            .get("workspace_id")
            .or_else(|| request.principal.get("workspace"))
            .and_then(Value::as_str)
        {
            acl_filter
                .equals
                .insert("workspace_id".to_owned(), json!(workspace));
        }
        if let Some(graph_space) = request.principal.get("graph_space").and_then(Value::as_str) {
            acl_filter
                .equals
                .insert("graph_space".to_owned(), json!(graph_space));
        }
        let mut acl_records = match graph
            .graph_records_of_kind_bounded(namespace, GraphEntityKind::Node, &acl_filter, 10_000)
            .await
        {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        acl_records
            .retain(|record| chroma_record_matches_principal_scope(record, &request.principal));
        let mut acl_index = GraphAclIndex::default();
        acl_index.extend_from_records(&acl_records, &input.truth_graph);
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (visible_nodes, visible_edges) = graph_records_visible_by_acl_with_index(
            &candidate_nodes,
            &candidate_edges,
            &acl_index,
            &principal,
        );
        let visible_ids = visible_nodes
            .iter()
            .chain(visible_edges.iter())
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();

        let mut ids = Vec::with_capacity(matches_by_query.len());
        let mut documents = Vec::with_capacity(matches_by_query.len());
        let mut metadatas = Vec::with_capacity(matches_by_query.len());
        let mut embeddings = Vec::with_capacity(matches_by_query.len());
        let mut distances = Vec::with_capacity(matches_by_query.len());
        for matches in matches_by_query {
            let matches = matches
                .into_iter()
                .filter(|matched| {
                    visible_ids.contains(matched.record.id.as_str())
                        && !matched
                            .record
                            .metadata
                            .get("lifecycle_status")
                            .and_then(Value::as_str)
                            .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
                })
                .collect::<Vec<_>>();
            ids.push(
                matches
                    .iter()
                    .map(|matched| json!(matched.record.id))
                    .collect::<Vec<_>>(),
            );
            if input.include.iter().any(|field| field == "documents") {
                documents.push(
                    matches
                        .iter()
                        .map(|matched| json!(matched.record.document))
                        .collect::<Vec<_>>(),
                );
            }
            if input.include.iter().any(|field| field == "metadatas") {
                metadatas.push(
                    matches
                        .iter()
                        .map(|matched| json!(matched.record.metadata))
                        .collect::<Vec<_>>(),
                );
            }
            if input.include.iter().any(|field| field == "embeddings") {
                embeddings.push(
                    matches
                        .iter()
                        .map(|matched| json!(matched.record.embedding))
                        .collect::<Vec<_>>(),
                );
            }
            if input.include.iter().any(|field| field == "distances") {
                distances.push(
                    matches
                        .iter()
                        .map(|matched| json!(matched.distance))
                        .collect::<Vec<_>>(),
                );
            }
        }
        let mut response = serde_json::Map::new();
        response.insert("ids".to_owned(), json!(ids));
        if input.include.iter().any(|field| field == "documents") {
            response.insert("documents".to_owned(), json!(documents));
        }
        if input.include.iter().any(|field| field == "metadatas") {
            response.insert("metadatas".to_owned(), json!(metadatas));
        }
        if input.include.iter().any(|field| field == "embeddings") {
            response.insert("embeddings".to_owned(), json!(embeddings));
        }
        if input.include.iter().any(|field| field == "distances") {
            response.insert("distances".to_owned(), json!(distances));
        }
        json_effect(
            StatusCode::OK,
            rust_native_attested(Value::Object(response)),
        )
    }

    fn get_run(&self, run_id: &str) -> Result<Option<ServerRun>, String> {
        self.store
            .get_server_run(run_id)
            .map_err(|error| error.to_string())
    }

    fn submit_run(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        self.submit_run_with_conversation_admission(request, None)
    }

    fn submit_run_with_conversation_admission(
        &self,
        request: &ApiEffectRequest,
        conversation_admission: Option<PreparedConversationAdmission>,
    ) -> ApiEffectResponse {
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
        let authenticated_owner = authenticated_run_owner_id(&request.principal);
        if input.user_id.as_deref().is_some_and(|requested| {
            authenticated_owner.is_none_or(|owner| owner != requested.trim())
        }) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "run user_id must match the authenticated principal",
            );
        }
        let run_user_id = authenticated_owner
            .map(str::to_owned)
            .or_else(|| input.user_id.clone());
        let owner_binding = runtime_owner_binding_v1(&request.principal, &self.capabilities);
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
        let worker_protocol = worker_protocol_for_runtime_kind(&runtime_kind);
        let mut event_payload = json!({
            "run_id": run_id,
            "run_kind": "workflow_runtime",
            "conversation_id": conversation_id,
            "workflow_id": workflow_id,
            "status": "queued",
            "turn_node_id": turn_node_id,
            "implementation_attestation": runtime_implementation_attestation(&runtime_kind),
        });
        if let Some(binding) = owner_binding {
            event_payload["runtime_owner_binding"] = json!(binding);
        }
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
        if is_answer_workflow_id(workflow_id)
            && (self.chroma_graph.is_none()
                || !answer_workflow_admission_ready(
                    workflow_id,
                    graph_plan.as_ref(),
                    &self.native_operations,
                ))
        {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_ANSWER_WORKFLOW_NOT_READY",
                "SQLite answer workflows require configured graph authority and a complete plan with compatible executors",
            );
        }
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
        let effective_node_versions = graph_plan
            .as_ref()
            .map(|plan| plan.node_versions.clone())
            .unwrap_or_else(|| {
                if input.node_versions.is_empty() {
                    effective_node_ops
                        .keys()
                        .map(|node_id| (node_id.clone(), "v1".to_owned()))
                        .collect()
                } else {
                    input.node_versions
                }
            });
        let effective_node_environment_ids = graph_plan
            .as_ref()
            .map(|plan| plan.node_environment_ids.clone())
            .unwrap_or(input.node_environment_ids);
        let effective_node_executor_targets = graph_plan
            .as_ref()
            .map(|plan| plan.node_executor_targets.clone())
            .unwrap_or(input.node_executor_targets);
        // Even a bare transport-level submission must hand the Python worker a
        // frozen operation.  Later continuation lanes already use this same
        // no-op fallback; leaving the admission lane as JSON null made the
        // worker correctly reject an otherwise valid durable run.
        let effective_start_op = effective_node_ops
            .get(&effective_start_node_id)
            .cloned()
            .unwrap_or_else(|| "noop".to_owned());
        let effective_start_op_version = effective_node_versions
            .get(&effective_start_node_id)
            .cloned()
            .unwrap_or_else(|| "v1".to_owned());
        let effective_start_environment_id = effective_node_environment_ids
            .get(&effective_start_node_id)
            .cloned();
        let effective_start_executor_target = effective_node_executor_targets
            .get(&effective_start_node_id)
            .cloned();
        let mut initial_state = input.initial_state.clone();
        // Runtime scope and owner are server-derived authority, never caller state.
        // Keep SQLite admission aligned with PostgreSQL before scheduling any step.
        initial_state.remove("_rt_graph_scope");
        initial_state.remove("_rt_run_owner_id");
        initial_state.insert(
            "_rt_graph_scope".to_owned(),
            security_scope_value(&request.principal),
        );
        if let Some(owner_id) = run_user_id.as_deref() {
            initial_state.insert("_rt_run_owner_id".to_owned(), json!(owner_id));
        }
        install_authoritative_node_executor_targets(
            &mut initial_state,
            &effective_node_executor_targets,
        );
        inject_authoritative_capability_state(
            &mut initial_state,
            &request.principal,
            &self.capabilities,
        );
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
        if effective_node_versions
            .values()
            .any(|version| version != "v1")
        {
            initial_state.insert(
                "_rt_node_versions".to_owned(),
                serde_json::to_value(&effective_node_versions)
                    .expect("runtime node versions serialize"),
            );
        }
        if !effective_node_environment_ids.is_empty() {
            initial_state.insert(
                "_rt_node_environment_ids".to_owned(),
                serde_json::to_value(&effective_node_environment_ids)
                    .expect("runtime node environment ids serialize"),
            );
        }
        initial_state.insert(
            "_rt_worker_protocol".to_owned(),
            Value::String(worker_protocol.to_owned()),
        );
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
            "worker_protocol": worker_protocol,
            "join_node_ids": effective_join_node_ids,
            "start_join_mask": effective_start_join_mask,
            "start_node_id": effective_start_node_id,
            "op": effective_start_op,
            "runtime_routes": effective_routes,
        });
        let mut worker_payload = worker_payload;
        if effective_start_op_version != "v1" {
            worker_payload["op_version"] = json!(effective_start_op_version);
        }
        if let Some(environment_id) = effective_start_environment_id.as_ref() {
            worker_payload["op_environment_id"] = json!(environment_id);
        }
        add_executor_target(
            &mut worker_payload,
            &effective_start_op,
            effective_start_executor_target.as_deref(),
        );
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
        if let Err((code, message)) = validate_native_operation_registry(
            &effective_node_ops,
            &effective_node_versions,
            &effective_node_executor_targets,
            &self.native_operations,
        ) {
            return effect_error(StatusCode::UNPROCESSABLE_ENTITY, code, message);
        }
        let max_queue = self.max_queue;
        let admission_actor_user_id = conversation_admission
            .as_ref()
            .map(|admission| admission.actor_user_id.clone());
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
                return Ok(RuntimeAdmissionOutcome::QueueFull);
            }
            if let Some(admission) = conversation_admission.as_ref() {
                let current_events = uow.latest_entity_events_by_metadata(
                    &admission.scope.namespace,
                    "node",
                    "conversation_id",
                    &admission.conversation_id,
                )?;
                let current_nodes = current_events
                    .iter()
                    .filter(|event| !event.op.eq_ignore_ascii_case("DELETE"))
                    .map(conversation_record_from_raw_event)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .filter(|record| {
                        !matches!(
                            conversation_record_string(record, "lifecycle_status").as_deref(),
                            Some("tombstoned" | "deleted")
                        )
                    })
                    .collect::<Vec<_>>();
                let current_ids = current_nodes
                    .iter()
                    .map(|record| record.id.clone())
                    .collect::<Vec<_>>();
                let current_acl = uow.graph_acl_records_for_entities(
                    &admission.scope,
                    "conversation",
                    &current_ids,
                )?;
                let mut acl_inputs = current_nodes;
                acl_inputs.extend(current_acl);
                let (visible_nodes, _) = graph_records_visible_by_acl(
                    &acl_inputs,
                    &[],
                    "conversation",
                    &admission.acl_principal,
                );
                let owner_matches = visible_nodes.iter().any(|record| {
                    conversation_record_string(record, "entity_type").as_deref()
                        == Some("conversation_start")
                        && conversation_record_string(record, "user_id").as_deref()
                            == Some(admission.owner_id.as_str())
                });
                let mut current_tail = visible_nodes
                    .iter()
                    .filter(|record| {
                        matches!(
                            conversation_record_string(record, "entity_type").as_deref(),
                            Some(
                                "conversation_start"
                                    | "conversation_turn"
                                    | "conversation_summary"
                                    | "assistant_turn"
                            )
                        ) && conversation_record_value(record, "in_conversation_chain")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(true)
                    })
                    .collect::<Vec<_>>();
                current_tail.sort_by_key(|record| {
                    (
                        conversation_record_i64(record, "turn_index").unwrap_or(-1),
                        record.id.clone(),
                    )
                });
                let actual_tail_id = current_tail.last().map(|record| record.id.as_str());
                if !owner_matches || actual_tail_id != admission.expected_tail_id.as_deref() {
                    return Ok(RuntimeAdmissionOutcome::ConversationChanged);
                }
            }
            uow.create_server_run(ServerRunCreate {
                run_id: run_id.clone(),
                conversation_id: conversation_id.to_owned(),
                workflow_id: workflow_id.to_owned(),
                user_id: run_user_id.clone(),
                user_turn_node_id: turn_node_id.clone(),
                status: "queued".to_owned(),
            })?;
            if let Some(admission) = conversation_admission.as_ref() {
                let graph_mutations = admission
                    .mutations
                    .iter()
                    .map(|(mutation, _)| mutation.clone())
                    .collect::<Vec<_>>();
                let acl_targets = acl_targets_for_graph_mutations(&graph_mutations).0;
                let target_ids = acl_target_entity_ids(&acl_targets);
                let existing_acl = uow.graph_acl_records_for_entities(
                    &admission.scope,
                    "conversation",
                    &target_ids,
                )?;
                let (acl_mutations, acl_jobs) = derive_default_acl_graph_effects_from_records(
                    &admission.scope,
                    &graph_mutations,
                    acl_targets,
                    existing_acl,
                    "conversation",
                    &admission.owner_id.to_ascii_lowercase(),
                    admission.acl_principal.security_scope.as_deref(),
                    &run_id,
                )
                .map_err(SqliteStoreError::TransactionAborted)?;
                for (mutation, envelope) in &admission.mutations {
                    validate_typed_graph_mutation(mutation, envelope)?;
                }
                let mut mutations = graph_mutations;
                mutations.extend(acl_mutations);
                let mut index_jobs = admission.index_jobs.clone();
                index_jobs.extend(acl_jobs);
                uow.apply_canonical_graph_mutation_batch(GraphMutationBatch {
                    mutations,
                    index_jobs,
                })?;
            }
            let created_event = uow.append_server_run_event(&run_id, "run.created", {
                let mut payload = event_payload.clone();
                if let Some(actor_user_id) = admission_actor_user_id.as_deref() {
                    payload["admission_actor"] = json!({
                        "user_id": actor_user_id,
                        "operation": "send_message",
                    });
                }
                serde_json::to_string(&payload)?
            })?;
            let mut persisted_worker_payload = worker_payload.clone();
            persisted_worker_payload["expected_event_seq"] = json!(created_event.seq);
            uow.project_lane_message(NewProjectedLaneMessage {
                message_id: message_id.clone(),
                namespace: "workflow".to_owned(),
                purpose: "system".to_owned(),
                inbox_id: "workflow-runtime".to_owned(),
                conversation_id: conversation_id.to_owned(),
                recipient_id: runtime_worker_recipient_for_operation(
                    &effective_start_op,
                    effective_start_environment_id.as_deref(),
                    effective_start_executor_target.as_deref(),
                ),
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
            Ok(RuntimeAdmissionOutcome::Queued)
        });
        match outcome {
            Ok(RuntimeAdmissionOutcome::QueueFull) => {
                return json_effect(
                    StatusCode::TOO_MANY_REQUESTS,
                    json!({
                        "admission": if matches!(priority_class.as_str(), "background" | "batch") { "rejected" } else { "deferred" },
                        "reason": "queue_full",
                        "max_queue": max_queue,
                    }),
                );
            }
            Ok(RuntimeAdmissionOutcome::ConversationChanged) => {
                return effect_error(
                    StatusCode::CONFLICT,
                    "KOGWISTAR_CONVERSATION_CHANGED",
                    "conversation owner or tail changed while the turn was being admitted",
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
            Ok(RuntimeAdmissionOutcome::Queued) => {}
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
        let recipient_ids = match effective_runtime_claim_recipients(&input.recipient_ids) {
            Ok(values) => values,
            Err(message) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    message,
                );
            }
        };
        match self.claim_runtime_work_items(input, recipient_ids) {
            Ok(work) => json_effect(StatusCode::OK, json!({"work": work})),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_CLAIM_CONFLICT",
                error,
            ),
        }
    }

    fn claim_runtime_work_items(
        &self,
        input: RuntimeClaimRequest,
        recipient_ids: Vec<String>,
    ) -> Result<Vec<RuntimeClaimedWork>, String> {
        let claim_owner = format!("{}#{}", input.claimed_by, uuid::Uuid::new_v4());
        let supports_graph_mutation_batch = self.chroma_graph.is_some();
        self.store
            .immediate_transaction(|uow| {
                let mut claimed = Vec::new();
                for recipient_id in &recipient_ids {
                    let remaining = input.limit.saturating_sub(claimed.len());
                    if remaining == 0 {
                        break;
                    }
                    claimed.extend(uow.claim_projected_lane_messages_filtered(
                        LaneMessageClaimFilter {
                            namespace: "workflow".to_owned(),
                            inbox_id: "workflow-runtime".to_owned(),
                            claimed_by: claim_owner.clone(),
                            message_ids: None,
                            run_id: input.run_id.clone(),
                            msg_type: None,
                            recipient_id: Some(recipient_id.clone()),
                            limit: remaining,
                            lease_seconds: input.lease_seconds,
                        },
                    )?);
                }
                let mut values = Vec::new();
                for lane in claimed {
                    let payload: Value =
                        serde_json::from_str(lane.payload_json.as_deref().unwrap_or("{}"))?;
                    let run_id = lane.run_id.clone().unwrap_or_default();
                    if uow
                        .get_server_run(&run_id)?
                        .is_some_and(|run| run.cancel_requested)
                    {
                        uow.update_projected_lane_message_status(
                            &lane.message_id,
                            "cancelled",
                            None,
                        )?;
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
                            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                                format!(
                                    "continuation lane message {:?} lacks expected_event_seq",
                                    lane.message_id
                                ),
                            )
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
                    let mut work = runtime_claimed_work(
                        lane,
                        &claim_owner,
                        run_id,
                        claimed_payload,
                        expected_event_seq,
                    );
                    if supports_graph_mutation_batch {
                        work.worker_effect_capabilities
                            .push("graph_mutation_batch_v1".to_owned());
                    }
                    values.push(work);
                }
                Ok(values)
            })
            .map_err(|error| error.to_string())
    }

    fn apply_runtime_result(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        self.apply_runtime_result_with_principal(request, &request.principal)
    }

    fn apply_runtime_result_with_principal(
        &self,
        request: &ApiEffectRequest,
        principal: &Value,
    ) -> ApiEffectResponse {
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
        self.apply_runtime_result_input(input, "python-worker", Some(principal))
    }

    fn apply_runtime_result_input(
        &self,
        input: RuntimeWorkerResultRequest,
        executor_target: &str,
        principal: Option<&Value>,
    ) -> ApiEffectResponse {
        if input.transition.is_some() == input.effect.is_some() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "exactly one of transition or effect is required",
            );
        }
        if input.effect.as_ref().is_some_and(|effect| {
            effect.graph_mutation_batch.is_some() && effect.graph_mutation_batch_v2.is_some()
        }) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "worker effect cannot carry both graph batch versions",
            );
        }
        let provider_event_acknowledgements = input
            .effect
            .as_ref()
            .map(|effect| effect.provider_event_acknowledgements.clone())
            .unwrap_or_default();
        if validate_provider_event_acknowledgements(&provider_event_acknowledgements).is_err() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "provider event acknowledgements must be unique, complete, and limited to 100",
            );
        }
        if !provider_event_acknowledgements.is_empty()
            && input.effect.as_ref().is_none_or(|effect| {
                effect.status != kogwistar_runtime::RuntimeWorkerEffectStatus::Success
            })
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "provider events can only be acknowledged with a successful durable handling step",
            );
        }
        if !provider_event_acknowledgements.is_empty() && self.chroma_graph.is_none() {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "SQLite provider-event acknowledgement requires the configured Chroma graph projection",
            );
        }
        let provider_event_publications = input
            .effect
            .as_ref()
            .map(|effect| effect.provider_event_publications.clone())
            .unwrap_or_default();
        if provider_event_acknowledgements.iter().any(|ack| {
            provider_event_publications
                .iter()
                .any(|publication| publication.event.event_id == ack.event_id)
        }) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "one worker effect cannot publish and acknowledge the same provider event",
            );
        }
        let provider_event_ack_scope = if provider_event_acknowledgements.is_empty() {
            None
        } else {
            let Some(principal) = principal else {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "provider event acknowledgement requires the run owner's current write_graph authority",
                );
            };
            if !capability_allowed_for_principal(
                principal,
                &self.capabilities,
                "write_graph",
                &["write_graph"],
            ) {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "provider event acknowledgement requires the run owner's current write_graph authority",
                );
            }
            let Some(owner_id) = authenticated_run_owner_id(principal).map(str::to_owned) else {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                    "provider event acknowledgement requires an authenticated run owner",
                );
            };
            let security_scope = security_scope_value(principal);
            let Some(namespace) = security_scope["storage_namespace"]
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .map(str::to_owned)
            else {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_SCOPE_FORBIDDEN",
                    "provider event acknowledgement has no admitted storage namespace",
                );
            };
            let scope = GraphScope {
                namespace,
                workspace_id: security_scope
                    .get("workspace")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned),
                graph_space: Some("conversation".to_owned()),
            };
            if !graph_mutation_scope_allowed(&scope, &security_scope) {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_SCOPE_FORBIDDEN",
                    "provider event acknowledgement scope is outside the run owner's admitted scope",
                );
            }
            Some((scope, owner_id, principal.clone()))
        };
        let graph_batch = input.effect.as_ref().and_then(|effect| {
            match (
                effect.graph_mutation_batch.as_ref(),
                effect.graph_mutation_batch_v2.as_ref(),
            ) {
                (Some(_), Some(_)) => Some(Err(
                    "worker effect cannot carry both graph batch versions".to_owned(),
                )),
                (Some(v1), None) => Some((|| {
                    if !provider_event_publications.is_empty() {
                        return Err(
                            "provider event publications require an ACL-aware graph batch V2"
                                .to_owned(),
                        );
                    }
                    let mutations = graph_mutation_batch_value(v1.clone())?;
                    validate_graph_mutation_envelopes(v1, &mutations)?;
                    validate_runtime_graph_mutations(&mutations)?;
                    let index_jobs = graph_index_jobs_value(v1, &mutations)?;
                    Ok((
                        GraphMutationBatch {
                            mutations,
                            index_jobs,
                        },
                        None,
                    ))
                })()),
                (None, Some(v2)) => Some((|| {
                    let (mutations, index_jobs) = graph_mutation_batch_v2_value(v2.clone())?;
                    validate_runtime_graph_mutations(&mutations)?;
                    if !provider_event_publications.is_empty() {
                        if v2.default_acl.truth_graph != "conversation" {
                            return Err(
                                "provider event publications require conversation ACL context"
                                    .to_owned(),
                            );
                        }
                        validate_provider_event_publications(
                            &provider_event_publications,
                            &mutations,
                            &input.handoff.run_id,
                        )?;
                    }
                    Ok((
                        GraphMutationBatch {
                            mutations,
                            index_jobs,
                        },
                        Some(v2.default_acl.truth_graph.clone()),
                    ))
                })()),
                (None, None) => None,
            }
        });
        let graph_batch = match graph_batch {
            Some(Ok((batch, default_acl_truth_graph))) => {
                if self.chroma_graph.is_none() {
                    return effect_error(
                        StatusCode::NOT_IMPLEMENTED,
                        "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                        "runtime graph effects require SQLite canonical events with configured Chroma projection",
                    );
                }
                let Some(principal) = principal else {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "runtime graph effects require an authenticated write_graph principal",
                    );
                };
                if !capability_allowed_for_principal(
                    principal,
                    &self.capabilities,
                    "write_graph",
                    &["write_graph"],
                ) {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        "runtime graph effects require write_graph capability",
                    );
                }
                let Some(owner_id) = authenticated_run_owner_id(principal).map(str::to_owned)
                else {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                        "runtime graph effects require an authenticated run owner",
                    );
                };
                let scope = security_scope_value(principal);
                if batch
                    .mutations
                    .iter()
                    .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &scope))
                {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_SCOPE_FORBIDDEN",
                        "runtime graph effect scope is outside the authenticated owner scope",
                    );
                }
                let effect_id = input
                    .effect
                    .as_ref()
                    .map(|effect| effect.effect_id.clone())
                    .unwrap_or_default();
                Some((
                    batch,
                    owner_id,
                    principal.clone(),
                    default_acl_truth_graph,
                    effect_id,
                    provider_event_publications.clone(),
                ))
            }
            Some(Err(error)) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
            None => None,
        };
        if let Some(effect) = input.effect.as_ref()
            && let Err(message) = validate_python_worker_submission_trace_binding(
                &effect.trace_events,
                &input.handoff,
            )
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_WORKER_TRACE",
                message,
            );
        }
        let runtime_run_id = input.handoff.run_id.clone();
        let outcome = self.store.immediate_transaction(|uow| {
            if let Some((batch, owner_id, principal, default_acl_truth_graph, effect_id, publications)) = graph_batch.as_ref() {
                if !capability_allowed_for_principal(
                    principal,
                    &self.capabilities,
                    "write_graph.commit",
                    &["write_graph"],
                ) {
                    return Err(kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                        "runtime graph owner no longer has write_graph capability".to_owned(),
                    ));
                }
                let run = uow.get_server_run(&input.handoff.run_id)?;
                if !run
                    .as_ref()
                    .is_some_and(|run| run.user_id.as_deref() == Some(owner_id.as_str()))
                {
                    return Err(kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                        "runtime graph effect owner does not match the durable run owner".to_owned(),
                    ));
                }
                if publications.iter().any(|publication| {
                    run.as_ref().is_none_or(|run| {
                        run.conversation_id != publication.conversation_id
                    })
                }) {
                    return Err(kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                        "provider event conversation does not match its durable run".to_owned(),
                    ));
                }
                let scope = security_scope_value(principal);
                if batch
                    .mutations
                    .iter()
                    .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &scope))
                {
                    return Err(kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                        "runtime graph effect scope is outside the current owner scope".to_owned(),
                    ));
                }
                let mut batch = batch.clone();
                if let Some(truth_graph) = default_acl_truth_graph {
                    let scope = batch.mutations.first().map(|mutation| mutation.scope.clone()).ok_or_else(|| {
                        kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                            "ACL-aware graph batch must not be empty".to_owned(),
                        )
                    })?;
                    let targets = acl_targets_for_graph_mutations(&batch.mutations).0;
                    let entity_ids = acl_target_entity_ids(&targets);
                    let existing = uow.graph_acl_records_for_entities(&scope, truth_graph, &entity_ids)?;
                    let security_scope = principal
                        .get("security_scope")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_ascii_lowercase);
                    let (acl_mutations, acl_jobs) = derive_default_acl_graph_effects_from_records(
                        &scope,
                        &batch.mutations,
                        targets,
                        existing,
                        truth_graph,
                        &owner_id.to_ascii_lowercase(),
                        security_scope.as_deref(),
                        effect_id,
                    )
                    .map_err(kogwistar_store_sqlite::SqliteStoreError::TransactionAborted)?;
                    batch.mutations.extend(acl_mutations);
                    batch.index_jobs.extend(acl_jobs);
                }
                uow.apply_canonical_graph_mutation_batch(batch)?;
                for publication in publications {
                    let event = &publication.event;
                    let message_id = format!("msg:{}", event.event_id);
                    let payload_json = serde_json::to_string(event).map_err(|error| {
                        kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                            format!("provider event serialization failed: {error}"),
                        )
                    })?;
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_err(|error| {
                            kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                                format!("system clock is before Unix epoch: {error}"),
                            )
                        })?
                        .as_secs() as i64;
                    if let Some(existing) = uow.get_projected_lane_message(&message_id)? {
                        let matches = existing.namespace == publication.namespace
                            && existing.purpose == "agent_event"
                            && existing.inbox_id == "agent-provider-events"
                            && existing.conversation_id == publication.conversation_id
                            && existing.recipient_id == "agent-event-consumer"
                            && existing.sender_id == "agent-runtime"
                            && existing.msg_type == "agent.provider_event"
                            && existing.run_id.as_deref() == Some(input.handoff.run_id.as_str())
                            && existing.step_id.as_deref() == Some(event.trace.step_id.as_str())
                            && existing.correlation_id.as_deref() == Some(event.event_id.as_str())
                            && existing.payload_json.as_deref() == Some(payload_json.as_str());
                        if !matches {
                            return Err(kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                                "provider event lane projection conflicts with the existing event".to_owned(),
                            ));
                        }
                    } else {
                        uow.project_lane_message(NewProjectedLaneMessage {
                            message_id,
                            namespace: publication.namespace.clone(),
                            purpose: "agent_event".to_owned(),
                            inbox_id: "agent-provider-events".to_owned(),
                            conversation_id: publication.conversation_id.clone(),
                            recipient_id: "agent-event-consumer".to_owned(),
                            sender_id: "agent-runtime".to_owned(),
                            msg_type: "agent.provider_event".to_owned(),
                            status: "pending".to_owned(),
                            created_at: now,
                            available_at: now,
                            run_id: Some(input.handoff.run_id.clone()),
                            step_id: Some(event.trace.step_id.clone()),
                            correlation_id: Some(event.event_id.clone()),
                            payload_json: Some(payload_json),
                            error_json: None,
                        })?;
                    }
                }
            }
            let handoff = input.handoff;
            let lane = uow
                .get_projected_lane_message(&handoff.message_id)?
                .ok_or_else(|| {
                    kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                        "claimed runtime lane does not exist".to_owned(),
                    )
                })?;
            validate_runtime_lane_executor(&lane, executor_target).map_err(|message| {
                kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                    message.to_owned(),
                )
            })?;
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
            if !applied.idempotent {
                if let Some((scope, owner_id, principal)) = provider_event_ack_scope.as_ref() {
                    if !capability_allowed_for_principal(
                        principal,
                        &self.capabilities,
                        "write_graph.commit",
                        &["write_graph"],
                    ) {
                        return Err(kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                            "provider event run owner no longer has write_graph authority".to_owned(),
                        ));
                    }
                    let run = uow.get_server_run(&runtime_run_id)?;
                    if !run.as_ref().is_some_and(|run| {
                        run.user_id.as_deref() == Some(owner_id.as_str())
                    }) {
                        return Err(kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                            "provider event run owner changed before acknowledgement".to_owned(),
                        ));
                    }
                    for acknowledgement in &provider_event_acknowledgements {
                        apply_sqlite_provider_event_acknowledgement(
                            uow,
                            acknowledgement,
                            &runtime_run_id,
                            &applied.reduced.state,
                            &transition.transition_id,
                            scope,
                            owner_id,
                            principal,
                        )?;
                    }
                }
            }
            if applied.idempotent {
                return Ok(applied);
            }
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
                            expected_event_seq: uow
                                .latest_server_run_event_seq(&transition.run_id)?,
                            resume_effect: None,
                        },
                    );
                    uow.project_lane_message(NewProjectedLaneMessage {
                        message_id,
                        namespace: "workflow".to_owned(),
                        purpose: "system".to_owned(),
                        inbox_id: "workflow-runtime".to_owned(),
                        conversation_id: transition.conversation_id.clone(),
                        recipient_id: runtime_worker_recipient_for_state(
                            &applied.reduced.state,
                            node_id,
                        ),
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
                        expected_event_seq: uow
                            .latest_server_run_event_seq(&transition.run_id)?,
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

    fn execute_native_runtime_step(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let input: RuntimeNativeExecuteRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let native_request = match native_step_request_from_payload(&input.handoff, input.payload) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::CONFLICT,
                    "KOGWISTAR_NATIVE_HANDOFF_CONFLICT",
                    error,
                );
            }
        };
        if native_request.operation == "aa_get_view_and_question" {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_NATIVE_ANSWER_VIEW_ASYNC_REQUIRED",
                "native answer-view execution requires the asynchronous application-service path",
            );
        }
        if native_request.operation == "aa_generate_answer_with_citations" {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_NATIVE_ANSWER_GENERATION_ASYNC_REQUIRED",
                "native answer generation requires the asynchronous application-service path",
            );
        }
        if native_request.operation == "aa_select_used_evidence"
            && crate::native_answer_generation::selection_snapshot_is_enabled(&native_request)
        {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_NATIVE_SELECTION_SNAPSHOT_ASYNC_REQUIRED",
                "default native evidence selection requires the asynchronous snapshot transaction path",
            );
        }
        self.execute_native_request(input.handoff, native_request)
    }

    async fn execute_native_runtime_step_async(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let input: RuntimeNativeExecuteRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let mut native_request =
            match native_step_request_from_payload(&input.handoff, input.payload) {
                Ok(value) => value,
                Err(error) => {
                    return effect_error(
                        StatusCode::CONFLICT,
                        "KOGWISTAR_NATIVE_HANDOFF_CONFLICT",
                        error,
                    );
                }
            };
        if native_request.operation == "aa_generate_answer_with_citations" {
            let result = match self
                .execute_sqlite_native_answer_generation(input.handoff.clone(), native_request)
                .await
            {
                Ok(result) => result,
                Err(_) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_CHAT_PROVIDER_UNAVAILABLE",
                        "Rust-native structured chat provider is temporarily unavailable; the work remains unacknowledged",
                    );
                }
            };
            return self.apply_runtime_result_input(
                result,
                kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                None,
            );
        } else if native_request.operation == "aa_get_view_and_question" {
            match self
                .native_answer_view_context(&native_request, Some(&request.principal))
                .await
            {
                Ok(view) => {
                    native_request
                        .ephemeral_context
                        .insert("authorized_answer_view".to_owned(), view);
                }
                Err(NativeEvidenceContextError::CallerNotRunOwner) => {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                        "native answer view requires the authenticated run owner",
                    );
                }
                Err(NativeEvidenceContextError::OwnerAuthorization) => {
                    return self.apply_runtime_result_input(
                        failed_native_result(
                            input.handoff,
                            &native_request,
                            "KOGWISTAR_CAPABILITY_FORBIDDEN",
                            "native answer view requires the run owner's current read_graph capability and admitted scope",
                        ),
                        kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                        None,
                    );
                }
                Err(NativeEvidenceContextError::GraphReadRejected) => {
                    return self.apply_runtime_result_input(
                        failed_native_result(
                            input.handoff,
                            &native_request,
                            "KOGWISTAR_ANSWER_VIEW_REJECTED",
                            "native answer view requires an active user turn in the admitted conversation",
                        ),
                        kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                        None,
                    );
                }
                Err(NativeEvidenceContextError::GraphReadUnavailable) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_ANSWER_VIEW_UNAVAILABLE",
                        "native answer-view storage is temporarily unavailable; the work remains unacknowledged",
                    );
                }
            }
        } else if native_request.operation == "aa_retrieve_candidates" {
            match self
                .native_candidate_context(&native_request, Some(&request.principal))
                .await
            {
                Ok(records) => {
                    native_request
                        .ephemeral_context
                        .insert("authorized_candidate_records".to_owned(), records);
                }
                Err(NativeEvidenceContextError::CallerNotRunOwner) => {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                        "native graph reads require the authenticated run owner",
                    );
                }
                Err(NativeEvidenceContextError::OwnerAuthorization) => {
                    return self.apply_runtime_result_input(
                        failed_native_result(
                            input.handoff,
                            &native_request,
                            "KOGWISTAR_CAPABILITY_FORBIDDEN",
                            "native candidate retrieval requires the run owner's current read_graph capability and admitted scope",
                        ),
                        kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                        None,
                    );
                }
                Err(NativeEvidenceContextError::GraphReadRejected) => {
                    return self.apply_runtime_result_input(
                        failed_native_result(
                            input.handoff,
                            &native_request,
                            "KOGWISTAR_CANDIDATE_RETRIEVAL_REJECTED",
                            "native candidate retrieval configuration or input is invalid",
                        ),
                        kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                        None,
                    );
                }
                Err(NativeEvidenceContextError::GraphReadUnavailable) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_CANDIDATE_RETRIEVAL_UNAVAILABLE",
                        "native candidate retrieval is temporarily unavailable; the work remains unacknowledged",
                    );
                }
            }
        } else if native_request.operation == "aa_materialize_evidence_pack" {
            match self
                .native_evidence_context(&native_request, Some(&request.principal))
                .await
            {
                Ok(records) => {
                    native_request
                        .ephemeral_context
                        .insert("authorized_evidence_records".to_owned(), records);
                }
                Err(NativeEvidenceContextError::CallerNotRunOwner) => {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                        "native evidence reads require the authenticated run owner",
                    );
                }
                Err(NativeEvidenceContextError::OwnerAuthorization) => {
                    return self.apply_runtime_result_input(
                        failed_native_result(
                            input.handoff,
                            &native_request,
                            "KOGWISTAR_CAPABILITY_FORBIDDEN",
                            "native evidence materialization requires the run owner's current read_graph capability and admitted scope",
                        ),
                        kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                        None,
                    );
                }
                Err(NativeEvidenceContextError::GraphReadRejected) => {
                    return self.apply_runtime_result_input(
                        failed_native_result(
                            input.handoff,
                            &native_request,
                            "KOGWISTAR_EVIDENCE_READ_FAILED",
                            "native evidence materialization could not read the selected knowledge records",
                        ),
                        kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                        None,
                    );
                }
                Err(NativeEvidenceContextError::GraphReadUnavailable) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_EVIDENCE_READ_UNAVAILABLE",
                        "native evidence storage is temporarily unavailable; the work remains unacknowledged",
                    );
                }
            }
        }
        if native_request.operation == "aa_select_used_evidence"
            && crate::native_answer_generation::selection_snapshot_is_enabled(&native_request)
        {
            if self
                .persist_sqlite_native_selection_context_snapshot(&native_request)
                .await
                .is_err()
            {
                return effect_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "KOGWISTAR_CONTEXT_SNAPSHOT_UNAVAILABLE",
                    "native evidence-selection context snapshot could not be committed; the work remains unacknowledged",
                );
            }
            let result = match native_result_from_request(
                input.handoff.clone(),
                native_request,
                &self.native_operations,
            ) {
                Ok(result) => result,
                Err(_) => {
                    return effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                        "native evidence selection rejected its bounded input",
                    );
                }
            };
            return self.apply_runtime_result_input(
                result,
                kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
                None,
            );
        }
        self.execute_native_request_async(input.handoff, native_request)
            .await
    }

    fn execute_native_request(
        &self,
        handoff: RecordedWorkerHandoff,
        native_request: kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> ApiEffectResponse {
        let effect = match kogwistar_runtime::execute_native_step_with_registry(
            &native_request,
            &self.native_operations,
        ) {
            Ok(effect) => effect,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                    error.to_string(),
                );
            }
        };
        let result = RuntimeWorkerResultRequest {
            handoff,
            transition: None,
            effect: Some(effect),
        };
        self.apply_runtime_result_input(
            result,
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
            None,
        )
    }

    async fn execute_native_request_async(
        &self,
        handoff: RecordedWorkerHandoff,
        native_request: kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> ApiEffectResponse {
        let mut result = match kogwistar_runtime::execute_native_step_with_registry(
            &native_request,
            &self.native_operations,
        ) {
            Ok(effect) => RuntimeWorkerResultRequest {
                handoff,
                transition: None,
                effect: Some(effect),
            },
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_NATIVE_EXECUTION_REJECTED",
                    error.to_string(),
                );
            }
        };
        let needs_graph_authority = result.effect.as_ref().is_some_and(|effect| {
            effect.graph_mutation_batch.is_some()
                || effect.graph_mutation_batch_v2.is_some()
                || !effect.provider_event_acknowledgements.is_empty()
        });
        let principal = if needs_graph_authority {
            match self
                .resolve_inline_native_owner_principal(&result.handoff.run_id, "write_graph")
                .await
            {
                Ok(principal) => Some(principal),
                Err(InlineOwnerPrincipalError::Denied(_)) => {
                    let effect = result
                        .effect
                        .as_mut()
                        .expect("graph authority was requested for an effect");
                    fail_inline_graph_effect(
                        effect,
                        "KOGWISTAR_GRAPH_AUTHORIZATION_DENIED",
                        "inline graph effects failed current owner authorization",
                    );
                    None
                }
                Err(InlineOwnerPrincipalError::Unavailable(_)) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_RUN_OWNER_UNAVAILABLE",
                        "inline graph effect could not revalidate the run owner",
                    );
                }
            }
        } else {
            None
        };
        self.apply_runtime_result_input(
            result,
            kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
            principal.as_ref(),
        )
    }
}

fn sqlite_native_answer_view_from_visible_records(
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
    conversation_id: &str,
    turn_id: &str,
) -> Result<Value, NativeEvidenceContextError> {
    fn record_document(record: &GraphRecord) -> Option<Value> {
        record
            .document
            .as_deref()
            .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
    }

    fn field<'a>(record: &'a GraphRecord, document: &'a Value, key: &str) -> Option<&'a Value> {
        let metadata = document.get("metadata").and_then(Value::as_object);
        document
            .get(key)
            .or_else(|| metadata.and_then(|metadata| metadata.get(key)))
            .or_else(|| record.metadata.get(key))
    }

    let Some(turn) = nodes.iter().find(|record| record.id == turn_id) else {
        return Err(NativeEvidenceContextError::GraphReadRejected);
    };
    let Some(turn_document) = record_document(turn) else {
        return Err(NativeEvidenceContextError::GraphReadRejected);
    };
    if field(turn, &turn_document, "entity_type").and_then(Value::as_str)
        != Some("conversation_turn")
        || field(turn, &turn_document, "conversation_id").and_then(Value::as_str)
            != Some(conversation_id)
        || field(turn, &turn_document, "role").and_then(Value::as_str) != Some("user")
        || field(turn, &turn_document, "lifecycle_status")
            .and_then(Value::as_str)
            .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
    {
        return Err(NativeEvidenceContextError::GraphReadRejected);
    }

    let record_value = |record: &GraphRecord| {
        json!({
            "id": record.id,
            "document": record.document,
            "metadata": record.metadata
        })
    };
    let context_nodes = nodes
        .iter()
        .filter(|record| {
            record_document(record).is_some_and(|document| {
                field(record, &document, "conversation_id").and_then(Value::as_str)
                    == Some(conversation_id)
            })
        })
        .map(record_value)
        .collect::<Vec<_>>();
    let conversation_document_id = format!("conv:{conversation_id}");
    let context_edges = edges
        .iter()
        .filter(|record| {
            record.metadata.get("doc_id").and_then(Value::as_str)
                == Some(conversation_document_id.as_str())
                || record
                    .metadata
                    .get("conversation_id")
                    .and_then(Value::as_str)
                    == Some(conversation_id)
        })
        .map(record_value)
        .collect::<Vec<_>>();

    Ok(json!({
        "turn": record_value(turn),
        "context_nodes": context_nodes,
        "context_edges": context_edges
    }))
}

fn chroma_record_matches_principal_scope(record: &GraphRecord, principal: &Value) -> bool {
    let workspace = principal
        .get("workspace_id")
        .or_else(|| principal.get("workspace"))
        .and_then(Value::as_str);
    if workspace.is_some_and(|workspace| {
        record.metadata.get("workspace_id").and_then(Value::as_str) != Some(workspace)
    }) {
        return false;
    }
    let graph_space = principal.get("graph_space").and_then(Value::as_str);
    if graph_space.is_some_and(|graph_space| {
        record.metadata.get("graph_space").and_then(Value::as_str) != Some(graph_space)
    }) {
        return false;
    }
    true
}

fn conversation_record_from_raw_event(
    event: &SqliteRawEntityEvent,
) -> Result<GraphRecord, SqliteStoreError> {
    let payload = serde_json::from_str::<Value>(&event.payload_json).map_err(|error| {
        SqliteStoreError::TransactionAborted(format!(
            "conversation event {:?} has invalid canonical JSON: {error}",
            event.event_id
        ))
    })?;
    let metadata = payload
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| {
            SqliteStoreError::TransactionAborted(format!(
                "conversation event {:?} has no metadata object",
                event.event_id
            ))
        })?;
    Ok(GraphRecord {
        id: event.entity_id.clone(),
        document: None,
        metadata,
        embedding: None,
    })
}

fn sqlite_native_candidate_values(
    candidates: &[GraphRecord],
    acl_records: &[GraphRecord],
    truth_graph: &str,
    principal: &GraphAclPrincipal,
) -> Value {
    let mut acl_inputs = candidates
        .iter()
        .filter(|record| {
            !record
                .metadata
                .get("lifecycle_status")
                .and_then(Value::as_str)
                .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
        })
        .cloned()
        .collect::<Vec<_>>();
    acl_inputs.extend_from_slice(acl_records);
    let (visible_candidates, _) =
        graph_records_visible_by_acl(&acl_inputs, &[], truth_graph, principal);
    let mut projected = visible_candidates
        .into_iter()
        .map(|record| {
            json!({
                "id": record.id,
                "label": record.metadata.get("label").cloned().unwrap_or(Value::Null),
                "summary": record.metadata.get("summary").cloned().unwrap_or(Value::Null),
                "doc": record.document.unwrap_or_default(),
            })
        })
        .collect::<Vec<_>>();
    projected.sort_by(|left, right| {
        left.get("id")
            .and_then(Value::as_str)
            .cmp(&right.get("id").and_then(Value::as_str))
    });
    Value::Array(projected)
}

fn apply_sqlite_provider_event_acknowledgement(
    uow: &mut SqliteUnitOfWork<'_>,
    acknowledgement: &kogwistar_runtime::ProviderEventAcknowledgementV1,
    run_id: &str,
    state: &RecordedRuntimeState,
    effect_id: &str,
    scope: &GraphScope,
    owner_id: &str,
    principal: &Value,
) -> Result<(), kogwistar_store_sqlite::SqliteStoreError> {
    let lane = uow
        .get_projected_lane_message(&acknowledgement.message_id)?
        .ok_or_else(|| {
            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                "provider event acknowledgement lane does not exist".to_owned(),
            )
        })?;
    let run = uow.get_server_run(run_id)?;
    if !run
        .as_ref()
        .is_some_and(|run| run.user_id.as_deref() == Some(owner_id))
    {
        return Err(
            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                "provider event run owner changed before acknowledgement".to_owned(),
            ),
        );
    }
    let event: kogwistar_contracts::DurableProviderEventV1 = lane
        .payload_json
        .as_deref()
        .and_then(|payload| serde_json::from_str(payload).ok())
        .ok_or_else(|| {
            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                "provider event payload is unavailable for causal linkage".to_owned(),
            )
        })?;
    if acknowledgement.message_id != format!("msg:{}", acknowledgement.event_id)
        || lane.namespace != scope.namespace
        || lane.purpose != "agent_event"
        || lane.inbox_id != "agent-provider-events"
        || lane.msg_type != "agent.provider_event"
        || lane.recipient_id != "agent-event-consumer"
        || lane.status != "claimed"
        || lane.claimed_by.as_deref() != Some(acknowledgement.claimed_by.as_str())
        || lane.run_id.as_deref() != Some(run_id)
        || lane.conversation_id
            != run
                .as_ref()
                .map(|run| run.conversation_id.as_str())
                .unwrap_or_default()
        || lane.correlation_id.as_deref() != Some(acknowledgement.event_id.as_str())
        || event.event_id != acknowledgement.event_id
        || event.trace.run_id != run_id
        || state.conversation_id != lane.conversation_id
        || state.run_id != run_id
    {
        return Err(
            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                "provider event acknowledgement is not bound to the active run claim".to_owned(),
            ),
        );
    }
    let handler_step_seq = state
        .state
        .get("agent_event_handling_step_seq")
        .and_then(Value::as_i64)
        .filter(|step_seq| *step_seq >= 0)
        .ok_or_else(|| {
            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                "provider event has no valid durable handler step sequence".to_owned(),
            )
        })?;
    let mut referenced_step_seqs = vec![handler_step_seq];
    if let Some(producer_step_seq) = event.trace.producer_step_seq {
        if producer_step_seq < 0 {
            return Err(
                kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                    "provider event has an invalid producer step sequence".to_owned(),
                ),
            );
        }
        referenced_step_seqs.push(producer_step_seq);
    }
    referenced_step_seqs.sort_unstable();
    referenced_step_seqs.dedup();
    let recorded_steps = uow.recorded_runtime_steps(run_id, &referenced_step_seqs)?;
    let snapshots = provider_event_step_snapshots(
        recorded_steps,
        run_id,
        &lane.conversation_id,
        &state.workflow_id,
        &referenced_step_seqs,
    )
    .map_err(kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict)?;
    let workflow_node_id = state
        .last_node_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                "provider event handling step has no workflow node identity".to_owned(),
            )
        })?;
    let step_seq = state.last_step_seq;
    if step_seq < 0
        || handler_step_seq >= step_seq
        || event
            .trace
            .producer_step_seq
            .is_some_and(|producer_step_seq| producer_step_seq >= step_seq)
    {
        return Err(
            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                "provider event causal steps must precede the acknowledgement step".to_owned(),
            ),
        );
    }
    let security_scope = security_scope_value(principal);
    if !graph_mutation_scope_allowed(scope, &security_scope) {
        return Err(
            kogwistar_store_sqlite::SqliteStoreError::RecordedRuntimeConflict(
                "provider event handling graph scope changed before commit".to_owned(),
            ),
        );
    }
    let mut causal_mutations = provider_event_ack_graph_mutations(
        &acknowledgement.event_id,
        run_id,
        &lane.conversation_id,
        &state.workflow_id,
        workflow_node_id,
        step_seq,
        effect_id,
        scope,
    );
    causal_mutations.extend(provider_event_step_link_mutations(
        &acknowledgement.event_id,
        run_id,
        &lane.conversation_id,
        event.trace.producer_step_seq,
        handler_step_seq,
        &snapshots,
        scope,
    ));
    let mut linked_step_nodes = BTreeSet::new();
    causal_mutations.retain(|mutation| {
        mutation.entity_kind != "node" || linked_step_nodes.insert(mutation.record.id.clone())
    });
    let (targets, _) = acl_targets_for_graph_mutations(&causal_mutations);
    let entity_ids = acl_target_entity_ids(&targets);
    let existing = uow.graph_acl_records_for_entities(scope, "conversation", &entity_ids)?;
    let (acl_mutations, acl_jobs) = derive_default_acl_graph_effects_from_records(
        scope,
        &causal_mutations,
        targets,
        existing,
        "conversation",
        &owner_id.to_ascii_lowercase(),
        security_scope["security_scope"].as_str(),
        &acknowledgement.event_id,
    )
    .map_err(kogwistar_store_sqlite::SqliteStoreError::TransactionAborted)?;
    causal_mutations.extend(acl_mutations);
    uow.apply_canonical_graph_mutation_batch(GraphMutationBatch {
        mutations: causal_mutations,
        index_jobs: acl_jobs,
    })?;
    uow.ack_claimed_projected_lane_message(
        &acknowledgement.message_id,
        &acknowledgement.claimed_by,
    )?;
    uow.append_server_run_event(
        run_id,
        "workflow.provider_event.handled.v1",
        serde_json::to_string(&json!({
            "event_id": acknowledgement.event_id,
            "message_id": acknowledgement.message_id,
            "workflow_node_id": workflow_node_id,
            "step_seq": step_seq,
            "effect_id": effect_id,
        }))?,
    )?;
    Ok(())
}
