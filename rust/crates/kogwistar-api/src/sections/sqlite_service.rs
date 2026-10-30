#[derive(Clone)]
pub struct SqliteRunApplicationService {
    store: SqliteStore,
    chroma_graph: Option<kogwistar_store_chroma::ChromaGraphStore>,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    max_queue: usize,
    capabilities: SharedCapabilityState,
    native_operations: kogwistar_runtime::NativeOperationRegistry,
}

#[derive(Clone)]
struct SqliteChromaGraphConfig {
    endpoint: String,
    metric: DistanceMetric,
    node_collection: String,
    edge_collection: String,
}

const STAGE1_RECOVERY_PROJECTION_SCHEMA_VERSION: i64 = 2;

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|number| number != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
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
        let model_class = match revision.entity_kind.as_str() {
            "node" => GraphModelClass::Node,
            "edge" => GraphModelClass::Edge,
            _ => {
                return Err(StoreError::Backend {
                    backend: "sqlite-chroma-projection".to_owned(),
                    message: "Stage-1 recovery supports only Node and Edge events".to_owned(),
                });
            }
        };
        GraphModelEnvelope::new(model_class, revision.payload.clone())
            .decode_persistent_v1()
            .map_err(|error| StoreError::InvalidEntityEventPayload {
                message: format!("canonical graph model cannot be projected: {error}"),
            })?;

        let mut document_payload = revision.payload.clone();
        let Some(entity) = document_payload.as_object_mut() else {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "canonical graph model payload must be an object".to_owned(),
            });
        };
        if entity.get("id").and_then(Value::as_str) != Some(revision.entity_id.as_str()) {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "canonical graph model ID does not match the event identity".to_owned(),
            });
        }
        entity.remove("embedding");
        if revision.entity_kind == "node" {
            entity.remove("metadata");
        }
        let document = kogwistar_contracts::canonical_json(&document_payload, false);

        let source_metadata = revision
            .payload
            .get("metadata")
            .and_then(Value::as_object);
        let mut metadata = source_metadata.cloned().unwrap_or_default();
        if metadata
            .get("namespace")
            .is_some_and(|value| {
                !value.is_null() && value.as_str() != Some(revision.namespace.as_str())
            })
        {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "canonical graph metadata namespace conflicts with its event scope".to_owned(),
            });
        }
        metadata.insert(
            "namespace".to_owned(),
            Value::String(revision.namespace.clone()),
        );
        if revision.entity_kind == "node" {
            metadata.retain(|_, value| !value.is_null());
            for field in ["doc_id", "label", "type", "summary", "domain_id", "canonical_entity_id"] {
                if let Some(value) = revision.payload.get(field).filter(|value| !value.is_null()) {
                    metadata.insert(field.to_owned(), value.clone());
                }
            }
            if let Some(value) = revision.payload.get("properties").filter(|value| !value.is_null()) {
                metadata.insert(
                    "properties".to_owned(),
                    Value::String(kogwistar_contracts::canonical_json(value, false)),
                );
            }
            if let Some(value) = revision.payload.get("mentions").filter(|value| !value.is_null()) {
                metadata.insert(
                    "mentions".to_owned(),
                    Value::String(kogwistar_contracts::canonical_json(value, false)),
                );
            }
        } else {
            for field in [
                "doc_id", "relation", "type", "summary", "domain_id", "canonical_entity_id",
            ] {
                if let Some(value) = revision.payload.get(field).filter(|value| !value.is_null()) {
                    metadata.insert(field.to_owned(), value.clone());
                }
            }
            for field in ["source_ids", "target_ids", "source_edge_ids", "target_edge_ids", "properties"] {
                if let Some(value) = revision.payload.get(field).filter(|value| !value.is_null()) {
                    metadata.insert(
                        field.to_owned(),
                        Value::String(kogwistar_contracts::canonical_json(value, false)),
                    );
                }
            }
            if let Some(value) = revision.payload.get("mentions").filter(|value| !value.is_null()) {
                metadata.insert(
                    "references".to_owned(),
                    Value::String(kogwistar_contracts::canonical_json(value, false)),
                );
            }
            for (target, sources) in [
                ("wf_priority", ["wf_priority", "priority"]),
                ("wf_is_default", ["wf_is_default", "is_default"]),
                ("wf_predicate", ["wf_predicate", "predicate"]),
                ("wf_multiplicity", ["wf_multiplicity", "multiplicity"]),
            ] {
                let value = sources
                    .iter()
                    .filter_map(|key| revision.payload["metadata"].get(*key))
                    .find(|value| json_truthy(value));
                if let Some(value) = value {
                    metadata.insert(target.to_owned(), value.clone());
                }
            }
            let node_endpoints = ["source_ids", "target_ids"]
                .iter()
                .filter_map(|key| revision.payload.get(*key).and_then(Value::as_array))
                .map(Vec::len)
                .sum::<usize>();
            let edge_endpoints = ["source_edge_ids", "target_edge_ids"]
                .iter()
                .filter_map(|key| revision.payload.get(*key).and_then(Value::as_array))
                .map(Vec::len)
                .sum::<usize>();
            metadata.insert("node_endpoint_count".to_owned(), Value::from(node_endpoints));
            metadata.insert("edge_endpoint_count".to_owned(), Value::from(edge_endpoints));
            metadata.insert(
                "total_endpoint_count".to_owned(),
                Value::from(node_endpoints + edge_endpoints),
            );
        }
        metadata
            .entry("lifecycle_status".to_owned())
            .or_insert_with(|| Value::String("active".to_owned()));
        if revision.entity_kind == "edge" {
            metadata
                .entry("redirect_to_id".to_owned())
                .or_insert(Value::Null);
        }

        let mut payload = serde_json::Map::new();
        payload.insert("id".to_owned(), Value::String(revision.entity_id.clone()));
        payload.insert("entity_kind".to_owned(), Value::String(revision.entity_kind.clone()));
        payload.insert("document".to_owned(), Value::String(document));
        payload.insert("metadata".to_owned(), Value::Object(metadata));
        payload.insert("source_fingerprint".to_owned(), Value::String(fingerprint.to_owned()));
        Ok(NamedProjectionWrite {
            payload,
            last_authoritative_seq: revision.revision,
            last_materialized_seq: revision.revision,
            projection_schema_version: STAGE1_RECOVERY_PROJECTION_SCHEMA_VERSION,
            materialization_status: "pending".to_owned(),
        })
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
    ) -> impl std::future::Future<Output = Result<kogwistar_store::ProjectionWriteOutcome, StoreError>> + Send
    {
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
    ) -> impl std::future::Future<Output = Result<kogwistar_store::ProjectionWriteOutcome, StoreError>> + Send
    {
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
                read_canonical_entity_revision(
                    &[event],
                    &namespace,
                    &entity_kind,
                    &record.id,
                )
                .ok()
                .flatten()
                .is_some_and(|revision| revision.revision == source_revision && !revision.is_deleted())
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
            records.retain(|record| {
                Self::stage2_record_is_current(record, latest.get(&record.id))
            });
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
                chroma_graph: None,
                embedding_provider: None,
                max_queue: runtime_max_queue(),
                capabilities: Arc::new(Mutex::new(CapabilityState::default())),
                native_operations: kogwistar_runtime::NativeOperationRegistry::default(),
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

    /// Add a separate Chroma graph projection to the SQLite metadata/runtime
    /// service. SQLite remains authoritative only for its existing metadata
    /// and queue responsibilities; graph reads are sourced from Chroma.
    pub fn with_chroma_graph(
        mut self,
        graph: kogwistar_store_chroma::ChromaGraphStore,
    ) -> Self {
        self.chroma_graph = Some(graph);
        self
    }

    pub fn with_embedding_provider(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(provider);
        self
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
                let fingerprint = RevisionFencedChromaProjectionStore::canonical_fingerprint(&current)
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
                    return Err("Stage-1 recovery did not persist the current canonical revision".to_owned());
                }
                // Never enqueue a revision that changed while the repair was
                // being written; a later reconciliation can repair its row.
                let latest = sqlite_canonical_entity_revision(
                    &self.store,
                    namespace,
                    kind_name,
                    &entity_id,
                )
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
                    .graph_records_of_kind_page(
                        namespace,
                        kind,
                        &filter,
                        offset,
                        page_size as u32,
                    )
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
                        if restored_stage1
                            .iter()
                            .any(|(restored_kind, restored_id)| {
                                *restored_kind == kind_name && restored_id == &record.id
                            })
                        {
                            continue;
                        }
                        let stage1 = RevisionFencedChromaProjectionStore::restore_stage1_from_revision(
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
    ) -> Result<(Vec<GraphRecord>, Vec<GraphRecord>), ApiEffectResponse> {
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
            filter
                .equals
                .insert("workspace_id".to_owned(), Value::String(workspace.to_owned()));
        }
        if let Some(graph_space) = request.principal.get("graph_space").and_then(Value::as_str) {
            filter
                .equals
                .insert("graph_space".to_owned(), Value::String(graph_space.to_owned()));
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
        if let Err(error) = self.filter_current_chroma_stage2(namespace, &mut nodes, &mut edges) {
            return Err(effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error,
            ));
        }
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (nodes, edges) = graph_records_visible_by_acl(&nodes, &edges, truth_graph, &principal);
        Ok((nodes, edges))
    }

    async fn chroma_graph_neighbors_effect(
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
        let (nodes, edges) = match self
            .chroma_graph_snapshot(request, input.truth_graph.trim())
            .await
        {
            Ok(records) => records,
            Err(response) => return response,
        };
        let neighbors = graph_neighbors_from_records(
            input.rid.trim(),
            &nodes,
            &edges,
            input.doc_id.as_deref(),
        );
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
        let (nodes, edges) = match self.chroma_graph_snapshot(request, "knowledge").await {
            Ok(records) => records,
            Err(response) => return response,
        };
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
        if input.src_id.trim().is_empty()
            || input.dst_id.trim().is_empty()
            || input.max_depth > 128
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "src_id and dst_id are required and max_depth must be <= 128",
            );
        }
        let (nodes, edges) = match self.chroma_graph_snapshot(request, "knowledge").await {
            Ok(records) => records,
            Err(response) => return response,
        };
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
        let (nodes, edges) = match self.chroma_graph_snapshot(request, "knowledge").await {
            Ok(records) => records,
            Err(response) => return response,
        };
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
            metadata
                .equals
                .insert("workspace_id".to_owned(), Value::String(workspace.to_owned()));
        }
        if let Some(graph_space) = request.principal.get("graph_space").and_then(Value::as_str) {
            metadata
                .equals
                .insert("graph_space".to_owned(), Value::String(graph_space.to_owned()));
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
            let nodes = match graph
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
        let acl_records = match graph
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
            ids.push(matches.iter().map(|matched| json!(matched.record.id)).collect::<Vec<_>>());
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
        let worker_protocol = worker_protocol_for_runtime_kind(&runtime_kind);
        let event_payload = json!({
            "run_id": run_id,
            "run_kind": "workflow_runtime",
            "conversation_id": conversation_id,
            "workflow_id": workflow_id,
            "status": "queued",
            "turn_node_id": turn_node_id,
            "implementation_attestation": runtime_implementation_attestation(&runtime_kind),
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
        if is_answer_workflow_id(workflow_id) {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_ANSWER_WORKFLOW_NOT_READY",
                "SQLite runtime has no conversation graph authority for answer workflows",
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
                    &claim_owner,
                    run_id,
                    claimed_payload,
                    expected_event_seq,
                ));
            }
                Ok(values)
            })
            .map_err(|error| error.to_string())
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
        self.apply_runtime_result_input(input, "python-worker")
    }

    fn apply_runtime_result_input(
        &self,
        input: RuntimeWorkerResultRequest,
        executor_target: &str,
    ) -> ApiEffectResponse {
        if input.transition.is_some() == input.effect.is_some() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "exactly one of transition or effect is required",
            );
        }
        if input
            .effect
            .as_ref()
            .is_some_and(|effect| {
                effect.graph_mutation_batch.is_some() || effect.graph_mutation_batch_v2.is_some()
            })
        {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "Worker graph effects require atomic graph authority with index jobs",
            );
        }
        if let Some(effect) = input.effect.as_ref()
            && let Err(message) =
                validate_python_worker_submission_trace_binding(
                    &effect.trace_events,
                    &input.handoff.claimed_by,
                )
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_WORKER_TRACE",
                message,
            );
        }
        let outcome = self.store.immediate_transaction(|uow| {
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
            handoff: input.handoff,
            transition: None,
            effect: Some(effect),
        };
        self.apply_runtime_result_input(result, kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET)
    }
}
