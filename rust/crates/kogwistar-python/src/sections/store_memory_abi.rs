// Snapshot/in-memory store bridge DTOs and adapter. Python parity: Kogwistar engine-core store bridge contracts.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StoreRequest {
    snapshot: StoreSnapshot,
    operation: StoreOperation,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoreSnapshot {
    #[serde(default)]
    records: Vec<SnapshotRecord>,
    #[serde(default)]
    events: Vec<SnapshotEvent>,
    #[serde(default)]
    cursors: Vec<SnapshotCursor>,
    #[serde(default)]
    projections: Vec<SnapshotProjection>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SnapshotRecord {
    namespace: String,
    id: String,
    #[serde(default)]
    document: Option<String>,
    #[serde(default)]
    metadata: Map<String, Value>,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SnapshotEvent {
    namespace: String,
    seq: i64,
    event_id: String,
    entity_kind: String,
    entity_id: String,
    op: String,
    payload: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SnapshotCursor {
    namespace: String,
    consumer: String,
    last_seq: i64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SnapshotProjection {
    namespace: String,
    key: String,
    payload: Map<String, Value>,
    last_authoritative_seq: i64,
    last_materialized_seq: i64,
    projection_schema_version: i64,
    materialization_status: String,
    updated_at_ms: i64,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum StoreOperation {
    GraphRecord {
        namespace: String,
        id: String,
    },
    GraphRecords {
        namespace: String,
        #[serde(default)]
        metadata: BTreeMap<String, Value>,
    },
    VectorQuery {
        namespace: String,
        embedding: Vec<f32>,
        limit: usize,
        #[serde(default)]
        metadata: BTreeMap<String, Value>,
        #[serde(default)]
        metric: SnapshotDistanceMetric,
    },
    ReplayEvents {
        namespace: String,
        after_seq: i64,
        limit: usize,
    },
    ReplayCursor {
        namespace: String,
        consumer: String,
    },
    LatestEventSeq {
        namespace: String,
    },
    NamedProjection {
        namespace: String,
        key: String,
    },
    NamedProjections {
        namespace: String,
    },
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SnapshotDistanceMetric {
    #[default]
    Cosine,
    L2,
    Ip,
}

impl From<SnapshotDistanceMetric> for DistanceMetric {
    fn from(value: SnapshotDistanceMetric) -> Self {
        match value {
            SnapshotDistanceMetric::Cosine => Self::Cosine,
            SnapshotDistanceMetric::L2 => Self::L2,
            SnapshotDistanceMetric::Ip => Self::InnerProduct,
        }
    }
}

fn metadata_filter(equals: BTreeMap<String, Value>) -> MetadataFilter {
    MetadataFilter { equals }
}

fn graph_record_json(record: GraphRecord) -> Value {
    json!({
        "id": record.id,
        "document": record.document,
        "metadata": record.metadata,
        "embedding": record.embedding,
    })
}

fn event_json(event: contracts::EntityEventEnvelope) -> Value {
    json!({
        "namespace": event.namespace,
        "seq": event.seq,
        "event_id": event.event_id,
        "entity_kind": event.entity_kind,
        "entity_id": event.entity_id,
        "op": event.op,
        "payload": event.payload,
    })
}

fn projection_json(projection: NamedProjection) -> Value {
    json!({
        "namespace": projection.namespace,
        "key": projection.key,
        "payload": projection.payload,
        "last_authoritative_seq": projection.last_authoritative_seq,
        "last_materialized_seq": projection.last_materialized_seq,
        "projection_schema_version": projection.projection_schema_version,
        "materialization_status": projection.materialization_status,
        "updated_at_ms": projection.updated_at_ms,
    })
}

fn workflow_design_snapshot_json(snapshot: WorkflowDesignSnapshot) -> Value {
    json!({
        "workflow_id": snapshot.workflow_id,
        "version": snapshot.version,
        "seq": snapshot.seq,
        "payload_json": snapshot.payload_json,
        "schema_version": snapshot.schema_version,
        "created_at_ms": snapshot.created_at_ms,
    })
}

fn workflow_design_delta_json(delta: WorkflowDesignDelta) -> Value {
    json!({
        "workflow_id": delta.workflow_id,
        "version": delta.version,
        "prev_version": delta.prev_version,
        "target_seq": delta.target_seq,
        "forward_json": delta.forward_json,
        "inverse_json": delta.inverse_json,
        "schema_version": delta.schema_version,
        "created_at_ms": delta.created_at_ms,
    })
}

fn workflow_design_snapshot_write(
    version: i64,
    seq: i64,
    payload_json: String,
    schema_version: i64,
) -> WorkflowDesignSnapshotWrite {
    WorkflowDesignSnapshotWrite {
        version,
        seq,
        payload_json,
        schema_version,
    }
}

fn workflow_design_delta_write(
    version: i64,
    prev_version: i64,
    target_seq: i64,
    forward_json: String,
    inverse_json: String,
    schema_version: i64,
) -> WorkflowDesignDeltaWrite {
    WorkflowDesignDeltaWrite {
        version,
        prev_version,
        target_seq,
        forward_json,
        inverse_json,
        schema_version,
    }
}

fn parse_store_request(payload_json: &str) -> Result<StoreRequest, (&'static str, String)> {
    let value: Value = serde_json::from_str(payload_json)
        .map_err(|error| (STORE_INVALID_JSON, format!("invalid JSON: {error}")))?;
    serde_json::from_value(value).map_err(|error| {
        (
            STORE_INVALID_PAYLOAD,
            format!("invalid store payload: {error}"),
        )
    })
}

fn store_memory_read_json_impl(payload_json: &str) -> Result<String, (&'static str, String)> {
    let request = parse_store_request(payload_json)?;
    block_on(async move {
        let store = InMemoryStore::new();
        for record in request.snapshot.records {
            store
                .upsert_graph_record(
                    &record.namespace,
                    GraphRecord {
                        id: record.id,
                        document: record.document,
                        metadata: record.metadata,
                        embedding: record.embedding,
                    },
                )
                .await
                .map_err(|error| (STORE_INVALID_PAYLOAD, error.to_string()))?;
        }
        store
            .restore_event_snapshot(
                request
                    .snapshot
                    .events
                    .into_iter()
                    .map(|event| contracts::EntityEventEnvelope {
                        namespace: event.namespace,
                        seq: event.seq,
                        event_id: event.event_id,
                        entity_kind: event.entity_kind,
                        entity_id: event.entity_id,
                        op: event.op,
                        payload: event.payload,
                    })
                    .collect(),
                request
                    .snapshot
                    .cursors
                    .into_iter()
                    .map(|cursor| ReplayCursor {
                        namespace: cursor.namespace,
                        consumer: cursor.consumer,
                        last_seq: cursor.last_seq,
                    })
                    .collect(),
            )
            .map_err(|error| (STORE_INVALID_PAYLOAD, error.to_string()))?;
        store
            .restore_projection_snapshot(
                request
                    .snapshot
                    .projections
                    .into_iter()
                    .map(|projection| NamedProjection {
                        namespace: projection.namespace,
                        key: projection.key,
                        payload: projection.payload,
                        last_authoritative_seq: projection.last_authoritative_seq,
                        last_materialized_seq: projection.last_materialized_seq,
                        projection_schema_version: projection.projection_schema_version,
                        materialization_status: projection.materialization_status,
                        updated_at_ms: projection.updated_at_ms,
                    })
                    .collect(),
            )
            .map_err(|error| (STORE_INVALID_PAYLOAD, error.to_string()))?;

        let value = match request.operation {
            StoreOperation::GraphRecord { namespace, id } => json!(store
                .graph_record(&namespace, &id)
                .await
                .map_err(|error| (STORE_OPERATION_FAILED, error.to_string()))?
                .map(graph_record_json)),
            StoreOperation::GraphRecords {
                namespace,
                metadata,
            } => Value::Array(
                store
                    .graph_records(&namespace, &metadata_filter(metadata))
                    .await
                    .map_err(|error| (STORE_OPERATION_FAILED, error.to_string()))?
                    .into_iter()
                    .map(graph_record_json)
                    .collect(),
            ),
            StoreOperation::VectorQuery {
                namespace,
                embedding,
                limit,
                metadata,
                metric,
            } => Value::Array(
                store
                    .vector_query(
                        &namespace,
                        &VectorQuery {
                            embedding,
                            limit,
                            metadata: metadata_filter(metadata),
                            metric: metric.into(),
                        },
                    )
                    .await
                    .map_err(|error| (STORE_OPERATION_FAILED, error.to_string()))?
                    .into_iter()
                    .map(|item| json!({"record": graph_record_json(item.record), "distance": item.distance}))
                    .collect(),
            ),
            StoreOperation::ReplayEvents {
                namespace,
                after_seq,
                limit,
            } => Value::Array(
                store
                    .replay_events(&namespace, after_seq, limit)
                    .await
                    .map_err(|error| (STORE_OPERATION_FAILED, error.to_string()))?
                    .into_iter()
                    .map(event_json)
                    .collect(),
            ),
            StoreOperation::ReplayCursor {
                namespace,
                consumer,
            } => {
                let cursor = store
                    .replay_cursor(&namespace, &consumer)
                    .await
                    .map_err(|error| (STORE_OPERATION_FAILED, error.to_string()))?;
                json!({"namespace": cursor.namespace, "consumer": cursor.consumer, "last_seq": cursor.last_seq})
            }
            StoreOperation::LatestEventSeq { namespace } => json!(store
                .latest_event_seq(&namespace)
                .await
                .map_err(|error| (STORE_OPERATION_FAILED, error.to_string()))?),
            StoreOperation::NamedProjection { namespace, key } => json!(store
                .named_projection(&namespace, &key)
                .await
                .map_err(|error| (STORE_OPERATION_FAILED, error.to_string()))?
                .map(projection_json)),
            StoreOperation::NamedProjections { namespace } => Value::Array(
                store
                    .named_projections(&namespace)
                    .await
                    .map_err(|error| (STORE_OPERATION_FAILED, error.to_string()))?
                    .into_iter()
                    .map(projection_json)
                    .collect(),
            ),
        };
        serde_json::to_string(&value).map_err(|error| {
            (
                STORE_OPERATION_INVALID,
                format!("cannot encode result: {error}"),
            )
        })
    })
}
