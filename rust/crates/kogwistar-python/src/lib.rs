use kogwistar_contracts as contracts;
use kogwistar_runtime::{RecordedRuntimeTransition, RecordedWorkerHandoff};
use kogwistar_store::{
    DistanceMetric, EntityRebuildRequest, EntityRecoveryReport, EntityRecoveryRequest,
    EventReadStore, GraphReadStore, GraphRecord, GraphWriteStore, LaneMessageFilter,
    MetadataFilter, NamedProjection, NamedProjectionWrite, NewProjectedLaneMessage,
    ProjectedLaneMessage, ProjectionReadStore, ReplayCursor, ServerRun, ServerRunCreate,
    ServerRunEvent, ServerRunUpdate, VectorQuery, WorkflowDesignDelta, WorkflowDesignDeltaWrite,
    WorkflowDesignSnapshot, WorkflowDesignSnapshotWrite,
};
use kogwistar_store_memory::InMemoryStore;
use kogwistar_store_postgres::{
    NewRawEntityEvent as PostgresNewRawEntityEvent, PostgresStore, PostgresStoreError,
    PostgresUnitOfWork, RawEntityEvent as PostgresRawEntityEvent,
};
use kogwistar_store_sqlite::{
    NewRawEntityEvent, RawEntityEvent, SqliteStore, SqliteStoreError, SqliteUnitOfWork,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::pin;
use std::sync::{Mutex, OnceLock, mpsc};
use std::task::{Context, Poll, Waker};
use std::thread::JoinHandle;
use tokio::runtime::{Builder as TokioRuntimeBuilder, Runtime as TokioRuntime};

create_exception!(kogwistar._rust, RustContractTypeError, PyTypeError);
create_exception!(kogwistar._rust, RustContractValueError, PyValueError);
create_exception!(kogwistar._rust, RustStoreValueError, PyValueError);

const STORE_INVALID_JSON: &str = "KOGWISTAR_STORE_INVALID_JSON";
const STORE_INVALID_PAYLOAD: &str = "KOGWISTAR_STORE_INVALID_PAYLOAD";
const STORE_OPERATION_INVALID: &str = "KOGWISTAR_STORE_OPERATION_INVALID";
const STORE_OPERATION_FAILED: &str = "KOGWISTAR_STORE_OPERATION_FAILED";
const STORE_TRANSACTION_ABORTED: &str = "KOGWISTAR_STORE_TRANSACTION_ABORTED";
const STORE_PERSISTENCE_FAILED: &str = "KOGWISTAR_STORE_PERSISTENCE_FAILED";
const STORE_EVENT_ID_NAMESPACE_COLLISION: &str = "KOGWISTAR_STORE_EVENT_ID_NAMESPACE_COLLISION";
const STORE_CURSOR_OUT_OF_RANGE: &str = "KOGWISTAR_STORE_CURSOR_OUT_OF_RANGE";
const STORE_CURSOR_REGRESSES: &str = "KOGWISTAR_STORE_CURSOR_REGRESSES";
const STORE_INVALID_SEQUENCE_VALUE: &str = "KOGWISTAR_STORE_INVALID_SEQUENCE_VALUE";
const STORE_INVALID_ENTITY_EVENT: &str = "KOGWISTAR_STORE_INVALID_ENTITY_EVENT";

fn default_run_status() -> String {
    "queued".to_owned()
}
fn default_run_limit() -> usize {
    100
}
fn default_run_event_limit() -> usize {
    500
}
fn default_namespace() -> String {
    "default".to_owned()
}
fn default_optional_namespace() -> Option<String> {
    Some(default_namespace())
}
fn default_max_retries() -> i64 {
    10
}
fn default_claim_limit() -> usize {
    50
}
fn default_lease_seconds() -> i64 {
    60
}
fn default_list_limit() -> usize {
    1000
}
fn default_lane_purpose() -> String {
    "user_visible".to_owned()
}
fn default_true() -> bool {
    true
}

fn decoded_run_json(raw: Option<String>) -> Result<Value, serde_json::Error> {
    match raw.as_deref() {
        None | Some("") => Ok(Value::Null),
        Some(raw) => serde_json::from_str(raw),
    }
}
fn server_run_json(run: ServerRun) -> Result<Value, serde_json::Error> {
    let status = run.status.clone();
    Ok(
        json!({"run_id":run.run_id,"conversation_id":run.conversation_id,"workflow_id":run.workflow_id,"user_id":run.user_id,"user_turn_node_id":run.user_turn_node_id,"assistant_turn_node_id":run.assistant_turn_node_id,"status":status,"cancel_requested":run.cancel_requested,"result":decoded_run_json(run.result_json)?,"error":decoded_run_json(run.error_json)?,"created_at_ms":run.created_at_ms,"updated_at_ms":run.updated_at_ms,"started_at_ms":run.started_at_ms,"finished_at_ms":run.finished_at_ms,"terminal":matches!(status.as_str(),"succeeded"|"failed"|"cancelled")}),
    )
}
fn server_run_event_json(event: ServerRunEvent) -> Result<Value, serde_json::Error> {
    let payload: Value = serde_json::from_str(&event.payload_json)?;
    let payload = if payload.is_null()
        || payload == Value::Bool(false)
        || payload == Value::String(String::new())
        || payload.as_f64().is_some_and(|number| number == 0.0)
        || payload.as_array().is_some_and(Vec::is_empty)
        || payload.as_object().is_some_and(Map::is_empty)
    {
        json!({})
    } else {
        payload
    };
    Ok(
        json!({"seq":event.seq,"run_id":event.run_id,"event_type":event.event_type,"payload":payload,"created_at_ms":event.created_at_ms}),
    )
}

fn index_job_json(job: kogwistar_store::IndexJob) -> Value {
    json!({"job_id":job.job_id,"namespace":job.namespace,"entity_kind":job.entity_kind,"entity_id":job.entity_id,"index_kind":job.index_kind,"coalesce_key":job.coalesce_key,"op":job.op,"status":job.status,"lease_until":job.lease_until,"next_run_at":job.next_run_at,"max_retries":job.max_retries,"retry_count":job.retry_count,"last_error":job.last_error,"payload_json":job.payload_json,"created_at":job.created_at,"updated_at":job.updated_at,"claim_token":job.claim_token,"claim_attempts":job.claim_attempts})
}
fn lane_message_json(row: ProjectedLaneMessage) -> Value {
    json!({"message_id":row.message_id,"namespace":row.namespace,"purpose":row.purpose,"inbox_id":row.inbox_id,"conversation_id":row.conversation_id,"recipient_id":row.recipient_id,"sender_id":row.sender_id,"msg_type":row.msg_type,"status":row.status,"seq":row.seq,"conversation_seq":row.conversation_seq,"claimed_by":row.claimed_by,"lease_until":row.lease_until,"retry_count":row.retry_count,"created_at":row.created_at,"available_at":row.available_at,"run_id":row.run_id,"step_id":row.step_id,"correlation_id":row.correlation_id,"payload_json":row.payload_json,"error_json":row.error_json,"prev_message_id":row.prev_message_id,"next_message_id":row.next_message_id,"inbox_tail_message_id":row.inbox_tail_message_id,"conversation_tail_message_id":row.conversation_tail_message_id})
}
#[allow(clippy::too_many_arguments)]
fn new_lane_message(
    message_id: String,
    namespace: String,
    purpose: String,
    inbox_id: String,
    conversation_id: String,
    recipient_id: String,
    sender_id: String,
    msg_type: String,
    status: String,
    created_at: i64,
    available_at: i64,
    run_id: Option<String>,
    step_id: Option<String>,
    correlation_id: Option<String>,
    payload_json: Option<String>,
    error_json: Option<String>,
) -> NewProjectedLaneMessage {
    NewProjectedLaneMessage {
        message_id,
        namespace,
        purpose,
        inbox_id,
        conversation_id,
        recipient_id,
        sender_id,
        msg_type,
        status,
        created_at,
        available_at,
        run_id,
        step_id,
        correlation_id,
        payload_json,
        error_json,
    }
}
#[allow(clippy::too_many_arguments)]
fn lane_filter(
    namespace: Option<String>,
    purpose: Option<String>,
    inbox_id: Option<String>,
    conversation_id: Option<String>,
    status: Option<String>,
    msg_type: Option<String>,
    sender_id: Option<String>,
    recipient_id: Option<String>,
    correlation_id: Option<String>,
    reply_to_message_id: Option<String>,
    created_at_gte: Option<i64>,
    created_at_lte: Option<i64>,
    available_at_gte: Option<i64>,
    available_at_lte: Option<i64>,
    limit: usize,
    newest_first: bool,
) -> LaneMessageFilter {
    LaneMessageFilter {
        namespace,
        purpose,
        inbox_id,
        conversation_id,
        status,
        msg_type,
        sender_id,
        recipient_id,
        correlation_id,
        reply_to_message_id,
        created_at_gte,
        created_at_lte,
        available_at_gte,
        available_at_lte,
        limit,
        newest_first,
    }
}
#[allow(clippy::too_many_arguments)]
fn new_index_job(
    job_id: String,
    namespace: String,
    entity_kind: String,
    entity_id: String,
    index_kind: String,
    op: String,
    payload_json: Option<String>,
    max_retries: i64,
) -> kogwistar_store::NewIndexJob {
    kogwistar_store::NewIndexJob {
        job_id,
        namespace,
        entity_kind,
        entity_id,
        index_kind,
        op,
        payload_json,
        max_retries,
    }
}

fn projection_write(
    payload: Map<String, Value>,
    last_authoritative_seq: i64,
    last_materialized_seq: i64,
    projection_schema_version: i64,
    materialization_status: String,
) -> NamedProjectionWrite {
    NamedProjectionWrite {
        payload,
        last_authoritative_seq,
        last_materialized_seq,
        projection_schema_version,
        materialization_status,
    }
}

fn contract_error(error: contracts::ContractError) -> PyErr {
    Python::attach(|py| contract_error_with_python(py, error))
}

fn contract_error_with_python(py: Python<'_>, error: contracts::ContractError) -> PyErr {
    let is_attribute_error = matches!(error, contracts::ContractError::StateUpdateTargetMustBeList);
    let is_type_error = matches!(
        error,
        contracts::ContractError::MetadataFilterWhereType { .. }
            | contracts::ContractError::MetadataFilterLogicalClausesNotIterable { .. }
            | contracts::ContractError::StateUpdateStateMustBeObject
            | contracts::ContractError::StateUpdateItemMustBePair
            | contracts::ContractError::StateUpdatePayloadMustBeObject
            | contracts::ContractError::StateUpdateTargetMustBeList
    );
    let result = if is_attribute_error {
        PyAttributeError::new_err(error.to_string())
    } else if is_type_error {
        RustContractTypeError::new_err(error.to_string())
    } else {
        RustContractValueError::new_err(error.to_string())
    };
    // Stable code is machine-facing; text remains diagnostic only.
    let _ = result.value(py).setattr("code", error.code());
    result
}

fn store_error(py: Python<'_>, code: &'static str, message: impl Into<String>) -> PyErr {
    let result = RustStoreValueError::new_err(message.into());
    let _ = result.value(py).setattr("code", code);
    result
}

fn block_on<T>(future: impl Future<Output = T>) -> T {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

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

#[derive(Default, Deserialize)]
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

/// Durable SQLite Phase-3 ABI. This accepts a database path, never a snapshot.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SqliteStoreRequest {
    path: String,
    #[serde(default)]
    transaction_id: Option<String>,
    operation: SqliteStoreOperation,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum SqliteStoreOperation {
    OpenInit,
    Close,
    BeginTransaction,
    CommitTransaction,
    RollbackTransaction,
    CurrentGlobalSeq,
    NextGlobalSeq,
    CurrentUserSeq {
        user_id: String,
    },
    NextUserSeq {
        user_id: String,
    },
    SetUserSeq {
        user_id: String,
        value: i64,
    },
    CurrentScopedSeq {
        scope_id: String,
    },
    NextScopedSeq {
        scope_id: String,
    },
    SetScopedSeq {
        scope_id: String,
        value: i64,
    },
    AllocEventSeq {
        namespace: String,
    },
    RawAppend {
        namespace: String,
        event_id: String,
        entity_kind: String,
        entity_id: String,
        op: String,
        payload_json: String,
    },
    ExclusiveRawReplay {
        namespace: String,
        after_seq: i64,
        limit: usize,
    },
    LatestRetainedEventSeq {
        namespace: String,
    },
    PruneEntityEventsAfter {
        namespace: String,
        to_seq: i64,
    },
    CursorGet {
        namespace: String,
        consumer: String,
    },
    LegacyCursorSet {
        namespace: String,
        consumer: String,
        last_seq: i64,
    },
    StrictCursorAdvance {
        namespace: String,
        consumer: String,
        last_seq: i64,
    },
    RecoverEntityProjection {
        namespace: String,
        consumer: String,
        projection_namespace: String,
        projection_key: String,
        batch_limit: usize,
        #[serde(default)]
        abort_after_projection: bool,
    },
    RebuildEntityProjection {
        namespace: String,
        consumer: String,
        projection_namespace: String,
        projection_key: String,
        #[serde(default)]
        abort_after_projection: bool,
    },
    GetNamedProjection {
        namespace: String,
        key: String,
    },
    ListNamedProjections {
        namespace: String,
    },
    ReplaceNamedProjection {
        namespace: String,
        key: String,
        payload: Map<String, Value>,
        last_authoritative_seq: i64,
        last_materialized_seq: i64,
        projection_schema_version: i64,
        materialization_status: String,
    },
    CompareAndSwapNamedProjection {
        namespace: String,
        key: String,
        payload: Map<String, Value>,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        last_authoritative_seq: i64,
        last_materialized_seq: i64,
        projection_schema_version: i64,
        materialization_status: String,
    },
    ClearNamedProjection {
        namespace: String,
        key: String,
    },
    ClearProjectionNamespace {
        namespace: String,
    },
    GetStage1NodeProjection {
        namespace: String,
        key: String,
    },
    ListStage1NodeProjections {
        namespace: String,
    },
    ReplaceStage1NodeProjection {
        namespace: String,
        key: String,
        payload: Map<String, Value>,
        last_authoritative_seq: i64,
        last_materialized_seq: i64,
        projection_schema_version: i64,
        materialization_status: String,
    },
    ClearStage1NodeProjection {
        namespace: String,
        key: String,
    },
    PutWorkflowDesignSnapshot {
        workflow_id: String,
        version: i64,
        seq: i64,
        payload_json: String,
        schema_version: i64,
    },
    GetWorkflowDesignSnapshot {
        workflow_id: String,
        max_version: i64,
        schema_version: i64,
    },
    ClearWorkflowDesignSnapshots {
        workflow_id: String,
    },
    PutWorkflowDesignDelta {
        workflow_id: String,
        version: i64,
        prev_version: i64,
        target_seq: i64,
        forward_json: String,
        inverse_json: String,
        schema_version: i64,
    },
    GetWorkflowDesignDelta {
        workflow_id: String,
        version: i64,
        schema_version: i64,
    },
    ClearWorkflowDesignDeltas {
        workflow_id: String,
    },
    CreateServerRun {
        run_id: String,
        conversation_id: String,
        workflow_id: String,
        #[serde(default)]
        user_id: Option<String>,
        user_turn_node_id: String,
        #[serde(default = "default_run_status")]
        status: String,
    },
    GetServerRun {
        run_id: String,
    },
    ListServerRuns {
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        workflow_id: Option<String>,
        #[serde(default)]
        conversation_id: Option<String>,
        #[serde(default = "default_run_limit")]
        limit: usize,
    },
    AppendServerRunEvent {
        run_id: String,
        event_type: String,
        payload_json: String,
    },
    ListServerRunEvents {
        run_id: String,
        #[serde(default)]
        after_seq: i64,
        #[serde(default = "default_run_event_limit")]
        limit: usize,
    },
    UpdateServerRun {
        run_id: String,
        status: String,
        assistant_turn_node_id: Option<String>,
        result_json: Option<String>,
        error_json: Option<String>,
        started_at_ms: Option<i64>,
        finished_at_ms: Option<i64>,
        #[serde(default)]
        cancel_requested: Option<bool>,
    },
    RequestServerRunCancel {
        run_id: String,
    },
    /// ADR-015 Phase-4 recorded-only durable runtime transition.  The result
    /// is already supplied by a worker; this ABI cannot dispatch work.
    ApplyRecordedRuntimeTransition {
        transition: Box<RecordedRuntimeTransition>,
        #[serde(default)]
        abort_after_writes: bool,
    },
    /// Complete one claimed Python-worker request and record its result in one
    /// Rust-owned SQLite transaction. No callback executes inside the UoW.
    ApplyClaimedRecordedRuntimeTransition {
        handoff: RecordedWorkerHandoff,
        transition: Box<RecordedRuntimeTransition>,
        #[serde(default)]
        abort_after_writes: bool,
    },
    /// Reopen/read recorded runtime state.  It never resumes or dispatches.
    ReadRecordedRuntimeState {
        run_id: String,
        workflow_id: String,
        conversation_id: String,
    },
    GetIndexAppliedFingerprint {
        #[serde(default = "default_namespace")]
        namespace: String,
        coalesce_key: String,
    },
    SetIndexAppliedFingerprint {
        #[serde(default = "default_namespace")]
        namespace: String,
        coalesce_key: String,
        #[serde(default)]
        applied_fingerprint: Option<String>,
        #[serde(default)]
        last_job_id: Option<String>,
    },
    EnqueueIndexJob {
        job_id: String,
        #[serde(default = "default_namespace")]
        namespace: String,
        entity_kind: String,
        entity_id: String,
        index_kind: String,
        op: String,
        #[serde(default)]
        payload_json: Option<String>,
        #[serde(default = "default_max_retries")]
        max_retries: i64,
    },
    ClaimIndexJobs {
        #[serde(default = "default_claim_limit")]
        limit: usize,
        #[serde(default = "default_lease_seconds")]
        lease_seconds: i64,
        #[serde(default = "default_optional_namespace")]
        namespace: Option<String>,
    },
    MarkIndexJobDone {
        job_id: String,
        #[serde(default)]
        claim_token: Option<String>,
    },
    MarkIndexJobFailed {
        job_id: String,
        error: String,
        #[serde(rename = "final", default = "default_true")]
        final_: bool,
        #[serde(default)]
        claim_token: Option<String>,
    },
    BumpRetryAndRequeue {
        job_id: String,
        error: String,
        next_run_at_seconds: i64,
        #[serde(default)]
        claim_token: Option<String>,
    },
    RenewIndexJobLease {
        job_id: String,
        claim_token: String,
        lease_seconds: i64,
    },
    RequeueIndexJobAtTail {
        job_id: String,
        payload_json: String,
        #[serde(default)]
        delay_seconds: i64,
        #[serde(default)]
        claim_token: Option<String>,
    },
    ListIndexJobs {
        #[serde(default = "default_optional_namespace")]
        namespace: Option<String>,
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        entity_kind: Option<String>,
        #[serde(default)]
        entity_id: Option<String>,
        #[serde(default)]
        index_kind: Option<String>,
        #[serde(default = "default_list_limit")]
        limit: usize,
    },
    ProjectLaneMessage {
        message_id: String,
        #[serde(default = "default_namespace")]
        namespace: String,
        #[serde(default = "default_lane_purpose")]
        purpose: String,
        inbox_id: String,
        conversation_id: String,
        recipient_id: String,
        sender_id: String,
        msg_type: String,
        status: String,
        created_at: i64,
        available_at: i64,
        #[serde(default)]
        run_id: Option<String>,
        #[serde(default)]
        step_id: Option<String>,
        #[serde(default)]
        correlation_id: Option<String>,
        #[serde(default)]
        payload_json: Option<String>,
        #[serde(default)]
        error_json: Option<String>,
    },
    GetProjectedLaneMessage {
        message_id: String,
    },
    UpdateProjectedLaneMessageStatus {
        message_id: String,
        status: String,
        #[serde(default)]
        error_json: Option<String>,
    },
    UpdateProjectedLaneMessageLinks {
        message_id: String,
        #[serde(default)]
        prev_message_id: Option<String>,
        #[serde(default)]
        next_message_id: Option<String>,
        #[serde(default)]
        inbox_tail_message_id: Option<String>,
        #[serde(default)]
        conversation_tail_message_id: Option<String>,
    },
    ListProjectedLaneMessages {
        #[serde(default = "default_optional_namespace")]
        namespace: Option<String>,
        #[serde(default)]
        purpose: Option<String>,
        #[serde(default)]
        inbox_id: Option<String>,
        #[serde(default)]
        conversation_id: Option<String>,
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        msg_type: Option<String>,
        #[serde(default)]
        sender_id: Option<String>,
        #[serde(default)]
        recipient_id: Option<String>,
        #[serde(default)]
        correlation_id: Option<String>,
        #[serde(default)]
        reply_to_message_id: Option<String>,
        #[serde(default)]
        created_at_gte: Option<i64>,
        #[serde(default)]
        created_at_lte: Option<i64>,
        #[serde(default)]
        available_at_gte: Option<i64>,
        #[serde(default)]
        available_at_lte: Option<i64>,
        #[serde(default = "default_list_limit")]
        limit: usize,
        #[serde(default)]
        newest_first: bool,
    },
    ClearProjectedLaneMessages {
        namespace: String,
    },
    ClaimProjectedLaneMessages {
        #[serde(default = "default_namespace")]
        namespace: String,
        inbox_id: String,
        claimed_by: String,
        #[serde(default = "default_claim_limit")]
        limit: usize,
        #[serde(default = "default_lease_seconds")]
        lease_seconds: i64,
    },
    AckProjectedLaneMessage {
        message_id: String,
        claimed_by: String,
    },
    RequeueProjectedLaneMessage {
        message_id: String,
        claimed_by: String,
        #[serde(default)]
        error_json: Option<String>,
        #[serde(default)]
        delay_seconds: i64,
    },
    DeadLetterProjectedLaneMessage {
        message_id: String,
        claimed_by: String,
        #[serde(default)]
        error_json: Option<String>,
    },
    Batch {
        operations: Vec<SqliteStoreOperation>,
        #[serde(default)]
        abort: bool,
    },
}

fn raw_event_json(event: RawEntityEvent) -> Value {
    json!({
        "namespace": event.namespace,
        "seq": event.seq,
        "event_id": event.event_id,
        "entity_kind": event.entity_kind,
        "entity_id": event.entity_id,
        "op": event.op,
        "payload_json": event.payload_json,
        "created_at": event.created_at,
    })
}

fn cursor_json(cursor: ReplayCursor) -> Value {
    json!({"namespace": cursor.namespace, "consumer": cursor.consumer, "last_seq": cursor.last_seq})
}

fn entity_recovery_report_json(report: EntityRecoveryReport) -> Value {
    json!({
        "processed_count": report.processed_count,
        "prior_cursor": report.prior_cursor,
        "new_cursor": report.new_cursor,
        "latest_authoritative_seq": report.latest_authoritative_seq,
        "caught_up": report.caught_up,
        "canonical_payload": report.canonical_payload,
        "digest": report.digest,
    })
}

fn entity_recovery_request(
    namespace: String,
    consumer: String,
    projection_namespace: String,
    projection_key: String,
    batch_limit: usize,
    abort_after_projection: bool,
) -> EntityRecoveryRequest {
    EntityRecoveryRequest {
        namespace,
        consumer,
        projection_namespace,
        projection_key,
        batch_limit,
        abort_after_projection,
    }
}

fn entity_rebuild_request(
    namespace: String,
    consumer: String,
    projection_namespace: String,
    projection_key: String,
    abort_after_projection: bool,
) -> EntityRebuildRequest {
    EntityRebuildRequest {
        namespace,
        consumer,
        projection_namespace,
        projection_key,
        abort_after_projection,
    }
}

fn new_raw_event(
    event_id: String,
    entity_kind: String,
    entity_id: String,
    op: String,
    payload_json: String,
) -> NewRawEntityEvent {
    NewRawEntityEvent {
        event_id,
        entity_kind,
        entity_id,
        op,
        payload_json,
    }
}

fn appended_raw_event_json(appended: kogwistar_store_sqlite::AppendedRawEvent) -> Value {
    let seq = appended.event.seq;
    json!({"seq": seq, "inserted": appended.inserted, "event": raw_event_json(appended.event)})
}

fn sqlite_store_operation_json(
    store: &SqliteStore,
    operation: SqliteStoreOperation,
) -> Result<Value, SqliteStoreError> {
    match operation {
        SqliteStoreOperation::OpenInit => Ok(json!({"initialized": true})),
        SqliteStoreOperation::Close => Err(SqliteStoreError::TransactionAborted(
            "SQLite close must be handled by the session".to_owned(),
        )),
        SqliteStoreOperation::BeginTransaction
        | SqliteStoreOperation::CommitTransaction
        | SqliteStoreOperation::RollbackTransaction => Err(SqliteStoreError::TransactionAborted(
            "SQLite transaction control must be handled by the session".to_owned(),
        )),
        SqliteStoreOperation::CurrentGlobalSeq => Ok(json!(store.current_global_seq()?)),
        SqliteStoreOperation::NextGlobalSeq => Ok(json!(store.next_global_seq()?)),
        SqliteStoreOperation::CurrentUserSeq { user_id } => {
            Ok(json!(store.current_user_seq(&user_id)?))
        }
        SqliteStoreOperation::NextUserSeq { user_id } => Ok(json!(store.next_user_seq(&user_id)?)),
        SqliteStoreOperation::SetUserSeq { user_id, value } => {
            store.set_user_seq(&user_id, value)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::CurrentScopedSeq { scope_id } => {
            Ok(json!(store.current_scoped_seq(&scope_id)?))
        }
        SqliteStoreOperation::NextScopedSeq { scope_id } => {
            Ok(json!(store.next_scoped_seq(&scope_id)?))
        }
        SqliteStoreOperation::SetScopedSeq { scope_id, value } => {
            store.set_scoped_seq(&scope_id, value)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::AllocEventSeq { namespace } => {
            Ok(json!(store.alloc_event_seq(&namespace)?))
        }
        SqliteStoreOperation::RawAppend {
            namespace,
            event_id,
            entity_kind,
            entity_id,
            op,
            payload_json,
        } => Ok(appended_raw_event_json(store.append_raw_entity_event(
            &namespace,
            new_raw_event(event_id, entity_kind, entity_id, op, payload_json),
        )?)),
        SqliteStoreOperation::ExclusiveRawReplay {
            namespace,
            after_seq,
            limit,
        } => Ok(Value::Array(
            store
                .replay_raw_events(&namespace, after_seq, limit)?
                .into_iter()
                .map(raw_event_json)
                .collect(),
        )),
        SqliteStoreOperation::LatestRetainedEventSeq { namespace } => {
            Ok(json!(store.latest_retained_event_seq(&namespace)?))
        }
        SqliteStoreOperation::PruneEntityEventsAfter { namespace, to_seq } => {
            Ok(json!(store.prune_entity_events_after(&namespace, to_seq)?))
        }
        SqliteStoreOperation::CursorGet {
            namespace,
            consumer,
        } => Ok(cursor_json(store.replay_cursor(&namespace, &consumer)?)),
        SqliteStoreOperation::LegacyCursorSet {
            namespace,
            consumer,
            last_seq,
        } => Ok(cursor_json(
            store.set_replay_cursor_legacy(&namespace, &consumer, last_seq)?,
        )),
        SqliteStoreOperation::StrictCursorAdvance {
            namespace,
            consumer,
            last_seq,
        } => Ok(cursor_json(store.strict_advance_replay_cursor(
            &namespace, &consumer, last_seq,
        )?)),
        SqliteStoreOperation::RecoverEntityProjection {
            namespace,
            consumer,
            projection_namespace,
            projection_key,
            batch_limit,
            abort_after_projection,
        } => Ok(entity_recovery_report_json(
            store.recover_entity_projection(entity_recovery_request(
                namespace,
                consumer,
                projection_namespace,
                projection_key,
                batch_limit,
                abort_after_projection,
            ))?,
        )),
        SqliteStoreOperation::RebuildEntityProjection {
            namespace,
            consumer,
            projection_namespace,
            projection_key,
            abort_after_projection,
        } => Ok(entity_recovery_report_json(
            store.rebuild_entity_projection(entity_rebuild_request(
                namespace,
                consumer,
                projection_namespace,
                projection_key,
                abort_after_projection,
            ))?,
        )),
        SqliteStoreOperation::GetNamedProjection { namespace, key } => Ok(store
            .get_named_projection(&namespace, &key)?
            .map(projection_json)
            .unwrap_or(Value::Null)),
        SqliteStoreOperation::ListNamedProjections { namespace } => Ok(Value::Array(
            store
                .list_named_projections(&namespace)?
                .into_iter()
                .map(projection_json)
                .collect(),
        )),
        SqliteStoreOperation::ReplaceNamedProjection {
            namespace,
            key,
            payload,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => {
            store.replace_named_projection(
                &namespace,
                &key,
                projection_write(
                    payload,
                    last_authoritative_seq,
                    last_materialized_seq,
                    projection_schema_version,
                    materialization_status,
                ),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::CompareAndSwapNamedProjection {
            namespace,
            key,
            payload,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => Ok(json!(store.compare_and_swap_named_projection(
            &namespace,
            &key,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            projection_write(
                payload,
                last_authoritative_seq,
                last_materialized_seq,
                projection_schema_version,
                materialization_status
            ),
        )?)),
        SqliteStoreOperation::ClearNamedProjection { namespace, key } => {
            store.clear_named_projection(&namespace, &key)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ClearProjectionNamespace { namespace } => {
            store.clear_projection_namespace(&namespace)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::GetStage1NodeProjection { namespace, key } => Ok(store
            .get_stage1_node_projection(&namespace, &key)?
            .map(projection_json)
            .unwrap_or(Value::Null)),
        SqliteStoreOperation::ListStage1NodeProjections { namespace } => Ok(Value::Array(
            store
                .list_stage1_node_projections(&namespace)?
                .into_iter()
                .map(projection_json)
                .collect(),
        )),
        SqliteStoreOperation::ReplaceStage1NodeProjection {
            namespace,
            key,
            payload,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => {
            store.replace_stage1_node_projection(
                &namespace,
                &key,
                projection_write(
                    payload,
                    last_authoritative_seq,
                    last_materialized_seq,
                    projection_schema_version,
                    materialization_status,
                ),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ClearStage1NodeProjection { namespace, key } => {
            store.clear_stage1_node_projection(&namespace, &key)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::PutWorkflowDesignSnapshot {
            workflow_id,
            version,
            seq,
            payload_json,
            schema_version,
        } => {
            store.put_workflow_design_snapshot(
                &workflow_id,
                workflow_design_snapshot_write(version, seq, payload_json, schema_version),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::GetWorkflowDesignSnapshot {
            workflow_id,
            max_version,
            schema_version,
        } => Ok(store
            .get_workflow_design_snapshot(&workflow_id, max_version, schema_version)?
            .map(workflow_design_snapshot_json)
            .unwrap_or(Value::Null)),
        SqliteStoreOperation::ClearWorkflowDesignSnapshots { workflow_id } => {
            store.clear_workflow_design_snapshots(&workflow_id)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::PutWorkflowDesignDelta {
            workflow_id,
            version,
            prev_version,
            target_seq,
            forward_json,
            inverse_json,
            schema_version,
        } => {
            store.put_workflow_design_delta(
                &workflow_id,
                workflow_design_delta_write(
                    version,
                    prev_version,
                    target_seq,
                    forward_json,
                    inverse_json,
                    schema_version,
                ),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::GetWorkflowDesignDelta {
            workflow_id,
            version,
            schema_version,
        } => Ok(store
            .get_workflow_design_delta(&workflow_id, version, schema_version)?
            .map(workflow_design_delta_json)
            .unwrap_or(Value::Null)),
        SqliteStoreOperation::ClearWorkflowDesignDeltas { workflow_id } => {
            store.clear_workflow_design_deltas(&workflow_id)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::CreateServerRun {
            run_id,
            conversation_id,
            workflow_id,
            user_id,
            user_turn_node_id,
            status,
        } => {
            store.create_server_run(ServerRunCreate {
                run_id,
                conversation_id,
                workflow_id,
                user_id,
                user_turn_node_id,
                status,
            })?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::GetServerRun { run_id } => Ok(store
            .get_server_run(&run_id)?
            .map(server_run_json)
            .transpose()?
            .unwrap_or(Value::Null)),
        SqliteStoreOperation::ListServerRuns {
            status,
            workflow_id,
            conversation_id,
            limit,
        } => Ok(Value::Array(
            store
                .list_server_runs(
                    status.as_deref(),
                    workflow_id.as_deref(),
                    conversation_id.as_deref(),
                    limit,
                )?
                .into_iter()
                .map(server_run_json)
                .collect::<Result<_, _>>()?,
        )),
        SqliteStoreOperation::AppendServerRunEvent {
            run_id,
            event_type,
            payload_json,
        } => Ok(server_run_event_json(store.append_server_run_event(
            &run_id,
            &event_type,
            payload_json,
        )?)?),
        SqliteStoreOperation::ListServerRunEvents {
            run_id,
            after_seq,
            limit,
        } => Ok(Value::Array(
            store
                .list_server_run_events(&run_id, after_seq, limit)?
                .into_iter()
                .map(server_run_event_json)
                .collect::<Result<_, _>>()?,
        )),
        SqliteStoreOperation::UpdateServerRun {
            run_id,
            status,
            assistant_turn_node_id,
            result_json,
            error_json,
            started_at_ms,
            finished_at_ms,
            cancel_requested,
        } => {
            store.update_server_run(
                &run_id,
                ServerRunUpdate {
                    status,
                    assistant_turn_node_id,
                    result_json,
                    error_json,
                    started_at_ms,
                    finished_at_ms,
                    cancel_requested,
                },
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::RequestServerRunCancel { run_id } => {
            store.request_server_run_cancel(&run_id)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ApplyRecordedRuntimeTransition {
            transition,
            abort_after_writes,
        } => serde_json::to_value(
            store.apply_recorded_runtime_transition(*transition, abort_after_writes)?,
        )
        .map_err(|error| SqliteStoreError::TransactionAborted(error.to_string())),
        SqliteStoreOperation::ApplyClaimedRecordedRuntimeTransition {
            handoff,
            transition,
            abort_after_writes,
        } => serde_json::to_value(store.apply_claimed_recorded_runtime_transition(
            handoff,
            *transition,
            abort_after_writes,
        )?)
        .map_err(|error| SqliteStoreError::TransactionAborted(error.to_string())),
        SqliteStoreOperation::ReadRecordedRuntimeState {
            run_id,
            workflow_id,
            conversation_id,
        } => serde_json::to_value(store.read_recorded_runtime_state(
            &run_id,
            &workflow_id,
            &conversation_id,
        )?)
        .map_err(|error| SqliteStoreError::TransactionAborted(error.to_string())),
        SqliteStoreOperation::GetIndexAppliedFingerprint {
            namespace,
            coalesce_key,
        } => Ok(json!(
            store.index_applied_fingerprint(&namespace, &coalesce_key)?
        )),
        SqliteStoreOperation::SetIndexAppliedFingerprint {
            namespace,
            coalesce_key,
            applied_fingerprint,
            last_job_id,
        } => {
            store.set_index_applied_fingerprint(
                &namespace,
                &coalesce_key,
                applied_fingerprint.as_deref(),
                last_job_id.as_deref(),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::EnqueueIndexJob {
            job_id,
            namespace,
            entity_kind,
            entity_id,
            index_kind,
            op,
            payload_json,
            max_retries,
        } => Ok(json!(store.enqueue_index_job(new_index_job(
            job_id,
            namespace,
            entity_kind,
            entity_id,
            index_kind,
            op,
            payload_json,
            max_retries
        ))?)),
        SqliteStoreOperation::ClaimIndexJobs {
            limit,
            lease_seconds,
            namespace,
        } => Ok(Value::Array(
            store
                .claim_index_jobs(limit, lease_seconds, namespace.as_deref())?
                .into_iter()
                .map(index_job_json)
                .collect(),
        )),
        SqliteStoreOperation::MarkIndexJobDone {
            job_id,
            claim_token,
        } => Ok(json!(
            store.mark_index_job_done(&job_id, claim_token.as_deref())?
        )),
        SqliteStoreOperation::MarkIndexJobFailed {
            job_id,
            error,
            final_,
            claim_token,
        } => {
            store.mark_index_job_failed(&job_id, &error, final_, claim_token.as_deref())?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::BumpRetryAndRequeue {
            job_id,
            error,
            next_run_at_seconds,
            claim_token,
        } => {
            store.bump_retry_and_requeue(
                &job_id,
                &error,
                next_run_at_seconds,
                claim_token.as_deref(),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::RenewIndexJobLease {
            job_id,
            claim_token,
            lease_seconds,
        } => Ok(json!(store.renew_index_job_lease(
            &job_id,
            &claim_token,
            lease_seconds
        )?)),
        SqliteStoreOperation::RequeueIndexJobAtTail {
            job_id,
            payload_json,
            delay_seconds,
            claim_token,
        } => {
            store.requeue_index_job_at_tail(
                &job_id,
                payload_json,
                delay_seconds,
                claim_token.as_deref(),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ListIndexJobs {
            namespace,
            status,
            entity_kind,
            entity_id,
            index_kind,
            limit,
        } => Ok(Value::Array(
            store
                .list_index_jobs(
                    namespace.as_deref(),
                    status.as_deref(),
                    entity_kind.as_deref(),
                    entity_id.as_deref(),
                    index_kind.as_deref(),
                    limit,
                )?
                .into_iter()
                .map(index_job_json)
                .collect(),
        )),
        SqliteStoreOperation::ProjectLaneMessage {
            message_id,
            namespace,
            purpose,
            inbox_id,
            conversation_id,
            recipient_id,
            sender_id,
            msg_type,
            status,
            created_at,
            available_at,
            run_id,
            step_id,
            correlation_id,
            payload_json,
            error_json,
        } => {
            store.project_lane_message(new_lane_message(
                message_id,
                namespace,
                purpose,
                inbox_id,
                conversation_id,
                recipient_id,
                sender_id,
                msg_type,
                status,
                created_at,
                available_at,
                run_id,
                step_id,
                correlation_id,
                payload_json,
                error_json,
            ))?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::GetProjectedLaneMessage { message_id } => Ok(store
            .get_projected_lane_message(&message_id)?
            .map(lane_message_json)
            .unwrap_or(Value::Null)),
        SqliteStoreOperation::UpdateProjectedLaneMessageStatus {
            message_id,
            status,
            error_json,
        } => {
            store.update_projected_lane_message_status(&message_id, &status, error_json)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::UpdateProjectedLaneMessageLinks {
            message_id,
            prev_message_id,
            next_message_id,
            inbox_tail_message_id,
            conversation_tail_message_id,
        } => {
            store.update_projected_lane_message_links(
                &message_id,
                prev_message_id,
                next_message_id,
                inbox_tail_message_id,
                conversation_tail_message_id,
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ListProjectedLaneMessages {
            namespace,
            purpose,
            inbox_id,
            conversation_id,
            status,
            msg_type,
            sender_id,
            recipient_id,
            correlation_id,
            reply_to_message_id,
            created_at_gte,
            created_at_lte,
            available_at_gte,
            available_at_lte,
            limit,
            newest_first,
        } => Ok(Value::Array(
            store
                .list_projected_lane_messages(lane_filter(
                    namespace,
                    purpose,
                    inbox_id,
                    conversation_id,
                    status,
                    msg_type,
                    sender_id,
                    recipient_id,
                    correlation_id,
                    reply_to_message_id,
                    created_at_gte,
                    created_at_lte,
                    available_at_gte,
                    available_at_lte,
                    limit,
                    newest_first,
                ))?
                .into_iter()
                .map(lane_message_json)
                .collect(),
        )),
        SqliteStoreOperation::ClearProjectedLaneMessages { namespace } => {
            Ok(json!(store.clear_projected_lane_messages(&namespace)?))
        }
        SqliteStoreOperation::ClaimProjectedLaneMessages {
            namespace,
            inbox_id,
            claimed_by,
            limit,
            lease_seconds,
        } => Ok(Value::Array(
            store
                .claim_projected_lane_messages(
                    &namespace,
                    &inbox_id,
                    &claimed_by,
                    limit,
                    lease_seconds,
                )?
                .into_iter()
                .map(lane_message_json)
                .collect(),
        )),
        SqliteStoreOperation::AckProjectedLaneMessage {
            message_id,
            claimed_by,
        } => {
            store.ack_projected_lane_message(&message_id, &claimed_by)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::RequeueProjectedLaneMessage {
            message_id,
            claimed_by,
            error_json,
            delay_seconds,
        } => {
            store.requeue_projected_lane_message(
                &message_id,
                &claimed_by,
                error_json,
                delay_seconds,
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::DeadLetterProjectedLaneMessage {
            message_id,
            claimed_by,
            error_json,
        } => {
            store.dead_letter_projected_lane_message(&message_id, &claimed_by, error_json)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::Batch { operations, abort } => store.immediate_transaction(|uow| {
            let mut results = Vec::with_capacity(operations.len());
            for operation in operations {
                results.push(sqlite_batch_operation_json(uow, operation)?);
            }
            if abort {
                return Err(SqliteStoreError::TransactionAborted(
                    "requested by SQLite JSON ABI abort flag".to_owned(),
                ));
            }
            Ok(Value::Array(results))
        }),
    }
}

fn sqlite_batch_operation_json(
    uow: &mut SqliteUnitOfWork<'_>,
    operation: SqliteStoreOperation,
) -> Result<Value, SqliteStoreError> {
    match operation {
        SqliteStoreOperation::NextGlobalSeq => Ok(json!(uow.next_global_seq()?)),
        SqliteStoreOperation::NextUserSeq { user_id } => Ok(json!(uow.next_user_seq(&user_id)?)),
        SqliteStoreOperation::SetUserSeq { user_id, value } => {
            uow.set_user_seq(&user_id, value)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::NextScopedSeq { scope_id } => {
            Ok(json!(uow.next_scoped_seq(&scope_id)?))
        }
        SqliteStoreOperation::SetScopedSeq { scope_id, value } => {
            uow.set_scoped_seq(&scope_id, value)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::AllocEventSeq { namespace } => {
            Ok(json!(uow.alloc_event_seq(&namespace)?))
        }
        SqliteStoreOperation::RawAppend {
            namespace,
            event_id,
            entity_kind,
            entity_id,
            op,
            payload_json,
        } => Ok(appended_raw_event_json(uow.append_raw_entity_event(
            &namespace,
            new_raw_event(event_id, entity_kind, entity_id, op, payload_json),
        )?)),
        SqliteStoreOperation::PruneEntityEventsAfter { namespace, to_seq } => {
            Ok(json!(uow.prune_entity_events_after(&namespace, to_seq)?))
        }
        SqliteStoreOperation::LegacyCursorSet {
            namespace,
            consumer,
            last_seq,
        } => Ok(cursor_json(
            uow.set_replay_cursor_legacy(&namespace, &consumer, last_seq)?,
        )),
        SqliteStoreOperation::StrictCursorAdvance {
            namespace,
            consumer,
            last_seq,
        } => Ok(cursor_json(uow.strict_advance_replay_cursor(
            &namespace, &consumer, last_seq,
        )?)),
        SqliteStoreOperation::ReplaceNamedProjection {
            namespace,
            key,
            payload,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => {
            uow.replace_named_projection(
                &namespace,
                &key,
                projection_write(
                    payload,
                    last_authoritative_seq,
                    last_materialized_seq,
                    projection_schema_version,
                    materialization_status,
                ),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::CompareAndSwapNamedProjection {
            namespace,
            key,
            payload,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => Ok(json!(uow.compare_and_swap_named_projection(
            &namespace,
            &key,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            projection_write(
                payload,
                last_authoritative_seq,
                last_materialized_seq,
                projection_schema_version,
                materialization_status
            ),
        )?)),
        SqliteStoreOperation::ClearNamedProjection { namespace, key } => {
            uow.clear_named_projection(&namespace, &key)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ClearProjectionNamespace { namespace } => {
            uow.clear_projection_namespace(&namespace)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ReplaceStage1NodeProjection {
            namespace,
            key,
            payload,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => {
            uow.replace_stage1_node_projection(
                &namespace,
                &key,
                projection_write(
                    payload,
                    last_authoritative_seq,
                    last_materialized_seq,
                    projection_schema_version,
                    materialization_status,
                ),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ClearStage1NodeProjection { namespace, key } => {
            uow.clear_stage1_node_projection(&namespace, &key)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::PutWorkflowDesignSnapshot {
            workflow_id,
            version,
            seq,
            payload_json,
            schema_version,
        } => {
            uow.put_workflow_design_snapshot(
                &workflow_id,
                workflow_design_snapshot_write(version, seq, payload_json, schema_version),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ClearWorkflowDesignSnapshots { workflow_id } => {
            uow.clear_workflow_design_snapshots(&workflow_id)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::PutWorkflowDesignDelta {
            workflow_id,
            version,
            prev_version,
            target_seq,
            forward_json,
            inverse_json,
            schema_version,
        } => {
            uow.put_workflow_design_delta(
                &workflow_id,
                workflow_design_delta_write(
                    version,
                    prev_version,
                    target_seq,
                    forward_json,
                    inverse_json,
                    schema_version,
                ),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ClearWorkflowDesignDeltas { workflow_id } => {
            uow.clear_workflow_design_deltas(&workflow_id)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::CreateServerRun {
            run_id,
            conversation_id,
            workflow_id,
            user_id,
            user_turn_node_id,
            status,
        } => {
            uow.create_server_run(ServerRunCreate {
                run_id,
                conversation_id,
                workflow_id,
                user_id,
                user_turn_node_id,
                status,
            })?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::AppendServerRunEvent {
            run_id,
            event_type,
            payload_json,
        } => Ok(server_run_event_json(uow.append_server_run_event(
            &run_id,
            &event_type,
            payload_json,
        )?)?),
        SqliteStoreOperation::UpdateServerRun {
            run_id,
            status,
            assistant_turn_node_id,
            result_json,
            error_json,
            started_at_ms,
            finished_at_ms,
            cancel_requested,
        } => {
            uow.update_server_run(
                &run_id,
                ServerRunUpdate {
                    status,
                    assistant_turn_node_id,
                    result_json,
                    error_json,
                    started_at_ms,
                    finished_at_ms,
                    cancel_requested,
                },
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::RequestServerRunCancel { run_id } => {
            uow.request_server_run_cancel(&run_id)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::SetIndexAppliedFingerprint {
            namespace,
            coalesce_key,
            applied_fingerprint,
            last_job_id,
        } => {
            uow.set_index_applied_fingerprint(
                &namespace,
                &coalesce_key,
                applied_fingerprint.as_deref(),
                last_job_id.as_deref(),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::EnqueueIndexJob {
            job_id,
            namespace,
            entity_kind,
            entity_id,
            index_kind,
            op,
            payload_json,
            max_retries,
        } => Ok(json!(uow.enqueue_index_job(new_index_job(
            job_id,
            namespace,
            entity_kind,
            entity_id,
            index_kind,
            op,
            payload_json,
            max_retries
        ))?)),
        SqliteStoreOperation::ClaimIndexJobs {
            limit,
            lease_seconds,
            namespace,
        } => Ok(Value::Array(
            uow.claim_index_jobs(limit, lease_seconds, namespace.as_deref())?
                .into_iter()
                .map(index_job_json)
                .collect(),
        )),
        SqliteStoreOperation::MarkIndexJobDone {
            job_id,
            claim_token,
        } => Ok(json!(
            uow.mark_index_job_done(&job_id, claim_token.as_deref())?
        )),
        SqliteStoreOperation::MarkIndexJobFailed {
            job_id,
            error,
            final_,
            claim_token,
        } => {
            uow.mark_index_job_failed(&job_id, &error, final_, claim_token.as_deref())?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::BumpRetryAndRequeue {
            job_id,
            error,
            next_run_at_seconds,
            claim_token,
        } => {
            uow.bump_retry_and_requeue(
                &job_id,
                &error,
                next_run_at_seconds,
                claim_token.as_deref(),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::RenewIndexJobLease {
            job_id,
            claim_token,
            lease_seconds,
        } => Ok(json!(uow.renew_index_job_lease(
            &job_id,
            &claim_token,
            lease_seconds
        )?)),
        SqliteStoreOperation::RequeueIndexJobAtTail {
            job_id,
            payload_json,
            delay_seconds,
            claim_token,
        } => {
            uow.requeue_index_job_at_tail(
                &job_id,
                payload_json,
                delay_seconds,
                claim_token.as_deref(),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ProjectLaneMessage {
            message_id,
            namespace,
            purpose,
            inbox_id,
            conversation_id,
            recipient_id,
            sender_id,
            msg_type,
            status,
            created_at,
            available_at,
            run_id,
            step_id,
            correlation_id,
            payload_json,
            error_json,
        } => {
            uow.project_lane_message(new_lane_message(
                message_id,
                namespace,
                purpose,
                inbox_id,
                conversation_id,
                recipient_id,
                sender_id,
                msg_type,
                status,
                created_at,
                available_at,
                run_id,
                step_id,
                correlation_id,
                payload_json,
                error_json,
            ))?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::UpdateProjectedLaneMessageStatus {
            message_id,
            status,
            error_json,
        } => {
            uow.update_projected_lane_message_status(&message_id, &status, error_json)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::UpdateProjectedLaneMessageLinks {
            message_id,
            prev_message_id,
            next_message_id,
            inbox_tail_message_id,
            conversation_tail_message_id,
        } => {
            uow.update_projected_lane_message_links(
                &message_id,
                prev_message_id,
                next_message_id,
                inbox_tail_message_id,
                conversation_tail_message_id,
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ClearProjectedLaneMessages { namespace } => {
            Ok(json!(uow.clear_projected_lane_messages(&namespace)?))
        }
        SqliteStoreOperation::ClaimProjectedLaneMessages {
            namespace,
            inbox_id,
            claimed_by,
            limit,
            lease_seconds,
        } => Ok(Value::Array(
            uow.claim_projected_lane_messages(
                &namespace,
                &inbox_id,
                &claimed_by,
                limit,
                lease_seconds,
            )?
            .into_iter()
            .map(lane_message_json)
            .collect(),
        )),
        SqliteStoreOperation::AckProjectedLaneMessage {
            message_id,
            claimed_by,
        } => {
            uow.ack_projected_lane_message(&message_id, &claimed_by)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::RequeueProjectedLaneMessage {
            message_id,
            claimed_by,
            error_json,
            delay_seconds,
        } => {
            uow.requeue_projected_lane_message(
                &message_id,
                &claimed_by,
                error_json,
                delay_seconds,
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::DeadLetterProjectedLaneMessage {
            message_id,
            claimed_by,
            error_json,
        } => {
            uow.dead_letter_projected_lane_message(&message_id, &claimed_by, error_json)?;
            Ok(Value::Null)
        }
        _ => Err(SqliteStoreError::TransactionAborted(
            "unsupported SQLite batch operation".to_owned(),
        )),
    }
}

fn sqlite_store_json_impl(payload_json: &str) -> Result<String, (&'static str, String)> {
    let value: Value = serde_json::from_str(payload_json)
        .map_err(|error| (STORE_INVALID_JSON, format!("invalid JSON: {error}")))?;
    validate_sqlite_operation(value.get("operation").ok_or_else(|| {
        (
            STORE_INVALID_PAYLOAD,
            "invalid SQLite store payload: missing operation".to_owned(),
        )
    })?)?;
    let request: SqliteStoreRequest = serde_json::from_value(value).map_err(|error| {
        (
            STORE_INVALID_PAYLOAD,
            format!("invalid SQLite store payload: {error}"),
        )
    })?;
    if matches!(&request.operation, SqliteStoreOperation::Close) {
        close_cached_sqlite_store(&request.path)?;
        return Ok("null".to_owned());
    }
    let entry = cached_sqlite_store(&request.path)?;
    let mut entry = entry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let result: Result<Value, SqliteStoreError> = (|| match request.operation {
        SqliteStoreOperation::BeginTransaction => {
            let transaction_id = request.transaction_id.ok_or_else(|| {
                SqliteStoreError::TransactionAborted(
                    "SQLite begin requires transaction_id".to_owned(),
                )
            })?;
            if entry.transaction_id.is_some() {
                return Err(SqliteStoreError::TransactionAborted(
                    "SQLite external transaction is already active".to_owned(),
                ));
            }
            entry.store.begin_external_transaction()?;
            entry.transaction_id = Some(transaction_id);
            Ok(Value::Null)
        }
        SqliteStoreOperation::CommitTransaction => {
            require_sqlite_transaction(&entry, request.transaction_id.as_deref())?;
            entry.store.commit_external_transaction()?;
            entry.transaction_id = None;
            Ok(Value::Null)
        }
        SqliteStoreOperation::RollbackTransaction => {
            require_sqlite_transaction(&entry, request.transaction_id.as_deref())?;
            entry.store.rollback_external_transaction()?;
            entry.transaction_id = None;
            Ok(Value::Null)
        }
        operation => {
            match (&entry.transaction_id, request.transaction_id.as_deref()) {
                (None, None) => {}
                (Some(active), Some(requested)) if active == requested => {}
                (Some(_), _) => {
                    return Err(SqliteStoreError::TransactionAborted(
                        "SQLite operation does not own the active transaction".to_owned(),
                    ));
                }
                (None, Some(_)) => {
                    return Err(SqliteStoreError::TransactionAborted(
                        "SQLite transaction_id is stale".to_owned(),
                    ));
                }
            }
            sqlite_store_operation_json(&entry.store, operation)
        }
    })();
    result
        .and_then(|value| {
            serde_json::to_string(&value).map_err(|error| {
                SqliteStoreError::TransactionAborted(format!("cannot encode result: {error}"))
            })
        })
        .map_err(|error| {
            let code = match error {
                SqliteStoreError::EventIdNamespaceCollision { .. } => {
                    STORE_EVENT_ID_NAMESPACE_COLLISION
                }
                SqliteStoreError::NegativeSequenceValue { .. } => STORE_INVALID_SEQUENCE_VALUE,
                SqliteStoreError::Store(kogwistar_store::StoreError::CursorOutOfRange {
                    ..
                }) => STORE_CURSOR_OUT_OF_RANGE,
                SqliteStoreError::Store(kogwistar_store::StoreError::CursorRegresses {
                    ..
                }) => STORE_CURSOR_REGRESSES,
                SqliteStoreError::Store(
                    kogwistar_store::StoreError::InvalidIdentifier { .. }
                    | kogwistar_store::StoreError::InvalidRecoveryBatchLimit
                    | kogwistar_store::StoreError::InvalidEntityEventPayload { .. }
                    | kogwistar_store::StoreError::UnsupportedEntityEventOperation { .. },
                ) => STORE_INVALID_ENTITY_EVENT,
                SqliteStoreError::TransactionAborted(ref message)
                    if message == "unsupported SQLite batch operation" =>
                {
                    STORE_OPERATION_INVALID
                }
                SqliteStoreError::TransactionAborted(_) => STORE_TRANSACTION_ABORTED,
                _ => STORE_PERSISTENCE_FAILED,
            };
            (code, error.to_string())
        })
}

fn require_sqlite_transaction(
    entry: &CachedSqliteStore,
    requested: Option<&str>,
) -> Result<(), SqliteStoreError> {
    match (&entry.transaction_id, requested) {
        (Some(active), Some(requested)) if active == requested => Ok(()),
        _ => Err(SqliteStoreError::TransactionAborted(
            "SQLite transaction control does not own the active transaction".to_owned(),
        )),
    }
}

/// Cache initialized path handles. Each handle owns one serialized SQLite
/// connection, avoiding repeated schema discovery and WAL open/teardown while
/// preserving the store transaction contract.
struct CachedSqliteStore {
    store: SqliteStore,
    transaction_id: Option<String>,
}

type SharedCachedSqliteStore = std::sync::Arc<Mutex<CachedSqliteStore>>;

fn cached_sqlite_store(path: &str) -> Result<SharedCachedSqliteStore, (&'static str, String)> {
    let path = PathBuf::from(path);
    let stores = sqlite_store_cache();
    let mut stores = stores
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    // A caller may remove a temporary database between calls. Evict the old
    // handle and initialize the recreated file.
    if path.exists()
        && let Some(store) = stores.get(&path)
    {
        return Ok(store.clone());
    }

    let store =
        SqliteStore::open(&path).map_err(|error| (STORE_PERSISTENCE_FAILED, error.to_string()))?;
    let entry = std::sync::Arc::new(Mutex::new(CachedSqliteStore {
        store,
        transaction_id: None,
    }));
    stores.insert(path, entry.clone());
    Ok(entry)
}

fn sqlite_store_cache() -> &'static Mutex<BTreeMap<PathBuf, SharedCachedSqliteStore>> {
    static STORES: OnceLock<Mutex<BTreeMap<PathBuf, SharedCachedSqliteStore>>> = OnceLock::new();
    STORES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn close_cached_sqlite_store(path: &str) -> Result<(), (&'static str, String)> {
    let path = PathBuf::from(path);
    let stores = sqlite_store_cache();
    let entry = stores
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&path)
        .cloned();
    if let Some(entry) = entry {
        let entry = entry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if entry.transaction_id.is_some() {
            return Err((
                STORE_INVALID_PAYLOAD,
                "cannot close SQLite store while transaction is active".to_owned(),
            ));
        }
    }
    stores
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(&path);
    Ok(())
}

fn validate_sqlite_operation(value: &Value) -> Result<(), (&'static str, String)> {
    let object = value.as_object().ok_or_else(|| {
        (
            STORE_INVALID_PAYLOAD,
            "invalid SQLite store payload: operation must be an object".to_owned(),
        )
    })?;
    if let Some(kind) = object.get("kind").and_then(Value::as_str)
        && kind == "batch"
        && let Some(ops) = object.get("operations").and_then(Value::as_array)
    {
        for op in ops {
            validate_sqlite_operation(op)?;
        }
    }
    Ok(())
}

/// Durable PostgreSQL Phase-3 ABI. Every call owns a Tokio runtime because
/// the native store performs socket I/O; this must never use the synchronous
/// no-op-waker helper used by isolated in-memory reads.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PostgresStoreRequest {
    dsn: String,
    schema: String,
    #[serde(default)]
    transaction_id: Option<String>,
    operation: PostgresStoreOperation,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum PostgresStoreOperation {
    BeginTransaction,
    CommitTransaction,
    RollbackTransaction,
    EnsureSchema,
    NextGlobalSeq,
    CurrentGlobalSeq,
    NextUserSeq {
        user_id: String,
    },
    CurrentUserSeq {
        user_id: String,
    },
    SetUserSeq {
        user_id: String,
        value: i64,
    },
    GetIndexAppliedFingerprint {
        #[serde(default = "default_namespace")]
        namespace: String,
        coalesce_key: String,
    },
    SetIndexAppliedFingerprint {
        #[serde(default = "default_namespace")]
        namespace: String,
        coalesce_key: String,
        #[serde(default)]
        applied_fingerprint: Option<String>,
        #[serde(default)]
        last_job_id: Option<String>,
    },
    CreateGraphSchema {
        embedding_dim: usize,
        #[serde(default)]
        nodes_table: Option<String>,
        #[serde(default)]
        edges_table: Option<String>,
        #[serde(default)]
        documents_table: Option<String>,
        #[serde(default)]
        domains_table: Option<String>,
    },
    AllocEventSeq {
        namespace: String,
    },
    RawAppend {
        namespace: String,
        event_id: String,
        entity_kind: String,
        entity_id: String,
        op: String,
        payload_json: String,
    },
    GraphMutation {
        namespace: String,
        #[serde(default)]
        workspace_id: Option<String>,
        #[serde(default)]
        graph_space: Option<String>,
        table: String,
        entity_kind: String,
        event_id: String,
        op: String,
        record: GraphMutationRecord,
        payload: Value,
        embedding_dim: usize,
    },
    GraphMetadataPatchMutation {
        namespace: String,
        #[serde(default)]
        workspace_id: Option<String>,
        #[serde(default)]
        graph_space: Option<String>,
        table: String,
        entity_kind: String,
        event_id: String,
        op: String,
        entity_id: String,
        #[serde(default)]
        document: Option<String>,
        #[serde(default)]
        metadata_patch: Map<String, Value>,
        payload: Value,
    },
    GraphDeleteMutation {
        namespace: String,
        #[serde(default)]
        workspace_id: Option<String>,
        #[serde(default)]
        graph_space: Option<String>,
        table: String,
        entity_kind: String,
        event_id: String,
        entity_id: String,
        payload: Value,
    },
    UpsertGraphProjection {
        namespace: String,
        #[serde(default)]
        workspace_id: Option<String>,
        #[serde(default)]
        graph_space: Option<String>,
        table: String,
        record: GraphMutationRecord,
        embedding_dim: usize,
    },
    PatchGraphProjectionMetadata {
        namespace: String,
        #[serde(default)]
        workspace_id: Option<String>,
        #[serde(default)]
        graph_space: Option<String>,
        table: String,
        entity_id: String,
        #[serde(default)]
        document: Option<String>,
        #[serde(default)]
        metadata_patch: Map<String, Value>,
        #[serde(default = "default_true")]
        patch_document_metadata: bool,
    },
    GraphProjectionRecords {
        namespace: String,
        #[serde(default)]
        workspace_id: Option<String>,
        #[serde(default)]
        graph_space: Option<String>,
        table: String,
        #[serde(default)]
        ids: Option<Vec<String>>,
        #[serde(default)]
        metadata: BTreeMap<String, Value>,
        #[serde(default = "default_list_limit")]
        limit: usize,
    },
    GraphProjectionVectorQuery {
        namespace: String,
        #[serde(default)]
        workspace_id: Option<String>,
        #[serde(default)]
        graph_space: Option<String>,
        table: String,
        embedding: Vec<f32>,
        #[serde(default = "default_list_limit")]
        limit: usize,
        #[serde(default)]
        metadata: BTreeMap<String, Value>,
        #[serde(default)]
        metric: SnapshotDistanceMetric,
        embedding_dim: usize,
    },
    ExclusiveRawReplay {
        namespace: String,
        after_seq: i64,
        limit: usize,
    },
    LatestRetainedEventSeq {
        namespace: String,
    },
    PruneEntityEventsAfter {
        namespace: String,
        to_seq: i64,
    },
    CursorGet {
        namespace: String,
        consumer: String,
    },
    LegacyCursorSet {
        namespace: String,
        consumer: String,
        last_seq: i64,
    },
    StrictCursorAdvance {
        namespace: String,
        consumer: String,
        last_seq: i64,
    },
    RecoverEntityProjection {
        namespace: String,
        consumer: String,
        projection_namespace: String,
        projection_key: String,
        batch_limit: usize,
        #[serde(default)]
        abort_after_projection: bool,
    },
    RebuildEntityProjection {
        namespace: String,
        consumer: String,
        projection_namespace: String,
        projection_key: String,
        #[serde(default)]
        abort_after_projection: bool,
    },
    GetNamedProjection {
        namespace: String,
        key: String,
    },
    ListNamedProjections {
        namespace: String,
    },
    ReplaceNamedProjection {
        namespace: String,
        key: String,
        payload: Map<String, Value>,
        last_authoritative_seq: i64,
        last_materialized_seq: i64,
        projection_schema_version: i64,
        materialization_status: String,
    },
    CompareAndSwapNamedProjection {
        namespace: String,
        key: String,
        payload: Map<String, Value>,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        last_authoritative_seq: i64,
        last_materialized_seq: i64,
        projection_schema_version: i64,
        materialization_status: String,
    },
    ClearNamedProjection {
        namespace: String,
        key: String,
    },
    ClearProjectionNamespace {
        namespace: String,
    },
    PutWorkflowDesignSnapshot {
        workflow_id: String,
        version: i64,
        seq: i64,
        payload_json: String,
        schema_version: i64,
    },
    GetWorkflowDesignSnapshot {
        workflow_id: String,
        max_version: i64,
        schema_version: i64,
    },
    ClearWorkflowDesignSnapshots {
        workflow_id: String,
    },
    PutWorkflowDesignDelta {
        workflow_id: String,
        version: i64,
        prev_version: i64,
        target_seq: i64,
        forward_json: String,
        inverse_json: String,
        schema_version: i64,
    },
    GetWorkflowDesignDelta {
        workflow_id: String,
        version: i64,
        schema_version: i64,
    },
    ClearWorkflowDesignDeltas {
        workflow_id: String,
    },
    CreateServerRun {
        run_id: String,
        conversation_id: String,
        workflow_id: String,
        #[serde(default)]
        user_id: Option<String>,
        user_turn_node_id: String,
        #[serde(default = "default_run_status")]
        status: String,
    },
    GetServerRun {
        run_id: String,
    },
    ListServerRuns {
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        workflow_id: Option<String>,
        #[serde(default)]
        conversation_id: Option<String>,
        #[serde(default = "default_run_limit")]
        limit: usize,
    },
    AppendServerRunEvent {
        run_id: String,
        event_type: String,
        payload_json: String,
    },
    ListServerRunEvents {
        run_id: String,
        #[serde(default)]
        after_seq: i64,
        #[serde(default = "default_run_event_limit")]
        limit: usize,
    },
    UpdateServerRun {
        run_id: String,
        status: String,
        assistant_turn_node_id: Option<String>,
        result_json: Option<String>,
        error_json: Option<String>,
        started_at_ms: Option<i64>,
        finished_at_ms: Option<i64>,
        #[serde(default)]
        cancel_requested: Option<bool>,
    },
    RequestServerRunCancel {
        run_id: String,
    },
    ApplyRecordedRuntimeTransition {
        transition: Box<RecordedRuntimeTransition>,
        #[serde(default)]
        abort_after_writes: bool,
    },
    ApplyClaimedRecordedRuntimeTransition {
        handoff: RecordedWorkerHandoff,
        transition: Box<RecordedRuntimeTransition>,
        #[serde(default)]
        abort_after_writes: bool,
    },
    ReadRecordedRuntimeState {
        run_id: String,
        workflow_id: String,
        conversation_id: String,
    },
    EnqueueIndexJob {
        job_id: String,
        #[serde(default = "default_namespace")]
        namespace: String,
        entity_kind: String,
        entity_id: String,
        index_kind: String,
        op: String,
        #[serde(default)]
        payload_json: Option<String>,
        #[serde(default = "default_max_retries")]
        max_retries: i64,
    },
    ClaimIndexJobs {
        #[serde(default = "default_claim_limit")]
        limit: usize,
        #[serde(default = "default_lease_seconds")]
        lease_seconds: i64,
        #[serde(default = "default_optional_namespace")]
        namespace: Option<String>,
    },
    MarkIndexJobDone {
        job_id: String,
        #[serde(default)]
        claim_token: Option<String>,
    },
    MarkIndexJobFailed {
        job_id: String,
        error: String,
        #[serde(rename = "final", default = "default_true")]
        final_: bool,
        #[serde(default)]
        claim_token: Option<String>,
    },
    BumpRetryAndRequeue {
        job_id: String,
        error: String,
        next_run_at_seconds: i64,
        #[serde(default)]
        claim_token: Option<String>,
    },
    RenewIndexJobLease {
        job_id: String,
        claim_token: String,
        lease_seconds: i64,
    },
    RequeueIndexJobAtTail {
        job_id: String,
        payload_json: String,
        #[serde(default)]
        delay_seconds: i64,
        #[serde(default)]
        claim_token: Option<String>,
    },
    ListIndexJobs {
        #[serde(default = "default_optional_namespace")]
        namespace: Option<String>,
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        entity_kind: Option<String>,
        #[serde(default)]
        entity_id: Option<String>,
        #[serde(default)]
        index_kind: Option<String>,
        #[serde(default = "default_list_limit")]
        limit: usize,
    },
    ProjectLaneMessage {
        message_id: String,
        #[serde(default = "default_namespace")]
        namespace: String,
        #[serde(default = "default_lane_purpose")]
        purpose: String,
        inbox_id: String,
        conversation_id: String,
        recipient_id: String,
        sender_id: String,
        msg_type: String,
        status: String,
        created_at: i64,
        available_at: i64,
        #[serde(default)]
        run_id: Option<String>,
        #[serde(default)]
        step_id: Option<String>,
        #[serde(default)]
        correlation_id: Option<String>,
        #[serde(default)]
        payload_json: Option<String>,
        #[serde(default)]
        error_json: Option<String>,
    },
    GetProjectedLaneMessage {
        message_id: String,
    },
    UpdateProjectedLaneMessageStatus {
        message_id: String,
        status: String,
        #[serde(default)]
        error_json: Option<String>,
    },
    UpdateProjectedLaneMessageLinks {
        message_id: String,
        #[serde(default)]
        prev_message_id: Option<String>,
        #[serde(default)]
        next_message_id: Option<String>,
        #[serde(default)]
        inbox_tail_message_id: Option<String>,
        #[serde(default)]
        conversation_tail_message_id: Option<String>,
    },
    ListProjectedLaneMessages {
        #[serde(default = "default_optional_namespace")]
        namespace: Option<String>,
        #[serde(default)]
        purpose: Option<String>,
        #[serde(default)]
        inbox_id: Option<String>,
        #[serde(default)]
        conversation_id: Option<String>,
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        msg_type: Option<String>,
        #[serde(default)]
        sender_id: Option<String>,
        #[serde(default)]
        recipient_id: Option<String>,
        #[serde(default)]
        correlation_id: Option<String>,
        #[serde(default)]
        reply_to_message_id: Option<String>,
        #[serde(default)]
        created_at_gte: Option<i64>,
        #[serde(default)]
        created_at_lte: Option<i64>,
        #[serde(default)]
        available_at_gte: Option<i64>,
        #[serde(default)]
        available_at_lte: Option<i64>,
        #[serde(default = "default_list_limit")]
        limit: usize,
        #[serde(default)]
        newest_first: bool,
    },
    ClearProjectedLaneMessages {
        namespace: String,
    },
    ClaimProjectedLaneMessages {
        #[serde(default = "default_namespace")]
        namespace: String,
        inbox_id: String,
        claimed_by: String,
        #[serde(default = "default_claim_limit")]
        limit: usize,
        #[serde(default = "default_lease_seconds")]
        lease_seconds: i64,
    },
    AckProjectedLaneMessage {
        message_id: String,
        claimed_by: String,
    },
    RequeueProjectedLaneMessage {
        message_id: String,
        claimed_by: String,
        #[serde(default)]
        error_json: Option<String>,
        #[serde(default)]
        delay_seconds: i64,
    },
    DeadLetterProjectedLaneMessage {
        message_id: String,
        claimed_by: String,
        #[serde(default)]
        error_json: Option<String>,
    },
    Batch {
        operations: Vec<PostgresStoreOperation>,
        #[serde(default)]
        abort: bool,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphMutationRecord {
    id: String,
    #[serde(default)]
    document: Option<String>,
    #[serde(default)]
    metadata: Map<String, Value>,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
}

fn postgres_raw_event_json(event: PostgresRawEntityEvent) -> Value {
    json!({
        "namespace": event.namespace,
        "seq": event.seq,
        "event_id": event.event_id,
        "entity_kind": event.entity_kind,
        "entity_id": event.entity_id,
        "op": event.op,
        "payload_json": event.payload_json,
        "created_at": event.created_at,
    })
}

fn postgres_new_raw_event(
    event_id: String,
    entity_kind: String,
    entity_id: String,
    op: String,
    payload_json: String,
) -> PostgresNewRawEntityEvent {
    PostgresNewRawEntityEvent {
        event_id,
        entity_kind,
        entity_id,
        op,
        payload_json,
    }
}

fn postgres_appended_raw_event_json(appended: kogwistar_store_postgres::AppendedRawEvent) -> Value {
    let seq = appended.event.seq;
    json!({"seq": seq, "inserted": appended.inserted, "event": postgres_raw_event_json(appended.event)})
}

fn graph_scope(
    namespace: String,
    workspace_id: Option<String>,
    graph_space: Option<String>,
) -> kogwistar_store::GraphScope {
    kogwistar_store::GraphScope {
        namespace,
        workspace_id,
        graph_space,
    }
}

fn graph_mutation_record(record: GraphMutationRecord) -> GraphRecord {
    GraphRecord {
        id: record.id,
        document: record.document,
        metadata: record.metadata,
        embedding: record.embedding,
    }
}

fn graph_mutation_json(applied: kogwistar_store::AppliedGraphMutation) -> Value {
    json!({
        "event": event_json(applied.event),
        "inserted": applied.inserted,
        "mutated": applied.mutated,
    })
}

fn optional_graph_mutation_json(applied: Option<kogwistar_store::AppliedGraphMutation>) -> Value {
    applied.map(graph_mutation_json).unwrap_or(Value::Null)
}

fn graph_table_names(
    nodes_table: Option<String>,
    edges_table: Option<String>,
    documents_table: Option<String>,
    domains_table: Option<String>,
) -> kogwistar_store_postgres::GraphTableNames {
    let defaults = kogwistar_store_postgres::GraphTableNames::default();
    kogwistar_store_postgres::GraphTableNames {
        nodes: nodes_table.unwrap_or(defaults.nodes),
        edges: edges_table.unwrap_or(defaults.edges),
        documents: documents_table.unwrap_or(defaults.documents),
        domains: domains_table.unwrap_or(defaults.domains),
    }
}

async fn postgres_store_operation_json(
    store: &PostgresStore,
    operation: PostgresStoreOperation,
) -> Result<Value, PostgresStoreError> {
    match operation {
        PostgresStoreOperation::BeginTransaction
        | PostgresStoreOperation::CommitTransaction
        | PostgresStoreOperation::RollbackTransaction => Err(
            postgres_session_error("PostgreSQL transaction control cannot be nested as an operation"),
        ),
        PostgresStoreOperation::EnsureSchema => {
            store.ensure_schema().await?;
            Ok(json!({"initialized": true}))
        }
        PostgresStoreOperation::NextGlobalSeq => Ok(json!(store.next_global_seq().await?)),
        PostgresStoreOperation::CurrentGlobalSeq => Ok(json!(store.current_global_seq().await?)),
        PostgresStoreOperation::NextUserSeq { user_id } => {
            Ok(json!(store.next_user_seq(&user_id).await?))
        }
        PostgresStoreOperation::CurrentUserSeq { user_id } => {
            Ok(json!(store.current_user_seq(&user_id).await?))
        }
        PostgresStoreOperation::SetUserSeq { user_id, value } => {
            store.set_user_seq(&user_id, value).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::GetIndexAppliedFingerprint {
            namespace,
            coalesce_key,
        } => Ok(json!(
            store
                .get_index_applied_fingerprint(&namespace, &coalesce_key)
                .await?
        )),
        PostgresStoreOperation::SetIndexAppliedFingerprint {
            namespace,
            coalesce_key,
            applied_fingerprint,
            last_job_id,
        } => {
            store
                .set_index_applied_fingerprint(
                    &namespace,
                    &coalesce_key,
                    applied_fingerprint,
                    last_job_id,
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::CreateGraphSchema {
            embedding_dim,
            nodes_table,
            edges_table,
            documents_table,
            domains_table,
        } => {
            store
                .create_graph_schema(
                    embedding_dim,
                    graph_table_names(nodes_table, edges_table, documents_table, domains_table),
                )
                .await?;
            Ok(json!({"initialized": true}))
        }
        PostgresStoreOperation::AllocEventSeq { namespace } => {
            Ok(json!(store.alloc_event_seq(&namespace).await?))
        }
        PostgresStoreOperation::RawAppend {
            namespace,
            event_id,
            entity_kind,
            entity_id,
            op,
            payload_json,
        } => Ok(postgres_appended_raw_event_json(
            store
                .append_raw_entity_event(
                    &namespace,
                    postgres_new_raw_event(event_id, entity_kind, entity_id, op, payload_json),
                )
            .await?,
        )),
        PostgresStoreOperation::GraphMutation {
            namespace,
            workspace_id,
            graph_space,
            table,
            entity_kind,
            event_id,
            op,
            record,
            payload,
            embedding_dim,
        } => Ok(graph_mutation_json(
            store
                .apply_graph_mutation(kogwistar_store::GraphMutation {
                    scope: graph_scope(namespace, workspace_id, graph_space),
                    table,
                    entity_kind,
                    event_id,
                    op,
                    payload,
                    record: graph_mutation_record(record),
                    embedding_dim,
                })
                .await?,
        )),
        PostgresStoreOperation::GraphMetadataPatchMutation {
            namespace,
            workspace_id,
            graph_space,
            table,
            entity_kind,
            event_id,
            op,
            entity_id,
            document,
            metadata_patch,
            payload,
        } => Ok(optional_graph_mutation_json(
            store
                .apply_graph_metadata_patch_mutation(
                    kogwistar_store_postgres::GraphMetadataPatchMutation {
                        scope: graph_scope(namespace, workspace_id, graph_space),
                        table,
                        entity_kind,
                        event_id,
                        op,
                        entity_id,
                        document,
                        metadata_patch,
                        payload,
                    },
                )
                .await?,
        )),
        PostgresStoreOperation::GraphDeleteMutation {
            namespace,
            workspace_id,
            graph_space,
            table,
            entity_kind,
            event_id,
            entity_id,
            payload,
        } => Ok(optional_graph_mutation_json(
            store
                .apply_graph_delete_mutation(
                    kogwistar_store_postgres::GraphDeleteMutation {
                        scope: graph_scope(namespace, workspace_id, graph_space),
                        table,
                        entity_kind,
                        event_id,
                        entity_id,
                        payload,
                    },
                )
                .await?,
        )),
        PostgresStoreOperation::UpsertGraphProjection {
            namespace,
            workspace_id,
            graph_space,
            table,
            record,
            embedding_dim,
        } => {
            store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: graph_scope(namespace, workspace_id, graph_space),
                    table,
                    record: graph_mutation_record(record),
                    embedding_dim,
                })
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::PatchGraphProjectionMetadata {
            namespace,
            workspace_id,
            graph_space,
            table,
            entity_id,
            document,
            metadata_patch,
            patch_document_metadata,
        } => Ok(json!(
            store
                .patch_graph_projection_metadata(
                    kogwistar_store_postgres::GraphProjectionMetadataPatch {
                        scope: graph_scope(namespace, workspace_id, graph_space),
                        table,
                        entity_id,
                        document,
                        metadata_patch,
                        patch_document_metadata,
                    },
                )
                .await?
        )),
        PostgresStoreOperation::GraphProjectionRecords {
            namespace,
            workspace_id,
            graph_space,
            table,
            ids,
            metadata,
            limit,
        } => Ok(Value::Array(
            store
                .graph_projection_records(kogwistar_store::GraphProjectionRead {
                    scope: graph_scope(namespace, workspace_id, graph_space),
                    table,
                    ids,
                    metadata: metadata_filter(metadata),
                    limit,
                })
                .await?
                .into_iter()
                .map(graph_record_json)
                .collect(),
        )),
        PostgresStoreOperation::GraphProjectionVectorQuery {
            namespace,
            workspace_id,
            graph_space,
            table,
            embedding,
            limit,
            metadata,
            metric,
            embedding_dim,
        } => Ok(Value::Array(
            store
                .graph_projection_vector_query(kogwistar_store::GraphProjectionVectorQuery {
                    scope: graph_scope(namespace, workspace_id, graph_space),
                    table,
                    query: VectorQuery {
                        embedding,
                        limit,
                        metadata: metadata_filter(metadata),
                        metric: metric.into(),
                    },
                    embedding_dim,
                })
                .await?
                .into_iter()
                .map(|item| json!({"record": graph_record_json(item.record), "distance": item.distance}))
                .collect(),
        )),
        PostgresStoreOperation::ExclusiveRawReplay {
            namespace,
            after_seq,
            limit,
        } => Ok(Value::Array(
            store
                .replay_raw_events(&namespace, after_seq, limit)
                .await?
                .into_iter()
                .map(postgres_raw_event_json)
                .collect(),
        )),
        PostgresStoreOperation::LatestRetainedEventSeq { namespace } => {
            Ok(json!(store.latest_retained_event_seq(&namespace).await?))
        }
        PostgresStoreOperation::PruneEntityEventsAfter { namespace, to_seq } => Ok(json!(
            store.prune_entity_events_after(&namespace, to_seq).await?
        )),
        PostgresStoreOperation::CursorGet {
            namespace,
            consumer,
        } => Ok(cursor_json(
            store.replay_cursor(&namespace, &consumer).await?,
        )),
        PostgresStoreOperation::LegacyCursorSet {
            namespace,
            consumer,
            last_seq,
        } => Ok(cursor_json(
            store
                .set_replay_cursor_legacy(&namespace, &consumer, last_seq)
                .await?,
        )),
        PostgresStoreOperation::StrictCursorAdvance {
            namespace,
            consumer,
            last_seq,
        } => Ok(cursor_json(
            store
                .strict_advance_replay_cursor(&namespace, &consumer, last_seq)
                .await?,
        )),
        PostgresStoreOperation::RecoverEntityProjection {
            namespace,
            consumer,
            projection_namespace,
            projection_key,
            batch_limit,
            abort_after_projection,
        } => Ok(entity_recovery_report_json(
            store
                .recover_entity_projection(entity_recovery_request(
                    namespace,
                    consumer,
                    projection_namespace,
                    projection_key,
                    batch_limit,
                    abort_after_projection,
                ))
                .await?,
        )),
        PostgresStoreOperation::RebuildEntityProjection {
            namespace,
            consumer,
            projection_namespace,
            projection_key,
            abort_after_projection,
        } => Ok(entity_recovery_report_json(
            store
                .rebuild_entity_projection(entity_rebuild_request(
                    namespace,
                    consumer,
                    projection_namespace,
                    projection_key,
                    abort_after_projection,
                ))
                .await?,
        )),
        PostgresStoreOperation::GetNamedProjection { namespace, key } => Ok(store
            .get_named_projection(&namespace, &key)
            .await?
            .map(projection_json)
            .unwrap_or(Value::Null)),
        PostgresStoreOperation::ListNamedProjections { namespace } => Ok(Value::Array(
            store
                .list_named_projections(&namespace)
                .await?
                .into_iter()
                .map(projection_json)
                .collect(),
        )),
        PostgresStoreOperation::ReplaceNamedProjection {
            namespace,
            key,
            payload,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => {
            store
                .replace_named_projection(
                    &namespace,
                    &key,
                    projection_write(
                        payload,
                        last_authoritative_seq,
                        last_materialized_seq,
                        projection_schema_version,
                        materialization_status,
                    ),
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::CompareAndSwapNamedProjection {
            namespace,
            key,
            payload,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => Ok(json!(
            store
                .compare_and_swap_named_projection(
                    &namespace,
                    &key,
                    expected_last_authoritative_seq,
                    expected_last_materialized_seq,
                    projection_write(
                        payload,
                        last_authoritative_seq,
                        last_materialized_seq,
                        projection_schema_version,
                        materialization_status
                    ),
                )
                .await?
        )),
        PostgresStoreOperation::ClearNamedProjection { namespace, key } => {
            store.clear_named_projection(&namespace, &key).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ClearProjectionNamespace { namespace } => {
            store.clear_projection_namespace(&namespace).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::PutWorkflowDesignSnapshot {
            workflow_id,
            version,
            seq,
            payload_json,
            schema_version,
        } => {
            store
                .put_workflow_design_snapshot(
                    &workflow_id,
                    workflow_design_snapshot_write(version, seq, payload_json, schema_version),
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::GetWorkflowDesignSnapshot {
            workflow_id,
            max_version,
            schema_version,
        } => Ok(store
            .get_workflow_design_snapshot(&workflow_id, max_version, schema_version)
            .await?
            .map(workflow_design_snapshot_json)
            .unwrap_or(Value::Null)),
        PostgresStoreOperation::ClearWorkflowDesignSnapshots { workflow_id } => {
            store.clear_workflow_design_snapshots(&workflow_id).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::PutWorkflowDesignDelta {
            workflow_id,
            version,
            prev_version,
            target_seq,
            forward_json,
            inverse_json,
            schema_version,
        } => {
            store
                .put_workflow_design_delta(
                    &workflow_id,
                    workflow_design_delta_write(
                        version,
                        prev_version,
                        target_seq,
                        forward_json,
                        inverse_json,
                        schema_version,
                    ),
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::GetWorkflowDesignDelta {
            workflow_id,
            version,
            schema_version,
        } => Ok(store
            .get_workflow_design_delta(&workflow_id, version, schema_version)
            .await?
            .map(workflow_design_delta_json)
            .unwrap_or(Value::Null)),
        PostgresStoreOperation::ClearWorkflowDesignDeltas { workflow_id } => {
            store.clear_workflow_design_deltas(&workflow_id).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::CreateServerRun {
            run_id,
            conversation_id,
            workflow_id,
            user_id,
            user_turn_node_id,
            status,
        } => {
            store
                .create_server_run(ServerRunCreate {
                    run_id,
                    conversation_id,
                    workflow_id,
                    user_id,
                    user_turn_node_id,
                    status,
                })
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::GetServerRun { run_id } => Ok(store
            .get_server_run(&run_id)
            .await?
            .map(server_run_json)
            .transpose()?
            .unwrap_or(Value::Null)),
        PostgresStoreOperation::ListServerRuns {
            status,
            workflow_id,
            conversation_id,
            limit,
        } => Ok(Value::Array(
            store
                .list_server_runs(
                    status.as_deref(),
                    workflow_id.as_deref(),
                    conversation_id.as_deref(),
                    limit,
                )
                .await?
                .into_iter()
                .map(server_run_json)
                .collect::<Result<_, _>>()?,
        )),
        PostgresStoreOperation::AppendServerRunEvent {
            run_id,
            event_type,
            payload_json,
        } => Ok(server_run_event_json(
            store
                .append_server_run_event(&run_id, &event_type, payload_json)
                .await?,
        )?),
        PostgresStoreOperation::ListServerRunEvents {
            run_id,
            after_seq,
            limit,
        } => Ok(Value::Array(
            store
                .list_server_run_events(&run_id, after_seq, limit)
                .await?
                .into_iter()
                .map(server_run_event_json)
                .collect::<Result<_, _>>()?,
        )),
        PostgresStoreOperation::UpdateServerRun {
            run_id,
            status,
            assistant_turn_node_id,
            result_json,
            error_json,
            started_at_ms,
            finished_at_ms,
            cancel_requested,
        } => {
            store
                .update_server_run(
                    &run_id,
                    ServerRunUpdate {
                        status,
                        assistant_turn_node_id,
                        result_json,
                        error_json,
                        started_at_ms,
                        finished_at_ms,
                        cancel_requested,
                    },
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::RequestServerRunCancel { run_id } => {
            store.request_server_run_cancel(&run_id).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ApplyRecordedRuntimeTransition {
            transition,
            abort_after_writes,
        } => serde_json::to_value(
            store
                .apply_recorded_runtime_transition(*transition, abort_after_writes)
                .await?,
        )
        .map_err(|error| PostgresStoreError::TransactionAborted(error.to_string())),
        PostgresStoreOperation::ApplyClaimedRecordedRuntimeTransition {
            handoff,
            transition,
            abort_after_writes,
        } => serde_json::to_value(
            store
                .apply_claimed_recorded_runtime_transition(
                    handoff,
                    *transition,
                    abort_after_writes,
                )
                .await?,
        )
        .map_err(|error| PostgresStoreError::TransactionAborted(error.to_string())),
        PostgresStoreOperation::ReadRecordedRuntimeState {
            run_id,
            workflow_id,
            conversation_id,
        } => serde_json::to_value(
            store
                .read_recorded_runtime_state(&run_id, &workflow_id, &conversation_id)
                .await?,
        )
        .map_err(|error| PostgresStoreError::TransactionAborted(error.to_string())),
        PostgresStoreOperation::EnqueueIndexJob {
            job_id,
            namespace,
            entity_kind,
            entity_id,
            index_kind,
            op,
            payload_json,
            max_retries,
        } => Ok(json!(
            store
                .enqueue_index_job(new_index_job(
                    job_id,
                    namespace,
                    entity_kind,
                    entity_id,
                    index_kind,
                    op,
                    payload_json,
                    max_retries
                ))
                .await?
        )),
        PostgresStoreOperation::ClaimIndexJobs {
            limit,
            lease_seconds,
            namespace,
        } => Ok(Value::Array(
            store
                .claim_index_jobs(limit, lease_seconds, namespace.as_deref())
                .await?
                .into_iter()
                .map(index_job_json)
                .collect(),
        )),
        PostgresStoreOperation::MarkIndexJobDone {
            job_id,
            claim_token,
        } => Ok(json!(
            store
                .mark_index_job_done(&job_id, claim_token.as_deref())
                .await?
        )),
        PostgresStoreOperation::MarkIndexJobFailed {
            job_id,
            error,
            final_,
            claim_token,
        } => {
            store
                .mark_index_job_failed(&job_id, &error, final_, claim_token.as_deref())
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::BumpRetryAndRequeue {
            job_id,
            error,
            next_run_at_seconds,
            claim_token,
        } => {
            store
                .bump_retry_and_requeue(
                    &job_id,
                    &error,
                    next_run_at_seconds,
                    claim_token.as_deref(),
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::RenewIndexJobLease {
            job_id,
            claim_token,
            lease_seconds,
        } => Ok(json!(
            store
                .renew_index_job_lease(&job_id, &claim_token, lease_seconds)
                .await?
        )),
        PostgresStoreOperation::RequeueIndexJobAtTail {
            job_id,
            payload_json,
            delay_seconds,
            claim_token,
        } => {
            store
                .requeue_index_job_at_tail(
                    &job_id,
                    payload_json,
                    delay_seconds,
                    claim_token.as_deref(),
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ListIndexJobs {
            namespace,
            status,
            entity_kind,
            entity_id,
            index_kind,
            limit,
        } => Ok(Value::Array(
            store
                .list_index_jobs(
                    namespace.as_deref(),
                    status.as_deref(),
                    entity_kind.as_deref(),
                    entity_id.as_deref(),
                    index_kind.as_deref(),
                    limit,
                )
                .await?
                .into_iter()
                .map(index_job_json)
                .collect(),
        )),
        PostgresStoreOperation::ProjectLaneMessage {
            message_id,
            namespace,
            purpose,
            inbox_id,
            conversation_id,
            recipient_id,
            sender_id,
            msg_type,
            status,
            created_at,
            available_at,
            run_id,
            step_id,
            correlation_id,
            payload_json,
            error_json,
        } => {
            store
                .project_lane_message(new_lane_message(
                    message_id,
                    namespace,
                    purpose,
                    inbox_id,
                    conversation_id,
                    recipient_id,
                    sender_id,
                    msg_type,
                    status,
                    created_at,
                    available_at,
                    run_id,
                    step_id,
                    correlation_id,
                    payload_json,
                    error_json,
                ))
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::GetProjectedLaneMessage { message_id } => Ok(store
            .get_projected_lane_message(&message_id)
            .await?
            .map(lane_message_json)
            .unwrap_or(Value::Null)),
        PostgresStoreOperation::UpdateProjectedLaneMessageStatus {
            message_id,
            status,
            error_json,
        } => {
            store
                .update_projected_lane_message_status(&message_id, &status, error_json)
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::UpdateProjectedLaneMessageLinks {
            message_id,
            prev_message_id,
            next_message_id,
            inbox_tail_message_id,
            conversation_tail_message_id,
        } => {
            store
                .update_projected_lane_message_links(
                    &message_id,
                    prev_message_id,
                    next_message_id,
                    inbox_tail_message_id,
                    conversation_tail_message_id,
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ListProjectedLaneMessages {
            namespace,
            purpose,
            inbox_id,
            conversation_id,
            status,
            msg_type,
            sender_id,
            recipient_id,
            correlation_id,
            reply_to_message_id,
            created_at_gte,
            created_at_lte,
            available_at_gte,
            available_at_lte,
            limit,
            newest_first,
        } => Ok(Value::Array(
            store
                .list_projected_lane_messages(lane_filter(
                    namespace,
                    purpose,
                    inbox_id,
                    conversation_id,
                    status,
                    msg_type,
                    sender_id,
                    recipient_id,
                    correlation_id,
                    reply_to_message_id,
                    created_at_gte,
                    created_at_lte,
                    available_at_gte,
                    available_at_lte,
                    limit,
                    newest_first,
                ))
                .await?
                .into_iter()
                .map(lane_message_json)
                .collect(),
        )),
        PostgresStoreOperation::ClearProjectedLaneMessages { namespace } => Ok(json!(
            store.clear_projected_lane_messages(&namespace).await?
        )),
        PostgresStoreOperation::ClaimProjectedLaneMessages {
            namespace,
            inbox_id,
            claimed_by,
            limit,
            lease_seconds,
        } => Ok(Value::Array(
            store
                .claim_projected_lane_messages(
                    &namespace,
                    &inbox_id,
                    &claimed_by,
                    limit,
                    lease_seconds,
                )
                .await?
                .into_iter()
                .map(lane_message_json)
                .collect(),
        )),
        PostgresStoreOperation::AckProjectedLaneMessage {
            message_id,
            claimed_by,
        } => {
            store
                .ack_projected_lane_message(&message_id, &claimed_by)
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::RequeueProjectedLaneMessage {
            message_id,
            claimed_by,
            error_json,
            delay_seconds,
        } => {
            store
                .requeue_projected_lane_message(&message_id, &claimed_by, error_json, delay_seconds)
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::DeadLetterProjectedLaneMessage {
            message_id,
            claimed_by,
            error_json,
        } => {
            store
                .dead_letter_projected_lane_message(&message_id, &claimed_by, error_json)
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::Batch { operations, abort } => {
            store
                .transaction(move |uow| {
                    Box::pin(async move {
                        let mut results = Vec::with_capacity(operations.len());
                        for operation in operations {
                            results.push(postgres_uow_operation_json(uow, operation).await?);
                        }
                        if abort {
                            return Err(PostgresStoreError::TransactionAborted(
                                "requested by PostgreSQL JSON ABI abort flag".to_owned(),
                            ));
                        }
                        Ok(Value::Array(results))
                    })
                })
                .await
        }
    }
}

async fn postgres_uow_operation_json(
    uow: &mut PostgresUnitOfWork<'_>,
    operation: PostgresStoreOperation,
) -> Result<Value, PostgresStoreError> {
    match operation {
        PostgresStoreOperation::NextGlobalSeq => Ok(json!(uow.next_global_seq().await?)),
        PostgresStoreOperation::CurrentGlobalSeq => Ok(json!(uow.current_global_seq().await?)),
        PostgresStoreOperation::NextUserSeq { user_id } => {
            Ok(json!(uow.next_user_seq(&user_id).await?))
        }
        PostgresStoreOperation::CurrentUserSeq { user_id } => {
            Ok(json!(uow.current_user_seq(&user_id).await?))
        }
        PostgresStoreOperation::SetUserSeq { user_id, value } => {
            uow.set_user_seq(&user_id, value).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::GetIndexAppliedFingerprint {
            namespace,
            coalesce_key,
        } => Ok(json!(
            uow.get_index_applied_fingerprint(&namespace, &coalesce_key)
                .await?
        )),
        PostgresStoreOperation::SetIndexAppliedFingerprint {
            namespace,
            coalesce_key,
            applied_fingerprint,
            last_job_id,
        } => {
            uow.set_index_applied_fingerprint(
                &namespace,
                &coalesce_key,
                applied_fingerprint,
                last_job_id,
            )
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::AllocEventSeq { namespace } => {
            Ok(json!(uow.alloc_event_seq(&namespace).await?))
        }
        PostgresStoreOperation::RawAppend {
            namespace,
            event_id,
            entity_kind,
            entity_id,
            op,
            payload_json,
        } => Ok(postgres_appended_raw_event_json(
            uow.append_raw_entity_event(
                &namespace,
                postgres_new_raw_event(event_id, entity_kind, entity_id, op, payload_json),
            )
            .await?,
        )),
        PostgresStoreOperation::GraphMutation {
            namespace,
            workspace_id,
            graph_space,
            table,
            entity_kind,
            event_id,
            op,
            record,
            payload,
            embedding_dim,
        } => Ok(graph_mutation_json(
            uow.apply_graph_mutation(kogwistar_store::GraphMutation {
                scope: graph_scope(namespace, workspace_id, graph_space),
                table,
                entity_kind,
                event_id,
                op,
                payload,
                record: graph_mutation_record(record),
                embedding_dim,
            })
            .await?,
        )),
        PostgresStoreOperation::ExclusiveRawReplay {
            namespace,
            after_seq,
            limit,
        } => Ok(Value::Array(
            uow.replay_raw_events(&namespace, after_seq, limit)
                .await?
                .into_iter()
                .map(postgres_raw_event_json)
                .collect(),
        )),
        PostgresStoreOperation::GraphMetadataPatchMutation {
            namespace,
            workspace_id,
            graph_space,
            table,
            entity_kind,
            event_id,
            op,
            entity_id,
            document,
            metadata_patch,
            payload,
        } => Ok(optional_graph_mutation_json(
            uow.apply_graph_metadata_patch_mutation(
                kogwistar_store_postgres::GraphMetadataPatchMutation {
                    scope: graph_scope(namespace, workspace_id, graph_space),
                    table,
                    entity_kind,
                    event_id,
                    op,
                    entity_id,
                    document,
                    metadata_patch,
                    payload,
                },
            )
            .await?,
        )),
        PostgresStoreOperation::GraphDeleteMutation {
            namespace,
            workspace_id,
            graph_space,
            table,
            entity_kind,
            event_id,
            entity_id,
            payload,
        } => Ok(optional_graph_mutation_json(
            uow.apply_graph_delete_mutation(kogwistar_store_postgres::GraphDeleteMutation {
                scope: graph_scope(namespace, workspace_id, graph_space),
                table,
                entity_kind,
                event_id,
                entity_id,
                payload,
            })
            .await?,
        )),
        PostgresStoreOperation::UpsertGraphProjection {
            namespace,
            workspace_id,
            graph_space,
            table,
            record,
            embedding_dim,
        } => {
            uow.upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: graph_scope(namespace, workspace_id, graph_space),
                table,
                record: graph_mutation_record(record),
                embedding_dim,
            })
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::PatchGraphProjectionMetadata {
            namespace,
            workspace_id,
            graph_space,
            table,
            entity_id,
            document,
            metadata_patch,
            patch_document_metadata,
        } => Ok(json!(
            uow.patch_graph_projection_metadata(
                kogwistar_store_postgres::GraphProjectionMetadataPatch {
                    scope: graph_scope(namespace, workspace_id, graph_space),
                    table,
                    entity_id,
                    document,
                    metadata_patch,
                    patch_document_metadata,
                },
            )
            .await?
        )),
        PostgresStoreOperation::PruneEntityEventsAfter { namespace, to_seq } => Ok(json!(
            uow.prune_entity_events_after(&namespace, to_seq).await?
        )),
        PostgresStoreOperation::LegacyCursorSet {
            namespace,
            consumer,
            last_seq,
        } => Ok(cursor_json(
            uow.set_replay_cursor_legacy(&namespace, &consumer, last_seq)
                .await?,
        )),
        PostgresStoreOperation::StrictCursorAdvance {
            namespace,
            consumer,
            last_seq,
        } => Ok(cursor_json(
            uow.strict_advance_replay_cursor(&namespace, &consumer, last_seq)
                .await?,
        )),
        PostgresStoreOperation::ReplaceNamedProjection {
            namespace,
            key,
            payload,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => {
            uow.replace_named_projection(
                &namespace,
                &key,
                projection_write(
                    payload,
                    last_authoritative_seq,
                    last_materialized_seq,
                    projection_schema_version,
                    materialization_status,
                ),
            )
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::CompareAndSwapNamedProjection {
            namespace,
            key,
            payload,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => Ok(json!(
            uow.compare_and_swap_named_projection(
                &namespace,
                &key,
                expected_last_authoritative_seq,
                expected_last_materialized_seq,
                projection_write(
                    payload,
                    last_authoritative_seq,
                    last_materialized_seq,
                    projection_schema_version,
                    materialization_status
                ),
            )
            .await?
        )),
        PostgresStoreOperation::ClearNamedProjection { namespace, key } => {
            uow.clear_named_projection(&namespace, &key).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ClearProjectionNamespace { namespace } => {
            uow.clear_projection_namespace(&namespace).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::PutWorkflowDesignSnapshot {
            workflow_id,
            version,
            seq,
            payload_json,
            schema_version,
        } => {
            uow.put_workflow_design_snapshot(
                &workflow_id,
                workflow_design_snapshot_write(version, seq, payload_json, schema_version),
            )
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ClearWorkflowDesignSnapshots { workflow_id } => {
            uow.clear_workflow_design_snapshots(&workflow_id).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::PutWorkflowDesignDelta {
            workflow_id,
            version,
            prev_version,
            target_seq,
            forward_json,
            inverse_json,
            schema_version,
        } => {
            uow.put_workflow_design_delta(
                &workflow_id,
                workflow_design_delta_write(
                    version,
                    prev_version,
                    target_seq,
                    forward_json,
                    inverse_json,
                    schema_version,
                ),
            )
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ClearWorkflowDesignDeltas { workflow_id } => {
            uow.clear_workflow_design_deltas(&workflow_id).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::CreateServerRun {
            run_id,
            conversation_id,
            workflow_id,
            user_id,
            user_turn_node_id,
            status,
        } => {
            uow.create_server_run(ServerRunCreate {
                run_id,
                conversation_id,
                workflow_id,
                user_id,
                user_turn_node_id,
                status,
            })
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::AppendServerRunEvent {
            run_id,
            event_type,
            payload_json,
        } => Ok(server_run_event_json(
            uow.append_server_run_event(&run_id, &event_type, payload_json)
                .await?,
        )?),
        PostgresStoreOperation::UpdateServerRun {
            run_id,
            status,
            assistant_turn_node_id,
            result_json,
            error_json,
            started_at_ms,
            finished_at_ms,
            cancel_requested,
        } => {
            uow.update_server_run(
                &run_id,
                ServerRunUpdate {
                    status,
                    assistant_turn_node_id,
                    result_json,
                    error_json,
                    started_at_ms,
                    finished_at_ms,
                    cancel_requested,
                },
            )
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::RequestServerRunCancel { run_id } => {
            uow.request_server_run_cancel(&run_id).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ApplyRecordedRuntimeTransition {
            transition,
            abort_after_writes,
        } => serde_json::to_value(
            uow.apply_recorded_runtime_transition(*transition, abort_after_writes)
                .await?,
        )
        .map_err(|error| PostgresStoreError::TransactionAborted(error.to_string())),
        PostgresStoreOperation::ApplyClaimedRecordedRuntimeTransition {
            handoff,
            transition,
            abort_after_writes,
        } => serde_json::to_value(
            uow.apply_claimed_recorded_runtime_transition(handoff, *transition, abort_after_writes)
                .await?,
        )
        .map_err(|error| PostgresStoreError::TransactionAborted(error.to_string())),
        PostgresStoreOperation::EnqueueIndexJob {
            job_id,
            namespace,
            entity_kind,
            entity_id,
            index_kind,
            op,
            payload_json,
            max_retries,
        } => Ok(json!(
            uow.enqueue_index_job(new_index_job(
                job_id,
                namespace,
                entity_kind,
                entity_id,
                index_kind,
                op,
                payload_json,
                max_retries
            ))
            .await?
        )),
        PostgresStoreOperation::ClaimIndexJobs {
            limit,
            lease_seconds,
            namespace,
        } => Ok(Value::Array(
            uow.claim_index_jobs(limit, lease_seconds, namespace.as_deref())
                .await?
                .into_iter()
                .map(index_job_json)
                .collect(),
        )),
        PostgresStoreOperation::MarkIndexJobDone {
            job_id,
            claim_token,
        } => Ok(json!(
            uow.mark_index_job_done(&job_id, claim_token.as_deref())
                .await?
        )),
        PostgresStoreOperation::MarkIndexJobFailed {
            job_id,
            error,
            final_,
            claim_token,
        } => {
            uow.mark_index_job_failed(&job_id, &error, final_, claim_token.as_deref())
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::BumpRetryAndRequeue {
            job_id,
            error,
            next_run_at_seconds,
            claim_token,
        } => {
            uow.bump_retry_and_requeue(
                &job_id,
                &error,
                next_run_at_seconds,
                claim_token.as_deref(),
            )
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::RenewIndexJobLease {
            job_id,
            claim_token,
            lease_seconds,
        } => Ok(json!(
            uow.renew_index_job_lease(&job_id, &claim_token, lease_seconds)
                .await?
        )),
        PostgresStoreOperation::RequeueIndexJobAtTail {
            job_id,
            payload_json,
            delay_seconds,
            claim_token,
        } => {
            uow.requeue_index_job_at_tail(
                &job_id,
                payload_json,
                delay_seconds,
                claim_token.as_deref(),
            )
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ProjectLaneMessage {
            message_id,
            namespace,
            purpose,
            inbox_id,
            conversation_id,
            recipient_id,
            sender_id,
            msg_type,
            status,
            created_at,
            available_at,
            run_id,
            step_id,
            correlation_id,
            payload_json,
            error_json,
        } => {
            uow.project_lane_message(new_lane_message(
                message_id,
                namespace,
                purpose,
                inbox_id,
                conversation_id,
                recipient_id,
                sender_id,
                msg_type,
                status,
                created_at,
                available_at,
                run_id,
                step_id,
                correlation_id,
                payload_json,
                error_json,
            ))
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::UpdateProjectedLaneMessageStatus {
            message_id,
            status,
            error_json,
        } => {
            uow.update_projected_lane_message_status(&message_id, &status, error_json)
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::UpdateProjectedLaneMessageLinks {
            message_id,
            prev_message_id,
            next_message_id,
            inbox_tail_message_id,
            conversation_tail_message_id,
        } => {
            uow.update_projected_lane_message_links(
                &message_id,
                prev_message_id,
                next_message_id,
                inbox_tail_message_id,
                conversation_tail_message_id,
            )
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ClearProjectedLaneMessages { namespace } => {
            Ok(json!(uow.clear_projected_lane_messages(&namespace).await?))
        }
        PostgresStoreOperation::ClaimProjectedLaneMessages {
            namespace,
            inbox_id,
            claimed_by,
            limit,
            lease_seconds,
        } => Ok(Value::Array(
            uow.claim_projected_lane_messages(
                &namespace,
                &inbox_id,
                &claimed_by,
                limit,
                lease_seconds,
            )
            .await?
            .into_iter()
            .map(lane_message_json)
            .collect(),
        )),
        PostgresStoreOperation::AckProjectedLaneMessage {
            message_id,
            claimed_by,
        } => {
            uow.ack_projected_lane_message(&message_id, &claimed_by)
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::RequeueProjectedLaneMessage {
            message_id,
            claimed_by,
            error_json,
            delay_seconds,
        } => {
            uow.requeue_projected_lane_message(&message_id, &claimed_by, error_json, delay_seconds)
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::DeadLetterProjectedLaneMessage {
            message_id,
            claimed_by,
            error_json,
        } => {
            uow.dead_letter_projected_lane_message(&message_id, &claimed_by, error_json)
                .await?;
            Ok(Value::Null)
        }
        _ => Err(PostgresStoreError::TransactionAborted(
            "unsupported PostgreSQL batch operation".to_owned(),
        )),
    }
}

fn postgres_store_error_code(error: &PostgresStoreError) -> &'static str {
    match error {
        PostgresStoreError::EventIdNamespaceCollision { .. } => STORE_EVENT_ID_NAMESPACE_COLLISION,
        PostgresStoreError::Store(kogwistar_store::StoreError::CursorOutOfRange { .. }) => {
            STORE_CURSOR_OUT_OF_RANGE
        }
        PostgresStoreError::Store(kogwistar_store::StoreError::CursorRegresses { .. }) => {
            STORE_CURSOR_REGRESSES
        }
        PostgresStoreError::Store(
            kogwistar_store::StoreError::InvalidIdentifier { .. }
            | kogwistar_store::StoreError::InvalidRecoveryBatchLimit
            | kogwistar_store::StoreError::InvalidEntityEventPayload { .. }
            | kogwistar_store::StoreError::UnsupportedEntityEventOperation { .. },
        ) => STORE_INVALID_ENTITY_EVENT,
        PostgresStoreError::TransactionAborted(message)
            if message == "unsupported PostgreSQL batch operation" =>
        {
            STORE_OPERATION_INVALID
        }
        PostgresStoreError::TransactionAborted(_) => STORE_TRANSACTION_ABORTED,
        // Backend failures deliberately remain persistence failures: they are
        // never idempotency/collision outcomes.
        PostgresStoreError::InvalidSchema { .. }
        | PostgresStoreError::EmptyNamespace
        | PostgresStoreError::Backend(_)
        | PostgresStoreError::InvalidPayload(_)
        | PostgresStoreError::RecordedRuntime(_)
        | PostgresStoreError::RecordedRuntimeConflict(_)
        | PostgresStoreError::Store(_) => STORE_PERSISTENCE_FAILED,
    }
}

fn postgres_runtime() -> Result<TokioRuntime, (&'static str, String)> {
    TokioRuntimeBuilder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| {
            (
                STORE_PERSISTENCE_FAILED,
                format!("cannot start Tokio runtime: {error}"),
            )
        })
}

const POSTGRES_EXTERNAL_ROLLBACK: &str = "external PostgreSQL transaction rollback requested";

enum PostgresSessionCommand {
    Operation {
        operation: Box<PostgresStoreOperation>,
        response: mpsc::Sender<Result<Value, PostgresStoreError>>,
    },
    Finish {
        commit: bool,
    },
}

struct PostgresSession {
    dsn: String,
    schema: String,
    sender: mpsc::Sender<PostgresSessionCommand>,
    worker: Option<JoinHandle<Result<(), PostgresStoreError>>>,
    finishing: bool,
}

fn postgres_sessions() -> &'static Mutex<BTreeMap<String, PostgresSession>> {
    static SESSIONS: OnceLock<Mutex<BTreeMap<String, PostgresSession>>> = OnceLock::new();
    SESSIONS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn postgres_session_error(message: impl Into<String>) -> PostgresStoreError {
    PostgresStoreError::TransactionAborted(message.into())
}

fn require_postgres_transaction_id(
    transaction_id: Option<String>,
) -> Result<String, PostgresStoreError> {
    transaction_id
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            postgres_session_error("PostgreSQL transaction control requires transaction_id")
        })
}

fn begin_postgres_session(
    dsn: String,
    schema: String,
    transaction_id: String,
) -> Result<(), PostgresStoreError> {
    let mut sessions = postgres_sessions()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if sessions.contains_key(&transaction_id) {
        return Err(postgres_session_error(
            "PostgreSQL transaction_id is already active",
        ));
    }
    if sessions
        .values()
        .any(|session| session.dsn == dsn && session.schema == schema)
    {
        return Err(postgres_session_error(
            "PostgreSQL external transaction is already active for this store",
        ));
    }

    let (sender, receiver) = mpsc::channel();
    let (started_sender, started_receiver) =
        mpsc::sync_channel::<Result<(), PostgresStoreError>>(1);
    let worker_dsn = dsn.clone();
    let worker_schema = schema.clone();
    let worker = std::thread::spawn(move || {
        let store = PostgresStore::from_dsn(&worker_dsn, &worker_schema)?;
        let runtime = postgres_runtime().map_err(|(_, message)| postgres_session_error(message))?;
        let result = runtime.block_on(store.transaction(move |uow| {
            Box::pin(async move {
                let _ = started_sender.send(Ok(()));
                loop {
                    let command = receiver.recv().map_err(|_| {
                        postgres_session_error(
                            "PostgreSQL transaction owner disconnected before commit or rollback",
                        )
                    })?;
                    match command {
                        PostgresSessionCommand::Operation {
                            operation,
                            response,
                        } => {
                            let result = postgres_uow_operation_json(uow, *operation).await;
                            let _ = response.send(result);
                        }
                        PostgresSessionCommand::Finish { commit: true } => return Ok(()),
                        PostgresSessionCommand::Finish { commit: false } => {
                            return Err(postgres_session_error(POSTGRES_EXTERNAL_ROLLBACK));
                        }
                    }
                }
            })
        }));
        match result {
            Err(PostgresStoreError::TransactionAborted(message))
                if message == POSTGRES_EXTERNAL_ROLLBACK =>
            {
                Ok(())
            }
            other => other,
        }
    });
    match started_receiver.recv() {
        Ok(Ok(())) => {
            sessions.insert(
                transaction_id,
                PostgresSession {
                    dsn,
                    schema,
                    sender,
                    worker: Some(worker),
                    finishing: false,
                },
            );
            Ok(())
        }
        Ok(Err(error)) => Err(error),
        Err(_) => match worker.join() {
            Ok(Err(error)) => Err(error),
            Ok(Ok(())) => Err(postgres_session_error(
                "PostgreSQL transaction worker exited before startup",
            )),
            Err(_) => Err(postgres_session_error(
                "PostgreSQL transaction worker panicked during startup",
            )),
        },
    }
}

fn postgres_session_operation(
    transaction_id: &str,
    operation: PostgresStoreOperation,
) -> Result<Value, PostgresStoreError> {
    let sender = {
        let sessions = postgres_sessions()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        sessions
            .get(transaction_id)
            .map(|session| session.sender.clone())
            .ok_or_else(|| postgres_session_error("PostgreSQL transaction_id is stale"))?
    };
    let (response_sender, response_receiver) = mpsc::channel();
    sender
        .send(PostgresSessionCommand::Operation {
            operation: Box::new(operation),
            response: response_sender,
        })
        .map_err(|_| postgres_session_error("PostgreSQL transaction worker is unavailable"))?;
    response_receiver
        .recv()
        .map_err(|_| postgres_session_error("PostgreSQL transaction worker dropped its response"))?
}

fn finish_postgres_session(transaction_id: &str, commit: bool) -> Result<(), PostgresStoreError> {
    let (sender, worker) = {
        let mut sessions = postgres_sessions()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let session = sessions
            .get_mut(transaction_id)
            .ok_or_else(|| postgres_session_error("PostgreSQL transaction_id is stale"))?;
        if session.finishing {
            return Err(postgres_session_error(
                "PostgreSQL transaction is already finishing",
            ));
        }
        session.finishing = true;
        let worker = session.worker.take().ok_or_else(|| {
            postgres_session_error("PostgreSQL transaction worker is unavailable")
        })?;
        (session.sender.clone(), worker)
    };
    sender
        .send(PostgresSessionCommand::Finish { commit })
        .map_err(|_| postgres_session_error("PostgreSQL transaction worker is unavailable"))?;
    let result = match worker.join() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(error),
        Err(_) => Err(postgres_session_error(
            "PostgreSQL transaction worker panicked",
        )),
    };
    postgres_sessions()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(transaction_id);
    result
}

fn require_no_active_postgres_session(dsn: &str, schema: &str) -> Result<(), PostgresStoreError> {
    let sessions = postgres_sessions()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if sessions
        .values()
        .any(|session| session.dsn == dsn && session.schema == schema)
    {
        return Err(postgres_session_error(
            "PostgreSQL operation does not own the active transaction",
        ));
    }
    Ok(())
}

fn require_postgres_fields(value: &Value, allowed: &[&str]) -> Result<(), (&'static str, String)> {
    let object = value.as_object().ok_or_else(|| {
        (
            STORE_INVALID_PAYLOAD,
            "invalid PostgreSQL store payload: operation must be an object".to_owned(),
        )
    })?;
    if let Some(field) = object
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err((
            STORE_INVALID_PAYLOAD,
            format!("invalid PostgreSQL store payload: unknown field {field:?}"),
        ));
    }
    Ok(())
}

fn validate_postgres_operation(value: &Value) -> Result<(), (&'static str, String)> {
    let operation = value.as_object().ok_or_else(|| {
        (
            STORE_INVALID_PAYLOAD,
            "invalid PostgreSQL store payload: operation must be an object".to_owned(),
        )
    })?;
    let kind = operation
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            (
                STORE_INVALID_PAYLOAD,
                "invalid PostgreSQL store payload: operation kind must be a string".to_owned(),
            )
        })?;
    let allowed = match kind {
        "begin_transaction"
        | "commit_transaction"
        | "rollback_transaction"
        | "ensure_schema"
        | "next_global_seq"
        | "current_global_seq" => &["kind"][..],
        "next_user_seq" | "current_user_seq" => &["kind", "user_id"][..],
        "set_user_seq" => &["kind", "user_id", "value"][..],
        "get_index_applied_fingerprint" => &["kind", "namespace", "coalesce_key"][..],
        "set_index_applied_fingerprint" => &[
            "kind",
            "namespace",
            "coalesce_key",
            "applied_fingerprint",
            "last_job_id",
        ][..],
        "create_graph_schema" => &[
            "kind",
            "embedding_dim",
            "nodes_table",
            "edges_table",
            "documents_table",
            "domains_table",
        ][..],
        "alloc_event_seq" => &["kind", "namespace"][..],
        "raw_append" => &[
            "kind",
            "namespace",
            "event_id",
            "entity_kind",
            "entity_id",
            "op",
            "payload_json",
        ][..],
        "graph_mutation" => &[
            "kind",
            "namespace",
            "workspace_id",
            "graph_space",
            "table",
            "entity_kind",
            "event_id",
            "op",
            "record",
            "payload",
            "embedding_dim",
        ][..],
        "graph_metadata_patch_mutation" => &[
            "kind",
            "namespace",
            "workspace_id",
            "graph_space",
            "table",
            "entity_kind",
            "event_id",
            "op",
            "entity_id",
            "document",
            "metadata_patch",
            "payload",
        ][..],
        "graph_delete_mutation" => &[
            "kind",
            "namespace",
            "workspace_id",
            "graph_space",
            "table",
            "entity_kind",
            "event_id",
            "entity_id",
            "payload",
        ][..],
        "upsert_graph_projection" => &[
            "kind",
            "namespace",
            "workspace_id",
            "graph_space",
            "table",
            "record",
            "embedding_dim",
        ][..],
        "patch_graph_projection_metadata" => &[
            "kind",
            "namespace",
            "workspace_id",
            "graph_space",
            "table",
            "entity_id",
            "document",
            "metadata_patch",
            "patch_document_metadata",
        ][..],
        "graph_projection_records" => &[
            "kind",
            "namespace",
            "workspace_id",
            "graph_space",
            "table",
            "ids",
            "metadata",
            "limit",
        ][..],
        "graph_projection_vector_query" => &[
            "kind",
            "namespace",
            "workspace_id",
            "graph_space",
            "table",
            "embedding",
            "limit",
            "metadata",
            "metric",
            "embedding_dim",
        ][..],
        "exclusive_raw_replay" => &["kind", "namespace", "after_seq", "limit"][..],
        "latest_retained_event_seq" => &["kind", "namespace"][..],
        "prune_entity_events_after" => &["kind", "namespace", "to_seq"][..],
        "cursor_get" => &["kind", "namespace", "consumer"][..],
        "legacy_cursor_set" | "strict_cursor_advance" => {
            &["kind", "namespace", "consumer", "last_seq"][..]
        }
        "recover_entity_projection" => &[
            "kind",
            "namespace",
            "consumer",
            "projection_namespace",
            "projection_key",
            "batch_limit",
            "abort_after_projection",
        ][..],
        "rebuild_entity_projection" => &[
            "kind",
            "namespace",
            "consumer",
            "projection_namespace",
            "projection_key",
            "abort_after_projection",
        ][..],
        "get_named_projection"
        | "clear_named_projection"
        | "get_stage1_node_projection"
        | "clear_stage1_node_projection" => &["kind", "namespace", "key"][..],
        "list_named_projections"
        | "clear_projection_namespace"
        | "list_stage1_node_projections" => &["kind", "namespace"][..],
        "replace_stage1_node_projection" => &[
            "kind",
            "namespace",
            "key",
            "payload",
            "last_authoritative_seq",
            "last_materialized_seq",
            "projection_schema_version",
            "materialization_status",
        ][..],
        "put_workflow_design_snapshot" => &[
            "kind",
            "workflow_id",
            "version",
            "seq",
            "payload_json",
            "schema_version",
        ][..],
        "get_workflow_design_snapshot" => {
            &["kind", "workflow_id", "max_version", "schema_version"][..]
        }
        "clear_workflow_design_snapshots" => &["kind", "workflow_id"][..],
        "put_workflow_design_delta" => &[
            "kind",
            "workflow_id",
            "version",
            "prev_version",
            "target_seq",
            "forward_json",
            "inverse_json",
            "schema_version",
        ][..],
        "get_workflow_design_delta" => &["kind", "workflow_id", "version", "schema_version"][..],
        "clear_workflow_design_deltas" => &["kind", "workflow_id"][..],
        "create_server_run" => &[
            "kind",
            "run_id",
            "conversation_id",
            "workflow_id",
            "user_id",
            "user_turn_node_id",
            "status",
        ][..],
        "get_server_run" | "request_server_run_cancel" => &["kind", "run_id"][..],
        "list_server_runs" => &["kind", "status", "workflow_id", "conversation_id", "limit"][..],
        "append_server_run_event" => &["kind", "run_id", "event_type", "payload_json"][..],
        "list_server_run_events" => &["kind", "run_id", "after_seq", "limit"][..],
        "update_server_run" => &[
            "kind",
            "run_id",
            "status",
            "assistant_turn_node_id",
            "result_json",
            "error_json",
            "started_at_ms",
            "finished_at_ms",
            "cancel_requested",
        ][..],
        "apply_recorded_runtime_transition" => &["kind", "transition", "abort_after_writes"][..],
        "apply_claimed_recorded_runtime_transition" => {
            &["kind", "handoff", "transition", "abort_after_writes"][..]
        }
        "read_recorded_runtime_state" => &["kind", "run_id", "workflow_id", "conversation_id"][..],
        "enqueue_index_job" => &[
            "kind",
            "job_id",
            "namespace",
            "entity_kind",
            "entity_id",
            "index_kind",
            "op",
            "payload_json",
            "max_retries",
        ][..],
        "claim_index_jobs" => &["kind", "limit", "lease_seconds", "namespace"][..],
        "mark_index_job_done" => &["kind", "job_id", "claim_token"][..],
        "mark_index_job_failed" => &["kind", "job_id", "error", "final", "claim_token"][..],
        "bump_retry_and_requeue" => &[
            "kind",
            "job_id",
            "error",
            "next_run_at_seconds",
            "claim_token",
        ][..],
        "renew_index_job_lease" => &["kind", "job_id", "claim_token", "lease_seconds"][..],
        "requeue_index_job_at_tail" => &[
            "kind",
            "job_id",
            "payload_json",
            "delay_seconds",
            "claim_token",
        ][..],
        "list_index_jobs" => &[
            "kind",
            "namespace",
            "status",
            "entity_kind",
            "entity_id",
            "index_kind",
            "limit",
        ][..],
        "project_lane_message" => &[
            "kind",
            "message_id",
            "namespace",
            "purpose",
            "inbox_id",
            "conversation_id",
            "recipient_id",
            "sender_id",
            "msg_type",
            "status",
            "created_at",
            "available_at",
            "run_id",
            "step_id",
            "correlation_id",
            "payload_json",
            "error_json",
        ][..],
        "get_projected_lane_message" => &["kind", "message_id"][..],
        "update_projected_lane_message_status" => {
            &["kind", "message_id", "status", "error_json"][..]
        }
        "update_projected_lane_message_links" => &[
            "kind",
            "message_id",
            "prev_message_id",
            "next_message_id",
            "inbox_tail_message_id",
            "conversation_tail_message_id",
        ][..],
        "list_projected_lane_messages" => &[
            "kind",
            "namespace",
            "purpose",
            "inbox_id",
            "conversation_id",
            "status",
            "msg_type",
            "sender_id",
            "recipient_id",
            "correlation_id",
            "reply_to_message_id",
            "created_at_gte",
            "created_at_lte",
            "available_at_gte",
            "available_at_lte",
            "limit",
            "newest_first",
        ][..],
        "clear_projected_lane_messages" => &["kind", "namespace"][..],
        "claim_projected_lane_messages" => &[
            "kind",
            "namespace",
            "inbox_id",
            "claimed_by",
            "limit",
            "lease_seconds",
        ][..],
        "ack_projected_lane_message" => &["kind", "message_id", "claimed_by"][..],
        "requeue_projected_lane_message" => &[
            "kind",
            "message_id",
            "claimed_by",
            "error_json",
            "delay_seconds",
        ][..],
        "dead_letter_projected_lane_message" => {
            &["kind", "message_id", "claimed_by", "error_json"][..]
        }
        "replace_named_projection" => &[
            "kind",
            "namespace",
            "key",
            "payload",
            "last_authoritative_seq",
            "last_materialized_seq",
            "projection_schema_version",
            "materialization_status",
        ][..],
        "compare_and_swap_named_projection" => &[
            "kind",
            "namespace",
            "key",
            "payload",
            "expected_last_authoritative_seq",
            "expected_last_materialized_seq",
            "last_authoritative_seq",
            "last_materialized_seq",
            "projection_schema_version",
            "materialization_status",
        ][..],
        "batch" => {
            require_postgres_fields(value, &["kind", "operations", "abort"])?;
            let operations = operation
                .get("operations")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    (
                        STORE_INVALID_PAYLOAD,
                        "invalid PostgreSQL store payload: batch operations must be an array"
                            .to_owned(),
                    )
                })?;
            for nested in operations {
                validate_postgres_operation(nested)?;
            }
            return Ok(());
        }
        _ => return Ok(()), // serde supplies stable invalid-operation diagnostics below.
    };
    require_postgres_fields(value, allowed)
}

fn postgres_store_json_impl(payload_json: &str) -> Result<String, (&'static str, String)> {
    let value: Value = serde_json::from_str(payload_json)
        .map_err(|error| (STORE_INVALID_JSON, format!("invalid JSON: {error}")))?;
    require_postgres_fields(&value, &["dsn", "schema", "transaction_id", "operation"])?;
    let operation = value.get("operation").ok_or_else(|| {
        (
            STORE_INVALID_PAYLOAD,
            "invalid PostgreSQL store payload: missing operation".to_owned(),
        )
    })?;
    validate_postgres_operation(operation)?;
    let request: PostgresStoreRequest = serde_json::from_value(value).map_err(|error| {
        (
            STORE_INVALID_PAYLOAD,
            format!("invalid PostgreSQL store payload: {error}"),
        )
    })?;
    let result: Result<Value, PostgresStoreError> = (|| match request.operation {
        PostgresStoreOperation::BeginTransaction => {
            let transaction_id = require_postgres_transaction_id(request.transaction_id)?;
            begin_postgres_session(request.dsn, request.schema, transaction_id)?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::CommitTransaction => {
            let transaction_id = require_postgres_transaction_id(request.transaction_id)?;
            finish_postgres_session(&transaction_id, true)?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::RollbackTransaction => {
            let transaction_id = require_postgres_transaction_id(request.transaction_id)?;
            finish_postgres_session(&transaction_id, false)?;
            Ok(Value::Null)
        }
        operation => {
            if let Some(transaction_id) = request.transaction_id {
                postgres_session_operation(&transaction_id, operation)
            } else {
                require_no_active_postgres_session(&request.dsn, &request.schema)?;
                let store = PostgresStore::from_dsn(&request.dsn, &request.schema)?;
                let runtime =
                    postgres_runtime().map_err(|(_, message)| postgres_session_error(message))?;
                runtime.block_on(postgres_store_operation_json(&store, operation))
            }
        }
    })();
    result
        .and_then(|value| {
            serde_json::to_string(&value).map_err(|error| {
                PostgresStoreError::TransactionAborted(format!("cannot encode result: {error}"))
            })
        })
        .map_err(|error| (postgres_store_error_code(&error), error.to_string()))
}

#[pyfunction]
fn stable_id(kind: &str, parts: Vec<String>) -> String {
    contracts::stable_id(kind, &parts).to_string()
}

#[pyfunction]
fn stable_id_json(payload_json: &str) -> PyResult<String> {
    contracts::stable_id_from_json(payload_json)
        .map(|value| value.to_string())
        .map_err(contract_error)
}

#[pyfunction]
fn canonical_json(payload_json: &str) -> PyResult<String> {
    contracts::canonical_json_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn evidence_pack_digest_hash(payload_json: &str) -> PyResult<String> {
    contracts::evidence_pack_digest_hash_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn metadata_filter_matches(payload_json: &str) -> PyResult<bool> {
    contracts::metadata_filter_matches_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn normalize_metadata_filter(payload_json: &str) -> PyResult<String> {
    contracts::normalize_metadata_filter_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn short_id_transform(payload_json: &str) -> PyResult<String> {
    contracts::short_id_transform_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn apply_state_update(payload_json: &str) -> PyResult<String> {
    contracts::apply_state_update_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn canonical_entity_event(payload_json: &str) -> PyResult<String> {
    contracts::canonical_entity_event_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn replay_entity_events(payload_json: &str) -> PyResult<String> {
    contracts::replay_entity_events_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn workflow_may_reach_join(payload_json: &str) -> PyResult<String> {
    contracts::workflow_may_reach_join_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn workflow_terminal_reachable(payload_json: &str) -> PyResult<bool> {
    contracts::workflow_terminal_reachable_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn runtime_select_route(payload_json: &str) -> PyResult<String> {
    kogwistar_runtime::select_runtime_route_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn runtime_plan_successors(payload_json: &str) -> PyResult<String> {
    kogwistar_runtime::plan_runtime_successors_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn runtime_apply_join_arrival(payload_json: &str) -> PyResult<String> {
    kogwistar_runtime::apply_runtime_join_arrival_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn runtime_decide_retry(payload_json: &str) -> PyResult<String> {
    kogwistar_runtime::decide_runtime_retry_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn runtime_plan_nested_invocation(payload_json: &str) -> PyResult<String> {
    kogwistar_runtime::plan_runtime_nested_invocation_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn runtime_decide_dispatch(payload_json: &str) -> PyResult<String> {
    kogwistar_runtime::decide_runtime_dispatch_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn runtime_decide_budget_suspend(payload_json: &str) -> PyResult<String> {
    kogwistar_runtime::decide_runtime_budget_suspend_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn runtime_scheduler_tick(payload_json: &str) -> PyResult<String> {
    kogwistar_runtime::tick_runtime_scheduler_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn api_health(payload_json: &str) -> PyResult<String> {
    kogwistar_api::health_response_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn api_authorize(payload_json: &str) -> PyResult<String> {
    kogwistar_api::authorize_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn api_sse_frame(payload_json: &str) -> PyResult<String> {
    kogwistar_api::sse_frame_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn api_mcp_result(payload_json: &str) -> PyResult<String> {
    kogwistar_api::mcp_result_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn api_cli_health(payload_json: &str) -> PyResult<String> {
    kogwistar_api::cli_health_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn api_run_server(py: Python<'_>) -> PyResult<()> {
    py.detach(|| {
        TokioRuntimeBuilder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .block_on(kogwistar_api::run_server_from_environment())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    })
}

/// Build an isolated Rust in-memory store from a JSON snapshot, then inspect it.
/// This boundary has no handle to Python-owned backend or meta-store state.
#[pyfunction]
fn store_memory_read_json(py: Python<'_>, payload_json: &str) -> PyResult<String> {
    store_memory_read_json_impl(payload_json)
        .map_err(|(code, message)| store_error(py, code, message))
}

#[pyfunction]
fn store_sqlite_json(py: Python<'_>, payload_json: &str) -> PyResult<String> {
    sqlite_store_json_impl(payload_json).map_err(|(code, message)| store_error(py, code, message))
}

#[pyfunction]
fn store_postgres_json(py: Python<'_>, payload_json: &str) -> PyResult<String> {
    let payload_json = payload_json.to_owned();
    py.detach(move || postgres_store_json_impl(&payload_json))
        .map_err(|(code, message)| store_error(py, code, message))
}

#[pymodule]
fn _rust(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("CONTRACT_VERSION", contracts::CONTRACT_VERSION)?;
    module.add(
        "RustContractTypeError",
        py.get_type::<RustContractTypeError>(),
    )?;
    module.add(
        "RustContractValueError",
        py.get_type::<RustContractValueError>(),
    )?;
    module.add("RustStoreValueError", py.get_type::<RustStoreValueError>())?;
    module.add_function(wrap_pyfunction!(stable_id, module)?)?;
    module.add_function(wrap_pyfunction!(stable_id_json, module)?)?;
    module.add_function(wrap_pyfunction!(canonical_json, module)?)?;
    module.add_function(wrap_pyfunction!(evidence_pack_digest_hash, module)?)?;
    module.add_function(wrap_pyfunction!(metadata_filter_matches, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_metadata_filter, module)?)?;
    module.add_function(wrap_pyfunction!(short_id_transform, module)?)?;
    module.add_function(wrap_pyfunction!(apply_state_update, module)?)?;
    module.add_function(wrap_pyfunction!(canonical_entity_event, module)?)?;
    module.add_function(wrap_pyfunction!(replay_entity_events, module)?)?;
    module.add_function(wrap_pyfunction!(workflow_may_reach_join, module)?)?;
    module.add_function(wrap_pyfunction!(workflow_terminal_reachable, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_select_route, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_plan_successors, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_apply_join_arrival, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_decide_retry, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_plan_nested_invocation, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_decide_dispatch, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_decide_budget_suspend, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_scheduler_tick, module)?)?;
    module.add_function(wrap_pyfunction!(api_health, module)?)?;
    module.add_function(wrap_pyfunction!(api_authorize, module)?)?;
    module.add_function(wrap_pyfunction!(api_sse_frame, module)?)?;
    module.add_function(wrap_pyfunction!(api_mcp_result, module)?)?;
    module.add_function(wrap_pyfunction!(api_cli_health, module)?)?;
    module.add_function(wrap_pyfunction!(api_run_server, module)?)?;
    module.add_function(wrap_pyfunction!(store_memory_read_json, module)?)?;
    module.add_function(wrap_pyfunction!(store_sqlite_json, module)?)?;
    module.add_function(wrap_pyfunction!(store_postgres_json, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn isolated_store_snapshot_reads_are_ordered_and_read_only() {
        let request = json!({
            "snapshot": {
                "records": [
                    {"namespace": "ns", "id": "z", "metadata": {"team": "red"}, "embedding": [1.0, 0.0]},
                    {"namespace": "ns", "id": "a", "metadata": {"team": "red"}, "embedding": [0.0, 1.0]}
                ],
                "events": [
                    {"namespace": "ns", "seq": 1, "event_id": "evt-1", "entity_kind": "node", "entity_id": "a", "op": "UPSERT", "payload": {"replacement": {"id": "a"}}},
                    {"namespace": "ns", "seq": 2, "event_id": "evt-2", "entity_kind": "node", "entity_id": "z", "op": "TOMBSTONE", "payload": {"tombstone": true}}
                ],
                "cursors": [{"namespace": "ns", "consumer": "replay", "last_seq": 1}]
            },
            "operation": {"kind": "graph_records", "namespace": "ns", "metadata": {"team": "red"}}
        });
        let result: Value =
            serde_json::from_str(&store_memory_read_json_impl(&request.to_string()).unwrap())
                .unwrap();
        assert_eq!(result[0]["id"], "z");
        assert_eq!(result[1]["id"], "a");

        let replay = json!({
            "snapshot": request["snapshot"].clone(),
            "operation": {"kind": "replay_events", "namespace": "ns", "after_seq": 0, "limit": 10}
        });
        let events: Value =
            serde_json::from_str(&store_memory_read_json_impl(&replay.to_string()).unwrap())
                .unwrap();
        assert_eq!(events[1]["payload"], json!({"tombstone": true}));
        assert_eq!(events[1]["op"], "TOMBSTONE");
    }

    #[test]
    fn isolated_store_reports_stable_input_codes() {
        assert_eq!(
            parse_store_request("not-json").err().unwrap().0,
            STORE_INVALID_JSON
        );
        assert_eq!(
            parse_store_request(r#"{"snapshot":{},"operation":{"kind":"nope"}}"#)
                .err()
                .unwrap()
                .0,
            STORE_INVALID_PAYLOAD
        );
    }

    #[test]
    fn sqlite_session_close_releases_cached_handle() {
        let path =
            std::env::temp_dir().join(format!("kogwistar-python-close-{}.db", std::process::id()));
        let path_text = path.to_string_lossy().into_owned();
        let init = json!({
            "path": path_text.clone(),
            "operation": {"kind": "open_init"}
        });
        sqlite_store_json_impl(&init.to_string()).unwrap();
        let close = json!({
            "path": path_text,
            "operation": {"kind": "close"}
        });
        sqlite_store_json_impl(&close.to_string()).unwrap();
        fs::remove_file(path).unwrap();
    }
}
