use kogwistar_contracts as contracts;
use kogwistar_runtime::{RecordedRuntimeTransition, RecordedWorkerHandoff};
use kogwistar_store::{
    DistanceMetric, EntityRebuildRequest, EntityRecoveryReport, EntityRecoveryRequest,
    EventReadStore, GraphMutationStore, GraphProjectionPageRead, GraphProjectionRead,
    GraphReadStore, GraphRecord, GraphWriteStore, LaneMessageClaimFilter, LaneMessageFilter,
    MetadataFilter, NamedProjection, NamedProjectionWrite, NewProjectedLaneMessage,
    ProjectedLaneMessage, ProjectionReadStore, ReplayCursor, ServerRun, ServerRunCreate,
    ServerRunEvent, ServerRunUpdate, VectorQuery, WorkflowDesignDelta, WorkflowDesignDeltaWrite,
    WorkflowDesignSnapshot, WorkflowDesignSnapshotWrite,
};
use kogwistar_store_memory::InMemoryStore;
use kogwistar_store_postgres::{
    GraphIndexTableNames, NamedProjectionCas as PostgresNamedProjectionCas,
    NewRawEntityEvent as PostgresNewRawEntityEvent, PostgresStore, PostgresStoreError,
    PostgresUnitOfWork, RawEntityEvent as PostgresRawEntityEvent,
};
use kogwistar_store_sqlite::{
    NamedProjectionCas, NewRawEntityEvent, RawEntityEvent, RestoredRawEntityEvent, SqliteStore,
    SqliteStoreError, SqliteUnitOfWork,
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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::task::{Context, Poll, Waker};
use std::thread::JoinHandle;
use tokio::runtime::{Builder as TokioRuntimeBuilder, Runtime as TokioRuntime};

create_exception!(kogwistar._rust, RustContractTypeError, PyTypeError);
create_exception!(kogwistar._rust, RustContractValueError, PyValueError);
create_exception!(kogwistar._rust, RustStoreValueError, PyValueError);

const STORE_INVALID_JSON: &str = "KOGWISTAR_STORE_INVALID_JSON";
const STORE_INVALID_PAYLOAD: &str = "KOGWISTAR_STORE_INVALID_PAYLOAD";
const STORE_SQLITE_ENGINE_MISMATCH: &str = "KOGWISTAR_SQLITE_ENGINE_MISMATCH";
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
fn default_node_index_table() -> String {
    "gke_node_index".to_owned()
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
    json!({"job_id":job.job_id,"namespace":job.namespace,"entity_kind":job.entity_kind,"entity_id":job.entity_id,"index_kind":job.index_kind,"coalesce_key":job.coalesce_key,"op":job.op,"status":job.status,"lease_until":job.lease_until,"next_run_at":job.next_run_at,"max_retries":job.max_retries,"retry_count":job.retry_count,"last_error":job.last_error,"payload_json":job.payload_json,"created_at":job.created_at,"updated_at":job.updated_at,"claim_token":job.claim_token,"claim_attempts":job.claim_attempts,"accepted_result_json":job.accepted_result_json,"accepted_result_sha256":job.accepted_result_sha256,"accepted_at":job.accepted_at})
}
fn accepted_index_job_result_json(result: kogwistar_store::AcceptedIndexJobResult) -> Value {
    json!({"status": result.status, "result_json": result.result_json, "result_sha256": result.result_sha256, "accepted_at": result.accepted_at})
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
fn lane_claim_filter(
    namespace: String,
    inbox_id: String,
    claimed_by: String,
    message_ids: Option<Vec<String>>,
    run_id: Option<String>,
    msg_type: Option<String>,
    recipient_id: Option<String>,
    limit: usize,
    lease_seconds: i64,
) -> LaneMessageClaimFilter {
    LaneMessageClaimFilter {
        namespace,
        inbox_id,
        claimed_by,
        message_ids,
        run_id,
        msg_type,
        recipient_id,
        limit,
        lease_seconds,
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

include!("sections/store_memory_abi.rs");
include!("sections/sqlite_store_contract.rs");
include!("sections/sqlite_store_operations.rs");
include!("sections/postgres_store_contract.rs");
include!("sections/postgres_store_operations.rs");
include!("sections/postgres_store_sessions.rs");
include!("sections/python_exports.rs");

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn cached_postgres_receiver_releases_lock_after_each_command() {
        let (sender, receiver) = mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        let (first_response, _) = mpsc::channel();
        let (second_response, _) = mpsc::channel();
        sender
            .send(CachedPostgresCommand::Close {
                response: first_response,
            })
            .unwrap();
        sender
            .send(CachedPostgresCommand::Close {
                response: second_response,
            })
            .unwrap();

        assert!(matches!(
            receive_cached_postgres_command(&receiver),
            Ok(CachedPostgresCommand::Close { .. })
        ));
        assert!(receiver.try_lock().is_ok());
        assert!(matches!(
            receive_cached_postgres_command(&receiver),
            Ok(CachedPostgresCommand::Close { .. })
        ));
    }

    #[test]
    fn failed_postgres_finish_still_joins_and_removes_session() {
        let transaction_id = format!("finish-failure-{}", std::process::id());
        let (sender, receiver) = mpsc::channel();
        drop(receiver);
        let worker = std::thread::spawn(|| Ok(()));
        postgres_sessions()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                transaction_id.clone(),
                PostgresSession {
                    dsn: "postgresql://unreachable/test".to_owned(),
                    schema: "finish_failure".to_owned(),
                    sender,
                    worker: Some(worker),
                    finishing: false,
                },
            );

        assert!(finish_postgres_session(&transaction_id, true).is_err());
        assert!(
            !postgres_sessions()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .contains_key(&transaction_id)
        );
    }

    #[test]
    fn managed_postgres_store_reuses_and_explicitly_closes_worker() {
        let context_id = format!("test-{}", std::process::id());
        let dsn = "postgresql://postgres:postgres@127.0.0.1/test";
        let first = cached_postgres_store(dsn, "test_schema", &context_id).unwrap();
        let second = cached_postgres_store(dsn, "test_schema", &context_id).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert!(
            cached_postgres_store(
                "postgresql://other:other@127.0.0.1/test",
                "test_schema",
                &context_id
            )
            .is_err()
        );

        close_cached_postgres_store(dsn, "test_schema", &context_id).unwrap();
        assert!(
            !postgres_store_cache()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .contains_key(&context_id)
        );
    }

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
    fn postgres_bridge_exposes_read_only_graph_authority_capability_operation() {
        validate_postgres_operation(&json!({
            "kind": "graph_authority_capability"
        }))
        .expect("capability discovery operation is part of the typed bridge");
        assert!(
            validate_postgres_operation(&json!({
                "kind": "graph_authority_capability",
                "write": true
            }))
            .is_err()
        );
    }

    #[test]
    fn postgres_graph_projection_page_is_optional_and_validated_at_the_bridge() {
        validate_postgres_operation(&json!({
            "kind": "graph_projection_records",
            "namespace": "conversation",
            "table": "gke_nodes",
            "after_id": "conversation-start-9",
            "limit": 100
        }))
        .expect("optional cursor must be accepted by the Python/Rust bridge");
        validate_postgres_operation(&json!({
            "kind": "graph_projection_records",
            "namespace": "conversation",
            "table": "gke_nodes",
            "limit": 100
        }))
        .expect("legacy unpaged graph reads remain accepted");
        assert!(
            validate_postgres_operation(&json!({
                "kind": "graph_projection_records",
                "namespace": "conversation",
                "table": "gke_nodes",
                "unknown_cursor": "bad"
            }))
            .is_err()
        );
    }

    #[test]
    fn postgres_bridge_accepts_typed_graph_mutation_envelope() {
        validate_postgres_operation(&json!({
            "kind": "graph_mutation",
            "model_envelope": {
                "schema_version": 1,
                "model_class": "node",
                "payload": {"id": "node-1"}
            }
        }))
        .expect("typed graph mutation envelopes must cross the Python/Rust bridge");
        assert!(
            validate_postgres_operation(&json!({
                "kind": "graph_mutation",
                "typed_envelope": {}
            }))
            .is_err()
        );
    }

    #[test]
    fn postgres_bridge_validates_atomic_graph_batch_nested_operations() {
        validate_postgres_operation(&json!({
            "kind": "batch",
            "abort": true,
            "operations": [
                {
                    "kind": "graph_mutation",
                    "namespace": "default",
                    "workspace_id": "workspace-1",
                    "graph_space": "knowledge",
                    "table": "gke_nodes",
                    "entity_kind": "node",
                    "event_id": "event-1",
                    "op": "ADD",
                    "record": {"id": "node-1"},
                    "payload": {},
                    "embedding_dim": 3,
                    "model_envelope": {
                        "schema_version": 1,
                        "model_class": "node",
                        "payload": {"id": "node-1"}
                    }
                },
                {
                    "kind": "enqueue_index_job",
                    "job_id": "job-1",
                    "namespace": "default",
                    "entity_kind": "node",
                    "entity_id": "node-1",
                    "index_kind": "node_docs",
                    "op": "UPSERT",
                    "payload_json": "{}",
                    "max_retries": 3
                }
            ]
        }))
        .expect("graph mutation and index intent must share the validated batch seam");

        assert!(
            validate_postgres_operation(&json!({
                "kind": "batch",
                "abort": true,
                "operations": [{
                    "kind": "graph_mutation",
                    "unexpected": "must fail closed"
                }]
            }))
            .is_err(),
            "unknown nested fields must be rejected before dispatch"
        );
    }

    #[test]
    fn postgres_bridge_rejects_nonportable_retry_count_before_dispatch() {
        validate_postgres_operation(&json!({
            "kind": "enqueue_index_job",
            "max_retries": i32::MAX
        }))
        .expect("the inclusive PostgreSQL INTEGER maximum is portable");
        validate_postgres_operation(&json!({
            "kind": "enqueue_index_job"
        }))
        .expect("legacy omitted retry count retains its default");

        let request = json!({
            "dsn": "postgresql://invalid/unreachable",
            "schema": "unused",
            "context_id": "unused-context",
            "operation": {
                "kind": "batch",
                "operations": [{
                    "kind": "enqueue_index_job",
                    "max_retries": i32::MAX as i64 + 1
                }]
            }
        });
        assert_eq!(
            postgres_store_json_impl(&request.to_string())
                .expect_err("invalid nested retry range must fail before DB dispatch")
                .0,
            STORE_INVALID_PAYLOAD
        );
    }

    #[test]
    fn postgres_bridge_accepts_closed_batched_graph_vector_query_contract() {
        let operation = json!({
            "kind": "graph_projection_vector_query_batch",
            "namespace": "default",
            "workspace_id": null,
            "graph_space": null,
            "table": "gke_nodes",
            "embeddings": [[1.0, 0.0], [0.0, 1.0]],
            "limit": 3,
            "metadata": {"team": "red"},
            "metric": "cosine",
            "embedding_dim": 2
        });
        validate_postgres_operation(&operation).expect("batch operation is allowlisted");
        serde_json::from_value::<PostgresStoreOperation>(operation.clone())
            .expect("batch operation matches the typed Rust contract");

        let mut invalid = operation;
        invalid["unexpected"] = json!(true);
        assert_eq!(
            validate_postgres_operation(&invalid)
                .expect_err("unknown batch fields must be rejected")
                .0,
            STORE_INVALID_PAYLOAD
        );
    }

    #[test]
    fn postgres_bridge_rejects_coercive_graph_numbers_before_dispatch() {
        let operations = [
            json!({
                "kind": "graph_mutation",
                "namespace": "default",
                "table": "gke_nodes",
                "entity_kind": "node",
                "event_id": "event-1",
                "op": "ADD",
                "record": {"id": "node-1", "embedding": [true]},
                "payload": {"id": "node-1"},
                "embedding_dim": 1
            }),
            json!({
                "kind": "graph_mutation",
                "namespace": "default",
                "table": "gke_nodes",
                "entity_kind": "node",
                "event_id": "event-1",
                "op": "ADD",
                "record": {"id": "node-1", "embedding": ["1"]},
                "payload": {"id": "node-1"},
                "embedding_dim": 1
            }),
            json!({
                "kind": "graph_mutation",
                "namespace": "default",
                "table": "gke_nodes",
                "entity_kind": "node",
                "event_id": "event-1",
                "op": "ADD",
                "record": {"id": "node-1"},
                "payload": {"id": "node-1"},
                "embedding_dim": true
            }),
            json!({
                "kind": "enqueue_index_job",
                "job_id": "job-1",
                "namespace": "default",
                "entity_kind": "node",
                "entity_id": "node-1",
                "index_kind": "node_docs",
                "op": "UPSERT",
                "max_retries": true
            }),
        ];

        for operation in operations {
            let request = json!({
                "dsn": "postgresql://invalid/unreachable",
                "schema": "unused",
                "context_id": "unused-context",
                "operation": operation
            });
            assert_eq!(
                postgres_store_json_impl(&request.to_string())
                    .expect_err("coercive numeric values must fail before DB dispatch")
                    .0,
                STORE_INVALID_PAYLOAD
            );
        }
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
