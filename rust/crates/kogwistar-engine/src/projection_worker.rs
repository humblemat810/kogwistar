//! Durable graph-projection worker for external graph/vector stores.
//!
//! The queue store is the authority for job state. A provider result is saved
//! before the graph store is changed, and the job is acknowledged only after
//! the idempotent upsert/delete succeeds. This lets a retry reuse the accepted
//! result after a crash between admission and projection.

use kogwistar_store::{
    GraphRecord, GraphWriteStore, IndexJob, IndexJobProviderInput, IndexJobWriteStore, StoreError,
    StoreResult,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::future::Future;

const ACTION_VERSION: i64 = 1;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectionWorkerSummary {
    pub claimed: usize,
    pub projected: usize,
    pub reused: usize,
    pub claim_lost: usize,
    pub retried: usize,
    pub failed: usize,
}

#[derive(Clone, Debug, PartialEq)]
enum ProjectionAction {
    Upsert(GraphRecord),
    Delete,
}

fn projection_error(message: impl Into<String>) -> StoreError {
    StoreError::InvalidEntityEventPayload {
        message: message.into(),
    }
}

fn backend_error(error: impl ToString) -> StoreError {
    StoreError::Backend {
        backend: "graph-projection-worker".to_owned(),
        message: error.to_string(),
    }
}

fn result_digest(result_json: &str) -> String {
    hex::encode(Sha256::digest(result_json.as_bytes()))
}

fn action_value(job: &IndexJob, record: Option<&GraphRecord>) -> Value {
    let operation = if job.op.eq_ignore_ascii_case("DELETE") {
        "DELETE"
    } else {
        "UPSERT"
    };
    json!({
        "version": ACTION_VERSION,
        "job_id": job.job_id,
        "namespace": job.namespace,
        "entity_kind": job.entity_kind,
        "entity_id": job.entity_id,
        "index_kind": job.index_kind,
        "operation": operation,
        "record": record.map(|record| json!({
            "id": record.id,
            "document": record.document,
            "metadata": record.metadata,
            "embedding": record.embedding,
        })),
    })
}

fn parse_action(job: &IndexJob, serialized: &str) -> StoreResult<ProjectionAction> {
    let value: Value = serde_json::from_str(serialized)
        .map_err(|error| projection_error(format!("invalid projection action JSON: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| projection_error("projection action must be an object"))?;
    let expected_fields = [
        "version",
        "job_id",
        "namespace",
        "entity_kind",
        "entity_id",
        "index_kind",
        "operation",
        "record",
    ];
    if object.len() != expected_fields.len()
        || expected_fields
            .iter()
            .any(|field| !object.contains_key(*field))
    {
        return Err(projection_error(
            "projection action has missing or unknown fields",
        ));
    }
    if job.namespace.trim().is_empty()
        || !matches!(job.entity_kind.as_str(), "node" | "edge")
        || job.entity_id.is_empty()
        || object.get("version").and_then(Value::as_i64) != Some(ACTION_VERSION)
        || object.get("job_id").and_then(Value::as_str) != Some(job.job_id.as_str())
        || object.get("namespace").and_then(Value::as_str) != Some(job.namespace.as_str())
        || object.get("entity_kind").and_then(Value::as_str) != Some(job.entity_kind.as_str())
        || object.get("entity_id").and_then(Value::as_str) != Some(job.entity_id.as_str())
        || object.get("index_kind").and_then(Value::as_str) != Some(job.index_kind.as_str())
    {
        return Err(projection_error(
            "projection action identity differs from its claimed queue job",
        ));
    }
    match (
        job.op.to_ascii_uppercase().as_str(),
        object.get("operation").and_then(Value::as_str),
    ) {
        ("DELETE", Some("DELETE")) => {
            if !object.get("record").is_some_and(Value::is_null) {
                return Err(projection_error("delete action must not contain a record"));
            }
            Ok(ProjectionAction::Delete)
        }
        ("UPSERT" | "ADD" | "REPLACE", Some("UPSERT")) => {
            let record = object
                .get("record")
                .and_then(Value::as_object)
                .ok_or_else(|| projection_error("upsert action requires a record object"))?;
            let expected_record_fields = ["id", "document", "metadata", "embedding"];
            if record.len() != expected_record_fields.len()
                || expected_record_fields
                    .iter()
                    .any(|field| !record.contains_key(*field))
            {
                return Err(projection_error(
                    "projection record has missing or unknown fields",
                ));
            }
            let id = record
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| projection_error("projection record id must be a string"))?;
            if id != job.entity_id || id.is_empty() {
                return Err(projection_error(
                    "projection record id differs from its queue job",
                ));
            }
            let document = match record.get("document") {
                Some(Value::Null) => None,
                Some(Value::String(value)) => Some(value.clone()),
                _ => {
                    return Err(projection_error(
                        "projection document must be string or null",
                    ));
                }
            };
            let metadata = record
                .get("metadata")
                .and_then(Value::as_object)
                .cloned()
                .ok_or_else(|| projection_error("projection metadata must be an object"))?;
            if metadata.contains_key("_kogwistar_namespace") {
                return Err(projection_error(
                    "projection metadata must not override the namespace field",
                ));
            }
            let embedding = parse_embedding(record.get("embedding"))?;
            Ok(ProjectionAction::Upsert(GraphRecord {
                id: id.to_owned(),
                document,
                metadata,
                embedding,
            }))
        }
        _ => Err(projection_error(
            "projection operation is unsupported or differs from its queue job",
        )),
    }
}

fn parse_embedding(value: Option<&Value>) -> StoreResult<Option<Vec<f32>>> {
    match value {
        Some(Value::Null) => Ok(None),
        Some(Value::Array(values)) => {
            let mut embedding = Vec::with_capacity(values.len());
            for value in values {
                let component = value
                    .as_f64()
                    .ok_or_else(|| projection_error("embedding components must be numbers"))?
                    as f32;
                if !component.is_finite() {
                    return Err(StoreError::NonFiniteVector);
                }
                embedding.push(component);
            }
            Ok(Some(embedding))
        }
        _ => Err(projection_error("embedding must be an array or null")),
    }
}

async fn retry_or_fail<Q: IndexJobWriteStore>(
    queue: &Q,
    job: &IndexJob,
    claim_token: &str,
    message: &str,
    summary: &mut ProjectionWorkerSummary,
) -> StoreResult<()> {
    let next_retry = job.retry_count + 1;
    if next_retry < job.max_retries {
        queue
            .bump_retry_and_requeue(
                &job.job_id,
                message,
                2_i64.pow(next_retry.clamp(1, 8) as u32 - 1),
                Some(claim_token),
            )
            .await?;
        summary.retried += 1;
    } else {
        queue
            .mark_index_job_failed(&job.job_id, message, true, Some(claim_token))
            .await?;
        summary.failed += 1;
    }
    Ok(())
}

/// Project one batch of durable queue jobs into a graph store.
///
/// The provider prepares the complete record (including any embedding) only
/// for non-delete jobs. Its result is durably admitted before the projection
/// write; retries reuse that result, and queue completion follows projection.
pub async fn run_graph_projection_jobs_once<Q, G, F, Fut>(
    queue: &Q,
    graph: &G,
    limit: usize,
    lease_seconds: i64,
    namespace: Option<&str>,
    provider: F,
) -> StoreResult<ProjectionWorkerSummary>
where
    Q: IndexJobWriteStore,
    G: GraphWriteStore,
    F: FnMut(IndexJob) -> Fut,
    Fut: Future<Output = StoreResult<GraphRecord>>,
{
    run_graph_projection_jobs_once_inner(queue, graph, limit, lease_seconds, namespace, provider)
        .await
}

/// Safe provider boundary that withholds the queue claim token and transition
/// metadata. The full-row API remains available for source compatibility.
pub async fn run_graph_projection_jobs_once_with_provider_input<Q, G, F, Fut>(
    queue: &Q,
    graph: &G,
    limit: usize,
    lease_seconds: i64,
    namespace: Option<&str>,
    mut provider: F,
) -> StoreResult<ProjectionWorkerSummary>
where
    Q: IndexJobWriteStore,
    G: GraphWriteStore,
    F: FnMut(IndexJobProviderInput) -> Fut,
    Fut: Future<Output = StoreResult<GraphRecord>>,
{
    run_graph_projection_jobs_once_inner(
        queue,
        graph,
        limit,
        lease_seconds,
        namespace,
        move |job| provider(IndexJobProviderInput::from(&job)),
    )
    .await
}

async fn run_graph_projection_jobs_once_inner<Q, G, F, Fut>(
    queue: &Q,
    graph: &G,
    limit: usize,
    lease_seconds: i64,
    namespace: Option<&str>,
    mut provider: F,
) -> StoreResult<ProjectionWorkerSummary>
where
    Q: IndexJobWriteStore,
    G: GraphWriteStore,
    F: FnMut(IndexJob) -> Fut,
    Fut: Future<Output = StoreResult<GraphRecord>>,
{
    let jobs = queue
        .claim_index_jobs(limit, lease_seconds, namespace)
        .await?;
    let mut summary = ProjectionWorkerSummary {
        claimed: jobs.len(),
        ..ProjectionWorkerSummary::default()
    };

    for job in jobs {
        let Some(claim_token) = job.claim_token.as_deref() else {
            return Err(backend_error("claimed index job has no claim token"));
        };
        let is_replay = job.accepted_result_json.is_some();
        let (action_json, accepted_digest) = if let Some(existing) =
            job.accepted_result_json.as_deref()
        {
            (
                existing.to_owned(),
                job.accepted_result_sha256.as_deref().map(str::to_owned),
            )
        } else {
            let candidate = if job.op.eq_ignore_ascii_case("DELETE") {
                None
            } else if matches!(
                job.op.to_ascii_uppercase().as_str(),
                "UPSERT" | "ADD" | "REPLACE"
            ) {
                match provider(job.clone()).await {
                    Ok(record) => Some(record),
                    Err(error) => {
                        retry_or_fail(queue, &job, claim_token, &error.to_string(), &mut summary)
                            .await?;
                        continue;
                    }
                }
            } else {
                queue
                    .mark_index_job_failed(
                        &job.job_id,
                        "unsupported graph projection operation",
                        true,
                        Some(claim_token),
                    )
                    .await?;
                summary.failed += 1;
                continue;
            };
            let candidate_json = action_value(&job, candidate.as_ref()).to_string();
            if let Err(error) = parse_action(&job, &candidate_json) {
                queue
                    .mark_index_job_failed(&job.job_id, &error.to_string(), true, Some(claim_token))
                    .await?;
                summary.failed += 1;
                continue;
            }
            match queue
                .accept_index_job_result(
                    &job.job_id,
                    claim_token,
                    &candidate_json,
                    &result_digest(&candidate_json),
                )
                .await?
            {
                admission if admission.status == "accepted" || admission.status == "existing" => {
                    let result_json = admission.result_json.ok_or_else(|| {
                        backend_error("queue accepted a projection action without returning it")
                    })?;
                    (result_json, admission.result_sha256)
                }
                admission if admission.status == "rejected" => {
                    summary.claim_lost += 1;
                    continue;
                }
                admission => {
                    return Err(backend_error(format!(
                        "unknown projection result admission status {:?}",
                        admission.status
                    )));
                }
            }
        };

        let digest_matches = accepted_digest
            .as_deref()
            .is_some_and(|expected| expected == result_digest(&action_json));
        if !digest_matches {
            queue
                .mark_index_job_failed(
                    &job.job_id,
                    "accepted projection action is missing or has an invalid SHA-256 digest",
                    true,
                    Some(claim_token),
                )
                .await?;
            summary.failed += 1;
            continue;
        }

        let action = match parse_action(&job, &action_json) {
            Ok(action) => action,
            Err(error) => {
                queue
                    .mark_index_job_failed(&job.job_id, &error.to_string(), true, Some(claim_token))
                    .await?;
                summary.failed += 1;
                continue;
            }
        };
        let projection = match action {
            ProjectionAction::Upsert(record) => {
                graph.upsert_graph_record(&job.namespace, record).await
            }
            ProjectionAction::Delete => {
                // A repeated delete after lost acknowledgement is already the
                // desired state, so `false` is still a successful projection.
                graph
                    .delete_graph_record(&job.namespace, &job.entity_id)
                    .await
                    .map(|_| ())
            }
        };
        if let Err(error) = projection {
            retry_or_fail(queue, &job, claim_token, &error.to_string(), &mut summary).await?;
            continue;
        }
        if queue
            .mark_index_job_done(&job.job_id, Some(claim_token))
            .await?
        {
            if is_replay {
                summary.reused += 1;
            } else {
                summary.projected += 1;
            }
        } else {
            summary.claim_lost += 1;
        }
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kogwistar_store::{
        DistanceMetric, GraphReadStore, IndexJobReadStore, MetadataFilter, NewIndexJob,
        VectorMatch, VectorQuery,
    };
    use kogwistar_store_memory::InMemoryStore;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    #[derive(Clone)]
    struct FailOnceGraph {
        inner: InMemoryStore,
        fail_next_upsert: Arc<AtomicBool>,
    }

    impl GraphReadStore for FailOnceGraph {
        async fn graph_record(
            &self,
            namespace: &str,
            id: &str,
        ) -> StoreResult<Option<GraphRecord>> {
            self.inner.graph_record(namespace, id).await
        }

        async fn graph_records(
            &self,
            namespace: &str,
            metadata: &MetadataFilter,
        ) -> StoreResult<Vec<GraphRecord>> {
            self.inner.graph_records(namespace, metadata).await
        }

        async fn vector_query(
            &self,
            namespace: &str,
            query: &VectorQuery,
        ) -> StoreResult<Vec<VectorMatch>> {
            self.inner.vector_query(namespace, query).await
        }
    }

    impl GraphWriteStore for FailOnceGraph {
        async fn upsert_graph_record(
            &self,
            namespace: &str,
            record: GraphRecord,
        ) -> StoreResult<()> {
            if self.fail_next_upsert.swap(false, Ordering::SeqCst) {
                return Err(StoreError::Backend {
                    backend: "test-graph".to_owned(),
                    message: "injected projection failure".to_owned(),
                });
            }
            self.inner.upsert_graph_record(namespace, record).await
        }

        async fn delete_graph_record(&self, namespace: &str, id: &str) -> StoreResult<bool> {
            self.inner.delete_graph_record(namespace, id).await
        }
    }

    async fn enqueue(queue: &InMemoryStore, op: &str) {
        queue
            .enqueue_index_job(NewIndexJob {
                job_id: "projection-job".to_owned(),
                namespace: "workspace".to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: "node-1".to_owned(),
                index_kind: "node_embedding".to_owned(),
                op: op.to_owned(),
                payload_json: None,
                max_retries: 3,
            })
            .await
            .unwrap();
    }

    fn candidate() -> GraphRecord {
        let mut record = GraphRecord::new("node-1");
        record.document = Some("current node".to_owned());
        record.metadata.insert("kind".to_owned(), json!("node"));
        record.embedding = Some(vec![0.25, 0.75]);
        record
    }

    #[tokio::test]
    async fn projection_action_matches_shared_python_wire_fixture() {
        let queue = InMemoryStore::new();
        enqueue(&queue, "UPSERT").await;
        let job = queue
            .claim_index_jobs(1, 30, Some("workspace"))
            .await
            .unwrap()
            .remove(0);
        let fixture = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-projection-action-v1.json"
        ))
        .trim();

        assert_eq!(action_value(&job, Some(&candidate())).to_string(), fixture);
        assert!(parse_action(&job, fixture).is_ok());
    }

    #[tokio::test]
    async fn accepted_projection_result_is_reused_after_failed_write() {
        let queue = InMemoryStore::new();
        enqueue(&queue, "UPSERT").await;
        let graph = FailOnceGraph {
            inner: InMemoryStore::new(),
            fail_next_upsert: Arc::new(AtomicBool::new(true)),
        };
        let provider_calls = Arc::new(AtomicUsize::new(0));
        let calls = provider_calls.clone();
        let first = run_graph_projection_jobs_once_with_provider_input(
            &queue,
            &graph,
            1,
            30,
            Some("workspace"),
            move |_input| {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(candidate()) }
            },
        )
        .await
        .unwrap();
        assert_eq!(first.retried, 1);
        assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
        let saved = queue
            .index_jobs(None, Some("PENDING"), None, None, None, 10)
            .await
            .unwrap();
        assert_eq!(saved.len(), 1);
        assert!(saved[0].accepted_result_json.is_some());

        std::thread::sleep(std::time::Duration::from_millis(1_100));
        let second = run_graph_projection_jobs_once_with_provider_input(
            &queue,
            &graph,
            1,
            30,
            Some("workspace"),
            move |_job| async { panic!("accepted projection must not rerun provider") },
        )
        .await
        .unwrap();
        assert_eq!(second.reused, 1);
        assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            graph
                .graph_record("workspace", "node-1")
                .await
                .unwrap()
                .unwrap()
                .document
                .as_deref(),
            Some("current node")
        );
        let done = queue
            .index_jobs(None, Some("DONE"), None, None, None, 10)
            .await
            .unwrap();
        assert_eq!(done.len(), 1);
    }

    #[tokio::test]
    async fn delete_projection_is_idempotent_and_does_not_call_provider() {
        let queue = InMemoryStore::new();
        enqueue(&queue, "DELETE").await;
        let graph = InMemoryStore::new();
        let summary = run_graph_projection_jobs_once_with_provider_input(
            &queue,
            &graph,
            1,
            30,
            Some("workspace"),
            move |_job| async { panic!("delete jobs do not need provider output") },
        )
        .await
        .unwrap();
        assert_eq!(summary.projected, 1);
        assert_eq!(summary.failed, 0);
        assert_eq!(
            graph
                .vector_query(
                    "workspace",
                    &VectorQuery {
                        embedding: vec![1.0, 0.0],
                        limit: 10,
                        metadata: MetadataFilter::default(),
                        metric: DistanceMetric::Cosine,
                    },
                )
                .await
                .unwrap(),
            Vec::<VectorMatch>::new()
        );
    }

    #[tokio::test]
    async fn accepted_action_with_wrong_scope_or_entity_fails_closed() {
        let queue = InMemoryStore::new();
        enqueue(&queue, "UPSERT").await;
        let claimed = queue
            .claim_index_jobs(1, 30, Some("workspace"))
            .await
            .unwrap()
            .remove(0);
        let action = json!({
            "version": ACTION_VERSION,
            "job_id": claimed.job_id,
            "namespace": "other-workspace",
            "entity_kind": claimed.entity_kind,
            "entity_id": claimed.entity_id,
            "index_kind": claimed.index_kind,
            "operation": "UPSERT",
            "record": {"id":"node-1", "document":"x", "metadata":{}, "embedding":null}
        })
        .to_string();
        let token = claimed.claim_token.unwrap();
        queue
            .accept_index_job_result("projection-job", &token, &action, &result_digest(&action))
            .await
            .unwrap();
        queue
            .bump_retry_and_requeue("projection-job", "test retry", 0, Some(&token))
            .await
            .unwrap();

        let summary = run_graph_projection_jobs_once_with_provider_input(
            &queue,
            &InMemoryStore::new(),
            1,
            30,
            Some("workspace"),
            move |_job| async { panic!("invalid accepted result must not call provider") },
        )
        .await
        .unwrap();
        assert_eq!(summary.failed, 1);
        assert!(
            queue
                .index_jobs(None, Some("FAILED"), None, None, None, 10)
                .await
                .unwrap()
                .iter()
                .any(|job| job.job_id == "projection-job")
        );
    }

    #[tokio::test]
    async fn accepted_action_with_mismatched_digest_fails_before_graph_write() {
        let queue = InMemoryStore::new();
        enqueue(&queue, "UPSERT").await;
        let claimed = queue
            .claim_index_jobs(1, 30, Some("workspace"))
            .await
            .unwrap()
            .remove(0);
        let action = action_value(&claimed, Some(&candidate())).to_string();
        let token = claimed.claim_token.unwrap();
        queue
            .accept_index_job_result("projection-job", &token, &action, "not-the-action-digest")
            .await
            .unwrap();
        queue
            .bump_retry_and_requeue("projection-job", "simulate restart", 0, Some(&token))
            .await
            .unwrap();

        let graph = InMemoryStore::new();
        let summary = run_graph_projection_jobs_once_with_provider_input(
            &queue,
            &graph,
            1,
            30,
            Some("workspace"),
            move |_job| async { panic!("invalid accepted result must not call provider") },
        )
        .await
        .unwrap();

        assert_eq!(summary.failed, 1);
        assert!(
            graph
                .graph_record("workspace", "node-1")
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(
            queue
                .index_jobs(None, Some("FAILED"), None, None, None, 10)
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn provider_record_with_wrong_identity_is_rejected_before_admission() {
        let queue = InMemoryStore::new();
        enqueue(&queue, "UPSERT").await;
        let summary = run_graph_projection_jobs_once(
            &queue,
            &InMemoryStore::new(),
            1,
            30,
            Some("workspace"),
            move |_job| async {
                let mut record = candidate();
                record.id = "different-node".to_owned();
                Ok(record)
            },
        )
        .await
        .unwrap();

        assert_eq!(summary.failed, 1);
        let failed = queue
            .index_jobs(None, Some("FAILED"), None, None, None, 10)
            .await
            .unwrap();
        assert_eq!(failed.len(), 1);
        assert!(failed[0].accepted_result_json.is_none());
    }
}
