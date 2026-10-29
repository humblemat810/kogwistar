//! Behavior and compile-time contracts for independently selected stores.

use kogwistar_store::{
    GraphRecord, GraphWriteStore, IndexJobClaimFilter, IndexJobReadStore, IndexJobWriteStore,
    NewIndexJob,
};
use kogwistar_store_chroma::{ChromaGraphStore, ChromaStoreConfig};
use kogwistar_store_memory::InMemoryStore;
use kogwistar_store_postgres::PostgresStore;
use kogwistar_store_sqlite::SqliteStore;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    env,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    task::JoinHandle,
};

#[test]
fn postgres_durable_queue_can_pair_with_chroma_graph_projection() {
    fn composition_bounds<Queue, Graph>()
    where
        Queue: IndexJobWriteStore,
        Graph: GraphWriteStore,
    {
    }

    // Queue durability and semantic projection are independent capabilities;
    // they do not need to share a transaction implementation.
    composition_bounds::<PostgresStore, ChromaGraphStore>();
}

async fn fake_chroma() -> (String, JoinHandle<Vec<(String, Value)>>) {
    fake_chroma_with_upsert_statuses(vec![200]).await
}

async fn fake_chroma_with_upsert_statuses(
    upsert_statuses: Vec<u16>,
) -> (String, JoinHandle<Vec<(String, Value)>>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fixture");
    let address = listener.local_addr().expect("fixture address");
    let handle = tokio::spawn(async move {
        let mut observed = Vec::new();
        let mut upsert_statuses = upsert_statuses.into_iter();
        for _ in 0..(2 + upsert_statuses.len()) {
            let (mut stream, _) = listener.accept().await.expect("accept request");
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 4096];
            let (header_end, body_size) = loop {
                let count = stream.read(&mut chunk).await.expect("read request");
                assert!(count > 0, "client closed incomplete request");
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..index]);
                    let length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if bytes.len() >= index + 4 + length {
                        break (index, length);
                    }
                }
            };
            let headers = String::from_utf8_lossy(&bytes[..header_end]);
            let request_line = headers.lines().next().expect("request line").to_owned();
            let body = &bytes[header_end + 4..header_end + 4 + body_size];
            let payload: Value = serde_json::from_slice(body).expect("JSON request body");
            let (status, response) = if request_line.contains("/collections HTTP") {
                (
                    200,
                    json!({
                        "id": format!("collection-{}", payload["name"].as_str().unwrap_or("unknown")),
                        "metadata":{"hnsw:space":"cosine"}
                    }),
                )
            } else {
                let status = upsert_statuses.next().expect("configured upsert response");
                let response = if status < 300 {
                    json!({"ids":["node-1"]})
                } else {
                    json!({"error":"injected temporary Chroma failure"})
                };
                (status, response)
            };
            let encoded = response.to_string();
            let status_text = if status < 300 {
                "200 OK"
            } else {
                "503 Service Unavailable"
            };
            let response_head = format!(
                "HTTP/1.1 {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                encoded.len(),
            );
            stream
                .write_all(response_head.as_bytes())
                .await
                .expect("write headers");
            stream
                .write_all(encoded.as_bytes())
                .await
                .expect("write body");
            observed.push((request_line, payload));
        }
        observed
    });
    (format!("http://{address}"), handle)
}

#[tokio::test]
async fn projection_worker_composes_queue_and_chroma_without_shared_transaction() {
    let queue = InMemoryStore::new();
    queue
        .enqueue_index_job(NewIndexJob {
            job_id: "job-1".to_owned(),
            namespace: "workspace-a".to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "node-1".to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: 3,
        })
        .await
        .unwrap();

    let (endpoint, server) = fake_chroma().await;
    let graph = ChromaGraphStore::connect_entity_collections(
        ChromaStoreConfig {
            endpoint: endpoint.clone(),
            collection: "test-graph-nodes".to_owned(),
            ..Default::default()
        },
        ChromaStoreConfig {
            endpoint,
            collection: "test-graph-edges".to_owned(),
            ..Default::default()
        },
    )
    .await
    .expect("connect fake Chroma");

    let summary =
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_with_provider_input(
            &queue,
            &graph,
            1,
            30,
            Some("workspace-a"),
            |input| async move {
                assert_eq!(input.namespace, "workspace-a");
                assert_eq!(input.entity_id, "node-1");
                let mut record = GraphRecord::new("node-1");
                record.document = Some("projected document".to_owned());
                record.metadata.insert("kind".to_owned(), json!("node"));
                record.embedding = Some(vec![0.25, 0.75]);
                Ok(record)
            },
        )
        .await
        .expect("project queued item");

    assert_eq!(summary.claimed, 1);
    assert_eq!(summary.projected, 1);
    let completed = queue
        .index_jobs(
            Some("workspace-a"),
            Some("DONE"),
            Some("node"),
            Some("node-1"),
            Some("node_embedding"),
            10,
        )
        .await
        .unwrap();
    assert_eq!(completed.len(), 1);

    let requests = server.await.expect("fake Chroma server");
    let upsert = requests
        .iter()
        .find(|(line, _)| line.contains("/upsert HTTP"))
        .expect("projection writes to Chroma");
    assert!(
        upsert
            .0
            .contains("/collections/collection-test-graph-nodes/upsert")
    );
    assert_eq!(upsert.1["ids"], json!(["node-1"]));
    assert_eq!(upsert.1["documents"], json!(["projected document"]));
    assert_eq!(
        upsert.1["metadatas"][0]["_kogwistar_namespace"],
        "workspace-a"
    );
}

#[tokio::test]
async fn sqlite_metadata_queue_composes_with_chroma_graph_projection() {
    fn composition_bounds<Queue, Graph>()
    where
        Queue: IndexJobWriteStore,
        Graph: GraphWriteStore,
    {
    }
    composition_bounds::<SqliteStore, ChromaGraphStore>();

    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("system clock after Unix epoch")
        .as_nanos();
    let database = env::temp_dir().join(format!(
        "kg-sqlite-chroma-projection-{}-{nonce}.sqlite3",
        std::process::id()
    ));
    let queue = SqliteStore::open(&database).expect("open SQLite metadata store");
    queue
        .enqueue_index_job(NewIndexJob {
            job_id: "job-sqlite-chroma".to_owned(),
            namespace: "workspace-a".to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "node-1".to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: 3,
        })
        .expect("enqueue durable projection job in SQLite");
    queue
        .enqueue_index_job(NewIndexJob {
            job_id: "job-sqlite-metadata".to_owned(),
            namespace: "workspace-a".to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "node-2".to_owned(),
            index_kind: "node_refs".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: 3,
        })
        .expect("enqueue unrelated metadata job");

    let (endpoint, server) = fake_chroma_with_upsert_statuses(vec![503, 200]).await;
    let graph = ChromaGraphStore::connect_entity_collections(
        ChromaStoreConfig {
            endpoint: endpoint.clone(),
            collection: "test-sqlite-queue-nodes".to_owned(),
            ..Default::default()
        },
        ChromaStoreConfig {
            endpoint,
            collection: "test-sqlite-queue-edges".to_owned(),
            ..Default::default()
        },
    )
    .await
    .expect("connect fake Chroma");
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let first_provider_calls = Arc::clone(&provider_calls);
    let first_summary =
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_filtered_with_provider_input(
            &queue,
            &graph,
            1,
            30,
            IndexJobClaimFilter {
                namespace: Some("workspace-a".to_owned()),
                index_kinds: vec!["node_embedding".to_owned()],
            },
            move |input| {
                first_provider_calls.fetch_add(1, Ordering::SeqCst);
                async move {
                    assert_eq!(input.namespace, "workspace-a");
                    assert_eq!(input.entity_id, "node-1");
                    let mut record = GraphRecord::new("node-1");
                    record.document = Some("SQLite metadata, Chroma graph".to_owned());
                    record.metadata.insert("kind".to_owned(), json!("node"));
                    record.embedding = Some(vec![0.25, 0.75]);
                    Ok(record)
                }
            },
        )
        .await
        .expect("first SQLite-to-Chroma projection attempt");

    assert_eq!(first_summary.retried, 1);
    assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
    std::thread::sleep(Duration::from_millis(1_100));
    let retry_provider_calls = Arc::clone(&provider_calls);
    let retry_summary =
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_filtered_with_provider_input(
            &queue,
            &graph,
            1,
            30,
            IndexJobClaimFilter {
                namespace: Some("workspace-a".to_owned()),
                index_kinds: vec!["node_embedding".to_owned()],
            },
            move |_input| {
                retry_provider_calls.fetch_add(1, Ordering::SeqCst);
                async {
                    let mut record = GraphRecord::new("node-1");
                    record.document = Some("unexpected provider replay".to_owned());
                    record.embedding = Some(vec![0.0, 1.0]);
                    Ok(record)
                }
            },
        )
        .await
        .expect("reuse SQLite-admitted result after Chroma recovers");
    assert_eq!(retry_summary.reused, 1);
    assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
    let completed = queue
        .index_jobs(
            Some("workspace-a"),
            Some("DONE"),
            Some("node"),
            Some("node-1"),
            Some("node_embedding"),
            10,
        )
        .await
        .expect("read completed SQLite queue item");
    assert_eq!(completed.len(), 1);
    let unrelated_pending = queue
        .index_jobs(
            Some("workspace-a"),
            Some("PENDING"),
            Some("node"),
            Some("node-2"),
            Some("node_refs"),
            10,
        )
        .await
        .expect("read unrelated metadata job");
    assert_eq!(unrelated_pending.len(), 1);

    let requests = server.await.expect("fake Chroma server");
    assert_eq!(requests.len(), 4);
    let upserts = requests
        .iter()
        .filter(|(line, _)| line.contains("/upsert HTTP"))
        .collect::<Vec<_>>();
    assert_eq!(upserts.len(), 2);
    assert!(
        upserts.iter().all(
            |(line, _)| line.contains("/collections/collection-test-sqlite-queue-nodes/upsert")
        )
    );
    assert!(
        upserts
            .iter()
            .all(|(_, request)| request["ids"] == json!(["node-1"]))
    );
    assert_eq!(
        upserts[0].1["documents"],
        json!(["SQLite metadata, Chroma graph"])
    );
    assert_eq!(upserts[1].1["documents"], upserts[0].1["documents"]);
    assert_eq!(upserts[1].1["embeddings"], upserts[0].1["embeddings"]);
    assert_eq!(
        upserts[1].1["metadatas"][0]["_kogwistar_namespace"],
        "workspace-a"
    );

    drop(graph);
    drop(queue);
    let _ = std::fs::remove_file(&database);
    let _ = std::fs::remove_file(format!("{}.kogwistar-authority.lock", database.display()));
    let _ = std::fs::remove_file(format!("{}-wal", database.display()));
    let _ = std::fs::remove_file(format!("{}-shm", database.display()));
}

#[tokio::test]
async fn rust_projection_worker_applies_python_wire_action_from_sqlite_queue() {
    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("system clock after Unix epoch")
        .as_nanos();
    let database = env::temp_dir().join(format!(
        "kg-python-action-rust-projection-{}-{nonce}.sqlite3",
        std::process::id()
    ));
    let queue = SqliteStore::open(&database).expect("open shared SQLite queue");
    queue
        .enqueue_index_job(NewIndexJob {
            job_id: "projection-job".to_owned(),
            namespace: "workspace".to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "node-1".to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: 3,
        })
        .expect("enqueue projection job");
    let claim = queue
        .claim_index_jobs(1, 30, Some("workspace"))
        .expect("claim projection job")
        .remove(0);
    let action = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-projection-action-v1.json"
    ))
    .trim();
    let digest = hex::encode(Sha256::digest(action.as_bytes()));
    let accepted = queue
        .accept_index_job_result(
            &claim.job_id,
            claim.claim_token.as_deref().expect("claim token"),
            action,
            &digest,
        )
        .expect("admit Python-compatible projection action");
    assert_eq!(accepted.status, "accepted");

    // Preserve the accepted action while making it available to a replacement
    // worker, modelling a crash after result admission and before projection.
    queue
        .bump_retry_and_requeue(
            &claim.job_id,
            "simulate worker loss after result admission",
            0,
            claim.claim_token.as_deref(),
        )
        .expect("release crashed worker claim");

    let (endpoint, server) = fake_chroma().await;
    let graph = ChromaGraphStore::connect_entity_collections(
        ChromaStoreConfig {
            endpoint: endpoint.clone(),
            collection: "test-python-action-rust-nodes".to_owned(),
            ..Default::default()
        },
        ChromaStoreConfig {
            endpoint,
            collection: "test-python-action-rust-edges".to_owned(),
            ..Default::default()
        },
    )
    .await
    .expect("connect fake Chroma");

    let summary =
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_with_provider_input(
            &queue,
            &graph,
            1,
            30,
            Some("workspace"),
            move |_input| async { panic!("accepted Python action must not call provider") },
        )
        .await
        .expect("apply Python-compatible action through Rust worker");

    assert_eq!(summary.claimed, 1);
    assert_eq!(summary.reused, 1);
    assert_eq!(
        queue
            .index_jobs(
                Some("workspace"),
                Some("DONE"),
                Some("node"),
                Some("node-1"),
                Some("node_embedding"),
                10,
            )
            .await
            .expect("read completed queue item")
            .len(),
        1
    );

    let requests = server.await.expect("fake Chroma server");
    let upsert = requests
        .iter()
        .find(|(line, _)| line.contains("/upsert HTTP"))
        .expect("Rust worker writes accepted Python action to Chroma");
    assert_eq!(upsert.1["ids"], json!(["node-1"]));
    assert_eq!(upsert.1["documents"], json!(["current node"]));
    assert_eq!(upsert.1["embeddings"], json!([[0.25, 0.75]]));
    assert_eq!(
        upsert.1["metadatas"][0]["_kogwistar_namespace"],
        "workspace"
    );

    drop(graph);
    drop(queue);
    let _ = std::fs::remove_file(&database);
    let _ = std::fs::remove_file(format!("{}.kogwistar-authority.lock", database.display()));
    let _ = std::fs::remove_file(format!("{}-wal", database.display()));
    let _ = std::fs::remove_file(format!("{}-shm", database.display()));
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN and an explicit integration-test run"]
async fn postgres_queue_projects_to_chroma_without_cross_store_transaction() {
    let dsn = env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("set KOGWISTAR_TEST_PG_DSN when running this ignored integration test");
    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("system clock after Unix epoch")
        .as_nanos();
    let schema = format!("kg_proj_chroma_{}_{}", std::process::id(), nonce);
    let queue = PostgresStore::from_dsn(&dsn, &schema).expect("construct isolated queue store");
    queue.ensure_schema().await.expect("create queue schema");
    queue
        .enqueue_index_job(NewIndexJob {
            job_id: "job-postgres-chroma".to_owned(),
            namespace: "workspace-a".to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "node-1".to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: 3,
        })
        .await
        .expect("enqueue in PostgreSQL");

    let (endpoint, server) = fake_chroma_with_upsert_statuses(vec![503, 200]).await;
    let graph = ChromaGraphStore::connect_entity_collections(
        ChromaStoreConfig {
            endpoint: endpoint.clone(),
            collection: "test-postgres-queue-nodes".to_owned(),
            ..Default::default()
        },
        ChromaStoreConfig {
            endpoint,
            collection: "test-postgres-queue-edges".to_owned(),
            ..Default::default()
        },
    )
    .await
    .expect("connect fake Chroma");
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let first_provider_calls = Arc::clone(&provider_calls);
    let first_summary =
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_with_provider_input(
            &queue,
            &graph,
            1,
            30,
            Some("workspace-a"),
            move |_input| {
                first_provider_calls.fetch_add(1, Ordering::SeqCst);
                async {
                    let mut record = GraphRecord::new("node-1");
                    record.document = Some("durable queue projection".to_owned());
                    record.metadata.insert("kind".to_owned(), json!("node"));
                    record.embedding = Some(vec![0.5, 0.5]);
                    Ok(record)
                }
            },
        )
        .await
        .expect("first PostgreSQL-to-Chroma projection attempt");
    assert_eq!(first_summary.retried, 1);
    assert_eq!(provider_calls.load(Ordering::SeqCst), 1);

    // Retry backoff is one second for the first failure. The accepted result
    // must survive the delay so the provider is not invoked again.
    std::thread::sleep(Duration::from_millis(1_100));
    let second_provider_calls = Arc::clone(&provider_calls);
    let second_summary =
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_with_provider_input(
            &queue,
            &graph,
            1,
            30,
            Some("workspace-a"),
            move |_job| {
                second_provider_calls.fetch_add(1, Ordering::SeqCst);
                async {
                    let mut record = GraphRecord::new("node-1");
                    record.document = Some("unexpected provider replay".to_owned());
                    record.embedding = Some(vec![0.0, 1.0]);
                    Ok(record)
                }
            },
        )
        .await
        .expect("replay accepted result after Chroma recovers");
    assert_eq!(second_summary.reused, 1);
    assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        queue
            .index_jobs(
                Some("workspace-a"),
                Some("DONE"),
                Some("node"),
                Some("node-1"),
                Some("node_embedding"),
                10,
            )
            .await
            .unwrap()
            .len(),
        1
    );
    let requests = server.await.expect("fake Chroma server");
    assert_eq!(requests.len(), 4);
    let upserts = requests
        .iter()
        .filter(|(line, _)| line.contains("/upsert HTTP"))
        .collect::<Vec<_>>();
    assert_eq!(upserts.len(), 2);
    assert_eq!(
        upserts[0].1["documents"],
        json!(["durable queue projection"])
    );
    assert_eq!(
        upserts[1].1["documents"],
        json!(["durable queue projection"])
    );

    drop(graph);
    drop(queue);
    let (client, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .expect("connect to remove isolated schema");
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
        .batch_execute(&format!("DROP SCHEMA IF EXISTS \"{schema}\" CASCADE"))
        .await
        .expect("drop isolated queue schema");
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN and an explicit integration-test run"]
async fn postgres_rust_worker_applies_python_wire_action_to_chroma() {
    let dsn = env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("set KOGWISTAR_TEST_PG_DSN when running this ignored integration test");
    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("system clock after Unix epoch")
        .as_nanos();
    let schema = format!("kg_py_action_{}_{}", std::process::id(), nonce);
    let queue = PostgresStore::from_dsn(&dsn, &schema).expect("construct isolated queue store");
    queue.ensure_schema().await.expect("create queue schema");
    let job_id = format!("python-action-{nonce}");
    let namespace = "workspace-python";
    queue
        .enqueue_index_job(NewIndexJob {
            job_id: job_id.clone(),
            namespace: namespace.to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "node-1".to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: 3,
        })
        .await
        .expect("enqueue PostgreSQL projection job");

    let claim = queue
        .claim_index_jobs(1, 30, Some(namespace))
        .await
        .expect("claim PostgreSQL projection job")
        .remove(0);
    let mut action: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-projection-action-v1.json"
    )))
    .expect("shared Python projection action fixture is valid");
    action["job_id"] = json!(job_id);
    action["namespace"] = json!(namespace);
    let action_json = action.to_string();
    let digest = hex::encode(Sha256::digest(action_json.as_bytes()));
    let claim_token = claim.claim_token.as_deref().expect("claim token");
    let accepted = queue
        .accept_index_job_result(&claim.job_id, claim_token, &action_json, &digest)
        .await
        .expect("admit shared Python-compatible projection action");
    assert_eq!(accepted.status, "accepted");
    queue
        .bump_retry_and_requeue(
            &claim.job_id,
            "simulate worker loss after Python result admission",
            0,
            Some(claim_token),
        )
        .await
        .expect("release expired worker claim while preserving accepted action");

    let (endpoint, server) = fake_chroma().await;
    let graph = ChromaGraphStore::connect_entity_collections(
        ChromaStoreConfig {
            endpoint: endpoint.clone(),
            collection: "test-postgres-python-action-nodes".to_owned(),
            ..Default::default()
        },
        ChromaStoreConfig {
            endpoint,
            collection: "test-postgres-python-action-edges".to_owned(),
            ..Default::default()
        },
    )
    .await
    .expect("connect fake Chroma");

    let summary =
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_with_provider_input(
            &queue,
            &graph,
            1,
            30,
            Some(namespace),
            move |_input| async { panic!("accepted Python action must not call provider") },
        )
        .await
        .expect("apply shared Python-compatible action with Rust worker");

    assert_eq!(summary.claimed, 1);
    assert_eq!(summary.reused, 1);
    let done = queue
        .index_jobs(
            Some(namespace),
            Some("DONE"),
            Some("node"),
            Some("node-1"),
            Some("node_embedding"),
            10,
        )
        .await
        .expect("read completed PostgreSQL queue item");
    assert_eq!(done.len(), 1);
    assert_eq!(
        done[0].accepted_result_json.as_deref(),
        Some(action_json.as_str())
    );

    let requests = server.await.expect("fake Chroma server");
    let upsert = requests
        .iter()
        .find(|(line, _)| line.contains("/upsert HTTP"))
        .expect("Rust worker applies accepted Python action to Chroma");
    assert_eq!(upsert.1["ids"], json!(["node-1"]));
    assert_eq!(upsert.1["documents"], json!(["current node"]));
    assert_eq!(upsert.1["embeddings"], json!([[0.25, 0.75]]));
    assert_eq!(upsert.1["metadatas"][0]["_kogwistar_namespace"], namespace);

    drop(graph);
    drop(queue);
    let (client, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .expect("connect to remove isolated schema");
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
        .batch_execute(&format!("DROP SCHEMA IF EXISTS \"{schema}\" CASCADE"))
        .await
        .expect("drop isolated queue schema");
}
